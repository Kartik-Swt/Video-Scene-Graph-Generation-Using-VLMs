from __future__ import annotations
import re
import json
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional, Protocol

import cv2
import numpy as np
import networkx as nx

def select_keyframes(video_path: str,
                     max_frames: int = 8,
                     min_gap_sec: float = 1.0,
                     hist_thresh: float = 0.35) -> list[tuple[int, float, np.ndarray]]:
    """
    Pick visually distinct keyframes via color-histogram scene-change detection.

    Returns list of (frame_index, timestamp_sec, BGR frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    keyframes = []
    prev_hist = None
    last_kept_t = -1e9
    idx = 0
    # stride: don't inspect every single frame for very long videos
    stride = max(1, int(fps // 3))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        t = idx / fps
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        take = False
        if prev_hist is None:
            take = True
        else:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > hist_thresh and (t - last_kept_t) >= min_gap_sec:
                take = True
        if take:
            keyframes.append((idx, t, frame.copy()))
            prev_hist = hist
            last_kept_t = t
        idx += 1

    cap.release()

    # if scene detection found too many, keep the most spread-out ones
    if len(keyframes) > max_frames:
        pick = np.linspace(0, len(keyframes) - 1, max_frames).astype(int)
        keyframes = [keyframes[i] for i in pick]
    # if too few (static video), pad with uniform samples
    if len(keyframes) < 2 and total > 0:
        cap = cv2.VideoCapture(video_path)
        for f in np.linspace(0, total - 1, min(max_frames, 3)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, fr = cap.read()
            if ok:
                keyframes.append((int(f), f / fps, fr))
        cap.release()
    return keyframes


class VLMBackend(Protocol):
    def generate(self, frames_bgr: list[np.ndarray], prompt: str,
                 max_new_tokens: int = 512) -> str:
        """Return the raw text the VLM produced for these frames + prompt."""
        ...


class QwenVLBackend:
    """
    Real backend using Qwen2.5-VL locally on GPU.
    Loads lazily so this file imports fine on a machine without the model.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda", max_pixels: int = 768 * 28 * 28,
                 use_flash_attn: bool = True):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        kwargs = dict(torch_dtype="auto", device_map=device)
        if use_flash_attn:
            try:
                kwargs["attn_implementation"] = "flash_attention_2"
                kwargs["torch_dtype"] = torch.bfloat16
            except Exception:
                pass
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)
        self.torch = torch

    def generate(self, frames_bgr, prompt, max_new_tokens=512):
        from qwen_vl_utils import process_vision_info
        from PIL import Image
      
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]


SCENE_GRAPH_PROMPT = """You are analyzing keyframes sampled in order from one short video.

Identify the MOST IMPORTANT entities (people, animals, key objects) and the
meaningful interactions or actions between them across these frames.

Rules:
- Return ONLY the salient relationships that describe what is happening.
- Use concrete action/semantic predicates (e.g. interviewing, hugging, holding,
  pointing_at, laughing_at, presenting, cooking, dancing_with, arguing_with).
- DO NOT use vague spatial predicates like "near", "next to", "beside", "above".
- Refer to recurring people consistently (person_1, person_2, host, guest...).
- At most 8 triples total for the whole video.

Return STRICT JSON only, no prose, in exactly this schema:
{"entities": ["...", "..."],
 "triples": [{"subject": "...", "predicate": "...", "object": "...", "salience": 0.0-1.0}],
 "scene_summary": "one short sentence"}"""

# Path B: a single rich caption per set of frames (cheap semantic baseline).
CAPTION_PROMPT = """Describe what is happening across these ordered keyframes from
one short video in 2-3 sentences. Focus on people, their actions, interactions,
mood, and setting. Be concrete. Return plain text only."""


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a possibly-messy VLM response."""
    # strip code fences
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    # find outermost {...}
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    # last-ditch: remove trailing commas
                    blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
                    try:
                        return json.loads(blob2)
                    except json.JSONDecodeError:
                        return None
    return None


@dataclass
class VLMTriple:
    subject: str
    predicate: str
    object: str
    salience: float = 0.5


def _norm_entity(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def build_scene_graph(video_path: str, vlm: VLMBackend,
                      max_frames: int = 8) -> dict:
    """
    Path C: keyframes -> VLM -> salient open-vocabulary triples -> graph.
    Returns dict with graph, triples, entities, summary, raw response.
    """
    kfs = select_keyframes(video_path, max_frames=max_frames)
    frames = [f for _, _, f in kfs]
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    raw = vlm.generate(frames, SCENE_GRAPH_PROMPT, max_new_tokens=512)
    parsed = _extract_json(raw) or {}
    triples_in = parsed.get("triples", []) if isinstance(parsed, dict) else []

    triples: list[VLMTriple] = []
    for t in triples_in:
        try:
            s = _norm_entity(str(t["subject"]))
            p = _norm_entity(str(t["predicate"]))
            o = _norm_entity(str(t["object"]))
            if not s or not o or not p:
                continue
            sal = float(t.get("salience", 0.5))
            triples.append(VLMTriple(s, p, o, max(0.0, min(1.0, sal))))
        except (KeyError, TypeError, ValueError):
            continue

    # build directed graph; node = entity, edge = predicate w/ salience weight
    G = nx.MultiDiGraph()
    ent_mentions = Counter()
    for tr in triples:
        ent_mentions[tr.subject] += 1
        ent_mentions[tr.object] += 1
    for ent, c in ent_mentions.items():
        is_person = any(k in ent for k in ("person", "host", "guest", "man",
                                           "woman", "people", "player", "child"))
        G.add_node(ent, label=ent, mentions=c, is_person=is_person)
    for tr in triples:
        G.add_edge(tr.subject, tr.object, key=tr.predicate,
                   predicate=tr.predicate, salience=tr.salience)

    return {
        "graph": G,
        "triples": [asdict(t) for t in triples],
        "entities": parsed.get("entities", []),
        "summary": parsed.get("scene_summary", ""),
        "n_keyframes": len(frames),
        "raw_response": raw,
    }


def scene_graph_features(G: nx.MultiDiGraph) -> dict:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    preds = Counter(nx.get_edge_attributes(G, "predicate").values())
    sal = [d["salience"] for *_, d in G.edges(data=True)]
    degrees = [d for _, d in G.degree()]
    n_person = sum(1 for _, d in G.nodes(data=True) if d.get("is_person"))
    human_edges = sum(1 for s, o, d in G.edges(data=True)
                      if G.nodes[s].get("is_person"))
    return {
        "num_entities": n_nodes,
        "num_relations": n_edges,
        "num_unique_predicates": len(preds),
        "num_persons": n_person,
        "human_subject_interactions": human_edges,
        "interaction_density": (n_edges / n_nodes) if n_nodes else 0.0,
        "graph_density": nx.density(G) if n_nodes > 1 else 0.0,
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(np.max(degrees)) if degrees else 0,
        "avg_salience": float(np.mean(sal)) if sal else 0.0,
        "max_salience": float(np.max(sal)) if sal else 0.0,
        "top_predicate": preds.most_common(1)[0][0] if preds else "",
    }
