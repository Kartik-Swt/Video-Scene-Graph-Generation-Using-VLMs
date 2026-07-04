# vlm-scene-graph

**Open-vocabulary video scene graphs from a local VLM (Qwen2.5-VL)** — no hand-defined object classes, no predicate rules. The VLM produces `(subject, predicate, object)` triples directly, which are assembled into a graph and reduced to a compact feature vector. Ships with a caption/embedding baseline and a resumable batch runner built for large corpora.

```
(host)  --interviewing-->  (guest)
(host)  --holding-->       (microphone)
(guest) --laughing_at-->   (host)
```

---

## Why

Classic scene graph generation needs a **fixed vocabulary** — a closed set of object and predicate classes you have to curate and train against. This project skips that entirely: a capable vision-language model emits open-vocabulary triples from a prompt, so the "ontology" lives in instructions, not a schema. If the model can see it, it can describe it.

The output — structured, queryable "who-did-what-to-whom" — is a general-purpose primitive useful for semantic video search, content moderation, robotics/embodied perception, VQA, summarization, activity analysis, accessibility, corpus analytics, and as features for a downstream model.

## How it works

```
video file
   │
   ▼
select_keyframes()          cheap, no model — scene-change detection.
   │  (≤8 visually distinct frames)     Cuts VLM calls by ~10×.
   ▼
vlm.generate()              one VLM call per video, all keyframes as a "video"
   │
   ├─► SCENE_GRAPH_PROMPT ─► parse JSON ─► triples ─► nx.MultiDiGraph ─► features   (Path C)
   │
   └─► CAPTION_PROMPT     ─► caption ─► sentence embedding                          (Path B)
```

- **Path C** — the salient, sparse scene graph (the main output).
- **Path B** — a caption + embedding baseline, so you can measure whether the structured graph actually beats a plain caption for *your* task.

Design choices that matter at scale: keyframe selection to minimize VLM calls, multi-frame-per-call reasoning, defensively-parsed strict JSON, and a **resumable** runner that writes one JSON per video and skips completed work on re-run.

## Install

Core pipeline (plumbing + mock backend, **no GPU required**):

```bash
pip install -e .
# or: pip install -r requirements.txt
```

Real Qwen2.5-VL backend (GPU). Install `torch` first, matched to your CUDA version (see pytorch.org):

```bash
pip install -e ".[gpu,embed]"
# or: pip install -r requirements-gpu.txt
# optional speedup: pip install flash-attn --no-build-isolation
```

## Quickstart (no GPU)

The pipeline depends on a `VLMBackend` *interface*, not a specific model, so you can run everything with a bundled mock backend:

```python
from vlm_scene_graph import build_scene_graph, scene_graph_features, MockVLM

sg = build_scene_graph("your_video.mp4", MockVLM(), max_frames=8)
print(sg["summary"])
print(sg["triples"])
print(scene_graph_features(sg["graph"]))
```

Or run the bundled example:

```bash
python examples/quickstart.py your_video.mp4
```

## Real usage (GPU)

```python
from vlm_scene_graph import QwenVLBackend, run_pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_parquet("data.parquet")   # needs an id column + a video-path column
vlm = QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

feats = run_pipeline(
    df, vlm, embedder=embedder,
    out_dir="vlm_graphs",
    id_col="videoId", path_col="videoPath", title_col="videoTitleEn",
    max_frames=8,
    limit=20,          # smoke-test first; drop for the full run
)
```

`run_pipeline` writes, under `out_dir/`:

| Path | Contents |
|---|---|
| `per_video/<id>.json` | triples, entities, summary, serialized graph, features, caption — one file per video |
| `viz/<id>.png` | a rendered scene graph (every `viz_every` videos) |
| `scene_graph_features.csv` | one flat feature row per video (the modeling table) |
| `features_partial.csv` | periodic flush, so a crash never loses everything |

**Resumable:** re-running skips any video whose JSON already exists — a crashed multi-hour job picks up where it left off.

## Feature vector

`scene_graph_features(G)` reduces a graph to a fixed dict:

| Feature | Meaning |
|---|---|
| `num_entities`, `num_relations` | graph size |
| `num_unique_predicates` | variety of interaction types |
| `num_persons`, `human_subject_interactions` | human presence / humans as actors |
| `interaction_density`, `graph_density` | how interactive / connected the scene is |
| `avg_degree`, `max_degree` | connectivity; hints at a central protagonist |
| `avg_salience`, `max_salience` | model confidence in importance |
| `top_predicate` | dominant action |

All are guarded against empty graphs (a video with no detected triples yields an all-zero row, not a crash).

## The `VLMBackend` interface

Any object with this method is a valid backend:

```python
def generate(self, frames_bgr: list[np.ndarray], prompt: str,
             max_new_tokens: int = 512) -> str: ...
```

This is why the whole pipeline is testable without a GPU: `MockVLM` returns canned JSON and everything else (keyframes, parsing, graph build, features, batch runner) runs on a laptop. Swap in `QwenVLBackend` — or your own backend for a different VLM — without touching the pipeline.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite runs entirely on the mock backend — no GPU, no model download.

## Project layout

```
vlm_scene_graph/
  core.py           keyframes, backends, prompts, JSON parsing, graph build, features
  batch_runner.py   resumable DataFrame runner + visualization
  mock_backend.py   GPU-free backend for tests/demos
examples/
  quickstart.py     single-video walkthrough
  run_batch.py      DataFrame batch walkthrough
tests/
  test_pipeline.py  parsing / graph / features via the mock
```

## License

MIT — see [LICENSE](LICENSE).
