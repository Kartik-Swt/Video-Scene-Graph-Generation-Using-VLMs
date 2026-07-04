<img width="1100" height="723" alt="image" src="https://github.com/user-attachments/assets/90c5acbd-81f5-4bd8-9490-cfa9583274ed" />

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

The output — structured, queryable "object->action->object" — is a general-purpose primitive useful for semantic video search, content moderation, robotics/embodied perception, VQA, summarization, activity analysis, accessibility, corpus analytics, and as features for a downstream model.

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
   ├─► SCENE_GRAPH_PROMPT ─► parse JSON ─► triples ─► nx.MultiDiGraph ─► features
```

## Usage

```python
from vlm_scene_graph import QwenVLBackend, run_pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_csv("data.csv")   # needs an id column + a video-path column
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

