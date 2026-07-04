from __future__ import annotations
import os
import json
import traceback

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from .core import (
    build_scene_graph, scene_graph_features,
)

def draw_scene_graph(G: nx.MultiDiGraph, title: str = "",
                     save_path: str | None = None, figsize=(10, 7)):
    if G.number_of_nodes() == 0:
        print(f"[{title}] empty graph")
        return
    pos = nx.spring_layout(G, k=1.5, seed=42)
    is_person = nx.get_node_attributes(G, "is_person")
    colors = ["#ff6b6b" if is_person.get(n) else "#4dabf7" for n in G.nodes]
    mentions = nx.get_node_attributes(G, "mentions")
    sizes = [800 + 500 * mentions.get(n, 1) for n in G.nodes]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, {n: n for n in G.nodes},
                            font_size=9, font_weight="bold")
    for u, v, k, d in G.edges(keys=True, data=True):
        w = 1 + 4 * d.get("salience", 0.5)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w, alpha=0.55,
                               edge_color="#495057", arrowsize=18,
                               connectionstyle="arc3,rad=0.1")
    elabels = {(u, v): d["predicate"] for u, v, k, d in G.edges(keys=True, data=True)}
    nx.draw_networkx_edge_labels(G, pos, elabels, font_size=8)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"saved -> {save_path}")
    else:
        plt.close()

def run_pipeline(df: pd.DataFrame,
                 vlm,
                 embedder=None,
                 out_dir: str = "vlm_graphs",
                 path_col: str = "videoPath",
                 id_col: str = "videoId",
                 title_col: str = "videoTitleEn",
                 do_scene_graph: bool = True,
                 save_viz: bool = True,
                 viz_every: int = 25,
                 max_frames: int = 8,
                 flush_every: int = 50,
                 limit: int | None = None) -> pd.DataFrame:
    
    os.makedirs(out_dir, exist_ok=True)
    json_dir = os.path.join(out_dir, "per_video")
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    sub = df if limit is None else df.head(limit)
    rows = []
    for i, (_, r) in enumerate(sub.iterrows()):
        vid = str(r[id_col])
        vpath = r[path_col]
        jpath = os.path.join(json_dir, f"{vid}.json")

        if os.path.exists(jpath):  # resume
            with open(jpath) as f:
                rec = json.load(f)
            rows.append(_row_from_record(rec, id_col, vid))
            continue

        rec = {id_col: vid}
        try:
            if do_scene_graph:
                sg = build_scene_graph(vpath, vlm, max_frames=max_frames)
                G = sg["graph"]
                rec["triples"] = sg["triples"]
                rec["entities"] = sg["entities"]
                rec["summary"] = sg["summary"]
                rec["graph_nodelink"] = nx.node_link_data(G)
                rec["features"] = scene_graph_features(G)
                if save_viz and (i % viz_every == 0):
                    draw_scene_graph(
                        G, title=str(r.get(title_col, vid))[:70],
                        save_path=os.path.join(viz_dir, f"{vid}.png"))
  
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"   !! {vid} failed: {rec['error']}")
            traceback.print_exc()

        with open(jpath, "w") as f:
            json.dump(rec, f)
        rows.append(_row_from_record(rec, id_col, vid))

        if (i + 1) % flush_every == 0:
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, "features_partial.csv"), index=False)
            print(f"[{i+1}/{len(sub)}] flushed")

    feat_df = pd.DataFrame(rows).fillna(0)
    feat_df.to_csv(os.path.join(out_dir, "scene_graph_features.csv"), index=False)
    print(f"\nfeatures -> {os.path.join(out_dir, 'scene_graph_features.csv')}")
    return feat_df


def _row_from_record(rec: dict, id_col: str, vid: str) -> dict:
    row = {id_col: vid}
    row.update(rec.get("features", {}))
    row["summary"] = rec.get("summary", "")
    if rec.get("error"):
        row["error"] = rec["error"]
    return row
