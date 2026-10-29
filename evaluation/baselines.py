"""Calibration baselines -> predictions/*.jsonl. No model/API needed.

  empty       : no nodes/edges (F1 floor = 0)
  random      : a few random in-vocab nodes + random edges (chance floor)
  caption_lex : nodes whose class word literally appears in the caption
                (shows how much the caption alone leaks -> floor for the
                'audio+video+caption' setting)
  template    : the corpus's most common graph, output verbatim for every
                clip (man + male_speech + background_music, one speaking
                edge); the content-free floor drawn in the ranking figure
  template_two_speaker : the two-speaker variant (adds woman +
                female_speech and an in_dialogue_with edge); the A2A floor

These calibrate the metric: a real model must clearly beat the templates.

  python baselines.py [gold.jsonl] [out_dir]
"""

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import VISUAL_CLASSES, AUDIO_CLASSES  # noqa: E402

GOLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "gold", "1k_result.jsonl")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("outputs", "baselines")

# frequency templates: the corpus's most common graph, and its two-speaker variant
TEMPLATE_NODES = [
    {"id": "v1", "class": "man", "attribute": "a man"},
    {"id": "a1", "class": "male_speech", "attribute": "speech"},
    {"id": "a2", "class": "background_music", "attribute": "background music"},
]
TEMPLATE_EDGES = [{"source": "v1", "edge": "speaking", "target": "a1"}]
TEMPLATE2_NODES = [
    {"id": "v1", "class": "man", "attribute": "a man"},
    {"id": "v2", "class": "woman", "attribute": "a woman"},
    {"id": "a1", "class": "male_speech", "attribute": "speech"},
    {"id": "a2", "class": "female_speech", "attribute": "speech"},
    {"id": "a3", "class": "background_music", "attribute": "background music"},
]
TEMPLATE2_EDGES = [
    {"source": "v1", "edge": "speaking", "target": "a1"},
    {"source": "v2", "edge": "speaking", "target": "a2"},
    {"source": "a1", "edge": "in_dialogue_with", "target": "a2"},
]

# a few caption synonyms -> canonical class (lexical baseline only)
SYN = {
    "person": "man", "people": "crowd", "lady": "woman", "guy": "man",
    "kid": "boy", "child": "boy", "music": "background_music",
    "song": "background_music", "singing": "background_music",
    "truck": "car", "puppy": "dog",
}


def load_gold():
    rows = []
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def caption_lex(caption):
    words = set(re.findall(r"[a-z_]+", (caption or "").lower()))
    vis, aud = [], []
    for w in words:
        c = SYN.get(w, w)
        if c in VISUAL_CLASSES and c not in [n["class"] for n in vis]:
            vis.append({"id": f"v{len(vis)+1}", "class": c, "attribute": ""})
        elif c in AUDIO_CLASSES and c not in [n["class"] for n in aud]:
            aud.append({"id": f"a{len(aud)+1}", "class": c, "attribute": ""})
    return vis + aud, []


def random_graph(rng):
    nv = rng.randint(1, 3)
    na = rng.randint(1, 2)
    vis = [{"id": f"v{i+1}", "class": rng.choice(sorted(VISUAL_CLASSES)), "attribute": ""}
           for i in range(nv)]
    aud = [{"id": f"a{i+1}", "class": rng.choice(sorted(AUDIO_CLASSES)), "attribute": ""}
           for i in range(na)]
    edges = []
    if len(vis) >= 1 and aud:
        edges.append({"source": vis[0]["id"], "edge": "speaking", "target": aud[0]["id"]})
    if len(vis) >= 2:
        edges.append({"source": vis[0]["id"], "edge": "next_to", "target": vis[1]["id"]})
    return vis + aud, edges


def main():
    os.makedirs(OUT, exist_ok=True)
    gold = load_gold()
    rng = random.Random(0)

    writers = {
        "empty": open(os.path.join(OUT, "baseline_empty.jsonl"), "w", encoding="utf-8"),
        "random": open(os.path.join(OUT, "baseline_random.jsonl"), "w", encoding="utf-8"),
        "caption_lex": open(os.path.join(OUT, "baseline_caption_lex.jsonl"), "w", encoding="utf-8"),
        "template": open(os.path.join(OUT, "baseline_template.jsonl"), "w", encoding="utf-8"),
        "template_two_speaker": open(os.path.join(OUT, "baseline_template_two_speaker.jsonl"), "w", encoding="utf-8"),
    }
    for r in gold:
        fn = r["filename"]
        writers["empty"].write(json.dumps({"filename": fn, "nodes": [], "edges": []}) + "\n")
        n, e = random_graph(rng)
        writers["random"].write(json.dumps({"filename": fn, "nodes": n, "edges": e}) + "\n")
        n, e = caption_lex(r.get("caption", ""))
        writers["caption_lex"].write(json.dumps({"filename": fn, "nodes": n, "edges": e}, ensure_ascii=False) + "\n")
        writers["template"].write(json.dumps({"filename": fn, "nodes": TEMPLATE_NODES, "edges": TEMPLATE_EDGES}) + "\n")
        writers["template_two_speaker"].write(json.dumps({"filename": fn, "nodes": TEMPLATE2_NODES, "edges": TEMPLATE2_EDGES}) + "\n")
    for w in writers.values():
        w.close()
    print(f"wrote 5 baselines to {OUT}/ over {len(gold)} clips")


if __name__ == "__main__":
    main()
