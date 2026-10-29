"""Render two complete canonical gold AVSG examples with real video stills.

The packaged example_data.json preserves complete source records and hashes.
The figure omits free-text attributes and captions; every node and edge is shown.
Run ``python make_examples.py`` using matplotlib, numpy, and Pillow. No access
to source videos is needed to rebuild from the supplied extracted PNG frames.
"""

from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parent
VISUAL = "#0072B2"
CROSS = "#D55E00"
AUDIO = "#7A5195"


def node(ax, center, label, audio=False):
    x, y = center
    width = max(0.18, len(label) * 0.0136 + 0.070)
    height = 0.22
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        facecolor="#F5EFF8" if audio else "#EBF4FA",
        edgecolor=AUDIO if audio else VISUAL, linewidth=0.8,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=9.0,
            fontweight="bold", color="#222222")
    return box


def arrow(ax, start, end, source_box, target_box, color, label, label_xy):
    ax.add_patch(FancyArrowPatch(
        start, end, patchA=source_box, patchB=target_box,
        arrowstyle="-|>", mutation_scale=9.5, linewidth=1.15,
        color=color, shrinkA=2, shrinkB=2,
    ))
    ax.text(*label_xy, label, ha="center", va="center", fontsize=8.7,
            color=color, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})


def verify(examples):
    expected = {
        "music": {
            "nodes": {"v1": "man", "v2": "guitar", "a1": "male_singing", "a2": "guitar_sound"},
            "edges": {("v1", "playing", "v2"), ("v1", "singing", "a1"), ("v1", "playing", "a2")},
        },
        "reaction": {
            "nodes": {"v1": "man", "v2": "dog", "a1": "male_speech", "a2": "bark"},
            "edges": {("v1", "holding", "v2"), ("v1", "speaking", "a1"),
                      ("v2", "barking", "a2"), ("a1", "responds_to", "a2")},
        },
    }
    for item in examples:
        record = item["gold_record"]
        digest = hashlib.sha256(json.dumps(
            record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        assert digest == item["gold_record_sha256"], "Frozen gold record hash changed"
        frame = ROOT / item["frame_path"]
        assert hashlib.sha256(frame.read_bytes()).hexdigest() == item["frame_sha256"]
        actual_nodes = {n["id"]: n["class"] for n in record["ground_truth_nodes"]}
        actual_edges = {(e["source"], e["edge"], e["target"]) for e in record["ground_truth_edges"]}
        assert actual_nodes == expected[item["key"]]["nodes"]
        assert actual_edges == expected[item["key"]]["edges"]
        assert len(record["ground_truth_edges"]) == len(actual_edges)


def draw_graph(ax, item):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = {"v1": (0.19, 0.80), "v2": (0.80, 0.80),
               "a1": (0.19, 0.18), "a2": (0.80, 0.18)}
    boxes = {}
    record = item["gold_record"]
    for n in record["ground_truth_nodes"]:
        boxes[n["id"]] = node(ax, centers[n["id"]], n["class"], n["id"].startswith("a"))
    label_positions = {
        ("v1", "v2"): (0.495, 0.885),
        ("v1", "a1"): (0.105, 0.49),
        ("v2", "a2"): (0.885, 0.49),
        ("v1", "a2"): (0.51, 0.49),
        ("a1", "a2"): (0.495, 0.26),
    }
    for edge in record["ground_truth_edges"]:
        source, target = edge["source"], edge["target"]
        color = VISUAL if target.startswith("v") else (AUDIO if source.startswith("a") else CROSS)
        arrow(ax, centers[source], centers[target], boxes[source], boxes[target],
              color, edge["edge"], label_positions[(source, target)])


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    data = json.loads((ROOT / "example_data.json").read_text(encoding="utf-8"))
    examples = data["examples"]
    verify(examples)
    fig = plt.figure(figsize=(7.0, 3.05))
    titles = ["(a) Sound source and instrument", "(b) Reaction between sound events"]
    for item, title, image_bottom, graph_bottom, title_y in zip(
        examples, titles, [0.558, 0.115], [0.548, 0.105], [0.957, 0.514]
    ):
        fig.text(0.015, title_y, title, fontsize=9.2, fontweight="bold", va="center")
        fig.text(0.665, title_y, "Complete gold graph", fontsize=8.5,
                 color="#555555", ha="center", va="center")
        image_ax = fig.add_axes([0.015, image_bottom, 0.285, 0.368])
        image_ax.imshow(Image.open(ROOT / item["frame_path"]))
        image_ax.axis("off")
        graph_ax = fig.add_axes([0.335, graph_bottom, 0.65, 0.375])
        draw_graph(graph_ax, item)

    # A subtle divider separates examples while leaving the data as the focus.
    fig.add_artist(Line2D([0.015, 0.985], [0.535, 0.535], transform=fig.transFigure,
                          linewidth=0.6, color="#DCE2E6"))
    handles = [Line2D([0], [0], color=color, linewidth=1.4, label=label) for color, label in [
        (VISUAL, "Visual-visual (V2V)"), (CROSS, "Visual-audio (V2A)"), (AUDIO, "Audio-audio (A2A)"),
    ]]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.50, 0.008), fontsize=8.2,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.8)
    output = ROOT / "figs" / "examples.pdf"
    fig.savefig(output, metadata={"Title": "Two complete gold AVSG examples with real video frames"})
    plt.close(fig)
    print(f"Wrote {output}; verified 8 nodes and 7 edges against frozen canonical records.")


if __name__ == "__main__":
    main()
