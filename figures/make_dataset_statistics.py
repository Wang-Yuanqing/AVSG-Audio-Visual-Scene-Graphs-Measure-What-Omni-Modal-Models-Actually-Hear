"""Rebuild the dataset-distribution figure from canonical gold annotations.

Within the repository, run ``python make_dataset_statistics.py``. For an
exported manuscript, supply ``--gold /path/to/1k_result.jsonl`` to recompute
the statistics, or omit it to render the included, source-hashed JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
GOLD = ROOT.parent / "data" / "gold" / "1k_result.jsonl"
STATS = ROOT / "dataset_statistics.json"
COLORS = {"V2V": "#446E8A", "V2A": "#A97349", "A2A": "#887997"}
MODALITY_COLORS = {"visual": COLORS["V2V"], "audio": COLORS["A2A"]}
DURATION_BINS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def summarize(gold_path: Path) -> dict:
    content = gold_path.read_bytes()
    records = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    sources = set()
    durations, node_counts, edge_counts = [], [], []
    families, family_clips, relations, modalities = Counter(), Counter(), Counter(), Counter()
    classes = {"visual": set(), "audio": set()}
    class_occurrences = {"visual": Counter(), "audio": Counter()}
    v2a_pairs = Counter()
    for record in records:
        filename = record["filename"]
        source, start, end = filename.removesuffix(".mp4").rsplit("_", 2)
        duration = float(end) - float(start)
        if duration <= 0:
            raise ValueError(f"Non-positive clip duration: {filename}")
        sources.add(source)
        durations.append(duration)
        nodes, edges = record["ground_truth_nodes"], record["ground_truth_edges"]
        node_counts.append(len(nodes))
        edge_counts.append(len(edges))
        node_classes = {node["id"]: node["class"] for node in nodes}
        node_ids = set(node_classes)
        for node in nodes:
            modality = "visual" if node["id"].startswith("v") else "audio"
            modalities[modality] += 1
            classes[modality].add(node["class"])
            class_occurrences[modality][node["class"]] += 1
        present = set()
        for edge in edges:
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                raise ValueError(f"Dangling edge: {filename}")
            source_modality = "V" if edge["source"].startswith("v") else "A"
            target_modality = "V" if edge["target"].startswith("v") else "A"
            family = source_modality + "2" + target_modality
            if family not in COLORS:
                raise ValueError(f"Unexpected edge family: {family}")
            families[family] += 1
            relations[family + ":" + edge["edge"]] += 1
            if family == "V2A":
                v2a_pairs[(node_classes[edge["source"]], node_classes[edge["target"]])] += 1
            present.add(family)
        family_clips.update(present)
    duration_histogram, _ = np.histogram(durations, DURATION_BINS)
    if int(duration_histogram.sum()) != len(records):
        raise ValueError("The displayed duration bins do not cover every clip.")
    source_totals, sound_totals = Counter(), Counter()
    for (source_class, sound_class), count in v2a_pairs.items():
        source_totals[source_class] += count
        sound_totals[sound_class] += count
    # Marginal selection is computed before restricting either matrix dimension.
    ranked_sources = sorted(source_totals, key=lambda key: (-source_totals[key], key))
    ranked_sounds = sorted(sound_totals, key=lambda key: (-sound_totals[key], key))
    shown_sources = [key for key in ranked_sources if source_totals[key] >= 5]
    shown_sounds = [key for key in ranked_sounds if sound_totals[key] >= 5]
    shown_matrix = [[v2a_pairs[source_class, sound_class] for sound_class in shown_sounds]
                    for source_class in shown_sources]
    shown_edges = sum(map(sum, shown_matrix))
    result = {
        "source": "data/gold/1k_result.jsonl",
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "clips": len(records),
        "source_videos": len(sources),
        "nodes": sum(node_counts),
        "edges": sum(edge_counts),
        "node_modalities": dict(modalities),
        "observed_class_counts": {key: len(value) for key, value in classes.items()},
        "duration_seconds": {
            "total": float(sum(durations)),
            "minimum": float(min(durations)),
            "median": float(np.median(durations)),
            "p90": float(np.quantile(durations, .9)),
            "maximum": float(max(durations)),
            "bin_edges": DURATION_BINS,
            "bin_counts": duration_histogram.tolist(),
        },
        "nodes_per_clip": dict(sorted(Counter(node_counts).items())),
        "edges_per_clip": dict(sorted(Counter(edge_counts).items())),
        "median_nodes": float(np.median(node_counts)),
        "median_edges": float(np.median(edge_counts)),
        "family_edges": dict(families),
        "family_clips": dict(family_clips),
        "typed_relation_counts": dict(sorted(relations.items())),
        "class_occurrence_counts": {key: dict(sorted(value.items()))
                                    for key, value in class_occurrences.items()},
        "classes_with_at_most_five_occurrences": sum(
            value <= 5 for modality in class_occurrences.values() for value in modality.values()),
        "rank_selection": {
            "nodes": "Descending occurrence count separately within each modality; alphabetical tie-break.",
            "relations": "Descending edge occurrence count across all observed modality-typed relations; alphabetical tie-break.",
        },
        "v2a_class_associations": {
            "interpretation": "Counts of annotated class-typed V2A edges; not instance-level source binding accuracy.",
            "source_totals": dict(sorted(source_totals.items())),
            "sound_totals": dict(sorted(sound_totals.items())),
            "occupied_class_pairs": len(v2a_pairs),
            "pair_counts": [{"visual_class": pair[0], "audio_class": pair[1], "edges": count}
                            for pair, count in sorted(v2a_pairs.items())],
            "display_selection": "Visual source and audio classes with at least 5 full-corpus V2A edges, independently ranked by full-corpus marginals; alphabetical tie-break.",
            "display_label_denominator": f"All {families['V2A']:,} V2A edges; percentages use full-corpus class marginals before pair restriction.",
            "display_link_width_points": "0.18 + 0.010 * edge_count; the intercept keeps rare class-pair links visible.",
            "display_marginal_threshold": 5,
            "display_nonzero_pairs": sum(count > 0 for row in shown_matrix for count in row),
            "display_sources": shown_sources,
            "display_sounds": shown_sounds,
            "display_matrix": shown_matrix,
            "display_edges": shown_edges,
            "total_edges": families["V2A"],
            "display_fraction": shown_edges / families["V2A"],
        },
    }
    if sum(families.values()) != result["edges"]:
        raise ValueError("Relation-family totals do not equal the edge count.")
    return result


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9199A1")
        ax.spines[side].set_linewidth(.6)
    ax.tick_params(axis="both", which="both", length=2.2, width=.5, labelsize=8.2)
    ax.set_axisbelow(True)


def audio_group(sound: str) -> str:
    if sound in {"male_speech", "female_speech", "male_singing", "female_singing",
                 "child_singing", "laughter", "crying", "panting", "coughing"}:
        return "voice"
    if sound in {"guitar_sound", "drum_sound", "piano_sound", "violin_sound",
                 "trumpet_sound", "flute_sound", "accordion_sound"}:
        return "music"
    if sound in {"crowd_sound", "applause"}:
        return "crowd"
    if sound in {"scraping", "crinkling"}:
        return "contact"
    if sound in {"bark", "meow", "chirp"}:
        return "animal"
    return "other"


def render_associations(fig, association: dict) -> None:
    """Bipartite class associations, with offset ports for readable curves.

    Geometry is in inches and line widths are in points. The positive width
    intercept retains rare pairs; it is stated explicitly in the frozen data.
    All printed percentages use full-corpus marginals, not the displayed subset.
    """
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    group_colors = {
        "voice": "#B09B76", "music": "#A4797D", "crowd": "#6C8CA0",
        "other": "#929798", "contact": "#958574", "animal": "#7D9A88",
    }
    sources, sounds = association["display_sources"], association["display_sounds"]
    matrix = np.asarray(association["display_matrix"])
    ax.text(3.08, 4.13, "(c) Visual-audio class associations", fontsize=9.4,
            fontweight="bold", ha="left", va="center")
    ax.text(3.08, 3.94,
            f"{association['display_edges']:,}/{association['total_edges']:,} V2A edges "
            f"({100 * association['display_fraction']:.1f}%)", fontsize=8.2)
    for index, (group, color) in enumerate(group_colors.items()):
        x = 3.13 + (index % 3) * 1.12
        y = 3.75 - (index // 3) * .16
        ax.plot([x, x + .16], [y, y], color=color, lw=2.4, solid_capstyle="butt")
        ax.text(x + .20, y, group, fontsize=8.2, va="center")
    ax.text(4.02, 3.39, "Visual source", fontsize=8.2, style="italic", ha="right")
    ax.text(5.44, 3.39, "Audio class", fontsize=8.2, style="italic")
    source_y = np.linspace(3.22, .14, len(sources))
    sound_y = np.linspace(3.22, .14, len(sounds))
    widths = np.where(matrix > 0, .18 + .010 * matrix, 0.0)
    source_height = widths.sum(axis=1) / 72
    sound_height = widths.sum(axis=0) / 72
    source_start = source_y + source_height / 2
    sound_start = sound_y + sound_height / 2
    source_offsets = np.zeros(len(sources))
    sound_offsets = np.zeros(len(sounds))
    links = []
    for i, source in enumerate(sources):
        for j, sound in enumerate(sounds):
            width = float(widths[i, j])
            if not width:
                continue
            half = width / 144
            y1 = source_start[i] - source_offsets[i] - half
            y2 = sound_start[j] - sound_offsets[j] - half
            source_offsets[i] += width / 72
            sound_offsets[j] += width / 72
            links.append((int(matrix[i, j]), width, y1, y2, sound))
    left, right = 4.08, 5.37
    for count, width, y1, y2, sound in sorted(links, reverse=True):
        path = MplPath([(left, y1), (left + .53, y1),
                        (right - .53, y2), (right, y2)],
                       [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, fill=False, color=group_colors[audio_group(sound)],
                               lw=width, alpha=.66, capstyle="butt", zorder=1))
    for source, y, height in zip(sources, source_y, source_height):
        height = max(height, .045)
        ax.add_patch(Rectangle((left - .047, y - height / 2), .047, height,
                               facecolor="#5F666B", edgecolor="none", zorder=3))
        percentage = 100 * association["source_totals"][source] / association["total_edges"]
        ax.text(left - .09, y, f"{source} {percentage:.1f}%", fontsize=8.2,
                ha="right", va="center")
    for sound, y, height in zip(sounds, sound_y, sound_height):
        height = max(height, .045)
        ax.add_patch(Rectangle((right, y - height / 2), .047, height,
                               facecolor=group_colors[audio_group(sound)], edgecolor="none", zorder=3))
        percentage = 100 * association["sound_totals"][sound] / association["total_edges"]
        ax.text(right + .09, y, f"{sound} {percentage:.1f}%", fontsize=8.2,
                ha="left", va="center")


def render(stats: dict, destination: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.3, "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(7, 4.3), facecolor="white")
    node_ax = fig.add_axes([.082, .625, .31, .29])
    relation_ax = fig.add_axes([.082, .115, .31, .29])
    for ax in (node_ax, relation_ax):
        style_axes(ax)
        ax.set_yscale("log")
        ax.set_ylim(.8, 1800)
        ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
        ax.minorticks_off()
        ax.grid(axis="y", color="#E3E6E8", lw=.6)
        ax.set_ylabel("Occurrences (log scale)", labelpad=4, fontsize=8.3)
    for modality, marker, line in (("visual", "o", "-"), ("audio", "s", "--")):
        ranked = sorted(stats["class_occurrence_counts"][modality].items(),
                        key=lambda item: (-item[1], item[0]))
        ranks = np.arange(1, len(ranked) + 1)
        node_ax.plot(ranks, [count for _, count in ranked], color=MODALITY_COLORS[modality],
                     marker=marker, linestyle=line, markersize=2.2, linewidth=1.0,
                     label=f"{modality.capitalize()} ({len(ranked)} classes)")
    node_ax.set_xlim(0, 76)
    node_ax.set_xticks([1, 20, 40, 60, 74])
    node_ax.set_xlabel("Class rank within modality", labelpad=3, fontsize=8.3)
    node_ax.set_title("(a) Node-class frequency", loc="left", fontsize=9.4,
                      fontweight="bold", pad=8)
    node_ax.legend(loc="upper right", frameon=False, fontsize=8.2,
                   handlelength=1.5, handletextpad=.5, borderaxespad=0,
                   labelspacing=.35)
    total_classes = sum(stats["observed_class_counts"].values())
    node_ax.text(.98, .52,
                 f"{stats['classes_with_at_most_five_occurrences']}/{total_classes} classes\noccur at most 5 times",
                 transform=node_ax.transAxes, ha="right", va="top", fontsize=8.2,
                 linespacing=1.3)

    ranked_relations = sorted(stats["typed_relation_counts"].items(),
                              key=lambda item: (-item[1], item[0]))
    relation_ranks = np.arange(1, len(ranked_relations) + 1)
    family_markers = {"V2V": "o", "V2A": "s", "A2A": "^"}
    for rank, (key, count) in zip(relation_ranks, ranked_relations):
        family = key.split(":", 1)[0]
        color = COLORS[family]
        relation_ax.vlines(rank, .8, count, color=color, alpha=.40, linewidth=.7)
        relation_ax.plot(rank, count, marker=family_markers[family], color=color, markersize=3.1,
                         markeredgewidth=0, zorder=3)
    relation_ax.set_xlim(0, len(ranked_relations) + 1)
    relation_ax.set_xticks([1, 10, 20, 30, 40, 52])
    relation_ax.set_xlabel("Typed-relation rank", labelpad=3, fontsize=8.3)
    relation_ax.set_title("(b) Typed-relation frequency", loc="left", fontsize=9.4,
                          fontweight="bold", pad=8)
    relation_counts = Counter(key.split(":", 1)[0] for key in stats["typed_relation_counts"])
    legend = [Line2D([0], [0], color=COLORS[family], marker=family_markers[family],
                     lw=0, markersize=4, label=f"{family} ({relation_counts[family]})")
              for family in ("V2V", "V2A", "A2A")]
    relation_ax.legend(handles=legend, loc="upper right", frameon=False,
                       fontsize=8.2, handlelength=.9, handletextpad=.35,
                       borderaxespad=0, labelspacing=.25)
    relation_ax.text(.98, .58,
                     f"speaking: {stats['typed_relation_counts']['V2A:speaking']:,}\n"
                     f"holding: {stats['typed_relation_counts']['V2V:holding']:,}",
                     transform=relation_ax.transAxes, ha="right", va="top",
                     fontsize=8.2, linespacing=1.3)
    render_associations(fig, stats["v2a_class_associations"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={"Title": "AVSG class and relation long tails, and class-typed visual-audio associations",
                                       "CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--statistics", type=Path, default=STATS)
    parser.add_argument("--output", type=Path, default=ROOT / "figs" / "dataset_statistics.pdf")
    args = parser.parse_args()
    gold_path = args.gold or (GOLD if GOLD.exists() else None)
    if gold_path is not None:
        stats = summarize(gold_path)
        args.statistics.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"Computed statistics from {gold_path}")
    else:
        stats = json.loads(args.statistics.read_text(encoding="utf-8"))
        print(f"Loaded frozen statistics from {args.statistics}")
    render(stats, args.output)
    print(f"{stats['clips']:,} clips; {stats['source_videos']} sources; {stats['nodes']:,} nodes; {stats['edges']:,} edges")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
