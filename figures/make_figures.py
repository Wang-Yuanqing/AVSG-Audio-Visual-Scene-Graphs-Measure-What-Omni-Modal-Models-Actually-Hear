"""Rebuild the manuscript's vector diagnostic and data-pipeline figures.

Run with Python and matplotlib from any directory:
    python make_figures.py

The diagnostic reads the frozen, paper-matched estimates rather than rounded
manuscript entries. Use --estimates to supply the CSV outside this repository.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
DEFAULT_ESTIMATES = ROOT / "figure_data.csv"
# A common order across both panels makes the two diagnostics comparable.
MODELS = [
    ("gemini-3.1-pro", "Gemini-3.1"),
    ("qwen3-omni-30b-think", "Qwen3-think"),
    ("qwen3-omni-30b-instruct", "Qwen3-instruct"),
    ("nemotron-3-nano-omni-think", "Nemotron-think"),
    ("nemotron-3-nano-omni", "Nemotron-instruct"),
    ("qwen2.5-omni-7b", "Qwen2.5"),
    ("minicpm-o-4.5", "MiniCPM-o"),
    ("interactiveomni-8b", "InteractiveOmni"),
    ("video-salmonn2plus-7b", "SALMONN 2+"),
    ("omnivinci", "OmniVinci"),
]
AV_COLOR = "#0072B2"  # Okabe-Ito blue.
NOAUDIO_COLOR = "#D55E00"  # Okabe-Ito vermilion.


def load_estimates(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    estimates = {}
    for row in rows:
        if (
            row["cohort"] == "full"
            and row["condition"] in ("raw", "noaudio")
            and row["metric"] in ("a2a", "audio_node")
        ):
            key = (row["model"], row["condition"], row["metric"])
            if key in estimates:
                raise ValueError(f"Duplicate estimate: {key}")
            if int(row["clips"]) != 1024 or int(row["source_videos"]) != 455:
                raise ValueError(f"Unexpected evaluation cohort: {key}")
            estimates[key] = row
    for model, _ in MODELS:
        for condition in ("raw", "noaudio"):
            for metric in ("a2a", "audio_node"):
                if (model, condition, metric) not in estimates:
                    raise ValueError(f"Missing estimate: {model}, {condition}, {metric}")
    return estimates


def make_audio_figure(estimates: dict, destination: Path) -> None:
    # Insert at its native 6.875-inch width: model labels remain 8.1 pt.
    # Keep this styling local so regenerating the diagnostic cannot restyle S1.
    with plt.rc_context({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.2, "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
        "text.color": "#252525", "axes.labelcolor": "#252525",
        "xtick.color": "#252525", "ytick.color": "#252525",
    }):
        _draw_audio_figure(estimates, destination)


def _draw_audio_figure(estimates: dict, destination: Path) -> None:
    width, height = 6.875, 3.0
    left, plot_width, plot_height = 1.23, 5.03, 1.30
    fig = plt.figure(figsize=(width, height))
    av_color = "#446E8A"
    noaudio_color = "#A97349"
    panels = [
        (1.53, "a2a", "f1", "(a) A2A F1"),
        (.18, "audio_node", "recall", "(b) Audio-class recall"),
    ]
    for panel_index, (bottom, metric, value_key, title) in enumerate(panels):
        ax = fig.add_axes([left / width, bottom / height,
                           plot_width / width, plot_height / height])
        ax.set_ylim(9.6, -.6)
        ax.set_xlim(-.02, .85)
        ax.set_yticks(range(len(MODELS)), [label for _, label in MODELS])
        ax.set_xticks([0, .2, .4, .6, .8], ["0", ".2", ".4", ".6", ".8"])
        ax.tick_params(axis="y", length=0, pad=4, labelsize=8.1)
        ax.tick_params(axis="x", length=0, pad=3, labelsize=8.2,
                       labelbottom=(panel_index == 1))
        for index in range(0, len(MODELS), 2):
            ax.axhspan(index - .5, index + .5, color="#F2F4F5", zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#A2A6A8")
        ax.spines["bottom"].set_linewidth(.6)
        ax.grid(axis="x", color="#DDE1E3", linewidth=.4, zorder=1)
        # Keep metric labels in the empty right-hand region of row 2.
        title_model = MODELS[2][0]
        assert max(float(estimates[(title_model, condition, metric)][value_key])
                   for condition in ("raw", "noaudio")) < .58
        ax.text(.60, 2, title, fontsize=9.0, fontweight="bold",
                ha="left", va="center")
        for index, (model, _) in enumerate(MODELS):
            av = float(estimates[(model, "raw", metric)][value_key])
            noaudio = float(estimates[(model, "noaudio", metric)][value_key])
            assert -.02 < min(av, noaudio)
            assert max(av, noaudio) < .85
            ax.plot([noaudio, av], [index, index],
                    lw=1.1, color="#91989D", zorder=2)
            ax.scatter(noaudio, index, s=11, marker="s", facecolor="white",
                       edgecolor=noaudio_color, linewidth=.85, zorder=3)
            ax.scatter(av, index, s=12, marker="o", color=av_color,
                       edgecolor="white", linewidth=.3, zorder=4)
            yfig = (bottom + plot_height * (9.6 - index) / 10.2) / height
            delta = f"{av - noaudio:+.3f}".replace("+0.", "+.").replace("-0.", "-.")
            fig.text(6.62 / width, yfig, delta,
                     ha="center", va="center", fontsize=8.1)
    legend_items = [
        Line2D([0], [0], marker="o", color=av_color, linestyle="none", markersize=3.5,
               label="Audio + video (AV)"),
        Line2D([0], [0], marker="s", color=noaudio_color, markerfacecolor="white",
               linestyle="none", markersize=3.5, label="Video only (V)"),
    ]
    fig.legend(handles=legend_items, loc="center left",
               bbox_to_anchor=(1.23 / width, 2.91 / height),
               ncol=2, frameon=False, fontsize=8.2, borderaxespad=0,
               handletextpad=.2, columnspacing=1.3)
    fig.text(6.24 / width, 2.855 / height, r"$\Delta$ = AV - V",
             ha="right", fontsize=8.2)
    fig.text(6.62 / width, 2.855 / height, r"$\Delta$", ha="center",
             fontsize=8.6, fontweight="bold")
    fig.savefig(destination, metadata={
        "Title": "Figure 4: compact full-width paired-score plots with a shared linear scale",
    })
    plt.close(fig)


def make_pipeline_figure(destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 1.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Screening", "32,490 source-batch clips\n4,348 human reviewed\n1,731 eligible"),
        ("Gemini 3.1 Pro prefill", "Fixed AVSG rubric\nVideo + audio input\nEditable initial graph"),
        ("Human correction", "1,024 completed clips\n455 source videos\n707 eligible unfinished"),
        ("Independent audits", "4 graph validators\n8 edge-audit volunteers\n330 edge judgments"),
    ]
    positions = [0.01, 0.265, 0.52, 0.775]
    box_width = 0.215
    for index, ((title, details), left) in enumerate(zip(steps, positions)):
        box = FancyBboxPatch((left, 0.15), box_width, 0.70,
                             boxstyle="round,pad=0.006,rounding_size=0.018",
                             linewidth=0.8, edgecolor="#A6B7C3", facecolor="#F5F8FA")
        ax.add_patch(box)
        ax.text(left + box_width / 2, 0.72, title, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color="#183F59")
        ax.text(left + box_width / 2, 0.43, details, ha="center", va="center",
                fontsize=8.1, linespacing=1.55, color="#222222")
        if index < len(steps) - 1:
            ax.annotate("", xy=(positions[index + 1] - 0.008, 0.5),
                        xytext=(left + box_width + 0.008, 0.5),
                        arrowprops={"arrowstyle": "-|>", "color": "#718391", "lw": 1.0})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(destination, metadata={"Title": "AVSG data construction and independent audits"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, default=DEFAULT_ESTIMATES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    estimates = load_estimates(args.estimates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_audio_figure(estimates, args.output_dir / "hearing.pdf")
    make_pipeline_figure(args.output_dir / "pipeline.pdf")
    print(f"Read full-cohort AV/noaudio estimates from {args.estimates}")
    print(f"Wrote {args.output_dir / 'hearing.pdf'}")
    print(f"Wrote {args.output_dir / 'pipeline.pdf'}")


if __name__ == "__main__":
    main()
