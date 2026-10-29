"""Build the workflow and annotation-interface figures from a real UI capture.

The PNG is an unmodified local browser screenshot of the running annotation
tool with a canonical gold example loaded. Only rectangular crops are made;
no controls, answers, or graph content are reconstructed. Cropped panels are
arranged for print, so the figures do not imply the original screen layout.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "figs" / "interface" / "annotation_full.png"
BLUE = "#0072B2"
INK = "#183F59"

# Coordinates refer to the unchanged 2341 x 1907 browser screenshot.
CROPS = {
    "video": (137, 360, 753, 684),
    "media": (137, 360, 753, 805),
    "graph": (755, 298, 1990, 1098),
    "edges": (1371, 1100, 1990, 1465),
    "controls": (755, 1100, 1990, 1465),
}


def panel(fig, screenshot, key, rect):
    ax = fig.add_axes(rect)
    ax.imshow(screenshot.crop(CROPS[key]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#D6DDE1")
    return ax


def workflow(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Screen + review", "4,348 reviewed", "1,731 eligible"),
        ("Model prefill", "Gemini 3.1 Pro", "Editable initial graph"),
        ("Human correction", "1,024 gold clips", "455 source videos"),
        ("Independent audits", "4 validators; 8 auditors", "330 edge judgments"),
    ]
    lefts = [0.015, 0.270, 0.525, 0.780]
    width = 0.205
    for index, ((title, line1, line2), left) in enumerate(zip(steps, lefts)):
        ax.add_patch(FancyBboxPatch(
            (left, 0.09), width, 0.79,
            boxstyle="round,pad=0.007,rounding_size=0.028",
            linewidth=0.8, edgecolor="#AABFCD", facecolor="#F4F8FB",
        ))
        # The small numbered tab makes the order easy to scan in grayscale.
        ax.text(left + 0.019, 0.88, str(index + 1), ha="center", va="center",
                fontsize=8.0, fontweight="bold", color="white",
                bbox={"boxstyle": "circle,pad=0.25", "facecolor": BLUE, "edgecolor": "white", "linewidth": 1.2})
        ax.text(left + width / 2, 0.675, title, ha="center", va="center",
                fontsize=8.3, fontweight="bold", color=INK)
        ax.text(left + width / 2, 0.422, line1, ha="center", va="center",
                fontsize=8.0, color="#222222")
        ax.text(left + width / 2, 0.205, line2, ha="center", va="center",
                fontsize=8.0, color="#222222")
        if index < 3:
            ax.annotate("", xy=(lefts[index + 1] - 0.010, 0.48),
                        xytext=(left + width + 0.012, 0.48),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#6D889A"})


def main_figure(screenshot):
    fig = plt.figure(figsize=(7.0, 3.4))
    workflow(fig.add_axes([0.0, 0.715, 1.0, 0.285]))
    fig.text(0.5, 0.662, "Human correction in the annotation tool", ha="center",
             va="center", fontsize=9.3, fontweight="bold", color=INK)
    fig.text(0.015, 0.603, "(a) Replay video and audio", fontsize=8.7,
             fontweight="bold", va="center")
    fig.text(0.515, 0.603, "(b) Edit directed relations", fontsize=8.7,
             fontweight="bold", va="center")
    panel(fig, screenshot, "video", [0.015, 0.060, 0.47, 0.525])
    panel(fig, screenshot, "edges", [0.515, 0.010, 0.47, 0.570])
    fig.savefig(ROOT / "figs" / "workflow.pdf", metadata={
        "Title": "AVSG workflow and cropped panels of the annotation interface",
        "Subject": "Actual browser screenshot crops; original layout rearranged for readability",
    })
    plt.close(fig)


def supplement_figure(screenshot):
    fig = plt.figure(figsize=(7.0, 7.3))
    fig.text(0.015, 0.972, "(a) Clip playback and caption", fontsize=9.0,
             fontweight="bold", va="center")
    fig.text(0.515, 0.972, "(c) Directed-edge editing", fontsize=9.0,
             fontweight="bold", va="center")
    panel(fig, screenshot, "media", [0.015, 0.620, 0.47, 0.325])
    panel(fig, screenshot, "edges", [0.515, 0.620, 0.47, 0.325])
    fig.text(0.015, 0.595, "(b) Complete live graph preview", fontsize=9.0,
             fontweight="bold", va="center")
    panel(fig, screenshot, "graph", [0.015, 0.010, 0.97, 0.565])
    fig.savefig(ROOT / "figs" / "annotation_interface.pdf", metadata={
        "Title": "Actual annotation interface: media, graph preview, and editing controls",
        "Subject": "Cropped screenshot panels rearranged for print; all four graph nodes and three edges are shown",
    })
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    screenshot = Image.open(SOURCE).convert("RGB")
    assert screenshot.size == (2341, 1907), "Update crop coordinates for a new screenshot size"
    main_figure(screenshot)
    supplement_figure(screenshot)
    print("Wrote figs/workflow.pdf (7 x 3.4 in) and figs/annotation_interface.pdf (7 x 7.3 in).")
    print("Source screenshot is unchanged; figures contain explicitly cropped, rearranged panels.")


if __name__ == "__main__":
    main()
