"""Build the paper overview from the committed results and source image."""

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 6.5, 3.7
INK = "#20303B"
TEAL = "#087E80"
OCHRE = "#A86619"
GRAY = "#89969F"


def main():
    results = json.loads((ROOT / "results/cv_cpu.json").read_text())
    summary = results["summary"]
    fixed = sum(f["fixed_position_metrics"]["correct"] for f in results["folds"])
    shuffled = summary["pre_generation_permuted_correct_30"]
    values = [fixed, summary["action_only"]["correct"],
              summary["pre_generation"]["correct"], mean(shuffled),
              summary["full"]["correct"]]
    labels = ["Fixed position", "Recipe baseline", "Pre-render",
              "Pre-render, shuffled", "Post-render"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
    canvas.axis("off")

    def text(x, y, content, **kwargs):
        return canvas.text(x, y, content, va="center", **kwargs)

    def arrow(start, end, color=GRAY):
        canvas.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=8,
            linewidth=1, color=color, shrinkA=0, shrinkB=0))

    def box(x, y, title, detail, color, fill):
        canvas.add_patch(FancyBboxPatch(
            (x, y - .30), 1.60, .60,
            boxstyle="round,pad=0.015,rounding_size=0.045",
            facecolor=fill, edgecolor=color, linewidth=.85))
        offset = .027 if "\n" not in detail else 0
        text(x + .8, y + .14 - offset, title, ha="center", fontsize=8, weight="bold")
        text(x + .8, y - .085 - offset, detail, ha="center", fontsize=7.3,
             linespacing=1.35)

    text(.06, 3.56, "a  Where selection happens", fontsize=9.5, weight="bold")
    text(6.43, 3.56, "Intended use", fontsize=7.5, ha="right", color=GRAY)

    thumbnail = fig.add_axes([.055 / WIDTH, 2.30 / HEIGHT, .70 / WIDTH, .70 / HEIGHT])
    thumbnail.imshow(Image.open(ROOT / "figures/coco_158945_source.png"))
    thumbnail.axis("off")
    text(.405, 2.19, "Source", ha="center", fontsize=7.5)
    text(.405, 1.98, "+ recipes", ha="center", fontsize=7.2)
    canvas.plot([.77, .87, .87], [2.65, 2.65, 2.12], color=GRAY, linewidth=1)
    canvas.plot([.87, .87], [2.65, 2.95], color=GRAY, linewidth=1)
    arrow((.87, 2.95), (1.05, 2.95), TEAL)
    arrow((.87, 2.12), (1.05, 2.12), OCHRE)

    text(1.06, 3.33, "Pre-render selection", fontsize=8, weight="bold", color=TEAL)
    box(1.07, 2.95, "Score recipes", "source CLIP + recipe\nMLP → (r, a)", TEAL, "#EDF7F6")
    box(2.93, 2.95, "Choose or reject", "highest r among\nrecipes with a > t", TEAL, "#EDF7F6")
    box(4.79, 2.95, "Render selection", "up to 1 output", TEAL, "#EDF7F6")
    arrow((2.69, 2.95), (2.90, 2.95), TEAL)
    arrow((4.55, 2.95), (4.76, 2.95), TEAL)

    text(1.06, 2.50, "Post-render selection", fontsize=8, weight="bold", color=OCHRE)
    box(1.07, 2.12, "Render candidates", "up to 12 outputs", OCHRE, "#FCF4E9")
    box(2.93, 2.12, "Score outputs", "add output CLIP\nMLP → (r, a)", OCHRE, "#FCF4E9")
    box(4.79, 2.12, "Choose or reject", "highest r among\noutputs with a > t", OCHRE, "#FCF4E9")
    arrow((2.69, 2.12), (2.90, 2.12), OCHRE)
    arrow((4.55, 2.12), (4.76, 2.12), OCHRE)
    text(.06, 1.66, "Source CLIP: image + masked subject. Recipe descriptors include subject count.",
         fontsize=7.1, color="#53616B")
    canvas.plot([.06, 6.43], [1.52, 1.52], color="#DCE2E5", linewidth=.7)

    text(.06, 1.37, "b  Results from the recorded candidate bank", fontsize=9.5, weight="bold")
    chart = fig.add_axes([1.49 / WIDTH, .35 / HEIGHT, 4.80 / WIDTH, .86 / HEIGHT])
    colors = [GRAY, GRAY, TEAL, "#B9DDDB", OCHRE]
    bars = chart.barh(range(5), values, height=.62, color=colors, zorder=3)
    bars[3].set_edgecolor(TEAL)
    bars[3].set_hatch("///")
    bars[3].set_linewidth(.5)
    chart.set_yticks(range(5), labels, fontsize=7.6)
    chart.invert_yaxis()
    chart.set_xlim(0, 100)
    chart.set_xticks([0, 20, 40, 60, 80, 100])
    chart.tick_params(axis="x", labelsize=7, length=0, pad=3)
    chart.tick_params(axis="y", length=0, pad=5)
    chart.grid(axis="x", color="#E2E7EA", linewidth=.6, zorder=0)
    for spine in chart.spines.values():
        spine.set_visible(False)
    for index, value in enumerate(values):
        chart.text(value + 1.5, index, f"{value:.1f}" if index == 3 else str(value),
                   fontsize=8, weight="bold", va="center")
    chart.set_xlabel("Correct decisions per 100 sources", fontsize=7.5, labelpad=3)

    # Preserve readable labels when the figure is scaled to the paper's text width.
    for artist in fig.findobj(match=matplotlib.text.Text):
        artist.set_fontsize(artist.get_fontsize() * 1.2)
    for extension in ("pdf", "svg", "png"):
        output = ROOT / "figures" / f"main_overview.{extension}"
        fig.savefig(output, dpi=240, facecolor="white")
        print(output.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
