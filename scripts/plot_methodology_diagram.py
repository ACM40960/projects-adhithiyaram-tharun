"""Static pipeline flowchart PNG: two tracks (classifier-ranked v1/v2,
physics-based v3) sharing a data stage and converging at validation."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import PROCESSED_DATA_DIR

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_COLOR = "#4E79A7"
PHYSICS_COLOR = "#76B7B2"
SHARED_COLOR = "#59595C"

# (label, x, y, width, height, facecolor, text_color, fontsize)
BOXES = [
    ("Stage 1\nData Pipeline & EDA", 1.0, 4.5, 2.3, 1.1, SHARED_COLOR, "white", 9.5),
    ("Stage 2\nFeature Engineering\n(leakage-safe)", 4.1, 6.7, 2.3, 1.1, CLASSIFIER_COLOR, "white", 9.5),
    ("Stage 3\nML Classifier\n(LogReg/RF/ET/XGB)", 7.5, 6.7, 2.3, 1.1, CLASSIFIER_COLOR, "white", 9.5),
    ("Stage 5\nSeason Monte Carlo\n(v1/v2, Gumbel-max)", 10.9, 6.7, 2.3, 1.1, CLASSIFIER_COLOR, "white", 9.5),
    ("Stage 4 / Phase 1\nTyre Degradation\n(OLS -> Bayesian\nhierarchical)", 4.1, 2.3, 2.8, 1.6, PHYSICS_COLOR, "white", 9),
    ("Phase 2\nLap-by-Lap\nRace Simulator", 7.5, 2.3, 2.3, 1.1, PHYSICS_COLOR, "white", 9.5),
    ("Phase 3\nPhysics Season\nMonte Carlo (v3)", 10.9, 2.3, 2.3, 1.1, PHYSICS_COLOR, "white", 9.5),
    ("Stage 6\nValidation\n(walk-forward +\ncalibration)", 14.3, 4.5, 2.6, 1.6, SHARED_COLOR, "white", 9),
    ("2026 Championship\nPredictions", 17.6, 4.5, 2.3, 1.1, "#2E2E30", "white", 9.5),
]

ARROWS = [
    (1.0, 5.0, 4.1, 6.7),
    (1.0, 4.0, 4.1, 2.3),
    (4.1, 6.7, 7.5, 6.7),
    (7.5, 6.7, 10.9, 6.7),
    (4.1, 2.3, 7.5, 2.3),
    (7.5, 2.3, 10.9, 2.3),
    (10.9, 6.7, 14.3, 4.9),
    (10.9, 2.3, 14.3, 4.1),
    (14.3, 4.5, 17.6, 4.5),
]


def add_box(ax, label, x, y, w, h, facecolor, textcolor, fontsize):
    box = FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=0, facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=fontsize, color=textcolor, fontweight="bold", linespacing=1.4)


def add_arrow(ax, x0, y0, x1, y1):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color="#8A8A8E", shrinkA=2, shrinkB=2,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16.5, 6.5))

    for label, x, y, w, h, facecolor, textcolor, fontsize in BOXES:
        add_box(ax, label, x, y, w, h, facecolor, textcolor, fontsize)

    for x0, y0, x1, y1 in ARROWS:
        add_arrow(ax, x0, y0, x1, y1)

    ax.text(0.2, 7.8, "Classifier-ranked track", fontsize=10.5, color=CLASSIFIER_COLOR, fontweight="bold")
    ax.text(0.2, 0.9, "Physics-based track (v3, production)", fontsize=10.5, color=PHYSICS_COLOR, fontweight="bold")

    ax.set_xlim(0, 20.2)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Final Methodology.png", dpi=200, bbox_inches="tight")
    print(f"Saved to {FIGURES_DIR / 'Final Methodology.png'}")


if __name__ == "__main__":
    main()
