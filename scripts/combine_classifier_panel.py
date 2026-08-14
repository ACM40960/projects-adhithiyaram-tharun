"""Combine confusion matrices and ROC curves into a single poster panel."""
import matplotlib.pyplot as plt
from matplotlib.image import imread

from src.config import PROCESSED_DATA_DIR

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].imshow(imread(FIGURES_DIR / "confusion_matrices.png"))
axes[0].axis("off")
axes[1].imshow(imread(FIGURES_DIR / "roc_curves.png"))
axes[1].axis("off")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "classifier_evaluation_panel.png", dpi=200)
print(f"Saved to {FIGURES_DIR / 'classifier_evaluation_panel.png'}")