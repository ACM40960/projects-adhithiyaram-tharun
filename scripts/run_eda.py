"""Exploratory data analysis on fetched race results."""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_combined_results() -> pd.DataFrame:
    """Load the combined CSV produced by fetch_all_seasons()."""
    path = RAW_DATA_DIR / "race_results_all.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_fetch` first to fetch data."
        )
    return pd.read_csv(path)


def summarize_structure(df: pd.DataFrame) -> None:
    """Log shape, dtypes, missingness, and season/era coverage."""
    logger.info("Shape: %s rows x %s columns", *df.shape)
    logger.info("Columns and dtypes:\n%s", df.dtypes)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    logger.info("Missing values per column:\n%s", missing if not missing.empty else "None")

    logger.info("Seasons covered: %s", sorted(df["season"].unique()))
    logger.info("Races per season:\n%s", df.groupby("season")["round"].nunique())
    logger.info("Rows per regulation era:\n%s", df.groupby("regulation_era").size())
    logger.info("Constructors per season:\n%s", df.groupby("season")["constructor"].nunique())


def plot_grid_vs_finish_by_era(df: pd.DataFrame) -> None:
    """Scatter of grid vs. finish position, split by regulation era."""
    finished = df.dropna(subset=["grid_position", "finish_position"]).copy()
    finished["finish_position"] = pd.to_numeric(finished["finish_position"], errors="coerce")
    finished = finished.dropna(subset=["finish_position"])

    eras = sorted(finished["regulation_era"].unique())
    fig, axes = plt.subplots(1, len(eras), figsize=(6 * len(eras), 6), squeeze=False)

    for ax, era in zip(axes[0], eras):
        subset = finished[finished["regulation_era"] == era]
        ax.scatter(subset["grid_position"], subset["finish_position"], alpha=0.3, s=15)
        ax.plot([1, 22], [1, 22], color="red", linestyle="--", label="grid = finish")
        ax.set_xlabel("Grid position")
        ax.set_ylabel("Finish position")
        ax.set_title(era)
        ax.legend()

    fig.tight_layout()
    out_path = FIGURES_DIR / "grid_vs_finish_by_era.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_points_distribution(df: pd.DataFrame) -> None:
    """Histogram of points scored per race entry."""
    fig, ax = plt.subplots(figsize=(6, 4))
    df["points"].plot(kind="hist", bins=26, ax=ax)
    ax.set_xlabel("Points scored")
    ax.set_title("Distribution of points scored per race entry")
    fig.tight_layout()

    out_path = FIGURES_DIR / "points_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_constructors_per_season(df: pd.DataFrame) -> None:
    """Bar chart of constructor count per season, after name normalization."""
    counts = df.groupby("season")["constructor"].nunique()

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Season")
    ax.set_ylabel("Distinct constructors")
    ax.set_title("Constructors per season (post name-normalization)")
    fig.tight_layout()

    out_path = FIGURES_DIR / "constructors_per_season.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    df = load_combined_results()
    summarize_structure(df)
    plot_grid_vs_finish_by_era(df)
    plot_points_distribution(df)
    plot_constructors_per_season(df)
    logger.info("EDA complete. Figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
