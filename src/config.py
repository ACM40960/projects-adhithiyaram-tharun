"""Project-wide configuration: paths, seasons, and constants.

See docs/market_analysis.md for the reasoning behind the season split and
the constructor name mapping.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = PROJECT_ROOT / "f1_cache"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_SEASONS = [2022, 2023, 2024]
VALIDATION_SEASON = 2025
PREDICTION_SEASON = 2026
ALL_SEASONS = TRAIN_SEASONS + [VALIDATION_SEASON, PREDICTION_SEASON]

RACE_SESSION = "R"

# Minimum delay between consecutive fastf1 requests. jolpica-f1
# (api.jolpi.ca), one of fastf1's data sources, enforces a public rate
# limit; fetching rounds back-to-back with no delay triggers 429s. This
# matters most on a cold cache, where there's no cached response to fall
# back to. See src/data_fetch.py.
FASTF1_REQUEST_DELAY_SECONDS = 1.5

REGULATION_ERA = {
    2022: "ground_effect_2022_2025",
    2023: "ground_effect_2022_2025",
    2024: "ground_effect_2022_2025",
    2025: "ground_effect_2022_2025",
    2026: "next_gen_2026",
}

# Fixed regardless of which seasons are present in a given DataFrame slice.
# pd.get_dummies only emits columns for categories it actually sees, and
# training data (2022-2025) vs. 2026 prediction data each contain only one
# era -- deriving era columns per-slice would give the two sides mismatched
# feature sets at predict time. See src/model.py, prepare_model_frame.
ERA_COLUMNS = [f"era_{era}" for era in sorted(set(REGULATION_ERA.values()))]

# Maps historical constructor names onto their 2026 identity so that
# team-level features join correctly across a rebrand. See
# docs/market_analysis.md, section 3. Verify against the exact strings
# fastf1 returns before relying on this for feature engineering.
CONSTRUCTOR_NAME_MAP = {
    "Alfa Romeo": "Audi",
    "Alfa Romeo Racing": "Audi",
    "Sauber": "Audi",
    "Kick Sauber": "Audi",
    "Stake F1 Team Kick Sauber": "Audi",
    "AlphaTauri": "Racing Bulls",
    "Scuderia AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Visa Cash App RB": "Racing Bulls",
    "Visa Cash App RB Formula One Team": "Racing Bulls",
}

# Teams with no prior-season history in this dataset; see docs/market_analysis.md.
NEW_ENTRANTS_2026 = ["Cadillac"]

for _directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MODELS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)