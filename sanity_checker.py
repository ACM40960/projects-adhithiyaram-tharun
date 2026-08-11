#manually check that the features are being calculated correctly for a given driver and season
import pandas as pd

df = pd.read_csv("data/processed/features.csv")

driver = "VER"  # swap for any 3-letter code you know appears in your data
season = 2021

check = (
    df[(df["driver_code"] == driver) & (df["season"] == season)]
    .sort_values("round")
    [["round", "finish_position", "driver_avg_finish_last3", "driver_points_last3", "driver_form_ewm"]]
)

print(check.to_string(index=False))