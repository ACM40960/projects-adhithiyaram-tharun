import pandas as pd
df = pd.read_csv("data/processed/features.csv")
print(df["grid_position"].isna().sum(), "missing grid_position values")
print(df["grid_position"].describe())