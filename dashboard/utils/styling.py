"""Shared Plotly theme and colour palette for the dashboard."""
MODEL_COLOR = "#4E79A7"
MARKET_COLOR = "#F28E2B"
PLOTLY_TEMPLATE = "plotly_white"

# Single-hue sequential ramp (light -> dark) for magnitude/probability heatmaps,
# same hue family as MODEL_COLOR so the palette reads as one system.
SEQUENTIAL_SCALE = "Blues"

# 7-slot categorical palette for the Home page's nav cards, CVD-safe.
NAV_CARD_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
]