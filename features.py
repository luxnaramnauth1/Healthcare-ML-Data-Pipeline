"""Stage 4 - feature engineering."""
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived shape features. Pure function: returns a new frame, never mutates input."""
    out = df.copy()
    out["area_to_perimeter_ratio"] = out["mean_area"] / out["mean_perimeter"].replace(0, float("nan"))
    out["worst_vs_mean_radius"] = out["worst_radius"] / out["mean_radius"].replace(0, float("nan"))
    out["concavity_x_compactness"] = out["mean_concavity"] * out["mean_compactness"]
    return out
