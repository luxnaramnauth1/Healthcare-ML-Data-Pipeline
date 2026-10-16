"""Stage 3 - cleaning."""
import pandas as pd

from pipeline.config import TARGET


def clean(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Drop rows without a label, drop duplicates, impute missing features with the median."""
    out = df.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    feature_cols = [c for c in out.columns if c != target]
    out[feature_cols] = out[feature_cols].fillna(out[feature_cols].median())
    return out
