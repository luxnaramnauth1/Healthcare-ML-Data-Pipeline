"""Stage 1 - ingestion.

Loads the Wisconsin Diagnostic Breast Cancer dataset (569 patients, 30 numeric
tumour measurements) and lands an untouched copy in the raw zone, so every later
stage can be re-run from an immutable source.

To use another healthcare dataset, replace `load_raw` and `TARGET` in config.py.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from pipeline.config import TARGET


def load_raw() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": TARGET})
    # scikit-learn encodes malignant=0 / benign=1. Flip it so that 1 marks the
    # condition of interest (malignant), which is what recall should be measured on.
    df[TARGET] = 1 - df[TARGET]
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def land_raw(df: pd.DataFrame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "breast_cancer_raw.csv"
    df.to_csv(path, index=False)
    return path
