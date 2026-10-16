"""Score patients with the trained model.

    python -m pipeline.predict                         # scores 5 sample rows
    python -m pipeline.predict --input patients.csv    # CSV with the 30 raw measurement columns
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from pipeline.config import OUTPUT, TARGET
from pipeline.features import engineer_features


def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    X = engineer_features(df.drop(columns=[TARGET], errors="ignore"))
    X = X[list(model.feature_names_in_)]
    result = pd.DataFrame(index=df.index)
    result["malignant_probability"] = model.predict_proba(X)[:, 1].round(4)
    result["prediction"] = (result["malignant_probability"] >= 0.5).map({True: "malignant", False: "benign"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, help="CSV of raw measurements (default: 5 sample rows)")
    parser.add_argument("--model", type=Path, default=OUTPUT / "models" / "model.joblib")
    args = parser.parse_args()

    model = joblib.load(args.model)
    if args.input:
        df = pd.read_csv(args.input)
        out = predict_df(model, df)
    else:
        df = pd.read_csv(OUTPUT / "raw" / "breast_cancer_raw.csv").sample(5, random_state=7)
        out = predict_df(model, df)
        out["actual"] = df[TARGET].map({1: "malignant", 0: "benign"})
    print(out.to_string())


if __name__ == "__main__":
    main()
