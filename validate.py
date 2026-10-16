"""Stage 2 - data-quality validation.

Hard failures raise `DataValidationError` and stop the run before any model is
trained. Soft issues are collected as warnings and written to the data-quality report.
"""
import pandas as pd

from pipeline.config import MAX_MISSING_FRACTION, MIN_CLASS_SHARE, MIN_ROWS, TARGET


class DataValidationError(Exception):
    """Raised when the data fails a hard quality check."""


def validate(df: pd.DataFrame, target: str = TARGET) -> dict:
    if target not in df.columns:
        raise DataValidationError(f"target column '{target}' is missing")

    errors: list[str] = []
    warnings: list[str] = []

    if len(df) < MIN_ROWS:
        errors.append(f"only {len(df)} rows, need at least {MIN_ROWS}")

    if not set(df[target].dropna().unique()) <= {0, 1}:
        errors.append("target must be binary (0/1)")

    features = df.drop(columns=[target])
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        errors.append(f"non-numeric feature columns: {non_numeric}")

    missing_fraction = float(df.isna().mean().mean())
    if missing_fraction > MAX_MISSING_FRACTION:
        errors.append(f"{missing_fraction:.1%} of values missing (limit {MAX_MISSING_FRACTION:.0%})")

    negatives = int((features.select_dtypes("number") < 0).sum().sum())
    if negatives:
        errors.append(f"{negatives} negative values in physical measurements")

    duplicates = int(df.duplicated().sum())
    if duplicates:
        warnings.append(f"{duplicates} duplicate rows (removed in cleaning)")

    balance = {str(k): float(v) for k, v in df[target].value_counts(normalize=True).round(3).items()}
    if balance and min(balance.values()) < MIN_CLASS_SHARE:
        warnings.append("severe class imbalance, consider resampling or class weights")

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_fraction": round(missing_fraction, 4),
        "duplicate_rows": duplicates,
        "negative_measurements": negatives,
        "class_balance": balance,
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }
    if errors:
        raise DataValidationError("; ".join(errors))
    return report
