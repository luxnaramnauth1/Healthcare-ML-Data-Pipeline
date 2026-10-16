import joblib
import pandas as pd
import pytest

from pipeline.features import engineer_features
from pipeline.ingest import load_raw
from pipeline.predict import predict_df
from pipeline.preprocess import clean
from pipeline.run_pipeline import run
from pipeline.validate import DataValidationError, validate


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_validate_passes_on_real_data(raw):
    report = validate(raw)
    assert report["passed"] is True
    assert report["rows"] == 569
    assert report["missing_fraction"] == 0


def test_validate_rejects_negative_measurements(raw):
    bad = raw.copy()
    bad.loc[0, "mean_radius"] = -1.0
    with pytest.raises(DataValidationError, match="negative"):
        validate(bad)


def test_validate_rejects_non_binary_target(raw):
    bad = raw.copy()
    bad.loc[0, "diagnosis"] = 7
    with pytest.raises(DataValidationError, match="binary"):
        validate(bad)


def test_validate_rejects_too_much_missing_data(raw):
    bad = raw.copy()
    feature_cols = [c for c in bad.columns if c != "diagnosis"]
    bad[feature_cols] = float("nan")
    with pytest.raises(DataValidationError, match="missing"):
        validate(bad)


def test_validate_rejects_missing_target(raw):
    with pytest.raises(DataValidationError, match="target column"):
        validate(raw.drop(columns=["diagnosis"]))


def test_clean_removes_duplicates_and_imputes(raw):
    dirty = raw.copy()
    dirty.loc[3, "mean_radius"] = float("nan")
    dirty = pd.concat([dirty, dirty.iloc[[10]]], ignore_index=True)
    cleaned = clean(dirty)
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) == len(raw)


def test_engineer_features_adds_columns_and_does_not_mutate(raw):
    before = list(raw.columns)
    out = engineer_features(raw)
    assert list(raw.columns) == before
    assert {"area_to_perimeter_ratio", "worst_vs_mean_radius", "concavity_x_compactness"} <= set(out.columns)


def test_end_to_end_run_and_predict(tmp_path, raw):
    metrics = run(output_dir=tmp_path, figures_dir=None)
    assert metrics["roc_auc"] > 0.95
    assert metrics["recall_malignant"] > 0.85
    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "reports" / "data_quality.json").exists()

    model = joblib.load(tmp_path / "models" / "model.joblib")
    preds = predict_df(model, raw.head(10))
    assert len(preds) == 10
    assert set(preds["prediction"]) <= {"malignant", "benign"}
    assert preds["malignant_probability"].between(0, 1).all()
