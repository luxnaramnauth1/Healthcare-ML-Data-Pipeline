"""Orchestrates the full pipeline.

    python -m pipeline.run_pipeline              # run everything, write figures to docs/images
    python -m pipeline.run_pipeline --skip-plots # metrics and model only
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

from pipeline import __version__
from pipeline.config import FIGURES_DIR, OUTPUT, TARGET
from pipeline.evaluate import evaluate, make_figures
from pipeline.features import engineer_features
from pipeline.ingest import land_raw, load_raw
from pipeline.preprocess import clean
from pipeline.train import select_best_model, split
from pipeline.validate import validate

log = logging.getLogger("pipeline")


def run(output_dir: Path = OUTPUT, figures_dir: Path | None = FIGURES_DIR) -> dict:
    started = time.time()
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    for d in (processed_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info("1/6 ingest")
    raw = load_raw()
    land_raw(raw, raw_dir)

    log.info("2/6 validate")
    quality = validate(raw)
    (reports_dir / "data_quality.json").write_text(json.dumps(quality, indent=2))

    log.info("3/6 clean")
    cleaned = clean(raw)

    log.info("4/6 engineer features")
    features = engineer_features(cleaned)
    features.to_csv(processed_dir / "features.csv", index=False)

    log.info("5/6 train (model selection by cross-validation)")
    X_train, X_test, y_train, y_test = split(features)
    best_name, model, cv_results = select_best_model(X_train, y_train)

    log.info("6/6 evaluate (best model: %s)", best_name)
    metrics = evaluate(model, X_test, y_test)
    metrics["selected_model"] = best_name
    metrics["model_comparison"] = cv_results
    metrics["train_rows"] = int(len(y_train))
    metrics["feature_count"] = int(X_train.shape[1])
    metrics["run"] = {
        "pipeline_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": TARGET,
        "seconds": round(time.time() - started, 1),
    }

    joblib.dump(model, models_dir / "model.joblib")
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if figures_dir is not None:
        make_figures(model, X_test, y_test, metrics, cv_results, best_name, figures_dir)

    log.info(
        "done in %.1fs | model=%s | accuracy=%.3f | recall=%.3f | roc_auc=%.3f",
        metrics["run"]["seconds"], best_name, metrics["accuracy"],
        metrics["recall_malignant"], metrics["roc_auc"],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--skip-plots", action="store_true", help="do not write report figures")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(figures_dir=None if args.skip_plots else FIGURES_DIR)


if __name__ == "__main__":
    main()
