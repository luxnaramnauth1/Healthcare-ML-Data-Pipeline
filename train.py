"""Stage 5 - model selection and training.

Three candidate models are compared with stratified cross-validation on the training
split only. The best one (by ROC-AUC) is refit on the full training split. The test
split is never touched until evaluation.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.config import CV_FOLDS, RANDOM_STATE, TARGET, TEST_SIZE


def candidate_models() -> dict[str, Pipeline]:
    def wrap(clf):
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])

    return {
        "logistic_regression": wrap(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "random_forest": wrap(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "gradient_boosting": wrap(GradientBoostingClassifier(random_state=RANDOM_STATE)),
    }


def split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def select_best_model(X_train: pd.DataFrame, y_train: pd.Series):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict[str, float]] = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        }
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    best_model = candidate_models()[best_name]
    best_model.fit(X_train, y_train)
    return best_name, best_model, results
