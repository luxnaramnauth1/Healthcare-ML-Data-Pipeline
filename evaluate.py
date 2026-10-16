"""Stage 6 - evaluation: metrics, permutation importance and report figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from pipeline.config import RANDOM_STATE  # noqa: E402

# Chart colours (single-series blue, neutral chrome) on a light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
BLUE_LIGHT = "#cde2fb"


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    imp = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE
    )
    importances = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_malignant": float(precision_score(y_test, preds)),
        "recall_malignant": float(recall_score(y_test, preds)),
        "f1_malignant": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_rows": int(len(y_test)),
        "top_features": {k: round(float(v), 4) for k, v in importances.head(10).items()},
    }


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_fig(w=6.4, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    _style(ax)
    return fig, ax


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_roc(model, X_test, y_test, path: Path, auc: float):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    fig, ax = _new_fig(5.2, 4.6)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(0.58, 0.5, "chance", color=MUTED, fontsize=9, rotation=34)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2)
    ax.set_xlabel("False positive rate", color=INK_2, fontsize=10)
    ax.set_ylabel("True positive rate (recall)", color=INK_2, fontsize=10)
    ax.set_title(f"ROC curve on held-out patients (AUC {auc:.3f})", loc="left", color=INK, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _save(fig, path)


def plot_confusion(cm: dict, path: Path):
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    cmap = LinearSegmentedColormap.from_list("seq_blue", [BLUE_LIGHT, BLUE_DARK])
    fig, ax = plt.subplots(figsize=(4.8, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max())
    for i in range(2):
        for j in range(2):
            dark_cell = mat[i, j] > mat.max() * 0.5
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=20,
                    color="#ffffff" if dark_cell else INK)
    ax.set_xticks([0, 1], ["Benign", "Malignant"], color=INK_2)
    ax.set_yticks([0, 1], ["Benign", "Malignant"], color=INK_2)
    ax.set_xlabel("Predicted", color=INK_2, fontsize=10)
    ax.set_ylabel("Actual", color=INK_2, fontsize=10)
    ax.set_title("Confusion matrix (test set)", loc="left", color=INK, fontsize=11)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    _save(fig, path)


def plot_importance(top_features: dict, path: Path):
    names = list(top_features.keys())[::-1]
    vals = [top_features[n] for n in names]
    fig, ax = _new_fig(6.6, 4.4)
    ax.grid(axis="y", visible=False)
    ax.barh(names, vals, height=0.5, color=BLUE)
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, y, f"{v:.3f}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], color=INK_2, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled", color=INK_2, fontsize=10)
    ax.set_title("What drives the predictions (top 10 features)", loc="left", color=INK, fontsize=11)
    _save(fig, path)


def plot_model_comparison(results: dict, best: str, path: Path):
    names = list(results.keys())
    means = [results[n]["cv_roc_auc_mean"] for n in names]
    stds = [results[n]["cv_roc_auc_std"] for n in names]
    order = np.argsort(means)
    fig, ax = _new_fig(6.6, 3.2)
    ax.grid(axis="y", visible=False)
    for pos, idx in enumerate(order):
        ax.errorbar(means[idx], pos, xerr=stds[idx], fmt="o", color=BLUE, ecolor=BLUE,
                    elinewidth=2, capsize=4, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        label = f"{means[idx]:.3f}" + ("  selected" if names[idx] == best else "")
        ax.text(means[idx], pos + 0.22, label, ha="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(names)), [names[i].replace("_", " ") for i in order], color=INK_2, fontsize=10)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.set_xlabel("Cross-validated ROC-AUC (mean, one standard deviation)", color=INK_2, fontsize=10)
    ax.set_title("Model selection on the training split", loc="left", color=INK, fontsize=11)
    _save(fig, path)


def make_figures(model, X_test, y_test, metrics: dict, cv_results: dict, best: str, directory: Path):
    plot_roc(model, X_test, y_test, directory / "roc_curve.png", metrics["roc_auc"])
    plot_confusion(metrics["confusion_matrix"], directory / "confusion_matrix.png")
    plot_importance(metrics["top_features"], directory / "feature_importance.png")
    plot_model_comparison(cv_results, best, directory / "model_comparison.png")
