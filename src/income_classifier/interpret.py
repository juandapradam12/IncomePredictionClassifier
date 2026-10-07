"""Interpretability helpers: threshold tuning, importance, fairness slices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from .evaluate import classification_metrics


def positive_proba(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model must expose predict_proba or decision_function.")


def tune_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    *,
    metric: str = "f1",
) -> dict[str, float]:
    """Pick the probability threshold that maximizes ``metric`` on labeled data.

    Uses the precision–recall curve candidate thresholds (plus 0.5).
    """
    y_true_arr = np.asarray(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_proba)
    # precision/recall arrays are one longer than thresholds.
    candidates = np.unique(np.concatenate([thresholds, [0.5]]))

    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for thr in candidates:
        preds = (y_proba >= thr).astype(int)
        prec = float(precision_score(y_true_arr, preds, zero_division=0))
        rec = float(recall_score(y_true_arr, preds, zero_division=0))
        f1 = float(f1_score(y_true_arr, preds, zero_division=0))
        score = {"precision": prec, "recall": rec, "f1": f1}[metric]
        if score > best[metric]:
            best = {
                "threshold": float(thr),
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
    return best


def metrics_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = (y_proba >= threshold).astype(int)
    return classification_metrics(y_true, preds, y_proba)


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 8,
    random_state: int = 42,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Column-level permutation importance on the raw feature frame."""
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    table = (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return table


def plot_permutation_importance(
    importance: pd.DataFrame,
    *,
    top_n: int = 12,
    title: str = "Permutation importance (ROC-AUC drop)",
    output_path: str | Path | None = None,
) -> Path | None:
    plot_df = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        color="#0B6E4F",
        alpha=0.9,
    )
    ax.set_xlabel("Mean importance")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is None:
        plt.close(fig)
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fairness_slices(
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    attributes: Iterable[str] = ("sex", "race"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-group selection rate / TPR / FPR / precision / recall / accuracy."""
    y_true_arr = np.asarray(y_true).astype(int)
    preds = (y_proba >= threshold).astype(int)
    rows: list[dict[str, Any]] = []

    for attr in attributes:
        if attr not in X.columns:
            continue
        series = X[attr].astype(str).fillna("Missing")
        for group, idx in series.groupby(series).groups.items():
            index = np.asarray(list(idx))
            yt = y_true_arr[index]
            yp = preds[index]
            if yt.size == 0:
                continue
            tp = int(((yt == 1) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            rows.append(
                {
                    "attribute": attr,
                    "group": group,
                    "n": int(yt.size),
                    "base_rate": float(yt.mean()),
                    "selection_rate": float(yp.mean()),
                    "tpr": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                    "fpr": float(fp / (fp + tn)) if (fp + tn) else np.nan,
                    "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                    "recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                    "accuracy": float((tp + tn) / yt.size),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["attribute", "n"], ascending=[True, False]
    ).reset_index(drop=True)
