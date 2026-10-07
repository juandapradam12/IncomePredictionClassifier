"""Evaluation helpers and model comparison utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute core binary classification metrics for the ``>50K`` class."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_proba)
        )
    return metrics


def _positive_proba(model: Any, X: Any) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Rank-preserving squash for AUC.
        return 1.0 / (1.0 + np.exp(-scores))
    return None


def compare_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit each model and return a metrics table sorted by ROC-AUC / F1."""
    rows: list[dict[str, Any]] = []

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = _positive_proba(pipe, X_test)
        metrics = classification_metrics(y_test, y_pred, y_proba)
        row = {"model": name, **metrics}
        rows.append(row)

        if verbose:
            print("=" * 72)
            print(name)
            print(classification_report(y_test, y_pred, digits=4))
            print(metrics)

    results = pd.DataFrame(rows)
    sort_cols = [c for c in ("roc_auc", "f1", "accuracy") if c in results.columns]
    return results.sort_values(sort_cols, ascending=False).reset_index(drop=True)
