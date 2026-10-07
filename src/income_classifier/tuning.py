"""Hyperparameter search helpers for the Adult income models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .pipeline import build_preprocessor


def _hist_param_distributions(random_state: int) -> dict[str, Any]:
    return {
        "model__learning_rate": np.logspace(-2.0, -0.5, 8),
        "model__max_depth": [3, 4, 5, 6, 8, None],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [10, 20, 40, 80],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        "model__max_iter": [150, 250, 400],
    }


def _rf_param_distributions() -> dict[str, Any]:
    return {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 12, 20, 30],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", 0.3, 0.5],
    }


def make_search_pipeline(model_name: str, *, random_state: int = 42) -> Pipeline:
    """Pipeline whose ``model`` step is suitable for RandomizedSearchCV."""
    if model_name == "hist_gradient_boosting":
        model: Any = HistGradientBoostingClassifier(
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=random_state,
        )
        scale = False
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
        scale = False
    else:
        raise ValueError(
            "Tuning currently supports 'hist_gradient_boosting' and 'random_forest'."
        )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=scale)),
            ("model", model),
        ]
    )


def tune_model(
    model_name: str,
    X,
    y,
    *,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """Run a stratified randomized search and return the fitted search object."""
    pipe = make_search_pipeline(model_name, random_state=random_state)
    if model_name == "hist_gradient_boosting":
        params = _hist_param_distributions(random_state)
    else:
        params = _rf_param_distributions()

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    search.fit(X, y)
    return search
