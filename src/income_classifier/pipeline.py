"""Preprocessing and model pipeline builders."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .adaboost import SimpleAdaBoost
from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """ColumnTransformer with median/most-frequent imputation + one-hot encoding.

    ``scale_numeric`` is useful for linear models; tree ensembles generally do
    not need scaled inputs.
    """
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_estimators(random_state: int = 42) -> dict[str, Any]:
    """Return named estimators used in the project benchmark."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        ),
        "adaboost_sklearn": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
            n_estimators=200,
            learning_rate=0.8,
            random_state=random_state,
        ),
        "adaboost_from_scratch": SimpleAdaBoost(
            n_estimators=100,
            learning_rate=1.0,
            use_sklearn_stumps=True,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def make_model_pipeline(
    model_name: str,
    *,
    random_state: int = 42,
) -> Pipeline:
    """Build a full preprocess → model pipeline for ``model_name``."""
    estimators = get_estimators(random_state=random_state)
    if model_name not in estimators:
        known = ", ".join(sorted(estimators))
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {known}")

    scale = model_name == "logistic_regression"
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=scale)),
            ("model", estimators[model_name]),
        ]
    )
