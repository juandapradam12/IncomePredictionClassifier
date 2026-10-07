"""Unit tests for AdaBoost, data loading, and analysis helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from income_classifier.adaboost import (  # noqa: E402
    SimpleAdaBoost,
    calc_alpha,
    calc_epsilon,
    default_weights,
    ent_from_split,
    find_splits,
    update_weights,
)
from income_classifier.data import (  # noqa: E402
    load_adult,
    load_adult_official_split,
    split_features_target,
)
from income_classifier.interpret import (  # noqa: E402
    fairness_slices,
    tune_threshold,
)
from income_classifier.pipeline import make_model_pipeline  # noqa: E402


def test_find_splits_midpoints():
    splits = find_splits(np.array([1.0, 2.0, 3.0, 3.0]))
    assert np.allclose(splits, [1.5, 2.5])


def test_find_splits_single_value():
    assert find_splits(np.array([1.0, 1.0, 1.0])).size == 0


def test_entropy_split_prefers_pure_cut():
    col = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    pure = ent_from_split(col, 0.5, labels)
    mixed = ent_from_split(col, -1.0, labels)
    assert pure < mixed


def test_weight_update_increases_mistakes():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])  # mistake on index 2
    weights = default_weights(4)
    alpha = calc_alpha(calc_epsilon(y_true, y_pred, weights))
    updated = update_weights(weights, alpha, y_true, y_pred)
    assert updated[2] > updated[0]
    assert pytest.approx(updated.sum(), rel=1e-9) == 1.0


def test_simple_adaboost_fits_and_predicts():
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        weights=[0.7, 0.3],
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = SimpleAdaBoost(n_estimators=40, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds)).issubset({0, 1})
    assert model.score(X_test, y_test) > 0.7


def test_load_adult_and_pipeline_smoke():
    df = load_adult()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = make_model_pipeline("logistic_regression", random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)
    assert pipe.score(X_test, y_test) > 0.75


def test_official_split_shapes_and_labels():
    train_df, test_df = load_adult_official_split()
    assert len(train_df) == 32561
    assert len(test_df) == 16281
    assert set(train_df["income"].unique()) == {"<=50K", ">50K"}
    assert set(test_df["income"].unique()) == {"<=50K", ">50K"}


def test_threshold_and_fairness_helpers():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9, 0.55, 0.2, 0.7, 0.3])
    best = tune_threshold(y_true, y_proba, metric="f1")
    assert 0.0 <= best["threshold"] <= 1.0
    assert best["f1"] >= 0.0

    X = pd.DataFrame(
        {
            "sex": ["Male", "Male", "Female", "Female", "Male", "Female", "Male", "Female"],
            "race": ["White"] * 8,
        }
    )
    table = fairness_slices(X, y_true, y_proba, attributes=["sex"], threshold=0.5)
    assert set(table["attribute"]) == {"sex"}
    assert table["n"].sum() == 8
