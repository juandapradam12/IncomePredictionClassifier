"""From-scratch AdaBoost with entropy decision stumps.

Educational implementation that mirrors the classic Freund & Schapire
weighting scheme used in the original project notebook, with bug fixes
and a scikit-learn-style estimator API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def find_splits(col: np.ndarray) -> np.ndarray:
    """Candidate midpoints between sorted unique values of a feature."""
    unique = np.sort(np.unique(col))
    if unique.size < 2:
        return np.array([])
    return (unique[:-1] + unique[1:]) / 2.0


def entropy(class1_n: float, class2_n: float) -> float:
    """Binary Shannon entropy for class counts."""
    if class1_n == 0 or class2_n == 0:
        return 0.0
    total = class1_n + class2_n
    proportions = np.array([class1_n / total, class2_n / total], dtype=float)
    return float((-proportions * np.log2(proportions)).sum())


def ent_from_split(col: np.ndarray, split_value: float, labels: np.ndarray) -> float:
    """Weighted entropy of a binary threshold split."""
    left = labels[col <= split_value]
    right = labels[col > split_value]
    if left.size == 0 or right.size == 0:
        return 1.0

    def _counts(node: np.ndarray) -> tuple[int, int]:
        c1 = int(np.count_nonzero(node))
        return c1, int(len(node) - c1)

    le_c1, le_c2 = _counts(left)
    g_c1, g_c2 = _counts(right)
    total = float(len(col))
    return (len(left) / total) * entropy(le_c1, le_c2) + (
        len(right) / total
    ) * entropy(g_c1, g_c2)


def pred_from_split(
    X: np.ndarray, y: np.ndarray, col_idx: int, split_value: float
) -> tuple[int, int]:
    """Majority-class predictions for left/right children of a stump."""
    col = X[:, col_idx]
    left = y[col <= split_value]
    right = y[col > split_value]

    def _pred(node: np.ndarray) -> int | None:
        if node.size == 0:
            return None
        c1 = int(np.count_nonzero(node))
        c0 = int(node.size - c1)
        if c1 > c0:
            return 1
        if c1 < c0:
            return 0
        return None

    left_pred, right_pred = _pred(left), _pred(right)
    if left_pred is None and right_pred is None:
        return 1, 1
    if left_pred is None:
        left_pred = 1 - int(right_pred)
    if right_pred is None:
        right_pred = 1 - int(left_pred)
    return int(left_pred), int(right_pred)


def simple_binary_tree_fit(X: np.ndarray, y: np.ndarray) -> tuple[int, float, int, int]:
    """Fit a depth-1 entropy stump by exhaustive threshold search."""
    best = (-1, -1.0, 1.0)
    for col_idx, col in enumerate(X.T):
        for split in find_splits(col):
            ent = ent_from_split(col, float(split), y)
            if ent < best[2]:
                best = (col_idx, float(split), ent)

    if best[0] < 0:
        # Degenerate feature matrix — predict global majority.
        majority = int(np.round(y.mean())) if y.size else 0
        return 0, 0.0, majority, majority

    left_pred, right_pred = pred_from_split(X, y, best[0], best[1])
    return best[0], best[1], left_pred, right_pred


def simple_binary_tree_predict(
    X: np.ndarray,
    col_idx: int,
    split_value: float,
    left_pred: int,
    right_pred: int,
) -> np.ndarray:
    """Predict with a fitted depth-1 stump."""
    col = X[:, col_idx]
    if left_pred == right_pred:
        return np.full(len(col), left_pred, dtype=int)
    if left_pred == 1:
        return (col <= split_value).astype(int)
    return (col > split_value).astype(int)


def default_weights(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) / n


def boot_strap_selection(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.choice(len(y), size=len(y), replace=True, p=weights)
    return X[indices], y[indices]


def calc_epsilon(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[y_true != y_pred].sum())


def calc_alpha(epsilon: float) -> float:
    # Clip away from {0, 1} for numerical stability.
    eps = float(np.clip(epsilon, 1e-12, 1 - 1e-12))
    return 0.5 * np.log((1.0 - eps) / eps)


def update_weights(
    weights: np.ndarray, alpha: float, y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    y_pm = np.where(y_true == 0, -1.0, 1.0)
    pred_pm = np.where(y_pred == 0, -1.0, 1.0)
    updated = weights * np.exp(-alpha * y_pm * pred_pm)
    return updated / updated.sum()


@dataclass
class _Stump:
    col_idx: int
    split_value: float
    left_pred: int
    right_pred: int
    alpha: float


class SimpleAdaBoost(BaseEstimator, ClassifierMixin):
    """AdaBoost with decision stumps (bootstrap + reweighting).

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds.
    learning_rate:
        Multiplier applied to each stump's alpha (1.0 recovers the classic form).
    use_sklearn_stumps:
        If True (default), fit ``DecisionTreeClassifier(max_depth=1)`` stumps
        for speed. If False, use the pure-numpy entropy stump search.
    random_state:
        Seed for bootstrap sampling.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        use_sklearn_stumps: bool = True,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.use_sklearn_stumps = use_sklearn_stumps
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "SimpleAdaBoost":
        X, y = check_X_y(X, y, dtype=None)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        weights = default_weights(len(y))
        self.estimators_: list[Any] = []
        self.estimator_weights_: list[float] = []
        self.estimator_errors_: list[float] = []

        for _ in range(self.n_estimators):
            bs_X, bs_y = boot_strap_selection(X, y, weights, rng)

            if self.use_sklearn_stumps:
                stump = DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, random_state=self.random_state
                )
                stump.fit(bs_X, bs_y)
                preds = stump.predict(X).astype(int)
                estimator: Any = stump
            else:
                col_idx, split_value, left_pred, right_pred = simple_binary_tree_fit(
                    bs_X, bs_y
                )
                preds = simple_binary_tree_predict(
                    X, col_idx, split_value, left_pred, right_pred
                )
                estimator = _Stump(col_idx, split_value, left_pred, right_pred, 0.0)

            epsilon = calc_epsilon(y, preds, weights)
            # Skip / stop if the stump is worse than chance.
            if epsilon >= 0.5:
                break

            alpha = self.learning_rate * calc_alpha(epsilon)
            if isinstance(estimator, _Stump):
                estimator.alpha = alpha

            self.estimators_.append(estimator)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(epsilon)
            weights = update_weights(weights, alpha, y, preds)

        if not self.estimators_:
            # Fallback constant classifier.
            majority = int(np.round(y.mean()))
            self.estimators_ = ["constant"]
            self.estimator_weights_ = [1.0]
            self.majority_class_ = majority
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=None)
        if self.estimators_ and self.estimators_[0] == "constant":
            score = 1.0 if self.majority_class_ == 1 else -1.0
            return np.full(X.shape[0], score)

        scores = np.zeros(X.shape[0], dtype=float)
        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            if isinstance(estimator, _Stump):
                preds = simple_binary_tree_predict(
                    X,
                    estimator.col_idx,
                    estimator.split_value,
                    estimator.left_pred,
                    estimator.right_pred,
                )
            else:
                preds = estimator.predict(X).astype(int)
            scores += alpha * np.where(preds == 1, 1.0, -1.0)
        return scores

    def predict(self, X: Any) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        # Squash decision scores into probabilities for metric compatibility.
        scores = self.decision_function(X)
        probs_pos = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probs_pos, probs_pos])
