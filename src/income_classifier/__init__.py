"""Income Prediction Classifier — Adult Census Income modeling toolkit."""

from .adaboost import SimpleAdaBoost
from .data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_adult,
    load_adult_official_split,
)
from .evaluate import classification_metrics, compare_models
from .interpret import fairness_slices, tune_threshold
from .pipeline import build_preprocessor, make_model_pipeline
from .train import train_and_evaluate

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "SimpleAdaBoost",
    "build_preprocessor",
    "classification_metrics",
    "compare_models",
    "fairness_slices",
    "load_adult",
    "load_adult_official_split",
    "make_model_pipeline",
    "train_and_evaluate",
    "tune_threshold",
]

__version__ = "1.2.0"
