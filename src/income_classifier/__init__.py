"""Income Prediction Classifier — Adult Census Income modeling toolkit."""

from .adaboost import SimpleAdaBoost
from .data import FEATURE_COLUMNS, TARGET_COLUMN, load_adult
from .evaluate import classification_metrics, compare_models
from .pipeline import build_preprocessor, make_model_pipeline
from .train import train_and_evaluate

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "SimpleAdaBoost",
    "build_preprocessor",
    "classification_metrics",
    "compare_models",
    "load_adult",
    "make_model_pipeline",
    "train_and_evaluate",
]

__version__ = "1.1.0"
