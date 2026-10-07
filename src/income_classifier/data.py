"""Load and clean the UCI Adult (Census Income) dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

COLUMN_NAMES: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

# Sampling weight — useful for population estimates, not as a predictive feature.
DROP_COLUMNS: list[str] = ["fnlwgt"]

# Redundant with education-num; keep ordinal education-num instead.
OPTIONAL_DROP: list[str] = ["education"]

NUMERIC_FEATURES: list[str] = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "income"


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "adult.data"


def load_adult(
    path: str | Path | None = None,
    *,
    drop_unknowns: bool = False,
    drop_education_label: bool = True,
) -> pd.DataFrame:
    """Load Adult Census Income CSV into a cleaned DataFrame.

    Parameters
    ----------
    path:
        Path to ``adult.data``. Defaults to the repository root file.
    drop_unknowns:
        If True, drop rows containing ``?`` in any categorical field.
        If False (default), keep them so the preprocessor can impute.
    drop_education_label:
        If True, drop the string ``education`` column (kept as ``education-num``).
    """
    data_path = Path(path) if path is not None else _default_data_path()
    df = pd.read_csv(
        data_path,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
    )

    # Normalize target labels (some Adult dumps use trailing periods).
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.replace(r"\.$", "", regex=True)
    )

    for col in CATEGORICAL_FEATURES + ["education", TARGET_COLUMN]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Treat Adult's missing-value sentinel as true missingness.
    df = df.replace("?", pd.NA)

    drop_cols: list[str] = list(DROP_COLUMNS)
    if drop_education_label:
        drop_cols.extend(OPTIONAL_DROP)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if drop_unknowns:
        df = df.dropna().reset_index(drop=True)

    return df


def split_features_target(
    df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with binary target ``1`` for ``>50K``."""
    cols = list(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")

    X = df[cols].copy()
    y = (df[TARGET_COLUMN] == ">50K").astype(int)
    return X, y
