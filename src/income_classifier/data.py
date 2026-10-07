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
PROTECTED_ATTRIBUTES: list[str] = ["sex", "race"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_data_path() -> Path:
    return _repo_root() / "adult.data"


def _default_test_path() -> Path:
    return _repo_root() / "adult.test"


def _clean_frame(
    df: pd.DataFrame,
    *,
    drop_unknowns: bool,
    drop_education_label: bool,
) -> pd.DataFrame:
    df = df.copy()

    # Normalize target labels (Adult test uses trailing periods).
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.replace(r"\.$", "", regex=True)
    )

    for col in CATEGORICAL_FEATURES + ["education", TARGET_COLUMN]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop malformed rows (blank lines sometimes appear in Adult dumps).
    df = df[df[TARGET_COLUMN].isin(["<=50K", ">50K"])].copy()

    # Treat Adult's missing-value sentinel as true missingness.
    df = df.replace("?", pd.NA)
    df = df.replace({"nan": pd.NA, "None": pd.NA})

    drop_cols: list[str] = list(DROP_COLUMNS)
    if drop_education_label:
        drop_cols.extend(OPTIONAL_DROP)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if drop_unknowns:
        df = df.dropna().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def load_adult(
    path: str | Path | None = None,
    *,
    drop_unknowns: bool = False,
    drop_education_label: bool = True,
    skiprows: int | None = None,
) -> pd.DataFrame:
    """Load an Adult CSV dump (``adult.data`` or ``adult.test``) into a DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV. Defaults to the repository ``adult.data``.
    drop_unknowns:
        If True, drop rows containing ``?`` in any field.
    drop_education_label:
        If True, drop the string ``education`` column (kept as ``education-num``).
    skiprows:
        Optional explicit row skip. For ``adult.test``, defaults to 1
        (skips the ``|1x3 Cross validator`` header line).
    """
    data_path = Path(path) if path is not None else _default_data_path()
    if skiprows is None and data_path.name.lower().endswith(".test"):
        skiprows = 1

    df = pd.read_csv(
        data_path,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        skiprows=skiprows or 0,
    )
    return _clean_frame(
        df,
        drop_unknowns=drop_unknowns,
        drop_education_label=drop_education_label,
    )


def load_adult_official_split(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    *,
    drop_unknowns: bool = False,
    drop_education_label: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official UCI Adult train/test split."""
    train = load_adult(
        train_path or _default_data_path(),
        drop_unknowns=drop_unknowns,
        drop_education_label=drop_education_label,
        skiprows=0,
    )
    test = load_adult(
        test_path or _default_test_path(),
        drop_unknowns=drop_unknowns,
        drop_education_label=drop_education_label,
    )
    return train, test


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
