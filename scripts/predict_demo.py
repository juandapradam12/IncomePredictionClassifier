#!/usr/bin/env python3
"""CLI demo: score a single Adult-style profile or a CSV of profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from income_classifier.data import FEATURE_COLUMNS  # noqa: E402
from income_classifier.interpret import positive_proba  # noqa: E402

EXAMPLE = {
    "age": 39,
    "education-num": 13,
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "workclass": "State-gov",
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States",
}


def _load_model(path: Path):
    if path.exists():
        return joblib.load(path)
    # Train a quick default model if none is saved yet.
    from income_classifier.data import load_adult_official_split, split_features_target
    from income_classifier.pipeline import make_model_pipeline

    train_df, _ = load_adult_official_split()
    X, y = split_features_target(train_df)
    pipe = make_model_pipeline("hist_gradient_boosting")
    pipe.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return pipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Adult income profiles.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "best_model.joblib",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV with Adult feature columns.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for >50K.",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Score the built-in example profile.",
    )
    args = parser.parse_args(argv)

    model = _load_model(args.model)

    if args.csv is not None:
        frame = pd.read_csv(args.csv)
    else:
        # Default: score the built-in example profile.
        frame = pd.DataFrame([EXAMPLE])

    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    X = frame[FEATURE_COLUMNS]
    proba = positive_proba(model, X)
    preds = (proba >= args.threshold).astype(int)

    out = frame.copy()
    out["proba_gt_50k"] = proba
    out["pred_gt_50k"] = preds
    out["label"] = out["pred_gt_50k"].map({1: ">50K", 0: "<=50K"})
    print(out[["proba_gt_50k", "pred_gt_50k", "label"]].to_string(index=False))
    if len(out) == 1:
        print(json.dumps({"profile": EXAMPLE if args.csv is None else "csv-row-0",
                          "probability": float(proba[0]),
                          "prediction": out.loc[0, "label"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
