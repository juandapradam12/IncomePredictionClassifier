"""Train / evaluate entrypoints for the Income Prediction Classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_adult, split_features_target
from .evaluate import compare_models
from .pipeline import get_estimators, make_model_pipeline


def train_and_evaluate(
    *,
    data_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    models: list[str] | None = None,
    drop_unknowns: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the full benchmark and return ``(metrics_table, fitted_pipelines)``."""
    df = load_adult(data_path, drop_unknowns=drop_unknowns)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    available = list(get_estimators(random_state=random_state))
    selected = models or available
    unknown = [m for m in selected if m not in available]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {available}")

    pipes = {
        name: make_model_pipeline(name, random_state=random_state) for name in selected
    }
    results = compare_models(
        pipes, X_train, X_test, y_train, y_test, verbose=verbose
    )
    return results, pipes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark income classifiers on the Adult Census dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to adult.data (defaults to repo root).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to train. Default: all.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--drop-unknowns",
        action="store_true",
        help="Drop rows with '?' missing categorical values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for metrics CSV / best model artifact.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    results, pipes = train_and_evaluate(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        models=args.models,
        drop_unknowns=args.drop_unknowns,
        verbose=not args.quiet,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    results.to_csv(metrics_path, index=False)

    best_name = results.iloc[0]["model"]
    best_pipe = pipes[best_name]
    model_path = args.output_dir / "best_model.joblib"
    joblib.dump(best_pipe, model_path)

    summary = {
        "best_model": best_name,
        "metrics": results.iloc[0].to_dict(),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nLeaderboard:")
    print(results.to_string(index=False))
    print(f"\nSaved metrics → {metrics_path}")
    print(f"Saved best model ({best_name}) → {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
