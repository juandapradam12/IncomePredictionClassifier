"""Train / evaluate entrypoints for the Income Prediction Classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import (
    PROTECTED_ATTRIBUTES,
    load_adult,
    load_adult_official_split,
    split_features_target,
)
from .evaluate import compare_models
from .interpret import (
    compute_permutation_importance,
    fairness_slices,
    metrics_at_threshold,
    plot_permutation_importance,
    positive_proba,
    tune_threshold,
)
from .pipeline import get_estimators, make_model_pipeline
from .tuning import tune_model


def _prepare_splits(
    *,
    split: str,
    data_path: str | Path | None,
    test_path: str | Path | None,
    test_size: float,
    random_state: int,
    drop_unknowns: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if split == "official":
        train_df, test_df = load_adult_official_split(
            train_path=data_path,
            test_path=test_path,
            drop_unknowns=drop_unknowns,
        )
        X_train, y_train = split_features_target(train_df)
        X_test, y_test = split_features_target(test_df)
        return X_train, X_test, y_train, y_test

    if split != "holdout":
        raise ValueError("split must be 'official' or 'holdout'")

    df = load_adult(data_path, drop_unknowns=drop_unknowns)
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_and_evaluate(
    *,
    data_path: str | Path | None = None,
    test_path: str | Path | None = None,
    split: str = "official",
    test_size: float = 0.2,
    random_state: int = 42,
    models: list[str] | None = None,
    drop_unknowns: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Run the benchmark and return ``(metrics, pipelines, meta)``."""
    X_train, X_test, y_train, y_test = _prepare_splits(
        split=split,
        data_path=data_path,
        test_path=test_path,
        test_size=test_size,
        random_state=random_state,
        drop_unknowns=drop_unknowns,
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
    meta = {
        "split": split,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return results, pipes, meta


def run_analysis(
    best_pipe: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    output_dir: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Threshold tuning, permutation importance, and fairness slices."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tune threshold on a validation slice of the training data to avoid
    # leaking the official test labels into the threshold choice.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )
    # Refit on the train portion only for threshold selection.
    from sklearn.base import clone

    temp = clone(best_pipe)
    temp.fit(X_tr, y_tr)
    val_proba = positive_proba(temp, X_val)
    threshold_info = tune_threshold(y_val, val_proba, metric="f1")

    test_proba = positive_proba(best_pipe, X_test)
    default_metrics = metrics_at_threshold(y_test, test_proba, 0.5)
    tuned_metrics = metrics_at_threshold(
        y_test, test_proba, threshold_info["threshold"]
    )

    importance = compute_permutation_importance(
        best_pipe,
        X_test,
        y_test,
        n_repeats=6,
        random_state=random_state,
    )
    importance_path = output_dir / "permutation_importance.csv"
    importance.to_csv(importance_path, index=False)
    fig_path = plot_permutation_importance(
        importance,
        output_path=output_dir.parent / "figures" / "permutation_importance.png"
        if (output_dir.parent / "figures").exists()
        or output_dir.name == "artifacts"
        else output_dir / "permutation_importance.png",
    )
    # Always also write under figures/ when running from repo root.
    repo_fig = Path("figures") / "permutation_importance.png"
    plot_permutation_importance(importance, output_path=repo_fig)

    fairness = fairness_slices(
        X_test,
        y_test,
        test_proba,
        attributes=PROTECTED_ATTRIBUTES,
        threshold=threshold_info["threshold"],
    )
    fairness_path = output_dir / "fairness_slices.csv"
    fairness.to_csv(fairness_path, index=False)

    analysis = {
        "threshold_selection": threshold_info,
        "metrics_at_0_5": default_metrics,
        "metrics_at_tuned_threshold": tuned_metrics,
        "importance_path": str(importance_path),
        "fairness_path": str(fairness_path),
        "importance_figure": str(repo_fig if repo_fig.exists() else fig_path),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(analysis, indent=2))
    return analysis


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
        "--test-path",
        type=Path,
        default=None,
        help="Path to adult.test (official split).",
    )
    parser.add_argument(
        "--split",
        choices=["official", "holdout"],
        default="official",
        help="Evaluation protocol. Default: official UCI train/test.",
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
        "--tune",
        choices=["hist_gradient_boosting", "random_forest"],
        default=None,
        help="Optionally run RandomizedSearchCV for one model before scoring.",
    )
    parser.add_argument("--tune-iter", type=int, default=16)
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run threshold tuning, permutation importance, and fairness slices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for metrics CSV / best model artifact.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    results, pipes, meta = train_and_evaluate(
        data_path=args.data_path,
        test_path=args.test_path,
        split=args.split,
        test_size=args.test_size,
        random_state=args.random_state,
        models=args.models,
        drop_unknowns=args.drop_unknowns,
        verbose=not args.quiet,
    )

    tuning_summary: dict[str, Any] | None = None
    if args.tune:
        if not args.quiet:
            print(f"\nTuning {args.tune} ({args.tune_iter} iterations)...")
        search = tune_model(
            args.tune,
            meta["X_train"],
            meta["y_train"],
            n_iter=args.tune_iter,
            random_state=args.random_state,
        )
        pipes[args.tune] = search.best_estimator_
        y_pred = search.best_estimator_.predict(meta["X_test"])
        y_proba = positive_proba(search.best_estimator_, meta["X_test"])
        from .evaluate import classification_metrics

        tuned_metrics = classification_metrics(meta["y_test"], y_pred, y_proba)
        # Replace / upsert row in results.
        results = results[results["model"] != args.tune]
        results = pd.concat(
            [results, pd.DataFrame([{"model": args.tune, **tuned_metrics}])],
            ignore_index=True,
        )
        sort_cols = [c for c in ("roc_auc", "f1", "accuracy") if c in results.columns]
        results = results.sort_values(sort_cols, ascending=False).reset_index(drop=True)
        tuning_summary = {
            "model": args.tune,
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "test_metrics": tuned_metrics,
        }
        if not args.quiet:
            print("Best params:", search.best_params_)
            print("CV score:", search.best_score_)
            print("Test metrics:", tuned_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    results.to_csv(metrics_path, index=False)

    # Also mirror under results/ for the docs when using official split.
    if args.split == "official":
        Path("results").mkdir(exist_ok=True)
        results.to_csv("results/official_benchmark_metrics.csv", index=False)

    best_name = results.iloc[0]["model"]
    best_pipe = pipes[best_name]
    model_path = args.output_dir / "best_model.joblib"
    joblib.dump(best_pipe, model_path)

    analysis = None
    if args.analyze:
        analysis = run_analysis(
            best_pipe,
            meta["X_train"],
            meta["X_test"],
            meta["y_train"],
            meta["y_test"],
            output_dir=args.output_dir,
            random_state=args.random_state,
        )
        Path("results").mkdir(exist_ok=True)
        for name in ("fairness_slices.csv", "permutation_importance.csv"):
            src = args.output_dir / name
            if src.exists():
                pd.read_csv(src).to_csv(Path("results") / name, index=False)

    summary = {
        "best_model": best_name,
        "split": args.split,
        "n_train": meta["n_train"],
        "n_test": meta["n_test"],
        "metrics": results.iloc[0].to_dict(),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "tuning": tuning_summary,
        "analysis": analysis,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    if tuning_summary:
        Path("results").mkdir(exist_ok=True)
        Path("results/tuning_summary.json").write_text(
            json.dumps(tuning_summary, indent=2, default=str)
        )

    print("\nLeaderboard:")
    print(results.to_string(index=False))
    print(f"\nSaved metrics → {metrics_path}")
    print(f"Saved best model ({best_name}) → {model_path}")
    if analysis:
        print(
            "Tuned threshold:",
            analysis["threshold_selection"]["threshold"],
            "F1@tuned:",
            analysis["metrics_at_tuned_threshold"]["f1"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
