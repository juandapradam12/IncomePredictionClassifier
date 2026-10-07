# Income Prediction Classifier

**Predict whether a person earns more than \$50K/year — and understand *how* boosting gets you there.**

This project pairs a **from-scratch AdaBoost implementation** (decision stumps, ε / α weight updates, bootstrap reweighting) with a **modern tabular ML benchmark** on the UCI Adult Census Income dataset. It started as an educational deep-dive into boosting; it now ships as a reproducible package that rivals scikit-learn’s AdaBoost and beats the original feature subset with histogram gradient boosting.

<p align="center">
  <img src="figures/benchmark_comparison.png" alt="Model benchmark comparison" width="900" />
</p>

## Why this project

| Pillar | What you get |
|--------|----------------|
| **Theory you can read** | AdaBoost math implemented in clear Python — not hidden inside a C++ extension |
| **Parity with sklearn** | Custom AdaBoost ≈ `AdaBoostClassifier` on the same Adult holdout (~85.8% vs ~85.9% accuracy) |
| **Stronger models** | Full feature set + `HistGradientBoostingClassifier` → **ROC-AUC 0.931**, best F1 for `>50K` |
| **Portfolio-ready** | Installable package, CLI benchmark, tests, and documentation — not just a notebook dump |

## Headline results

Stratified 80/20 split on `adult.data` (`random_state=42`):

| Model | ROC-AUC | F1 (`>50K`) | Accuracy |
|-------|---------|-------------|----------|
| **Hist Gradient Boosting** | **0.931** | **0.720** | 0.835 |
| Random Forest | 0.921 | 0.713 | 0.837 |
| AdaBoost (sklearn) | 0.914 | 0.668 | **0.859** |
| **AdaBoost (from scratch)** | 0.908 | 0.673 | 0.858 |
| Logistic Regression | 0.907 | 0.682 | 0.808 |

Full metrics and design notes: [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

## Quickstart

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# 2. Run the full benchmark
python scripts/run_benchmark.py --output-dir artifacts

# 3. Tests
pytest -q
```

Programmatic use:

```python
from income_classifier import make_model_pipeline, load_adult
from income_classifier.data import split_features_target
from sklearn.model_selection import train_test_split

X, y = split_features_target(load_adult())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = make_model_pipeline("hist_gradient_boosting")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## What’s inside

```text
src/income_classifier/     # data → preprocess → models → evaluate
scripts/run_benchmark.py   # one-command leaderboard + saved artifact
notebooks/                 # walkthrough + modern benchmark (+ 2020 archive)
tests/                     # stump math, AdaBoost, pipeline smoke tests
docs/DOCUMENTATION.md      # algorithms, metrics, API, limitations
adult.data                 # UCI Adult training dump
```

### Notebooks

1. [`notebooks/01_adaboost_walkthrough.ipynb`](notebooks/01_adaboost_walkthrough.ipynb) — build stumps and AdaBoost step by step.
2. [`notebooks/02_modern_benchmark.ipynb`](notebooks/02_modern_benchmark.ipynb) — EDA + package benchmark.
3. [`notebooks/archive_original_2020.ipynb`](notebooks/archive_original_2020.ipynb) — original exploratory notebook (historical).

## What improved vs the original notebook

1. **Features that matter** — capital gain/loss, marital status, relationship, and native country (not just six hand-picked columns).
2. **Honest missing data** — `?` treated as missing and imputed inside a `ColumnTransformer` pipeline.
3. **Fixed stump search** — `find_splits` no longer zeros out candidate thresholds.
4. **Modern learner** — HistGradientBoosting with early stopping and class balancing.
5. **Engineering** — package layout, CLI, tests, MIT license, and docs you can share in a portfolio.

## Dataset

[UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) (Kohavi & Becker). Prediction task: `income > 50K` vs `<= 50K`. See `adult.names` for the original codebook and historical baselines (~84–86% accuracy for classical systems after unknown removal).

## Stack

Python 3.10+ · pandas · NumPy · scikit-learn · matplotlib · pytest

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Juan David Prada Malagon** — data science / ML engineering.
