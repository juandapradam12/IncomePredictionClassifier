# Income Prediction Classifier

**Predict whether a person earns more than \$50K/year — and understand *how* boosting gets you there.**

This project pairs a **from-scratch AdaBoost implementation** (decision stumps, ε / α weight updates, bootstrap reweighting) with a **modern tabular ML toolkit** on the UCI Adult Census Income dataset. It started as an educational deep-dive into boosting; it now ships as a reproducible package with official train/test evaluation, tuning, interpretability, fairness slices, CI, and a small demo app.

<p align="center">
  <img src="figures/benchmark_comparison.png" alt="Model benchmark comparison" width="900" />
</p>

## Why this project

| Pillar | What you get |
|--------|----------------|
| **Theory you can read** | AdaBoost math implemented in clear Python — not hidden inside a C++ extension |
| **Parity with sklearn** | Custom AdaBoost tracks `AdaBoostClassifier` closely on Adult |
| **Stronger models** | Full feature set + tuned `HistGradientBoostingClassifier` |
| **Honest evaluation** | Official UCI `adult.data` / `adult.test` split, not only a random holdout |
| **Portfolio-ready** | Package, CLI, Streamlit demo, tests, CI, model card, fairness report |

## Headline results (official UCI test set)

Train on `adult.data`, evaluate on `adult.test`:

| Model | ROC-AUC | F1 (`>50K`) | Accuracy |
|-------|---------|-------------|----------|
| **Hist Gradient Boosting** | **0.927** | **0.707** | 0.833 |
| Random Forest | 0.916 | 0.699 | 0.833 |
| AdaBoost (sklearn) | 0.910 | 0.650 | **0.857** |
| AdaBoost (from scratch) | 0.907 | 0.654 | 0.854 |
| Logistic Regression | 0.903 | 0.671 | 0.806 |

F1 rises to **~0.718** after validation-chosen threshold tuning (`≈0.70` instead of 0.5). Checked-in metrics: [`results/official_benchmark_metrics.csv`](results/official_benchmark_metrics.csv).

Full write-up, permutation importance, and fairness slices: [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) · [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

<p align="center">
  <img src="figures/permutation_importance.png" alt="Permutation importance" width="720" />
</p>

## Quickstart

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# 2. Official-split benchmark (+ threshold / importance / fairness)
python scripts/run_benchmark.py --split official --analyze --output-dir artifacts

# 3. Optional: randomized hyperparameter search for HGB
python scripts/run_benchmark.py --split official --tune hist_gradient_boosting --tune-iter 16 --analyze

# 4. Tests
pytest -q

# 5. Demo (CLI)
python scripts/predict_demo.py --example

# 6. Demo (Streamlit UI)
streamlit run app/streamlit_app.py
```

Programmatic use:

```python
from income_classifier import make_model_pipeline, load_adult_official_split
from income_classifier.data import split_features_target

train_df, test_df = load_adult_official_split()
X_train, y_train = split_features_target(train_df)
X_test, y_test = split_features_target(test_df)

model = make_model_pipeline("hist_gradient_boosting")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## What’s inside

```text
src/income_classifier/     # data, AdaBoost, pipelines, tuning, interpretability
scripts/run_benchmark.py   # leaderboard + optional --tune / --analyze
scripts/predict_demo.py    # score one profile or a CSV
app/streamlit_app.py       # interactive demo
notebooks/                 # walkthrough + benchmark (+ clearly labeled 2020 archive)
tests/                     # unit + smoke tests
.github/workflows/ci.yml   # pytest + logistic smoke on PRs
docs/DOCUMENTATION.md      # algorithms, metrics, API
docs/MODEL_CARD.md         # intended use, limitations, ethics
adult.data / adult.test    # official UCI split
```

### Notebooks

See [`notebooks/README.md`](notebooks/README.md). Start with `01_adaboost_walkthrough.ipynb`. The 2020 notebook is kept only as a **historical archive**.

## What improved vs the original notebook

1. **Official train/test protocol** with `adult.test`
2. **Features that matter** — capital gain/loss, marital status, relationship, native country
3. **Honest missing data** — `?` imputed inside a `ColumnTransformer`
4. **Fixed stump search** — `find_splits` no longer zeros out candidate thresholds
5. **Modern learner + tuning** — HistGradientBoosting with randomized search
6. **Threshold tuning, permutation importance, fairness slices**
7. **Engineering** — package, CI, pinned deps, model card, Streamlit demo

## Dataset

[UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) (Kohavi & Becker). Prediction task: `income > 50K` vs `<= 50K`. See `adult.names` for the original codebook.

## Stack

Python 3.10+ · pandas · NumPy · scikit-learn · matplotlib · pytest · streamlit

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Juan David Prada Malagon** — data science / ML engineering.
