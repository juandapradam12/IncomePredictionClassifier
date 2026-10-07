# Documentation — Income Prediction Classifier

## 1. Problem statement

Predict whether a U.S. Census respondent earns **more than \$50,000 per year** (`>50K`) or not (`<=50K`), using demographic and employment attributes from the [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset (Kohavi & Becker, 1996).

This is a classic **imbalanced binary classification** problem (~24% positive class). Accuracy alone is misleading; the project reports **ROC-AUC**, **average precision**, **precision**, **recall**, and **F1** for the `>50K` class.

## 2. Project goals

1. **Pedagogy** — Implement AdaBoost from first principles (stumps, ε, α, weight updates) so the boosting math is inspectable, not a black box.
2. **Validation** — Show that the from-scratch booster closely matches scikit-learn’s `AdaBoostClassifier` on the same task.
3. **Performance** — Improve over the original 2020 notebook by using the full informative feature set and modern tabular learners (especially histogram-based gradient boosting).
4. **Reproducibility** — Package the pipeline so results can be regenerated with one command.

## 3. Dataset

| Item | Detail |
|------|--------|
| Source | UCI ML Repository — Adult |
| File in repo | `adult.data` (training split dump, 32,561 rows) |
| Target | `income` ∈ {`<=50K`, `>50K`} |
| Missing values | Encoded as `?` in `workclass`, `occupation`, `native-country` |

### Features used (v1.1)

| Type | Columns |
|------|---------|
| Numeric | `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` |
| Categorical | `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country` |

**Dropped on purpose**

- `fnlwgt` — census sampling weight, not a personal attribute useful for prediction.
- `education` — redundant with ordinal `education-num`.

The original notebook used a smaller subset (`age`, `workclass`, `education-num`, `occupation`, `sex`, `hours-per-week`). Restoring capital gains/losses and family/relationship structure is the largest modeling upgrade: those fields are strongly associated with high income.

## 4. Preprocessing

Implemented in `income_classifier.pipeline.build_preprocessor`:

1. Strip whitespace and normalize target labels.
2. Map `?` → missing (`pd.NA`).
3. **Numeric:** median imputation (optional `StandardScaler` for logistic regression).
4. **Categorical:** most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`.
5. Stratified train/test split (default 80/20, `random_state=42`).

Unknown categories at inference time are ignored by the encoder rather than crashing the pipeline.

## 5. Algorithms

### 5.1 From-scratch AdaBoost (`SimpleAdaBoost`)

Classic Freund & Schapire reweighting with decision stumps:

\[
\varepsilon_t = \sum_i w_t(i)\,\mathbb{1}\{y_i \ne f_t(x_i)\}
\qquad
\alpha_t = \tfrac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}
\]

\[
\hat{w}_{t+1}(i) = w_t(i)\,e^{-\alpha_t y_i f_t(x_i)}
\qquad
w_{t+1}(i) = \frac{\hat{w}_{t+1}(i)}{\sum_j \hat{w}_{t+1}(j)}
\]

\[
F(x) = \mathrm{sign}\Big(\sum_t \alpha_t f_t(x)\Big)
\]

Implementation notes:

- Bootstrap sampling with current weights (as in the original notebook).
- Numerical clipping of \(\varepsilon\) away from `{0,1}`.
- Optional pure-NumPy entropy stump search (`use_sklearn_stumps=False`) vs fast sklearn depth-1 trees.
- sklearn-compatible `fit` / `predict` / `predict_proba` / `decision_function`.

**Bug fixed from the 2020 notebook:** `find_splits` previously overwrote the midpoint array with an empty `np.array()`, which would break the pure stump search.

### 5.2 Benchmark model zoo

| Key | Algorithm | Why it’s included |
|-----|-----------|-------------------|
| `logistic_regression` | L2 logistic regression, balanced class weights | Strong linear baseline + calibrated-ish probabilities |
| `random_forest` | Random Forest (300 trees) | Bagging ensemble reference from the original write-up |
| `adaboost_sklearn` | sklearn AdaBoost (200 stumps) | Reference implementation |
| `adaboost_from_scratch` | This repo’s AdaBoost | Educational parity check |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier` | Strong modern default for medium tabular data |

Histogram gradient boosting typically wins here because it captures non-linear interactions (e.g. education × occupation × capital gains) with regularization and early stopping, without needing manual feature crosses.

## 6. Benchmark results

### 6.1 Official UCI train/test (default)

Train on `adult.data`, evaluate on `adult.test`. Metrics for the positive class `>50K` are written to `results/official_benchmark_metrics.csv` when you run:

```bash
python scripts/run_benchmark.py --split official --analyze --output-dir artifacts
```

### 6.2 Internal holdout (optional)

For quick iteration you can still use a stratified 80/20 split of `adult.data`:

```bash
python scripts/run_benchmark.py --split holdout --output-dir artifacts
```

Previous holdout snapshot (for continuity with earlier README numbers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. Precision |
|-------|----------|-----------|--------|----|---------|----------------|
| Hist Gradient Boosting | 0.833 | 0.602 | 0.858 | **0.707** | **0.927** | **0.824** |
| Random Forest | 0.833 | 0.609 | 0.818 | 0.699 | 0.916 | 0.793 |
| AdaBoost (sklearn) | **0.857** | **0.771** | 0.562 | 0.650 | 0.910 | 0.779 |
| AdaBoost (from scratch) | 0.854 | 0.746 | 0.582 | 0.654 | 0.907 | 0.770 |
| Logistic Regression | 0.806 | 0.559 | 0.840 | 0.671 | 0.903 | 0.755 |

Threshold tuning on a train validation slice selects ≈0.70 and lifts HGB F1 on `adult.test` from 0.707 → **~0.718**.

![Benchmark comparison](../figures/benchmark_comparison.png)

### How to read the table

- **AdaBoost (from scratch) ≈ AdaBoost (sklearn)** — the educational implementation is faithful.
- **HistGradientBoosting** leads on ranking metrics (ROC-AUC / AP) and F1 under class-balanced training — best default production candidate in this repo.
- **sklearn AdaBoost** posts the highest raw accuracy/precision but lower recall; useful when false positives are costly.

Reproduce:

```bash
python scripts/run_benchmark.py --output-dir artifacts
```

## 7. Repository layout

```text
.
├── adult.data / adult.test / adult.names  # official UCI split + codebook
├── src/income_classifier/       # Installable package
│   ├── adaboost.py              # From-scratch AdaBoost + stump utilities
│   ├── data.py                  # Loading / cleaning
│   ├── pipeline.py              # Preprocess + model zoo
│   ├── evaluate.py              # Metrics + comparison
│   └── train.py                 # CLI entrypoint
├── scripts/run_benchmark.py     # One-command benchmark
├── notebooks/
│   ├── 01_adaboost_walkthrough.ipynb
│   ├── 02_modern_benchmark.ipynb
│   └── archive_original_2020.ipynb
├── tests/                       # Unit + smoke tests
├── results/benchmark_metrics.csv
├── figures/benchmark_comparison.png
├── docs/DOCUMENTATION.md        # This file
└── docs/MODEL_CARD.md           # Intended use / ethics
```

## 8. API quick reference

```python
from income_classifier import load_adult, train_and_evaluate, make_model_pipeline
from income_classifier.data import split_features_target
from sklearn.model_selection import train_test_split

df = load_adult()
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = make_model_pipeline("hist_gradient_boosting")
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

# Or run the full leaderboard:
metrics, models = train_and_evaluate()
print(metrics)
```

## 9. Testing

```bash
pytest -q
```

Coverage includes stump/split utilities, weight updates, AdaBoost fit/predict on synthetic data, and an end-to-end Adult + logistic pipeline smoke test.

## 10. Design choices & limitations

- Official `adult.test` is shipped; default evaluation uses that protocol. `--split holdout` remains available for quick iteration.
- From-scratch AdaBoost uses bootstrap sampling (notebook-faithful). sklearn’s AdaBoost uses sample weights directly on the stump; both are valid AdaBoost variants and land in a similar accuracy band.
- No fairness auditing is performed. Adult includes protected attributes (`sex`, `race`); deploying income models in real decisions requires separate bias analysis and governance.
- Hyperparameters are strong defaults, not an exhaustive grid search.

## 11. Threshold tuning, importance, and fairness

```bash
python scripts/run_benchmark.py --split official --analyze --output-dir artifacts
```

Produces:

- `artifacts/analysis_summary.json` — F1-optimal threshold (chosen on a train validation slice) and test metrics at 0.5 vs tuned threshold
- `results/permutation_importance.csv` + `figures/permutation_importance.png`
- `results/fairness_slices.csv` — selection rate / TPR / FPR by `sex` and `race`

## 12. Hyperparameter tuning

```bash
python scripts/run_benchmark.py --split official --tune hist_gradient_boosting --tune-iter 16
```

Uses stratified `RandomizedSearchCV` (see `income_classifier.tuning`). Best params are saved to `results/tuning_summary.json`.

## 13. Demo surfaces

- CLI: `python scripts/predict_demo.py --example`
- UI: `streamlit run app/streamlit_app.py`

## 14. CI

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` and a logistic-regression official-split smoke test on pushes/PRs.

## 15. References


- Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.* KDD.
- Freund, Y. & Schapire, R. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting.*
- UCI Adult dataset: https://archive.ics.uci.edu/dataset/2/adult
- scikit-learn User Guide — Ensembles: https://scikit-learn.org/stable/modules/ensemble.html
