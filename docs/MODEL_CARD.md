# Model Card — Adult Income Prediction Classifier

## Model details

| Field | Value |
|-------|--------|
| Name | Income Prediction Classifier |
| Version | 1.2.0 |
| Task | Binary classification: `>50K` vs `<=50K` annual income |
| Primary model | `HistGradientBoostingClassifier` (scikit-learn) inside a preprocess pipeline |
| Supporting models | Logistic regression, Random Forest, AdaBoost (sklearn + from-scratch) |
| Authors | Juan David Prada Malagon |
| License | MIT |
| Intended use | **Educational / portfolio demonstration** of boosting, tabular ML, and evaluation practice |

## Intended users

- Students and practitioners learning ensemble methods
- Recruiters / hiring managers reviewing a public portfolio project
- Researchers comparing simple baselines on UCI Adult

**Out of scope:** credit decisions, hiring, insurance, government benefit eligibility, or any consequential decisioning about individuals.

## Training data

- **Source:** [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) (Kohavi & Becker, 1994 Census extract)
- **Train file:** `adult.data` (32,561 rows)
- **Official test file:** `adult.test` (16,281 rows)
- **Target prevalence (`>50K`):** ~24%
- **Features used:** age, education-num, capital-gain, capital-loss, hours-per-week, workclass, marital-status, occupation, relationship, race, sex, native-country
- **Excluded:** `fnlwgt` (sampling weight), string `education` (redundant with `education-num`)
- **Missingness:** `?` values imputed inside the sklearn pipeline (median / most-frequent)

## Evaluation

Default protocol: **official UCI train/test split**.

Metrics reported: accuracy, precision, recall, F1 (positive = `>50K`), ROC-AUC, average precision.  
Additional analyses:

- Probability **threshold tuning** for F1 on a training validation slice
- **Permutation importance** on the official test set
- **Fairness slices** by `sex` and `race` (selection rate, TPR, FPR, precision, recall)

See `results/official_benchmark_metrics.csv`, `results/fairness_slices.csv`, and `docs/DOCUMENTATION.md`.

## Ethical considerations

Adult includes **protected attributes** (`sex`, `race`) that historically correlate with income in the 1994 Census extract. Models trained on this data can encode and amplify disparities.

- Fairness slice tables are provided for transparency, **not** as evidence that the model is fair enough to deploy.
- Do not use predictions to approve/deny opportunities for individuals.
- If adapting this pipeline to a real product, remove or carefully justify protected attributes, run a full fairness review, and involve domain stakeholders.

## Caveats & limitations

- 1994 U.S. Census economics; not a current income model
- Labels are a coarse \$50K cutoff, not continuous income
- From-scratch AdaBoost uses bootstrap sampling (notebook-faithful variant)
- Hyperparameter search is randomized and finite — not an exhaustive optimum
- Streamlit / CLI demos are illustrations only

## How to reproduce

```bash
pip install -r requirements.txt
pip install -e .
python scripts/run_benchmark.py --split official --analyze --output-dir artifacts
pytest -q
```
