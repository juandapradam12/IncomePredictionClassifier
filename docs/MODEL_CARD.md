# Scope of the income case study

| Item | Detail |
|------|--------|
| Task | Binary income classification: `>50K` vs `<=50K` |
| Primary model | Histogram gradient boosting pipeline |
| Supporting models | Logistic regression, random forest, AdaBoost (sklearn + from scratch) |
| Data | UCI Adult (`adult.data` / `adult.test`) |
| Author | Juan David Prada Malagon |

This repository is a **case study**: explain AdaBoost, compare tabular models, and surface ranking / threshold / fairness behavior on a public census extract.

It is **not** intended for credit, hiring, insurance, benefits, or any decision that affects a real person. Adult contains protected attributes (`sex`, `race`); the fairness plots in the README and documentation are diagnostic only.
