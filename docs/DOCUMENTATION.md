# Income prediction from census attributes

This write-up is the case study behind the repo: predict whether a person in the [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) extract earns more than $50K/year, expose the AdaBoost math used to attack that problem, and show how a modern tabular model behaves on the official test split.

It is not an install guide. For the narrative landing page, see the [README](../README.md).

## Use case

Given demographic and employment fields collected in a 1994 U.S. Census extract, classify each person as:

- `<=50K` — annual income at or below $50,000
- `>50K` — annual income above $50,000

The label is imbalanced. Roughly **76%** of training rows are `<=50K`, so majority-class guessing is a weak answer to the use case even when accuracy looks high.

<p align="center">
  <img src="../figures/class_balance.png" alt="Class balance" width="560" />
</p>

<p align="center"><em>Figure 1.</em> Train-set label counts. Evaluation focuses on ranking quality and F1 for the minority high-income class.</p>

### Signals that matter for this case

Three patterns show why the problem is learnable and why the original narrow feature subset left signal on the table:

<p align="center">
  <img src="../figures/income_by_education.png" alt="Income rate by education" width="680" />
</p>

<p align="center"><em>Figure 2.</em> High-income rate grows with years of education.</p>

<p align="center">
  <img src="../figures/income_by_marital_status.png" alt="Income rate by marital status" width="720" />
</p>

<p align="center"><em>Figure 3.</em> Marital status separates income rates sharply in this extract — especially married civilian spouses versus never-married adults.</p>

<p align="center">
  <img src="../figures/income_by_capital_gain.png" alt="Income mix by capital gain" width="560" />
</p>

<p align="center"><em>Figure 4.</em> Capital gains are rare, but when present they flip the income mix toward <code>&gt;50K</code>.</p>

Features kept for modeling:

| Type | Fields |
|------|--------|
| Numeric | `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` |
| Categorical | `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country` |

`fnlwgt` is dropped as a sampling weight; string `education` is dropped as redundant with `education-num`. Missing `?` values are treated as missingness and imputed inside the model pipeline.

## AdaBoost as the first answer to the use case

The project’s original intent was to make boosting inspectable: fit weak decision stumps, upweight the mistakes, and combine stumps into a stronger classifier.

Weighted stump error and stump influence:

$$
\varepsilon_t = \sum_{i=1}^{n} w_t(i)\,\mathbb{1}\{y_i \neq f_t(x_i)\}
\qquad
\alpha_t = \frac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}
$$

Sample-weight update and final score:

$$
\hat{w}_{t+1}(i) = w_t(i)\,e^{-\alpha_t y_i f_t(x_i)}
\qquad
w_{t+1}(i) = \frac{\hat{w}_{t+1}(i)}{\sum_j \hat{w}_{t+1}(j)}
$$

$$
F(x) = \mathrm{sign}\!\left(\sum_{t=1}^{T} \alpha_t f_t(x)\right)
$$

In code this lives in `SimpleAdaBoost` (`src/income_classifier/adaboost.py`): bootstrap sampling with the current weights, numerically stable $\varepsilon_t$, and a sklearn-style `fit` / `predict` API. The educational stump search also exposes entropy-based split finding; the production path uses depth-1 trees for speed.

On Adult, the from-scratch booster lands next to sklearn’s AdaBoost — close enough to treat the math as faithful, not merely decorative.

## Comparing answers to the same use case

All numbers below use the official protocol: train on `adult.data`, score on `adult.test`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. Precision |
|-------|----------|-----------|--------|----|---------|----------------|
| Hist Gradient Boosting | 0.833 | 0.602 | 0.858 | **0.707** | **0.927** | **0.824** |
| Random Forest | 0.833 | 0.609 | 0.818 | 0.699 | 0.916 | 0.793 |
| AdaBoost (sklearn) | **0.857** | **0.771** | 0.562 | 0.650 | 0.910 | 0.779 |
| AdaBoost (from scratch) | 0.854 | 0.746 | 0.582 | 0.654 | 0.907 | 0.770 |
| Logistic Regression | 0.806 | 0.559 | 0.840 | 0.671 | 0.903 | 0.755 |

<p align="center">
  <img src="../figures/benchmark_comparison.png" alt="Benchmark comparison" width="900" />
</p>

<p align="center"><em>Figure 5.</em> Gradient boosting is the best ranker and the best F1 for <code>&gt;50K</code>. AdaBoost is more precise/conservative: higher accuracy, lower recall on the high-income class.</p>

<p align="center">
  <img src="../figures/roc_curves.png" alt="ROC curves" width="520" />
</p>

<p align="center"><em>Figure 6.</em> ROC curves on <code>adult.test</code>. Custom AdaBoost tracks sklearn AdaBoost; histogram gradient boosting dominates the upper-left region.</p>

### Operating point for the use case

Predicting “high income” is not only about ranking. The probability cut changes who is flagged:

<p align="center">
  <img src="../figures/threshold_tradeoff.png" alt="Threshold tradeoff" width="720" />
</p>

<p align="center"><em>Figure 7.</em> Precision, recall, and F1 versus decision threshold for the best model. A validation-chosen cut near 0.65–0.70 improves F1 on test from about 0.707 to about 0.72 versus the default 0.5.</p>

## Interpretation for this census task

<p align="center">
  <img src="../figures/permutation_importance.png" alt="Permutation importance" width="720" />
</p>

<p align="center"><em>Figure 8.</em> Permutation importance on the official test set. Marital status and capital gain are the largest contributors to ROC-AUC, then age and education — the same structure visible in the exploratory plots.</p>

## Equity snapshot (not a deployment claim)

Adult encodes `sex` and `race`. For this case study, the best model’s outcomes are sliced so gaps are visible:

<p align="center">
  <img src="../figures/fairness_by_sex.png" alt="Fairness by sex" width="780" />
</p>

<p align="center"><em>Figure 9.</em> Selection rate and true-positive rate by sex on <code>adult.test</code> at the F1-tuned threshold. Men are flagged and recovered as high income more often than women. These plots are diagnostic for the case study; they are not a fairness certification.</p>

Checked-in tables: [`results/fairness_slices.csv`](../results/fairness_slices.csv), [`results/permutation_importance.csv`](../results/permutation_importance.csv), [`results/official_benchmark_metrics.csv`](../results/official_benchmark_metrics.csv).

## Scope of the case

- Historical 1994 Census economics, not a current income product
- Binary \$50K cutoff, not continuous earnings
- Protected attributes are present; consequential use would need separate governance
- From-scratch AdaBoost uses bootstrap reweighting (notebook-faithful variant of AdaBoost)

## References

- Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.* KDD.
- Freund, Y. & Schapire, R. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting.*
- UCI Adult dataset: https://archive.ics.uci.edu/dataset/2/adult
