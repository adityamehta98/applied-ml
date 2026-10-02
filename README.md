# churn-predictor

Predict which Telco customers will churn, so retention can target them before they leave.

## Data
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
about 7,043 customers, 26.5 percent churn. Put the CSV at `data/telco.csv` (gitignored).

## Metric
Churn is imbalanced and missing a churner is the costly mistake, so the headline metric is
PR-AUC (average precision) with recall, not accuracy.

## Baseline comparison (stratified 5-fold, PR-AUC)
Run `python -c "from src.data import load_data; from src.model import compare_models; print(compare_models(load_data('data/telco.csv')))"`
and paste your numbers here. A majority-class dummy sits at roughly the churn base rate;
both real models should beat it, with gradient boosting usually on top.

| model | PR-AUC |
| --- | --- |
| dummy (most frequent) | ~0.27 |
| logistic regression | fill in |
| gradient boosting | fill in |

# interview-prep

Written, rehearsable answers and drills for AI Engineer interviews. Every answer is in my
own words and tied to something I built, so it comes out fluently under pressure.

- `notes/ml-answers.md`: the core machine-learning questions (week 1).

Structure I use for every answer: claim, why, example (from a project), trade-off.
