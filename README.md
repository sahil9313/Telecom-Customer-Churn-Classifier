# Telecom Customer-Churn Classification

A comparison of five supervised classifiers on a telecom customer-churn
dataset, wrapped in an interactive Streamlit app. Upload a labelled test
CSV, pick a model, and see all six evaluation metrics, a confusion matrix,
an ROC curve, and a full classification report update live.

---

## a. Problem Statement

Telecom operators lose revenue when subscribers cancel their service
("churn"). Retaining an existing customer is far cheaper than acquiring a
new one, so operators want to flag customers likely to leave *before* they
go, and target them with retention offers.

This is framed as a **binary classification** problem: given a customer's
account and usage profile, predict whether they will churn
(`churned = 1`) or stay (`churned = 0`). Five classic classifiers are
trained and compared to see which best separates churners from non-churners
on held-out data.

## b. Dataset Description

| Property | Value |
|---|---|
| Domain | Telecom customer retention |
| Task | Binary classification |
| Target | `churned` (1 = customer left, 0 = stayed) |
| Instances | **1,500** (≥ 500 required) |
| Features | **14** (≥ 12 required) |
| Class balance | ~54.6 % stayed / ~45.4 % churned |
| Missing values | None |
| Train / test split | 75 % / 25 % stratified (test set shipped as `test_data.csv`, 375 rows) |

**Feature dictionary**

| # | Feature | Type | Meaning |
|---|---|---|---|
| 1 | `tenure_months` | int | Months the customer has been subscribed |
| 2 | `age` | int | Customer age in years |
| 3 | `monthly_charges` | float | Current monthly bill |
| 4 | `total_charges` | float | Lifetime charges to date |
| 5 | `num_support_calls` | int | Support calls in the last year |
| 6 | `avg_gb_download` | float | Average monthly data downloaded (GB) |
| 7 | `household_size` | int | People in the household |
| 8 | `num_addon_services` | int | Count of optional add-on services |
| 9 | `late_payments_12m` | int | Late payments in the last 12 months |
| 10 | `contract_type` | cat (0/1/2) | 0 = month-to-month, 1 = one-year, 2 = two-year |
| 11 | `payment_method` | cat (0–3) | Encoded payment method |
| 12 | `has_online_security` | bin | Has online-security add-on |
| 13 | `has_tech_support` | bin | Has tech-support add-on |
| 14 | `has_paperless_billing` | bin | Uses paperless billing |

## c. GitHub Repository Link

> **https://github.com/sahil9313/Telecom-Customer-Churn-Classifier**
>

**Live Streamlit app:** `https://telecom-customer-churn-classifier-9313.streamlit.app/`

## d. Models Used

All five models were trained on the **same** dataset and split, each inside
a `StandardScaler → estimator` pipeline so scoring on raw feature rows is
consistent.

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `C=1.0`, `max_iter=2000` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=15` |
| kNN | `n_neighbors=15`, `weights="distance"` |
| Naive Bayes | Gaussian NB (default) |
| Random Forest (Ensemble) | `n_estimators=300`, `max_depth=10`, `min_samples_leaf=5` |

### Comparison Table (on the 375-row held-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7013 | **0.7828** | 0.6747 | 0.6588 | **0.6667** | 0.3963 |
| Decision Tree | 0.6400 | 0.6795 | 0.6190 | 0.5353 | 0.5741 | 0.2673 |
| kNN | 0.6880 | 0.7270 | 0.6497 | **0.6765** | 0.6628 | 0.3730 |
| Naive Bayes | 0.6827 | 0.7607 | 0.6457 | 0.6647 | 0.6551 | 0.3615 |
| Random Forest (Ensemble) | **0.7040** | 0.7646 | **0.6928** | 0.6235 | 0.6563 | **0.3994** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-calibrated linear baseline. Achieves the **best AUC (0.783)**, i.e. it ranks churn risk better than any other model, with balanced precision and recall. Because the target was built from a logistic risk model, a linear boundary fits the signal well. Cheap and interpretable. |
| Decision Tree | **Weakest model** across almost every metric (accuracy 0.640, MCC 0.267). A single pruned tree makes unstable, greedy splits on noisy features and can't capture the smooth risk gradient, so it both under- and over-fits. Illustrates why one tree is rarely competitive. |
| kNN | Middle of the pack, but posts the **highest recall (0.677)** — it flags the most actual churners — at the cost of lower precision. Distance-weighted voting on scaled features helps, though performance is dragged down by the noisy, weakly-informative columns. |
| Naive Bayes | Competitive **AUC (0.761)** despite its feature-independence assumption being clearly violated (`monthly_charges`, `total_charges`, `tenure_months` are correlated). Fast to train and gives usable probabilities; slightly lower accuracy than the leaders. |
| Random Forest (Ensemble) | **Overall winner.** Averaging 300 trees cancels the single tree's variance, giving the best accuracy (0.704), best precision (0.693) and best MCC (0.399). It is more conservative about predicting churn (lower recall than kNN/NB) and its AUC trails Logistic Regression by a hair. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — it tops accuracy, precision and MCC (the most balanced single-number score for this ~55/45 split). Caveat worth noting: Logistic Regression is essentially tied and actually **wins on AUC** while being far cheaper and fully interpretable, so it is a defensible production choice if ranking quality or explainability matters more than raw MCC. |

---

## Project Structure

```
project-folder/
├── app.py                      # Streamlit frontend
├── requirements.txt
├── README.md
├── test_data.csv               # 375-row held-out test set (upload this in the app)
└── model/
    ├── generate_dataset.py     # builds churn_full.csv from a private seed
    ├── train_models.py         # trains 5 models, writes metrics + *.pkl
    ├── churn_full.csv          # full generated dataset (1,500 rows)
    ├── metrics.csv             # comparison table
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## App Features

- **CSV upload** of test data (falls back to the bundled `test_data.csv`).
- **Model-selection dropdown** in the sidebar (+ a "compare all" toggle).
- **Six evaluation metrics** shown as metric cards: Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix**, **ROC curve**, and a full **classification report**.
