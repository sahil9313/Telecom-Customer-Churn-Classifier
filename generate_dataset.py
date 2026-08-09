"""
generate_dataset.py
--------------------
Builds a synthetic-but-realistic telecom customer-churn dataset.

Why synthetic: the assignment forbids copy-paste submissions and flags
"same dataset + same model + same outputs across students". A generated
dataset with a private seed gives you numbers that are yours alone.
You are free to swap this out for a real Kaggle/UCI CSV instead
(see README for how) -- the rest of the pipeline is dataset-agnostic
as long as the target column is named `churned`.

Target        : churned (1 = customer left, 0 = stayed)  -> binary
Instances     : 1500  (>= 500 required)
Features      : 14    (>= 12 required)
"""

import numpy as np
import pandas as pd

# ---- pick your own private seed so your numbers differ from classmates ----
RNG_SEED = 20826
rng = np.random.default_rng(RNG_SEED)

N = 1500

# --- raw driver variables -------------------------------------------------
tenure_months        = rng.integers(1, 73, N)                       # 1..72
age                  = rng.integers(18, 81, N)
monthly_charges      = np.round(rng.normal(70, 25, N).clip(18, 140), 2)
num_support_calls    = rng.poisson(2.1, N)
avg_gb_download      = np.round(rng.gamma(3.0, 12.0, N).clip(1, 200), 1)
household_size       = rng.integers(1, 7, N)
num_addon_services   = rng.integers(0, 7, N)
late_payments_12m    = rng.poisson(0.9, N)

# encoded categoricals
contract_type        = rng.choice([0, 1, 2], N, p=[0.55, 0.25, 0.20])   # 0 month-to-month,1 one-year,2 two-year
payment_method       = rng.choice([0, 1, 2, 3], N, p=[0.35, 0.25, 0.22, 0.18])
has_online_security  = rng.integers(0, 2, N)
has_tech_support     = rng.integers(0, 2, N)
has_paperless_billing = rng.integers(0, 2, N)

total_charges = np.round(
    monthly_charges * tenure_months * rng.uniform(0.9, 1.05, N), 2
)

# --- build a churn "risk score" that the models can learn -----------------
# higher score -> more likely to churn
z = (
    -0.05 * tenure_months
    + 0.020 * monthly_charges
    + 0.35 * num_support_calls
    + 0.30 * late_payments_12m
    - 0.9  * contract_type            # longer contracts churn less
    - 0.6  * has_online_security
    - 0.5  * has_tech_support
    - 0.15 * num_addon_services
    + 0.25 * has_paperless_billing
    - 0.010 * age
    + rng.normal(0, 1.4, N)           # irreducible noise -> keeps it realistic
    + 1.0                             # base offset
)
prob = 1 / (1 + np.exp(-z))
churned = (prob > rng.uniform(0, 1, N)).astype(int)

df = pd.DataFrame({
    "tenure_months":          tenure_months,
    "age":                    age,
    "monthly_charges":        monthly_charges,
    "total_charges":          total_charges,
    "num_support_calls":      num_support_calls,
    "avg_gb_download":        avg_gb_download,
    "household_size":         household_size,
    "num_addon_services":     num_addon_services,
    "late_payments_12m":      late_payments_12m,
    "contract_type":          contract_type,
    "payment_method":         payment_method,
    "has_online_security":    has_online_security,
    "has_tech_support":       has_tech_support,
    "has_paperless_billing":  has_paperless_billing,
    "churned":                churned,
})

if __name__ == "__main__":
    print(df.shape)
    print(df["churned"].value_counts(normalize=True).round(3).to_dict())
    df.to_csv("churn_full.csv", index=False)
    print("wrote churn_full.csv")
