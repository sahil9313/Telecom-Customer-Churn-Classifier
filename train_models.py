"""
train_models.py
---------------
Trains the five required classifiers on the churn dataset, evaluates six
metrics for each, and saves:
  - one self-contained pipeline per model  -> model/<name>.pkl
  - the held-out test split                -> test_data.csv   (repo root)
  - the metric comparison table            -> model/metrics.csv / metrics.json

Each saved pipeline bundles a StandardScaler + the estimator, so the
Streamlit app can call .predict() / .predict_proba() on raw feature rows
without needing to re-fit or re-scale anything.

Run from inside the model/ folder:  python train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET = "churned"
SPLIT_SEED = 20826
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def build_models():
    """Return {display_name: (filename, estimator)}."""
    return {
        "Logistic Regression": (
            "logistic_regression.pkl",
            LogisticRegression(max_iter=2000, C=1.0, random_state=SPLIT_SEED),
        ),
        "Decision Tree": (
            "decision_tree.pkl",
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=15,
                                   random_state=SPLIT_SEED),
        ),
        "kNN": (
            "knn.pkl",
            KNeighborsClassifier(n_neighbors=15, weights="distance"),
        ),
        "Naive Bayes": (
            "naive_bayes.pkl",
            GaussianNB(),
        ),
        "Random Forest (Ensemble)": (
            "random_forest.pkl",
            RandomForestClassifier(n_estimators=300, max_depth=10,
                                   min_samples_leaf=5, n_jobs=-1,
                                   random_state=SPLIT_SEED),
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    df = pd.read_csv(os.path.join(HERE, "churn_full.csv"))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SPLIT_SEED
    )

    # Persist the held-out test set for the Streamlit uploader.
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(os.path.join(REPO_ROOT, "test_data.csv"), index=False)
    print(f"Wrote test_data.csv  ({test_df.shape[0]} rows)")

    results = {}
    for name, (fname, estimator) in build_models().items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = evaluate(y_test, y_pred, y_proba)

        joblib.dump(pipe, os.path.join(HERE, fname))
        print(f"Saved {fname:26s} -> {results[name]}")

    # Comparison table
    table = pd.DataFrame(results).T
    table.index.name = "ML Model"
    table.to_csv(os.path.join(HERE, "metrics.csv"))
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Comparison table ===")
    print(table.to_string())
    winner = table["MCC"].idxmax()
    print(f"\nBest by MCC: {winner}")


if __name__ == "__main__":
    main()
