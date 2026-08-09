"""
Telecom Churn Classifier - Streamlit demo
==========================================
Interactive frontend for five pre-trained classifiers.

Features (mapped to the rubric):
  a) CSV upload of TEST data
  b) Model-selection dropdown
  c) Evaluation-metric display (Accuracy, AUC, Precision, Recall, F1, MCC)
  d) Confusion matrix + full classification report

The models are trained offline by model/train_models.py and loaded here
from model/*.pkl. This keeps the app light enough for Streamlit's free tier.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)

TARGET = "churned"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_FILES = {
    "Logistic Regression":       "logistic_regression.pkl",
    "Decision Tree":             "decision_tree.pkl",
    "kNN":                       "knn.pkl",
    "Naive Bayes":               "naive_bayes.pkl",
    "Random Forest (Ensemble)":  "random_forest.pkl",
}

st.set_page_config(page_title="Churn Classifier Lab",
                   page_icon="📡", layout="wide")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded


@st.cache_data
def load_bundled_test():
    path = os.path.join(HERE, "test_data.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")
models = load_models()

if not models:
    st.error(
        "No trained models found in `model/`. "
        "Run `python model/train_models.py` before launching the app."
    )
    st.stop()

model_names = list(models.keys())
selected = st.sidebar.selectbox("Choose a model", model_names, index=len(model_names) - 1)
compare_all = st.sidebar.checkbox("Compare all models", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Expected columns**\n\n14 feature columns + a `churned` target "
    "(1 = left, 0 = stayed). Upload the provided `test_data.csv` or your own."
)


# --------------------------------------------------------------------------- #
# Header + data intake
# --------------------------------------------------------------------------- #
st.title("📡 Telecom Customer-Churn Classifier")
st.caption(
    "Five classifiers trained on a 14-feature churn dataset. "
    "Upload a labelled test CSV to score them live."
)

uploaded = st.file_uploader("Upload TEST data (CSV)", type=["csv"])
bundled = load_bundled_test()

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.success(f"Loaded uploaded file — {data.shape[0]} rows × {data.shape[1]} cols.")
elif bundled is not None:
    data = bundled
    st.info("No file uploaded — using the bundled `test_data.csv`. "
            "Upload your own CSV above to override.")
else:
    st.warning("Upload a CSV to continue.")
    st.stop()

if TARGET not in data.columns:
    st.error(f"The CSV must contain a `{TARGET}` target column. "
             f"Found: {list(data.columns)}")
    st.stop()

with st.expander("Preview data"):
    st.dataframe(data.head(20), use_container_width=True)

X = data.drop(columns=[TARGET])
y = data[TARGET].astype(int)


# --------------------------------------------------------------------------- #
# Scoring helper
# --------------------------------------------------------------------------- #
def score(model):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba


# --------------------------------------------------------------------------- #
# Single-model view
# --------------------------------------------------------------------------- #
st.header(f"Results — {selected}")

y_pred, y_proba = score(models[selected])
m = compute_metrics(y, y_pred, y_proba)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy",  f"{m['Accuracy']:.3f}")
c2.metric("AUC",       f"{m['AUC']:.3f}")
c3.metric("Precision", f"{m['Precision']:.3f}")
c4.metric("Recall",    f"{m['Recall']:.3f}")
c5.metric("F1",        f"{m['F1']:.3f}")
c6.metric("MCC",       f"{m['MCC']:.3f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", cbar=False,
                xticklabels=["Stayed", "Churned"],
                yticklabels=["Stayed", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("ROC curve")
    fpr, tpr, _ = roc_curve(y, y_proba)
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
    ax2.plot(fpr, tpr, lw=2, label=f"AUC = {m['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

st.subheader("Classification report")
report = classification_report(
    y, y_pred, target_names=["Stayed (0)", "Churned (1)"],
    output_dict=True, zero_division=0,
)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)


# --------------------------------------------------------------------------- #
# Compare-all view
# --------------------------------------------------------------------------- #
if compare_all:
    st.header("All models on this test set")
    rows = {}
    for name, mdl in models.items():
        yp, ypr = score(mdl)
        rows[name] = compute_metrics(y, yp, ypr)
    comp = pd.DataFrame(rows).T.round(4)
    comp.index.name = "ML Model"

    st.dataframe(
        comp.style.highlight_max(axis=0, color="#1b5e20")
                  .format("{:.4f}"),
        use_container_width=True,
    )
    winner = comp["MCC"].idxmax()
    st.success(f"🏆 Best by MCC on this data: **{winner}** "
               f"(MCC = {comp.loc[winner, 'MCC']:.4f})")

st.caption("Built for the ML classification assignment · models trained offline, "
           "loaded read-only here.")
