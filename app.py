"""
Census Income Classifier Explorer
----------------------------------
Streamlit app to demonstrate 6 classification models trained on the UCI
Adult (Census Income) dataset. Upload a CSV of held-out records (matching
test_data.csv), pick a model, and inspect how it performs.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(page_title="Census Income Classifier Explorer", page_icon="📊", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "model", "artifacts")

NUMERIC_FEATURES = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "income"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(ARTIFACT_DIR, MODEL_FILES[name]))


@st.cache_data
def load_training_metrics():
    return pd.read_csv(os.path.join(ARTIFACT_DIR, "metrics.csv"), index_col=0)


def transform_features(preprocessor, df):
    X = preprocessor.transform(df[FEATURE_COLUMNS])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("📊 Census Income Classifier Explorer")
st.caption(
    "Compare 5 classification models trained on the UCI Adult (Census Income) dataset — "
    "predicting whether an individual earns more than $50K/year from census attributes."
)

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload test CSV (same schema as test_data.csv)", type=["csv"]
    )
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.markdown(
        "**Expected columns:**\n\n"
        f"`{', '.join(FEATURE_COLUMNS)}, income`"
    )

if uploaded_file is None:
    st.info("👈 Showing results on the repo's `test_data.csv` by default — upload your own CSV in the sidebar to override.")
    default_path = os.path.join(BASE_DIR, "test_data.csv")
    if not os.path.exists(default_path):
        st.error(f"Default test data not found at {default_path}")
        st.stop()
    data = pd.read_csv(default_path)
else:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

missing_cols = [c for c in FEATURE_COLUMNS + [TARGET] if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
    st.stop()

data = data.dropna(subset=FEATURE_COLUMNS + [TARGET]).reset_index(drop=True)
y_true = (data[TARGET].astype(str).str.strip().str.rstrip(".") == ">50K").astype(int)

preprocessor = load_preprocessor()
model = load_model(model_name)

X_transformed = transform_features(preprocessor, data)
y_pred = model.predict(X_transformed)
y_prob = model.predict_proba(X_transformed)[:, 1]

metrics = compute_metrics(y_true, y_pred, y_prob)

st.subheader(f"Results — {model_name}")
st.caption(f"Evaluated on {len(data)} uploaded records.")

cols = st.columns(6)
for col, (metric_name, value) in zip(cols, metrics.items()):
    col.metric(metric_name, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"]).plot(ax=ax, colorbar=False, cmap="Blues")
    st.pyplot(fig)

with right:
    st.markdown("#### ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    ax2.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

st.markdown("#### Classification Report")
report = classification_report(y_true, y_pred, target_names=["<=50K", ">50K"], output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.markdown("---")
st.subheader("All Models — Training-Time Comparison")
st.caption("Metrics computed on a 20% held-out split during training (see README for details).")
st.dataframe(load_training_metrics().round(4), use_container_width=True)
