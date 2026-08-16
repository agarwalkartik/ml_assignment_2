"""
Train and evaluate 6 classification models on the UCI Adult (Census Income) dataset.

Predicts whether a person's income exceeds $50K/year based on census attributes.
Saves fitted models + preprocessing pipeline to model/artifacts/ for reuse in the
Streamlit app, and writes a held-out test split to test_data.csv for submission.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
TARGET = "income"


def load_raw():
    train = pd.read_csv(
        os.path.join(DATA_DIR, "adult.data"),
        names=COLUMNS, sep=r",\s*", engine="python", na_values="?",
    )
    test = pd.read_csv(
        os.path.join(DATA_DIR, "adult.test"),
        names=COLUMNS, sep=r",\s*", engine="python", na_values="?", skiprows=1,
    )
    test[TARGET] = test[TARGET].str.rstrip(".")
    df = pd.concat([train, test], ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    df[TARGET] = (df[TARGET] == ">50K").astype(int)
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
    }


def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_raw()
    print(f"Loaded {len(df)} rows after dropping missing values.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
        X_test_t = X_test_t.toarray()

    results = {}
    for name, model in get_models().items():
        print(f"Training {name}...")
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        y_prob = model.predict_proba(X_test_t)[:, 1]
        results[name] = evaluate(y_test, y_pred, y_prob)

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(ARTIFACT_DIR, f"{fname}.joblib"))

    metrics_df = pd.DataFrame(results).T
    metrics_df = metrics_df.round(4)
    metrics_df.to_csv(os.path.join(ARTIFACT_DIR, "metrics.csv"))
    print("\n=== Comparison Table ===")
    print(metrics_df.to_string())

    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_df.reset_index().rename(columns={"index": "Model"}).to_dict(orient="records"), f, indent=2)

    # Held-out test data for submission: raw (untransformed) feature columns + true label,
    # sampled from the test split so the Streamlit app can run real inference on it.
    test_export = X_test.copy()
    test_export[TARGET] = y_test.map({0: "<=50K", 1: ">50K"}).values
    test_export = test_export.sample(n=min(1000, len(test_export)), random_state=42)
    test_export.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)
    print(f"\nWrote {len(test_export)} rows to test_data.csv")


if __name__ == "__main__":
    main()
