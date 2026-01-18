# src/models/train_classification.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_classification_features(df: pd.DataFrame):
    feature_cols = [
        "Charging Rate (kW)",
        "abs_temperature",
        "soc_change",
        "Charging Duration (hours)",
        "Vehicle Age (years)",
        "charger_risk"
    ]

    X = df[feature_cols]
    y = df["battery_risk_level"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder, feature_cols


def train_classification_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(X_test, y_test, model, label_encoder):
    preds = model.predict(X_test)

    print("Classification Evaluation")
    print("-------------------------")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        preds,
        target_names=label_encoder.classes_
    ))


def run_classification_pipeline(data_path: str, model_path: str, encoder_path: str):
    df = load_dataset(data_path)

    X, y, label_encoder, feature_cols = prepare_classification_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_classification_model(X_train, y_train)
    evaluate_classification_model(X_test, y_test, model, label_encoder)

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"Classification model saved at {model_path}")
    print(f"Label encoder saved at {encoder_path}")

    return model, label_encoder, feature_cols
