# src/models/train_regression.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_regression_features(df: pd.DataFrame):
    feature_cols = [
        "Charging Rate (kW)",
        "abs_temperature",
        "soc_change",
        "Charging Duration (hours)",
        "Vehicle Age (years)",
        "charger_risk"
    ]

    X = df[feature_cols]
    y = df["battery_stress_score"]

    return X, y, feature_cols


def train_regression_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(X_test, y_test, model):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Regression Evaluation")
    print("---------------------")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


def run_regression_pipeline(data_path: str, model_path: str):
    df = load_dataset(data_path)

    X, y, feature_cols = prepare_regression_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_regression_model(X_train, y_train)
    evaluate_regression_model(X_test, y_test, model)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Regression model saved at {model_path}")

    return model, feature_cols