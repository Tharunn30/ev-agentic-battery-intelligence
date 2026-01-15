# src/data/preprocess.py

import pandas as pd

def preprocess_ev_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses EV charging dataset.
    """

    df = df.copy()

    # -----------------------------
    # 1. Datetime conversion
    # -----------------------------
    datetime_cols = ["Charging Start Time", "Charging End Time"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove rows with invalid timestamps
    df = df.dropna(subset=datetime_cols)

    # -----------------------------
    # 2. Handle missing values
    # -----------------------------
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Fill numeric NaNs with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaNs with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # -----------------------------
    # 3. Sanity checks
    # -----------------------------
    # Charging duration should be positive
    df = df[df["Charging Duration (hours)"] > 0]

    # SOC values should be between 0 and 100
    df = df[
        (df["State of Charge (Start %)"] >= 0) &
        (df["State of Charge (End %)"] <= 100)
    ]

    # -----------------------------
    # 4. Standardize categorical text
    # -----------------------------
    df["Charger Type"] = df["Charger Type"].str.strip()
    df["User Type"] = df["User Type"].str.strip()
    df["Vehicle Model"] = df["Vehicle Model"].str.strip()

    return df
