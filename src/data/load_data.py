# src/data/load_data.py

import pandas as pd

def load_ev_charging_data(file_path: str) -> pd.DataFrame:
    """
    Loads the EV charging dataset from CSV.
    """
    df = pd.read_csv(file_path)

    required_columns = [
        "Charging Rate (kW)",
        "Temperature (°C)",
        "State of Charge (Start %)",
        "State of Charge (End %)",
        "Charging Duration (hours)",
        "Charger Type",
        "Charging Start Time",
        "Charging End Time"
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df
