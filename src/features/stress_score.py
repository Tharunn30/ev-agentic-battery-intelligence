# src/features/stress_score.py

import numpy as np
import pandas as pd

def normalize(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min())

def compute_battery_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes battery stress score and risk level.
    """

    # Normalize numerical stress factors
    df["norm_charging_rate"] = normalize(df["Charging Rate (kW)"])
    df["norm_temperature"] = normalize(df["Temperature (°C)"].abs())

    soc_change = df["State of Charge (End %)"] - df["State of Charge (Start %)"]
    df["norm_soc_change"] = normalize(soc_change)

    # Charger risk mapping
    charger_risk_map = {
        "DC Fast Charger": 1.0,
        "Level 2": 0.6,
        "Level 1": 0.3
    }
    df["charger_risk"] = df["Charger Type"].map(charger_risk_map).fillna(0.5)

    # Battery Stress Score
    df["battery_stress_score"] = (
        0.35 * df["norm_charging_rate"] +
        0.30 * df["norm_temperature"] +
        0.20 * df["norm_soc_change"] +
        0.15 * df["charger_risk"]
    )

    # Risk categorization
    def risk_level(score):
        if score < 0.33:
            return "Low"
        elif score < 0.66:
            return "Medium"
        return "High"

    df["battery_risk_level"] = df["battery_stress_score"].apply(risk_level)

    return df
