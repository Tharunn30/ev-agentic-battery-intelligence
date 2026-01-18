# src/explainability/shap_regression.py

import os
import shap
import pandas as pd
import matplotlib.pyplot as plt


def run_shap_regression(
    model,
    data_path: str,
    feature_cols: list,
    output_path: str
):
    df = pd.read_csv(data_path)
    X = df[feature_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_cols,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Regression SHAP plot saved at {output_path}")
