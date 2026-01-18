# src/explainability/shap_classification.py

import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_shap_classification(
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

    # ---- Handle different SHAP output formats safely ----
    if isinstance(shap_values, list):
        # Binary or multiclass → pick the most important class
        # For binary classification, index 1 = positive class
        shap_to_plot = shap_values[1]
    elif isinstance(shap_values, np.ndarray):
        # New SHAP versions: shape = (samples, features, classes)
        shap_to_plot = shap_values[:, :, 1]
    else:
        raise ValueError("Unexpected SHAP values format")

    plt.figure()
    shap.summary_plot(
        shap_to_plot,
        X,
        feature_names=feature_cols,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Classification SHAP plot saved at {output_path}")
