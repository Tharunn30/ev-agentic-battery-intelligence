# # main.py

# from src.data.load_data import load_ev_charging_data
# from src.data.preprocess import preprocess_ev_data
# from src.features.stress_score import (
#     add_engineered_features,
#     compute_battery_stress_score
# )
# from src.config.settings import (
#     RAW_DATA_PATH,
#     CLEAN_DATA_PATH,
#     STRESS_DATA_PATH
# )


# def main():
#     # -----------------------------
#     # Milestone 1: Load & preprocess
#     # -----------------------------
#     df_raw = load_ev_charging_data(RAW_DATA_PATH)
#     df_clean = preprocess_ev_data(df_raw)
#     df_clean.to_csv(CLEAN_DATA_PATH, index=False)

#     # -----------------------------
#     # Milestone 2: Feature engineering
#     # -----------------------------
#     df_features = add_engineered_features(df_clean)
#     df_stress = compute_battery_stress_score(df_features)

#     df_stress.to_csv(STRESS_DATA_PATH, index=False)

#     print("Milestone 2 completed: Battery stress features and labels generated.")


# if __name__ == "__main__":
#     main()

#--------------------------------------------

# main.py - Regression Model Training

# from src.models.train_regression import run_regression_pipeline
# from src.config.settings import STRESS_DATA_PATH, REG_MODEL_PATH


# def main():
#     run_regression_pipeline(
#         data_path=STRESS_DATA_PATH,
#         model_path=REG_MODEL_PATH
#     )

#     print("Milestone 3A completed: Regression model trained.")


# if __name__ == "__main__":
#     main()

#------------------------------


# main.py - Classification Model Training

from src.models.train_classification import run_classification_pipeline
from src.config.settings import (
    STRESS_DATA_PATH,
    CLF_MODEL_PATH,
    LABEL_ENCODER_PATH
)


def main():
    run_classification_pipeline(
        data_path=STRESS_DATA_PATH,
        model_path=CLF_MODEL_PATH,
        encoder_path=LABEL_ENCODER_PATH
    )

    print("Milestone 3B completed: Classification model trained.")


if __name__ == "__main__":
    main()
