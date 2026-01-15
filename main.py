# main.py

from src.data.load_data import load_ev_charging_data
from src.data.preprocess import preprocess_ev_data
from src.config.settings import RAW_DATA_PATH, CLEAN_DATA_PATH

def main():
    # Load raw data
    df = load_ev_charging_data(RAW_DATA_PATH)

    # Preprocess
    df_clean = preprocess_ev_data(df)

    # Save cleaned dataset
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)

    print("Milestone 1 completed: Clean dataset saved.")

if __name__ == "__main__":
    main()
