import os
import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED, COLS_TO_DROP, TARGET_COL, CATEGORICAL_COLS, RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split

def load_raw_data() -> pd.DataFrame:
    filepath = os.path.join(DATA_RAW, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
    return pd.read_csv(filepath)

def inspect_data(df) -> None:
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nInfo:")
    df.info()
    print("\nDescribe:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicates:")
    print(df.duplicated().sum())

def clean_data(df) -> pd.DataFrame:
    df_cleaned = df.drop(columns=COLS_TO_DROP)
    df_cleaned = df_cleaned.drop_duplicates()
    if TARGET_COL in df_cleaned.columns:
        df_cleaned[TARGET_COL] = df_cleaned[TARGET_COL].map({'Yes': 1, 'No': 0})
    return df_cleaned

def encode_features(df) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=True)
    return df_encoded

def get_train_test_split(df) -> tuple:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def save_processed_data(df, filename):
    filepath = os.path.join(DATA_PROCESSED, filename)
    df.to_csv(filepath, index=False)

def get_preprocessing_pipeline():
    def pipeline(df):
        df_clean = clean_data(df)
        df_encoded = encode_features(df_clean)
        return df_encoded
    return pipeline
