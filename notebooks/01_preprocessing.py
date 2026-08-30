import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_raw_data, inspect_data, clean_data, encode_features, save_processed_data, get_train_test_split

print("## Section 1: Loading Data")
df_raw = load_raw_data()
print("Raw data loaded successfully.")

print("\n## Section 2: Inspecting Data")
inspect_data(df_raw)

print("\n## Section 3: Cleaning Data")
df_clean = clean_data(df_raw)
print(f"Cleaned data shape: {df_clean.shape}")

print("\n## Section 4: Encoding Features")
df_encoded = encode_features(df_clean)
print(f"Encoded data shape: {df_encoded.shape}")

print("\n## Section 5: Saving Processed Data")
save_processed_data(df_encoded, 'processed_employee_attrition.csv')
print("Processed data saved to data/processed/processed_employee_attrition.csv")

print("\n## Section 6: Train-Test Split")
X_train, X_test, y_train, y_test = get_train_test_split(df_encoded)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("\nClass distribution in train set:")
print(y_train.value_counts(normalize=True))
print("\nClass distribution in test set:")
print(y_test.value_counts(normalize=True))

print("\n## Summary of Preprocessing")
print("- Dropped irrelevant columns (EmployeeCount, StandardHours, Over18, EmployeeNumber)")
print("- Removed duplicates")
print("- Mapped Target (Attrition) to 0/1")
print("- One-hot encoded categorical features")
print("- Split into 80/20 train/test sets with stratification")
