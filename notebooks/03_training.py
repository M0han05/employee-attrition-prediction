import os
import sys
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR
from src.preprocessing import get_train_test_split
from src.training import train_all_models, save_model

print("1. Loading processed data...")
df = pd.read_csv(os.path.join(DATA_PROCESSED, 'processed_employee_attrition.csv'))

print("2. Regenerating train/test split...")
X_train, X_test, y_train, y_test = get_train_test_split(df)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

print("3. Training baseline models...")
trained_models = train_all_models(X_train, y_train)

print("4. Saving models...")
for name, model in trained_models.items():
    filename = name.lower().replace(' ', '_') + '.pkl'
    save_path = os.path.join(MODELS_DIR, filename)
    save_model(model, save_path)
    print(f"Saved {name} to {filename}")

print("\nTraining completed.")
