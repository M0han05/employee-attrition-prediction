import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.config import RANDOM_STATE, MODELS_DIR

def get_baseline_models() -> dict:
    return {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
    }

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train, y_train) -> dict:
    models = get_baseline_models()
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_model(model, X_train, y_train)
    return trained_models

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)
