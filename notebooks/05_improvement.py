import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE
from src.preprocessing import get_train_test_split
from src.evaluation import evaluate_model

print("1. Loading processed data and regenerating splits...")
df = pd.read_csv(os.path.join(DATA_PROCESSED, 'processed_employee_attrition.csv'))
X_train, X_test, y_train, y_test = get_train_test_split(df)
feature_columns = list(X_train.columns)

print("2. Applying SMOTE to training data only...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Original train shape: {X_train.shape}, {y_train.value_counts().to_dict()}")
print(f"SMOTE train shape: {X_train_sm.shape}, {y_train_sm.value_counts().to_dict()}")

print("\n3. Baseline vs SMOTE Cross-Validation (5-fold, F1 Score)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scorer = make_scorer(f1_score)

models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring=scorer)
    print(f"{name} CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n4. Hyperparameter tuning (RandomizedSearchCV)...")

rf_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

print("Tuning Random Forest...")
rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=RANDOM_STATE), 
                               param_distributions=rf_param_grid, n_iter=30, cv=5, 
                               scoring='f1', random_state=RANDOM_STATE, n_jobs=-1)
rf_random.fit(X_train_sm, y_train_sm)
print(f"Best RF F1: {rf_random.best_score_:.4f}")

print("Tuning XGBoost...")
xgb_random = RandomizedSearchCV(estimator=XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'), 
                                param_distributions=xgb_param_grid, n_iter=30, cv=5, 
                                scoring='f1', random_state=RANDOM_STATE, n_jobs=-1)
xgb_random.fit(X_train_sm, y_train_sm)
print(f"Best XGB F1: {xgb_random.best_score_:.4f}")

if xgb_random.best_score_ > rf_random.best_score_:
    best_model = xgb_random.best_estimator_
    best_model_name = 'XGBoost'
else:
    best_model = rf_random.best_estimator_
    best_model_name = 'Random Forest'

print(f"\nBest Model selected: {best_model_name}")

print("\n5. Feature Importance...")
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
plt.figure(figsize=(10, 8))
plt.title(f"Top 20 Feature Importances - {best_model_name}")
plt.bar(range(20), importances[indices], align="center")
plt.xticks(range(20), [feature_columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, '..', 'visualizations', 'feature_importance.png'), dpi=150)
plt.close()
print("Saved feature_importance.png")

print("\n6. Threshold Tuning (Precision-Recall)...")
y_proba = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal Threshold for F1: {best_threshold:.4f}")

print("\n7. Final Evaluation...")
y_pred_tuned = (y_proba >= best_threshold).astype(int)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
final_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_tuned),
    'precision': precision_score(y_test, y_pred_tuned),
    'recall': recall_score(y_test, y_pred_tuned),
    'f1': f1_score(y_test, y_pred_tuned),
    'roc_auc': roc_auc_score(y_test, y_proba)
}
print(pd.DataFrame([final_metrics], index=[f"{best_model_name} (Tuned)"]))

print("\n8. Saving Final Model...")
final_save_dict = {
    'model': best_model,
    'feature_columns': feature_columns,
    'threshold': float(best_threshold)
}
joblib.dump(final_save_dict, os.path.join(MODELS_DIR, 'best_model.pkl'))
joblib.dump(feature_columns, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
print("Saved best_model.pkl and feature_columns.pkl")
