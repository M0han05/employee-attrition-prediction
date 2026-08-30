import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR, VIZ_DIR
from src.preprocessing import get_train_test_split
from src.training import load_model
from src.evaluation import evaluate_all_models, get_classification_report, plot_confusion_matrix, plot_roc_curves, plot_metrics_comparison

print("1. Loading test data and models...")
df = pd.read_csv(os.path.join(DATA_PROCESSED, 'processed_employee_attrition.csv'))
_, X_test, _, y_test = get_train_test_split(df)

models_to_load = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

loaded_models = {}
for name, filename in models_to_load.items():
    loaded_models[name] = load_model(os.path.join(MODELS_DIR, filename))

print("2. Evaluating models...")
results_df = evaluate_all_models(loaded_models, X_test, y_test)

print("\n3. Classification Reports:")
for name, model in loaded_models.items():
    print(f"\n--- {name} ---")
    print(get_classification_report(model, X_test, y_test))

print("4. Generating confusion matrices...")
for name, model in loaded_models.items():
    save_path = os.path.join(VIZ_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
    plot_confusion_matrix(model, X_test, y_test, name, save_path)

print("5. Generating ROC curves...")
plot_roc_curves(loaded_models, X_test, y_test, os.path.join(VIZ_DIR, 'roc_curves.png'))

print("6. Generating metrics comparison...")
plot_metrics_comparison(results_df.T.to_dict(), os.path.join(VIZ_DIR, 'metrics_comparison.png'))

print("\n7. Comparison Table:")
print(results_df)

print("\nEvaluation completed. All plots saved to visualizations directory.")
