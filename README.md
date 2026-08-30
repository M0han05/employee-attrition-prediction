# Employee Attrition Prediction Project

An end-to-end Machine Learning project to predict employee attrition (binary classification) using the IBM HR Analytics dataset, complete with exploratory data analysis, pipeline implementation, hyperparameter tuning, model improvement (SMOTE + threshold optimization), and a stunning Flask web application interface.

## Project Structure

```
employee-attrition-prediction/
├── data/
│   ├── raw/                          # Original dataset CSV
│   └── processed/                    # Cleaned full dataset (splits generated in-memory)
├── notebooks/
│   ├── 01_preprocessing.py           # Preprocessing script (Section 2)
│   ├── 02_eda.py                     # Exploratory Data Analysis & Viz (Section 3)
│   ├── 03_training.py                # Model training (Section 4)
│   ├── 04_evaluation.py              # Model evaluation & comparisons (Section 5)
│   └── 05_improvement.py             # Model tuning, CV, SMOTE & selection (Section 6)
├── src/
│   ├── __init__.py
│   ├── config.py                     # Feature names, paths, configs
│   ├── preprocessing.py              # Cleaning and helper functions
│   ├── training.py                   # Baseline training utilities
│   └── evaluation.py                 # Plotting and metric utilities
├── models/
│   ├── best_model.pkl                # Final trained pipeline model dict
│   ├── feature_columns.pkl           # Saved feature columns list
│   ├── logistic_regression.pkl       # Baseline Logistic Regression
│   ├── decision_tree.pkl             # Baseline Decision Tree
│   ├── random_forest.pkl             # Baseline Random Forest
│   └── xgboost.pkl                   # Baseline XGBoost
├── visualizations/                   # Generated PNG plots & charts
├── app/
│   ├── app.py                        # Flask main application (Section 7)
│   ├── static/
│   │   └── style.css                 # Premium custom styling (Glassmorphism)
│   └── templates/
│       ├── index.html                # Prediction Input Form
│       └── result.html               # Prediction Results Dashboard
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation (this file)
└── .gitignore                        # Git ignore patterns
```

---

## 1. Problem Identification
- **Objective**: Predict whether an employee is likely to leave the organization.
- **Problem Type**: Binary Classification (`Attrition` column: `Yes` / `No`).
- **Business Value**: Enable organizations to proactively identify high-risk employees, understand key drivers of attrition (like salary, overtime, job role), and implement retention strategies to mitigate recruitment/onboarding costs.

---

## 2. Dataset & Preprocessing
The project uses the fictional **IBM HR Analytics Employee Attrition & Performance** dataset (1,470 rows, 35 features).
- **Inspection**: Inspected schema, datatypes, descriptions, null counts (0 null values), and duplicate rows (0 duplicates).
- **Irrelevant Features Removed**: 
  - `EmployeeCount` (always `1`)
  - `StandardHours` (always `80`)
  - `Over18` (always `'Y'`)
  - `EmployeeNumber` (unique identifier)
- **Target Encoding**: Mapped target `'Attrition'` (`Yes` -> `1`, `No` -> `0`).
- **Feature Encoding**: Categorical nominal variables (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`) were one-hot encoded using `pd.get_dummies(..., drop_first=True)` to prevent the dummy variable trap.
- **Data Split**: Performed an 80/20 train/test split stratified by the target column to preserve class balance ratios (84% No / 16% Yes) and prevent data leakage during preprocessing.

---

## 3. EDA & Visualizations
We performed extensive EDA and generated 12 visualizations (saved to `visualizations/`):
- **attrition_distribution.png**: Highlights severe class imbalance (83.9% No, 16.1% Yes).
- **attrition_by_overtime.png**: Shows a massive spike in attrition rate for employees working overtime.
- **monthly_income_by_attrition.png**: Demonstrates that employees who leave have a lower median monthly income.
- **age_distribution_by_attrition.png**: Visualizes that younger employees are more prone to attrition.
- **jobrole_vs_attrition.png**: Shows Sales Representatives and Lab Technicians have the highest attrition rates.
- **correlation_heatmap.png**: Illustrates strong correlations between JobLevel, MonthlyIncome, and TotalWorkingYears.

---

## 4. ML Algorithm Implementation
Four baseline algorithms were trained using their default hyperparameters on the 80/20 split:
1. **Logistic Regression** (max_iter=1000)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier** (eval_metric='logloss')

---

## 5. Model Evaluation
The baseline models were evaluated on the test set using multiple metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 86.1% | 68.8% | 23.4% | 34.9% | 0.793 |
| **Decision Tree** | 76.5% | 31.0% | 38.3% | 34.3% | 0.611 |
| **Random Forest** | 83.3% | 41.7% | 10.6% | 16.9% | 0.786 |
| **XGBoost** | 86.1% | 65.0% | 27.7% | 38.8% | 0.764 |

- *Observation*: Baseline models suffer from extremely low recall on the positive class (`Attrition = Yes`) due to the class imbalance. For example, Random Forest only caught 10.6% of leaving employees.

---

## 6. Model Improvement
To improve performance, several optimization steps were conducted:
1. **Class Balancing (SMOTE)**: Applied SMOTE to the training set only, balancing the classes to 986 instances of both 'No' and 'Yes' classes.
2. **Cross-Validation**: Performed Stratified 5-Fold Cross Validation.
3. **Hyperparameter Tuning**: Tuned Random Forest and XGBoost using `RandomizedSearchCV` (scoring on F1-score):
   - **Best CV Model**: Random Forest (CV F1 Score: 0.9070)
4. **Threshold Tuning**: Tuned the decision boundary using a Precision-Recall curve to optimize the F1-score:
   - **Optimal Decision Threshold**: `0.2750`
5. **Comparison (Before vs After)**:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (Baseline)** | 83.3% | 41.7% | 10.6% | 16.9% | 0.786 |
| **Random Forest (SMOTE + Tuned + Threshold)** | **72.8%** | **34.0%** | **74.5%** | **46.7%** | **0.752** |

- *Impact*: Although overall accuracy dropped slightly, **Recall increased from 10.6% to 74.5%**, and the **F1-score improved from 16.9% to 46.7%**. In employee attrition prediction, identifying 74.5% of leaving employees (high recall) is highly preferred over a model with high accuracy that fails to detect employees leaving.

The final pipeline was saved as `models/best_model.pkl`.

---

## 7. Application / UI
We built a premium, responsive **Flask web application** that lets HR departments enter employee profiles and get attrition risk predictions.

- **Preprocessing Alignment**: Encodes raw categorical dropdown values directly into one-hot variables matching the 44 features expected by the trained Random Forest model.
- **Threshold Matching**: Integrates the optimized threshold of `0.2750` to make prediction decisions.
- **Glassmorphism Interface**: Featuring custom CSS styling, dark/indigo colors, smooth layouts, validation checks, progress sliders, and a dynamic prediction gauge widget.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / Command line to execute scripts

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/employee-attrition-prediction.git
   cd employee-attrition-prediction
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the scripts end-to-end (to regenerate models & visualizations):
   ```bash
   python notebooks/01_preprocessing.py
   python notebooks/02_eda.py
   python notebooks/03_training.py
   python notebooks/04_evaluation.py
   python notebooks/05_improvement.py
   ```
4. Start the Flask application:
   ```bash
   python app/app.py
   ```
5. Open your browser and navigate to `http://localhost:5000` to interact with the application.
