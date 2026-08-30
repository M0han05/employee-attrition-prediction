import os
import traceback
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'premium_attrition_predictor_secret'

# Global variable for the model dict
model_dict = None

def load_model():
    global model_dict
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl'))
    try:
        if os.path.exists(model_path):
            model_dict = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}.")
        else:
            print(f"Warning: Model file not found at {model_path}. Please train and save the model.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_model()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_dict is None:
        return render_template('result.html', error="Model not loaded. Please ensure the model file exists in ../models/best_model.pkl.")
    
    try:
        # Extract form data
        form_data = request.form.to_dict()
        
        # Define numeric columns
        numeric_cols = [
            'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
            'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
            'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        
        input_dict = {}
        for key, value in form_data.items():
            if key in numeric_cols:
                input_dict[key] = float(value) if '.' in str(value) else int(value)
            else:
                input_dict[key] = value
                
        # Initialize input dictionary for the model
        feature_columns = model_dict['feature_columns']
        model_input = {col: 0 for col in feature_columns}
        
        # Fill numeric values
        for col in numeric_cols:
            if col in input_dict:
                model_input[col] = input_dict[col]
                
        # Fill categorical values (mapping to one-hot columns)
        cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        for col in cat_cols:
            if col in input_dict:
                val = input_dict[col]
                one_hot_col = f"{col}_{val}"
                if one_hot_col in model_input:
                    model_input[one_hot_col] = 1
                    
        # Create DataFrame with exact column ordering
        df_encoded = pd.DataFrame([model_input])[feature_columns]
        
        # Predict
        model = model_dict['model']
        threshold = model_dict.get('threshold', 0.5)
        
        probabilities = model.predict_proba(df_encoded)[0]
        prob_no = probabilities[0]
        prob_yes = probabilities[1]
        
        prediction = "Yes" if prob_yes >= threshold else "No"
        probability_percent = round(prob_yes * 100, 1)
        
        return render_template('result.html', 
                               prediction=prediction, 
                               probability=probability_percent,
                               input_details=input_dict)
        
    except Exception as e:
        traceback.print_exc()
        return render_template('result.html', error=f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
