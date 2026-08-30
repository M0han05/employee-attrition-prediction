import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import load_raw_data
from src.config import VIZ_DIR

# Set colors
colors = {'No': '#2ecc71', 'Yes': '#e74c3c'}
palette = ['#2ecc71', '#e74c3c']

print("Loading raw data for EDA...")
df = load_raw_data()

print("1. attrition_distribution.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=df, x='Attrition', palette=palette, ax=axes[0])
axes[0].set_title('Attrition Count')
attrition_counts = df['Attrition'].value_counts()
axes[1].pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', colors=[colors[k] for k in attrition_counts.index])
axes[1].set_title('Attrition Percentage')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'attrition_distribution.png'), dpi=150)
plt.close()
print("Insight: The dataset is imbalanced, with a much higher percentage of 'No' compared to 'Yes'.")

print("2. attrition_by_department.png")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition', palette=palette)
plt.title('Attrition by Department')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'attrition_by_department.png'), dpi=150)
plt.close()
print("Insight: Research & Development has the highest count of both attritions, but Sales might have a higher rate relative to its size.")

print("3. attrition_by_overtime.png")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='OverTime', hue='Attrition', palette=palette)
plt.title('Attrition by OverTime')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'attrition_by_overtime.png'), dpi=150)
plt.close()
print("Insight: Employees working overtime show a significantly higher proportion of attrition compared to those who don't.")

print("4. attrition_by_marital_status.png")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='MaritalStatus', hue='Attrition', palette=palette)
plt.title('Attrition by Marital Status')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'attrition_by_marital_status.png'), dpi=150)
plt.close()
print("Insight: Single employees have the highest rate of attrition.")

print("5. attrition_by_business_travel.png")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='BusinessTravel', hue='Attrition', palette=palette)
plt.title('Attrition by Business Travel')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'attrition_by_business_travel.png'), dpi=150)
plt.close()
print("Insight: Those who travel frequently seem to have a higher attrition rate.")

print("6. age_distribution_by_attrition.png")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['Attrition'] == 'No'], x='Age', label='No', fill=True, color=colors['No'])
sns.kdeplot(data=df[df['Attrition'] == 'Yes'], x='Age', label='Yes', fill=True, color=colors['Yes'])
plt.title('Age Distribution by Attrition')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'age_distribution_by_attrition.png'), dpi=150)
plt.close()
print("Insight: Younger employees are more likely to leave.")

print("7. monthly_income_by_attrition.png")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', palette=palette)
plt.title('Monthly Income by Attrition')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'monthly_income_by_attrition.png'), dpi=150)
plt.close()
print("Insight: Employees who left generally had lower median monthly incomes.")

print("8. jobrole_vs_attrition.png")
plt.figure(figsize=(12, 8))
role_attrition = pd.crosstab(df['JobRole'], df['Attrition'], normalize='index') * 100
role_attrition.plot(kind='barh', stacked=True, color=[colors['No'], colors['Yes']], figsize=(12, 8))
plt.title('Percentage Attrition by Job Role')
plt.xlabel('Percentage')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'jobrole_vs_attrition.png'), dpi=150)
plt.close()
print("Insight: Sales Representatives and Laboratory Technicians have the highest relative attrition rates.")

print("9. correlation_heatmap.png")
plt.figure(figsize=(16, 12))
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("Insight: There are strong positive correlations among JobLevel, MonthlyIncome, and TotalWorkingYears.")

print("10. years_at_company_by_attrition.png")
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Attrition', y='YearsAtCompany', palette=palette)
plt.title('Years at Company by Attrition')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'years_at_company_by_attrition.png'), dpi=150)
plt.close()
print("Insight: Attrition is highest among newer employees with fewer years at the company.")

print("11. job_satisfaction_by_attrition.png")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='JobSatisfaction', hue='Attrition', palette=palette)
plt.title('Attrition by Job Satisfaction')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'job_satisfaction_by_attrition.png'), dpi=150)
plt.close()
print("Insight: Lower job satisfaction slightly increases the likelihood of attrition, though the trend is not overwhelmingly dominant.")

print("12. distance_from_home_by_attrition.png")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['Attrition'] == 'No'], x='DistanceFromHome', label='No', fill=True, color=colors['No'])
sns.kdeplot(data=df[df['Attrition'] == 'Yes'], x='DistanceFromHome', label='Yes', fill=True, color=colors['Yes'])
plt.title('Distance From Home by Attrition')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'distance_from_home_by_attrition.png'), dpi=150)
plt.close()
print("Insight: Employees living further from home show a slightly higher tendency to leave.")

print("\nEDA completed. All visualizations saved to visualizations directory.")
