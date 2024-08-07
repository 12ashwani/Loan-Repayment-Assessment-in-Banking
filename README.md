Loan Repayment Assessment in Banking
KnowledgeHut AI Hackathon Report
actua.

Dataset
The dataset consists of two files:

train_loan_data.csv: 80,000 records with 28 features, including the target variable loan_status.
test_loan_data.csv: 20,000 records with 27 features (excluding loan_status).
test_results.csv: Contains the target loan_status for the test dataset, used for model evaluation.
Features Description
earliest_cr_line: The month the borrower's earliest reported credit line was opened.
emp_title: The job title supplied by the borrower when applying for the loan.
fico_range_high: The upper boundary range of the borrower’s FICO at loan origination.
fico_range_low: The lower boundary range of the borrower’s FICO at loan origination.
grade: LC assigned loan grade.
application_type: Indicates whether the loan is an individual or joint application.
initial_list_status: The initial listing status of the loan (W or F).
num_actv_bc_tl: Number of currently active bankcard accounts.
mort_acc: Number of mortgage accounts.
tot_cur_bal: Total current balance of all accounts.
open_acc: Number of open credit lines in the borrower's credit file.
pub_rec: Number of derogatory public records.
pub_rec_bankruptcies: Number of public record bankruptcies.
purpose: Category provided by the borrower for the loan request.
revol_bal: Total credit revolving balance.
title: The loan title provided by the borrower.
total_acc: Total number of credit lines in the borrower's credit file.
verification_status: Indicates if income was verified by LC.
addr_state: The state provided by the borrower in the loan application.
annual_inc: Self-reported annual income provided by the borrower.
emp_length: Employment length in years.
home_ownership: Home ownership status provided by the borrower.
int_rate: Interest rate on the loan.
loan_amnt: The listed amount of the loan applied for by the borrower.
sub_grade: LC assigned loan subgrade.
term: Number of payments on the loan.
revol_util: Revolving line utilization rate.
loan_status: Status of the loan (target variable).
Task
Data Split: Decide the data split ratio for training, validation, and test sets. Ensure randomness in the split.
Class Imbalance: Address potential class imbalance in the loan_status feature.
Feature Normalization: Decide on feature normalization techniques.
Data Preprocessing: Perform tasks such as outlier detection, missing value handling, and feature selection before training the model.
Model Building: Build the model using various machine learning techniques, including ensemble methods (Bagging and Boosting).
Model Evaluation: Evaluate the model using cross-validation, grid search, and the F1 score as the evaluation metric.
Methodology
1. Data Split
The dataset will be split into training, validation, and test sets with a ratio of 70:15:15. Randomness will be ensured using the train_test_split function from scikit-learn with a fixed random seed.

2. Class Imbalance
The class distribution of the loan_status feature will be analyzed. If an imbalance is found, techniques such as SMOTE (Synthetic Minority Over-sampling Technique) or class weights adjustment will be applied.

3. Feature Normalization
Continuous features will be normalized using techniques such as Min-Max Scaling or Standard Scaling to ensure all features contribute equally to the model.

4. Data Preprocessing
Outlier Detection: Identify and handle outliers using IQR (Interquartile Range) or Z-score methods.
Missing Values: Handle missing values using imputation techniques like mean, median, or mode imputation.
Feature Selection: Select relevant features based on correlation analysis and feature importance from models.
5. Model Building
EDA: Perform Exploratory Data Analysis (EDA) to understand the data distribution and relationships between features.
Feature Engineering: Create new features or transform existing ones to improve model performance.
Ensemble Techniques: Implement Bagging (e.g., Random Forest) and Boosting (e.g., XGBoost) techniques to improve model accuracy.
Cross-Validation and Grid Search: Use cross-validation to evaluate model performance and grid search for hyperparameter tuning.
Model Evaluation: Evaluate the model using the F1 score.
Implementation
1. Statistics Descriptive Analysis
python
Copy code
import pandas as pd

# Load the dataset
train_data = pd.read_csv('train_loan_data.csv')

# Descriptive statistics
train_data.describe()
2. Exploratory Data Analysis (EDA)
python
Copy code
import matplotlib.pyplot as plt
import seaborn as sns

# EDA
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_status', data=train_data)
plt.title('Loan Status Distribution')
plt.show()
3. Data Preprocessing
python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Handle missing values
imputer = SimpleImputer(strategy='median')
train_data_imputed = imputer.fit_transform(train_data)

# Split the data
X = train_data_imputed.drop('loan_status', axis=1)
y = train_data_imputed['loan_status']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
4. Model Building and Evaluation
python
Copy code
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

# Model training
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Prediction
y_pred = rf.predict(X_val_scaled)

# Evaluation
f1 = f1_score(y_val, y_pred, pos_label='Paid')
print(f'F1 Score: {f1}')

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_
# Conclusion
This report provides a comprehensive approach to building a machine learning model for loan repayment assessment. The process includes data preprocessing, feature engineering, model building, and evaluation. The model's performance will be evaluated using the F1 score, ensuring an effective classification of loan repayment statuses. The results, including the final model and evaluation metrics, will be presented in a Jupyter Notebook, along with a PowerPoint presentation and a video walkthrough. The entire project will be shared via a GitHub repository.
