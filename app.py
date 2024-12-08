

import numpy as np


import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer

# Load the trained model
model = pickle.load(open('model_logi.pkl', 'rb'))

# Load label encoder (if needed)
le = LabelEncoder()  # Replace with actual loaded encoder if used during training

# Define the function to preprocess input data
def preprocess_data(df):
    # Assuming similar preprocessing steps as in your training code
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Handle outliers (if applicable)
    def handle_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound),
                              df[column].median(), df[column])
        return df

    for column in numerical_columns:
        df = handle_outliers(df, column)

    # Encode categorical columns (if applicable)
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])

    # Scale numerical columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Apply Yeo-Johnson transformation (if applicable)
    cols = []  # Replace with the actual list of columns used for transformation
    pt = PowerTransformer(method='yeo-johnson')
    df_yeojohnson_transformed = df.copy()
    df_yeojohnson_transformed[cols] = pt.fit_transform(df_yeojohnson_transformed[cols])

    return df_yeojohnson_transformed

# Streamlit app
st.title('Loan Status Prediction App')

# Input fields for user data
addr_state = st.selectbox('Address State', ['AL', 'AK', 'AZ', ...])  # Replace with actual state options
annual_inc = st.number_input('Annual Income')
earliest_cr_line = st.date_input('Earliest Credit Line')
emp_title = st.text_input('Employment Title')
int_rate = st.number_input('Interest Rate')
loan_amnt = st.number_input('Loan Amount')
tot_cur_bal = st.number_input('Total Current Balance')
open_acc = st.number_input('Open Accounts')
revol_bal = st.number_input('Revolving Balance')
revol_util = st.number_input('Revolving Utilization Rate')
sub_grade = st.selectbox('Sub Grade', ['A1', 'A2', 'A3', ...])  # Replace with actual sub grade options
total_acc = st.number_input('Total Accounts')

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'addr_state': [addr_state],
    'annual_inc': [annual_inc],
    'earliest_cr_line': [earliest_cr_line],
    'emp_title': [emp_title],
    'int_rate': [int_rate],
    'loan_amnt': [loan_amnt],
    'tot_cur_bal': [tot_cur_bal],
    'open_acc': [open_acc],
    'revol_bal': [revol_bal],
    'revol_util': [revol_util],
    'sub_grade': [sub_grade],
    'total_acc': [total_acc]
})

# Preprocess the input data
preprocessed_data = preprocess_data(input_data)

# Make prediction
if st.button('Predict'):
    prediction = model.predict(preprocessed_data)
    predicted_category = le.inverse_transform(prediction)
    st.write('Predicted Loan Status:', predicted_category[0])

