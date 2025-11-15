import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

# Load the pre-trained model and preprocessing objects
model = joblib.load('churn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
min_max_scaler = joblib.load('min_max_scaler.pkl')

# Define the order of features used during training
# This should match the 'selected_features' list used during model training
feature_order = [
    'CreditScore', 'Geography', 'Gender', 'Age',
    'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary'
]

# Streamlit app title and description
st.title('Customer Churn Prediction App')
st.write('Enter customer details to predict if they will churn.')

# Create input fields
credit_score = st.slider('Credit Score', 350, 850, 600)
geography_options = ['France', 'Spain', 'Germany']
geography = st.selectbox('Geography', geography_options)
gender_options = ['Female', 'Male']
gender = st.selectbox('Gender', gender_options)
age = st.slider('Age', 18, 92, 35)
tenure = st.slider('Tenure (years with bank)', 0, 10, 5)
balance = st.number_input('Balance', 0.0, 250000.0, 50000.0, format="%.2f")
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.checkbox('Has Credit Card?')
is_active_member = st.checkbox('Is Active Member?')
estimated_salary = st.number_input('Estimated Salary', 0.0, 200000.0, 100000.0, format="%.2f")

# Convert boolean checkboxes to integer (0 or 1)
has_cr_card_val = 1 if has_cr_card else 0
is_active_member_val = 1 if is_active_member else 0

# Prediction button
if st.button('Predict Churn'):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card_val,
        'IsActiveMember': is_active_member_val,
        'EstimatedSalary': estimated_salary
    }])

    # Apply Label Encoding to 'Geography' and 'Gender'
    # Ensure the input categories are known to the encoder
    input_data['Geography'] = label_encoder.transform(input_data['Geography'])
    input_data['Gender'] = label_encoder.transform(input_data['Gender'])

    # Apply Min-Max Scaling to numerical features
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    input_data[numerical_cols] = min_max_scaler.transform(input_data[numerical_cols])
    
    # Ensure the order of columns matches the training data
    input_data = input_data[feature_order]

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The customer is likely to churn. (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'The customer is not likely to churn. (Probability: {prediction_proba[0]:.2f})')
