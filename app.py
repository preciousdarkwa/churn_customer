import streamlit as st
import pandas as pd
import joblib
import numpy as np
 
# --------------------------------
# Load model and preprocessing objects
# --------------------------------
model = joblib.load('churn_model.pkl')
 
# Separate encoders for Geography and Gender (BEST PRACTICE)
geo_encoder = joblib.load('geo_encoder.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
 
# Load scaler
min_max_scaler = joblib.load('min_max_scaler.pkl')
 
# Feature order (must match training)
feature_order = [
    'CreditScore', 'Geography', 'Gender', 'Age',
    'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary'
]
 
# --------------------------------
# Streamlit UI
# --------------------------------
st.title('Customer Churn Prediction App')
st.write('Enter customer details to predict if they will churn.')
 
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
 
has_cr_card_val = 1 if has_cr_card else 0
is_active_member_val = 1 if is_active_member else 0
 
# --------------------------------
# SAFE LABEL ENCODING HELPER
# --------------------------------
def safe_label_transform(encoder, value):
    """Safely transform label-encoded values, including unseen ones."""
    if value not in encoder.classes_:
        # Extend encoder to include new category
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]
 
# --------------------------------
# Prediction
# --------------------------------
if st.button('Predict Churn'):
 
    # Build user input DataFrame
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
 
    # Safe encoding (prevents "unseen label" crash)
    input_data['Geography'] = input_data['Geography'].apply(
        lambda x: safe_label_transform(geo_encoder, x)
    )
    input_data['Gender'] = input_data['Gender'].apply(
        lambda x: safe_label_transform(gender_encoder, x)
    )
 
    # Numerical scaling
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    input_data[numerical_cols] = min_max_scaler.transform(input_data[numerical_cols])
 
    # Reorder columns
    input_data = input_data[feature_order]
 
    # Predict
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]
 
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The customer is likely to churn. (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'The customer is not likely to churn. (Probability: {prediction_proba[0]:.2f})')
