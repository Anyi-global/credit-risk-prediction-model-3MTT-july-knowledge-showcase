import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and label encoders
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('encoder.pkl')

# --------- Streamlit Page Setup ----------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.markdown("""
    <style>
        body, .stApp {
            background-color: #000000;
            color: white;
        }

        /* Make form label text white */
        label, .stTextInput label, .stSelectbox label, .stNumberInput label, .stSlider label {
            color: white !important;
        }

        /* Subheader and title */
        h1, h2, h3, .stMarkdown, .css-1v0mbdj, .css-1d391kg {
            color: white !important;
        }

        /* Make button text and background more visible */
        .stButton>button {
            background-color: #1a1a1a !important;
            color: white !important;
            border: 1px solid white;
        }

        .stButton>button:hover {
            background-color: #333333 !important;
            color: #00ffcc !important;
            border: 1px solid #00ffcc;
        }

    </style>
""", unsafe_allow_html=True)

st.title("Credit Risk Prediction App")
st.write("Enter borrower details to predict the likelihood of defaulting on a loan.")

# Input fields
person_age = st.number_input("Age", min_value=18, max_value=100)
person_income = st.number_input("Monthly Income")

home_type = st.selectbox("Home Type", label_encoders['home_type'].classes_)
work_experience = st.number_input("Work Experience (Years)", min_value=0.0)

loan_intent = st.selectbox("Loan Intent", label_encoders['loan_intent'].classes_)
loan_grade = st.selectbox("Loan Grade", label_encoders['loan_grade'].classes_)

loan_amount = st.number_input("Loan Amount")
loan_int_rate = st.number_input("Interest Rate (%)")
loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0)

# --- Auto-calculate loan_percent_income ---
annual_income = person_income * 12
loan_percent_income = annual_income / loan_amount if annual_income > 0 else 0.0

st.markdown(f"**Loan as % of Income:** `{loan_percent_income:.2f}`")

previous_default = st.selectbox("Previous Default?", label_encoders['previous_default'].classes_)
credit_score = st.number_input("Credit History Length (Years)", min_value=0)

# Encode categorical variables using saved label encoders
home_type_encoded = label_encoders['home_type'].transform([home_type])[0]
loan_intent_encoded = label_encoders['loan_intent'].transform([loan_intent])[0]
loan_grade_encoded = label_encoders['loan_grade'].transform([loan_grade])[0]
previous_default_encoded = label_encoders['previous_default'].transform([previous_default])[0]

# Split input into numeric and categorical parts
numeric_features = np.array([[
    person_age,
    person_income,
    work_experience,
    loan_amount,
    loan_int_rate,
    loan_percent_income,
    credit_score
]])

categorical_features = np.array([[
    home_type_encoded,
    loan_intent_encoded,
    loan_grade_encoded,
    previous_default_encoded
]])

# Scale only numeric part
numeric_scaled = scaler.transform(numeric_features)

# Concatenate scaled numeric and categorical inputs
input_final = np.concatenate([numeric_scaled, categorical_features], axis=1)

# Predict and display result
if st.button("Predict Default Risk"):
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability: {probability:.2f})")
