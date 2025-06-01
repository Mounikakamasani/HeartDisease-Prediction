import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open("model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

st.title("Heart Disease Prediction App")
st.markdown("Enter the patient details below to predict heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)")
chol = st.number_input("Cholesterol (chol)")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG results (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)")
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise (oldpeak)")
slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    sex_val = 1 if sex == "Male" else 0
    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")
