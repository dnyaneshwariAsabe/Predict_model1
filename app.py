import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Health Predictor", page_icon="🩺", layout="centered")

# Custom CSS for a centered, attractive interface
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .centered-icon {
        display: flex;
        justify-content: center;
        font-size: 70px;
        margin-bottom: 10px;
    }
    .report-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Header Section
st.markdown('<div class="centered-icon">🩺</div>', unsafe_html=True)
st.markdown("<h1 style='text-align: center;'>Diabetes Risk Assessment</h1>", unsafe_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter patient details below to predict health risk.</p>", unsafe_html=True)

st.divider()

# Input Form
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=1)
        glucose = st.slider("Glucose Level", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 140, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, value=80)
        bmi = st.number_input("BMI (Body Mass Index)", format="%.1f", value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", format="%.3f", value=0.470)
        age = st.number_input("Age (Years)", min_value=1, step=1, value=30)

st.write("") # Spacer

# Prediction Logic
if st.button("Analyze Results"):
    # Features must be in the exact order the model expects
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.divider()
    
    if prediction[0] == 1:
        st.error(f"### High Risk Detected")
        st.write(f"The model predicts a high probability of diabetes with **{probability[0][1]*100:.1f}%** confidence.")
    else:
        st.success(f"### Low Risk Detected")
        st.write(f"The model predicts a low probability of diabetes with **{probability[0][0]*100:.1f}%** confidence.")

    st.warning("Disclaimer: This tool is for educational purposes and is not a substitute for professional medical diagnosis.")
