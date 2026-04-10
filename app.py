import streamlit as st
import pandas as pd
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import requests

# Page Configuration
st.set_page_config(page_title="Health Diagnostic Assistant", layout="wide")

# Custom CSS for Animation and Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        animation: fadeIn 1.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """, unsafe_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Assets
lottie_health = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_5njp3v83.json")
model = pickle.load(open('model (1).pkl', 'rb'))

# App Layout
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter the patient's clinical metrics in the sidebar to get a prediction.")

with st.sidebar:
    st.header("Patient Data Input")
    st_lottie(lottie_health, height=150, key="health_icon")
    
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 140, 70)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", format="%.3f")
    age = st.number_input("Age", min_value=1, step=1)

# Prediction Logic
if st.button("Generate Diagnostic Report"):
    # Prepare features in the exact order the model was trained
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("### Prediction: High Risk")
            st.write("The model suggests a high likelihood of diabetes.")
        else:
            st.success("### Prediction: Low Risk")
            st.write("The model suggests a low likelihood of diabetes.")
            
    with col2:
        st.metric(label="Confidence Level", value=f"{max(probability[0])*100:.2f}%")
        st.progress(max(probability[0]))

    st.info("**Note:** This is a machine learning prediction and should not replace professional medical advice.")
