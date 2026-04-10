import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration - Standard and stable
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

# Load the model with error handling
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found. Please ensure it is in the same folder.")
        return None

model = load_model()

# Centered Header using standard Markdown
st.markdown("<h1 style='text-align: center;'>🩺 Health Diagnostic Tool</h1>", unsafe_html=True)
st.write("---")

if model:
    # Creating two columns for a balanced UI
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
        glucose = st.slider("Glucose Level", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, value=0)
        bmi = st.number_input("BMI", format="%.1f", value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", format="%.3f", value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=25)

    st.write("---")

    # Centered Button
    if st.button("Predict Results", use_container_width=True):
        # Arrange features for the KNN model
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        
        # Display Result
        if prediction[0] == 1:
            st.error("### Prediction: High Risk of Diabetes")
        else:
            st.success("### Prediction: Low Risk of Diabetes")
            
        st.info("This is an AI prediction. Please consult a doctor for official medical advice.")
