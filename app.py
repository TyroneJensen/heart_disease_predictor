import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal):
    # Create a feature array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict_proba(features_scaled)[0]
    
    return {
        "No Heart Disease": float(prediction[0]),
        "Heart Disease": float(prediction[1])
    }

# Create Gradio interface
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Sex (0=female, 1=male)"),
        gr.Number(label="Chest Pain Type (0-3)"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="Fasting Blood Sugar > 120 (1=true, 0=false)"),
        gr.Number(label="Resting ECG (0-2)"),
        gr.Number(label="Maximum Heart Rate"),
        gr.Number(label="Exercise Induced Angina (1=yes, 0=no)"),
        gr.Number(label="ST Depression"),
        gr.Number(label="Slope of ST Segment (0-2)"),
        gr.Number(label="Number of Major Vessels (0-3)"),
        gr.Number(label="Thal (3=normal, 6=fixed defect, 7=reversible defect)")
    ],
    outputs=gr.Label(label="Prediction"),
    title="Heart Disease Predictor",
    description="Enter patient information to predict heart disease probability"
)

if __name__ == "__main__":
    iface.launch()
