import gradio as gr
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import json
import os

# ── Load Model & Scaler ───────────────────────────────────────────────
# In HF Spaces, files are in the root directory
try:
    model = load_model("saved_model/heart_attack_model.keras")
    scaler = joblib.load("saved_model/scaler.pkl")
    with open("saved_model/metadata.json", "r") as f:
        metadata = json.load(f)
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fallback/Mock for testing if files are missing in build
    raise e

# ── Prediction Function ───────────────────────────────────────────────
def predict_heart_attack(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Mapping for categorical inputs (if used as dropdowns)
    # Ensure inputs match the training schema exactly
    
    input_data = np.array([[
        float(age),
        float(sex),
        float(cp),
        float(trestbps),
        float(chol),
        float(fbs),
        float(restecg),
        float(thalach),
        float(exang),
        float(oldpeak),
        float(slope),
        float(ca),
        float(thal)
    ]])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prob = float(model.predict(input_scaled, verbose=0)[0][0])
    
    label = "High Risk 🚨" if prob > 0.5 else "Low Risk ✅"
    confidence = prob if prob > 0.5 else 1 - prob
    
    return {
        "Risk Assessment": label,
        "Probability": f"{prob:.1%}",
        "Confidence": f"{confidence:.1%}"
    }

# ── Create Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(title="Heart Attack Risk Predictor") as demo:
    gr.Markdown("# ❤️ Heart Attack Risk Predictor")
    gr.Markdown("Enter patient details below to estimate heart attack risk using a Neural Network.")
    
    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=63)
            sex = gr.Radio(label="Sex", choices=[("Male", 1), ("Female", 0)], value=1)
            cp = gr.Slider(label="Chest Pain Type (0-3)", minimum=0, maximum=3, step=1, value=3)
            trestbps = gr.Number(label="Resting Blood Pressure (mm Hg)", value=145)
            chol = gr.Number(label="Cholesterol (mg/dl)", value=233)
            fbs = gr.Radio(label="Fasting Blood Sugar > 120 mg/dl", choices=[("True", 1), ("False", 0)], value=1)
            restecg = gr.Slider(label="Resting ECG Results (0-2)", minimum=0, maximum=2, step=1, value=0)
            
        with gr.Column():
            thalach = gr.Number(label="Max Heart Rate Achieved", value=150)
            exang = gr.Radio(label="Exercise Induced Angina", choices=[("Yes", 1), ("No", 0)], value=0)
            oldpeak = gr.Number(label="ST Depression (Oldpeak)", value=2.3)
            slope = gr.Slider(label="Slope of Peak Exercise ST (0-2)", minimum=0, maximum=2, step=1, value=0)
            ca = gr.Slider(label="Number of Major Vessels (0-3)", minimum=0, maximum=3, step=1, value=0)
            thal = gr.Slider(label="Thalassemia (0-3)", minimum=0, maximum=3, step=1, value=1)
            
    btn = gr.Button("Predict Risk", variant="primary")
    
    output = gr.JSON(label="Prediction Results")
    
    btn.click(
        predict_heart_attack,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=output
    )
    
    gr.Markdown("---")
    gr.Markdown("### Model Info")
    gr.Markdown(f"- **Features:** {len(metadata['features'])}")
    gr.Markdown(f"- **Accuracy (Test):** ~80%")

# ── Launch ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
