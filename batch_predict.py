"""
Batch Prediction on Heart Attack Dataset
Loads the dataset, predicts risk for EVERY row, and saves the results.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Heart Attack Data Set.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
OUTPUT_PATH = os.path.join(BASE_DIR, "heart_attack_with_predictions.csv")

# ── load Resources ────────────────────────────────────────────────────
print(f"📂 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print(f"Start Loading model and scaler...")
model = load_model(os.path.join(MODEL_DIR, "heart_attack_model.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# ── Preprocess Features ───────────────────────────────────────────────
# We need to ensure we use the exact same columns as training (excluding target)
# Auto-detect target again to drop it
target_candidates = ['target', 'output', 'label', 'class', 'result']
target_col = None
for col in df.columns:
    if col.strip().lower() in target_candidates:
        target_col = col
        break

if target_col:
    print(f"Target column detected: '{target_col}' (Dropping for prediction)")
    X = df.drop(columns=[target_col])
else:
    X = df.copy()

# Handle missing values (same logic as training)
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
cat_cols = X.select_dtypes(exclude=[np.number]).columns
if len(cat_cols) > 0:
    # simple fill for batch script
    X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

# Scale features
X_scaled = scaler.transform(X)

# ── Make Predictions ──────────────────────────────────────────────────
print(f"🔮 Predicting on {len(df)} patients...")
predictions = model.predict(X_scaled, verbose=1)

# Add results to dataframe
df['Predicted_Probability'] = predictions.flatten()
df['Predicted_Risk_Label'] = (df['Predicted_Probability'] > 0.5).astype(int)
df['Risk_Level'] = df['Predicted_Probability'].apply(
    lambda p: "High Risk" if p > 0.5 else "Low Risk"
)

# ── Save & Show ───────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Predictions saved to: {OUTPUT_PATH}")

print("\n── Sample Results (First 5 Rows) ─────────────────────────────")
# Show relevant columns + predictions
cols_to_show = ['age', 'sex', 'cp', 'chol', 'target', 'Predicted_Risk_Label', 'Risk_Level', 'Predicted_Probability']
# Filter columns that actually exist
cols_to_show = [c for c in cols_to_show if c in df.columns]

print(df[cols_to_show].head(10).to_string(index=False))
print("────────────────────────────────────────────────────────────────")

# Calculate Accuracy on this full dataset (since we have labels)
if target_col:
    correct = (df[target_col] == df['Predicted_Risk_Label']).sum()
    total = len(df)
    print(f"\nOverall Accuracy on Full Dataset: {correct}/{total} ({correct/total:.2%})")
