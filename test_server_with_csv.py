"""
Test Server with CSV Data
Sends rows from 'Heart Attack Data Set.csv' to the running API (localhost:8000).
"""

import requests
import pandas as pd
import time
import json

# Load dataset
df = pd.read_csv("Heart Attack Data Set.csv")
print(f"📂 Loaded {len(df)} rows from dataset.")

# API Endpoint
url = "http://localhost:8000/predict"

print(f"🚀 Sending first 5 patients to Server at {url}...\n")

success_count = 0
for i, row in df.head(5).iterrows():
    # Convert row to dictionary matches API schema
    payload = {
        "age": float(row['age']),
        "sex": float(row['sex']),
        "cp": float(row['cp']),
        "trestbps": float(row['trestbps']),
        "chol": float(row['chol']),
        "fbs": float(row['fbs']),
        "restecg": float(row['restecg']),
        "thalach": float(row['thalach']),
        "exang": float(row['exang']),
        "oldpeak": float(row['oldpeak']),
        "slope": float(row['slope']),
        "ca": float(row['ca']),
        "thal": float(row['thal'])
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        latency = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Patient {i+1}: Server says -> {result['risk']} ({result['probability']:.1%})  [Latency: {latency:.0f}ms]")
            success_count += 1
        else:
            print(f"❌ Patient {i+1}: Failed. Status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server. Is 'python api.py' running?")
        break

print(f"\n✨ Test Complete. {success_count}/5 requests successful.")
