
import subprocess
import time
import requests
import sys
import os

def run_demo():
    print("🚀 Starting API Server...")
    # Start the API server in a separate process
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=r"d:\Classification",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to boot
    time.sleep(5)
    
    try:
        # Define a high-risk sample patient
        # (Age 63, Male, CP 3, High BP, etc.)
        payload = {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
            "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }
        
        print("\n💉 Sending Patient Data to Prediction Endpoint...")
        print(f"   Input: {payload}")
        
        url = "http://127.0.0.1:8000/predict"
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ PREDICTION RECEIVED!")
            print("="*40)
            print(f"  Risk Evaluation : {result['risk'].upper()}")
            print(f"  Probability     : {result['probability']:.2%}")
            print(f"  Confidence      : {result['confidence']:.2%}")
            print("="*40)
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
    finally:
        print("\n🛑 Stopping Server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    run_demo()
