---
title: Heart Attack Risk Classification
emoji: ❤️
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---
# ❤️ Heart Attack Risk Classification

A lightweight neural network for binary classification of heart attack risk, built with **TensorFlow/Keras** and deployable via **FastAPI**.

---

## 📁 Project Structure

```
Classification/
├── Heart Attack Data Set.csv   # Source dataset (303 samples, 14 features)
├── heart_attack_classifier.py  # Training script (preprocessing → training → evaluation → plots)
├── api.py                      # FastAPI deployment server
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── saved_model/                # Generated after training
│   ├── heart_attack_model.keras
│   ├── scaler.pkl
│   └── metadata.json
└── plots/                      # Generated after training
    ├── training_curves.png
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 🧠 Model Architecture

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Dense (ReLU) | 64 | 896 |
| Dropout (0.3) | 64 | 0 |
| Dense (ReLU) | 32 | 2,080 |
| Dropout (0.2) | 32 | 0 |
| Dense (ReLU) | 16 | 528 |
| Dense (ReLU) | 8 | 136 |
| Dense (Sigmoid) | 1 | 9 |

**Total Parameters:** ~3,649

---

## 📊 Dataset

- **Source:** Heart Attack Data Set (UCI / Kaggle)
- **Samples:** 303
- **Features:** 13 medical indicators
- **Target:** Binary (0 = No Risk, 1 = Risk)

### Features:
| Feature | Description |
|---------|------------|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0-3) |
| `thal` | Thalassemia (0-3) |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python heart_attack_classifier.py
```

This will:
- Preprocess the dataset (scaling, encoding, splitting)
- Train the neural network (up to 100 epochs with early stopping)
- Save the model to `saved_model/`
- Generate evaluation plots in `plots/`
- Print accuracy, confusion matrix, and classification report

### 3. Run the API
```bash
python api.py
```

The API starts at `http://localhost:8000`. Interactive docs available at `http://localhost:8000/docs`.

### 4. Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

**Response:**
```json
{
  "risk": "High Risk",
  "probability": 0.8723,
  "confidence": 0.8723,
  "label": 1
}
```

---

## 📈 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/features` | List expected features |
| `POST` | `/predict` | Predict heart attack risk |
| `GET` | `/docs` | Interactive Swagger docs |

---

## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 (with early stopping) |
| Batch Size | 32 |
| Validation Split | 20% |
| Test Split | 20% |
| Early Stopping Patience | 10 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

---

## 📊 Visualizations

After training, four plots are generated in the `plots/` directory:

1. **Training Curves** — Accuracy and loss over epochs
2. **ROC Curve** — Receiver Operating Characteristic with AUC score
3. **Confusion Matrix** — True vs predicted labels heatmap
4. **Feature Importance** — Input layer weight magnitudes per feature

---

## 📝 License

This project is for educational and research purposes.
