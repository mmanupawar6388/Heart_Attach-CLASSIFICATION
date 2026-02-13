"""
Heart Attack Classification - Neural Network
Optimized binary classifier with visualizations, model saving, and evaluation.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for fast rendering
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import json

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Heart Attack Data Set.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {'primary': '#6C5CE7', 'secondary': '#00CEC9', 'accent': '#FD79A8', 'bg': '#2D3436'}

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print(f"{'='*60}")
print(f"  Heart Attack Risk Classification")
print(f"{'='*60}")
print(f"  Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════
# 2. AUTO-DETECT TARGET
# ══════════════════════════════════════════════════════════════════════
target_candidates = ['target', 'output', 'label', 'class', 'result']
target_col = None
for col in df.columns:
    if col.strip().lower() in target_candidates:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]
    print(f"  ⚠ Using last column as target: '{target_col}'")
else:
    print(f"  Target column: '{target_col}'")

print(f"  Class distribution: {dict(df[target_col].value_counts())}")

# ══════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
X = df.drop(columns=[target_col])
y = df[target_col].values
feature_names = list(X.columns)

# Handle missing values
missing = X.isnull().sum().sum()
if missing > 0:
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])
    print(f"  Missing values filled: {missing}")
else:
    print(f"  Missing values: None")

# Encode categoricals
cat_cols = X.select_dtypes(exclude=[np.number]).columns
if len(cat_cols) > 0:
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
    print(f"  Encoded categoricals: {list(cat_cols)}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for API use
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"{'='*60}\n")

# ══════════════════════════════════════════════════════════════════════
# 4. BUILD IMPROVED MODEL
# ══════════════════════════════════════════════════════════════════════
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ══════════════════════════════════════════════════════════════════════
# 5. TRAIN
# ══════════════════════════════════════════════════════════════════════
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("\n── Training ────────────────────────────────────────────────")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ══════════════════════════════════════════════════════════════════════
# 6. SAVE MODEL
# ══════════════════════════════════════════════════════════════════════
model_path = os.path.join(MODEL_DIR, "heart_attack_model.keras")
model.save(model_path)

# Save metadata
metadata = {
    "features": feature_names,
    "target": target_col,
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "input_shape": int(X_train.shape[1]),
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {os.path.join(MODEL_DIR, 'scaler.pkl')}")

# ══════════════════════════════════════════════════════════════════════
# 7. EVALUATE
# ══════════════════════════════════════════════════════════════════════
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Training Accuracy   : {train_acc:.4f}")
print(f"  Validation Accuracy : {val_acc:.4f}")
print(f"  Test Accuracy       : {test_acc:.4f}")
print(f"  Test Loss           : {test_loss:.4f}")
print(f"{'='*60}")

# Predictions
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n── Confusion Matrix ────────────────────────────────────────")
print(f"                Predicted 0    Predicted 1")
print(f"  Actual 0      {cm[0][0]:>8}       {cm[0][1]:>8}")
print(f"  Actual 1      {cm[1][0]:>8}       {cm[1][1]:>8}")

print(f"\n── Classification Report ───────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=['No Risk', 'Risk']))

# ══════════════════════════════════════════════════════════════════════
# 8. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════
print("Generating plots...")

# --- Plot 1: Training & Validation Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#1a1a2e')

for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#333')

epochs_range = range(1, len(history.history['accuracy']) + 1)

# Accuracy
axes[0].plot(epochs_range, history.history['accuracy'], color=COLORS['primary'],
             linewidth=2.5, label='Train Accuracy', marker='o', markersize=3)
axes[0].plot(epochs_range, history.history['val_accuracy'], color=COLORS['secondary'],
             linewidth=2.5, label='Val Accuracy', marker='s', markersize=3)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(facecolor='#16213e', edgecolor='#333', labelcolor='white', fontsize=10)

# Loss
axes[1].plot(epochs_range, history.history['loss'], color=COLORS['accent'],
             linewidth=2.5, label='Train Loss', marker='o', markersize=3)
axes[1].plot(epochs_range, history.history['val_loss'], color=COLORS['secondary'],
             linewidth=2.5, label='Val Loss', marker='s', markersize=3)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
axes[1].legend(facecolor='#16213e', edgecolor='#333', labelcolor='white', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

# --- Plot 2: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.tick_params(colors='white')

ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['primary'])
ax.plot([0, 1], [0, 1], '--', color='#666', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12, color='white')
ax.set_ylabel('True Positive Rate', fontsize=12, color='white')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold', color='white')
ax.legend(facecolor='#16213e', edgecolor='#333', labelcolor='white', fontsize=11)
for spine in ax.spines.values():
    spine.set_color('#333')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

# --- Plot 3: Confusion Matrix Heatmap ---
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Risk', 'Risk'])
disp.plot(ax=ax, cmap='RdPu', colorbar=False, values_format='d')
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='white')
ax.set_xlabel('Predicted', fontsize=12, color='white')
ax.set_ylabel('Actual', fontsize=12, color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('#333')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

# --- Plot 4: Feature Importance (Permutation-based approximation via weight magnitude) ---
first_layer_weights = np.abs(model.layers[0].get_weights()[0])  # shape: (n_features, 64)
importance = first_layer_weights.mean(axis=1)  # Average weight magnitude per feature
sorted_idx = np.argsort(importance)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.tick_params(colors='white')

bars = ax.barh(range(len(sorted_idx)), importance[sorted_idx], color=COLORS['primary'], edgecolor='none')
# Highlight top 3
for i in range(-1, -4, -1):
    bars[i].set_color(COLORS['accent'])

ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10, color='white')
ax.set_xlabel('Mean |Weight|', fontsize=12, color='white')
ax.set_title('Feature Importance (Input Layer Weights)', fontsize=14, fontweight='bold', color='white')
for spine in ax.spines.values():
    spine.set_color('#333')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print(f"\n✅ Plots saved to: {PLOTS_DIR}/")
print(f"   • training_curves.png")
print(f"   • roc_curve.png")
print(f"   • confusion_matrix.png")
print(f"   • feature_importance.png")

# Final summary
print(f"\n{'='*60}")
print(f"  AUC Score: {roc_auc:.4f}")
print(f"  Model Size: {os.path.getsize(model_path) / 1024:.1f} KB")
print(f"{'='*60}")
print(f"  ✅ All done! Run the API with: python api.py")
print(f"{'='*60}")
