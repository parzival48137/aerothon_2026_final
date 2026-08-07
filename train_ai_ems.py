"""
train_ai_ems.py — Aerothon 2026 | Trains the APEX-AI phase-awareness model

Trains a compact MLP classifier to predict loiter CHARGING/ELECTRIC state
from telemetry, using the physically-verified deterministic APEX as the
imitation-learning target. Saves the fitted model + feature scaler to disk
for the dashboard/controller to load at inference time.
"""
import os, sys, time, joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "loiter_training_data.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["soc", "power_demand_frac", "fuel_frac", "prev_state_charging"]
LABEL = "label_charging"

print("Loading training data...")
df = pd.read_csv(DATA)
print(f"  {len(df):,} samples, {df[LABEL].mean()*100:.1f}% CHARGING class")

X = df[FEATURES].values
y = df[LABEL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

print("\nTraining MLP classifier (compact: 16-8 hidden units)...")
t0 = time.time()
clf = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=200,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42,
)
clf.fit(X_train_s, y_train)
print(f"  Trained in {time.time()-t0:.1f}s, {clf.n_iter_} iterations")

y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"\n  VALIDATION (held-out 20% of episodes' timesteps):")
print(f"  Accuracy vs deterministic APEX: {acc*100:.2f}%")
print(f"  F1 score:                        {f1:.4f}")
print(f"  Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")

joblib.dump(clf, os.path.join(MODEL_DIR, "apex_ai_classifier.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "apex_ai_scaler.joblib"))
with open(os.path.join(MODEL_DIR, "apex_ai_features.txt"), "w") as f:
    f.write("\n".join(FEATURES))

print(f"\nSaved model to {MODEL_DIR}/apex_ai_classifier.joblib")
print(f"Architecture: {clf.hidden_layer_sizes} hidden units, "
      f"{sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_):,} "
      f"trainable parameters")
