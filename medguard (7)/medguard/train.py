"""
train.py — Train all 7 ML models on ECU-IoHT dataset
Run this ONCE to train and save all models:
    python train.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

print("=" * 55)
print("  MedGuard — ML Training Pipeline")
print("  Dataset: ECU-IoHT")
print("=" * 55)

# ── 1. LOAD DATASET ────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_excel('data/ECU_IoHT.xlsx', sheet_name='ECU-IoHT')
print(f"      Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. FEATURE ENGINEERING ─────────────────────────────
print("\n[2/6] Engineering features...")

# Encode Protocol → numeric
le_proto = LabelEncoder()
df['Protocol_enc'] = le_proto.fit_transform(df['Protocol'].astype(str))

# Encode Source and Destination → numeric
le_src = LabelEncoder()
le_dst = LabelEncoder()
df['Source_enc']      = le_src.fit_transform(df['Source'].astype(str))
df['Destination_enc'] = le_dst.fit_transform(df['Destination'].astype(str))

# Extract packet size from Length
df['pkt_size'] = df['Length'].fillna(0).astype(float)

# Time delta (proxy for packet rate)
df['time_delta'] = df['Time'].diff().fillna(0).abs()

# Info length as a proxy feature
df['info_len'] = df['Info'].astype(str).apply(len)

# Final feature columns
FEATURES = [
    'Protocol_enc',
    'Source_enc',
    'Destination_enc',
    'pkt_size',
    'time_delta',
    'info_len',
]

# Target: Binary — 0 = Normal, 1 = Attack
df['label'] = (df['Type'] == 'Attack').astype(int)

X = df[FEATURES].values
y = df['label'].values

print(f"      Features: {FEATURES}")
print(f"      Normal: {(y==0).sum():,}  |  Attack: {(y==1).sum():,}")

# ── 3. SCALE ───────────────────────────────────────────
print("\n[3/6] Scaling features (MinMaxScaler)...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. SPLIT ───────────────────────────────────────────
print("\n[4/6] Splitting — 80% train / 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. TRAIN ALL 7 MODELS ──────────────────────────────
print("\n[5/6] Training 7 ML models...\n")

models = {
    "XGBoost":       XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "CatBoost":      CatBoostClassifier(iterations=100, random_state=42, verbose=0),
    "MLP":           MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
    "SVM":           SVC(kernel='rbf', probability=True, random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Logistic Reg.": LogisticRegression(max_iter=500, random_state=42),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"  Training {name}...", end=' ', flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
    rec  = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
    f1   = round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    trained_models[name] = model
    print(f"Accuracy: {acc}%  F1: {f1}%")

# ── 6. SAVE EVERYTHING ────────────────────────────────
print("\n[6/6] Saving models and preprocessors...")
os.makedirs('models', exist_ok=True)

# Save each trained model
for name, model in trained_models.items():
    fname = name.lower().replace(' ', '_').replace('.', '') + '.pkl'
    with open(f'models/{fname}', 'wb') as f:
        pickle.dump(model, f)

# Save scaler and encoders
with open('models/scaler.pkl',   'wb') as f: pickle.dump(scaler,   f)
with open('models/le_proto.pkl', 'wb') as f: pickle.dump(le_proto, f)
with open('models/le_src.pkl',   'wb') as f: pickle.dump(le_src,   f)
with open('models/le_dst.pkl',   'wb') as f: pickle.dump(le_dst,   f)

# Save results summary
with open('models/results.pkl', 'wb') as f: pickle.dump(results, f)

# Save feature names
with open('models/features.pkl', 'wb') as f: pickle.dump(FEATURES, f)

# ── SUMMARY ───────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING COMPLETE — Results Summary")
print("=" * 55)
print(f"  {'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
print("  " + "-" * 53)
for name, r in results.items():
    print(f"  {name:<18} {r['accuracy']:>9}% {r['precision']:>9}% {r['recall']:>9}% {r['f1']:>7}%")
print("=" * 55)
print("\n  Models saved to /models/")
print("  You can now run: python app.py\n")