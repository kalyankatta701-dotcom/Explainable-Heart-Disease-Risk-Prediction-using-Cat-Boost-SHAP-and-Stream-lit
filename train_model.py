# -------------------------------
# train_model.py (USING CATBOOST)
# -------------------------------

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier # Import CatBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

print("🚀 Starting Heart Disease Model Training (CatBoost)...")

# --- Fetch UCI Heart Disease Dataset ---
heart_data = fetch_ucirepo(id=45)
data = pd.concat([heart_data.data.features, heart_data.data.targets], axis=1)

# Clean up data: drop rows with missing values
data = data.dropna()

print("Dataset loaded and cleaned successfully!")

# --- Define features and target ---
X = data.drop(columns='num')
y = data['num']

# --- CRITICAL: Convert to Binary Classification (0=Low Risk, 1=High Risk) ---
y = (y > 0).astype(int)
print(f"Target variable converted to binary (0 or 1). Value counts:\n{y.value_counts()}")

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Train CatBoost model ---
model = CatBoostClassifier(
    iterations=150,
    learning_rate=0.1,
    depth=5,
    random_seed=42,
    loss_function='Logloss', # Appropriate for binary classification
    verbose=0, # Suppress training output
    allow_writing_files=False 
)

print("🧠 Training CatBoost model...")
model.fit(X_train, y_train)

# --- Evaluate model ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {acc:.2f}")

# --- Save trained model ---
model.save_model("heart_disease_catboost.cbm")
print("💾 Model saved as 'heart_disease_catboost.cbm'")

print("🎉 Training complete!")