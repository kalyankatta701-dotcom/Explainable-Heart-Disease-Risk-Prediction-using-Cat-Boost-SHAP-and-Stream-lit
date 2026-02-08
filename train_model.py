"""
Heart Disease Prediction Model Training Script
This script trains a CatBoost classifier on heart disease data and saves the model.
"""

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def train_model():
    """Train and save the CatBoost model for heart disease prediction."""
    
    # Load the data
    print("Loading heart disease dataset...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'heart_disease.csv')
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize and train CatBoost classifier
    print("\nTraining CatBoost model...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=100
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'catboost_heart_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names for later use
    feature_names_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"Feature names saved to: {feature_names_path}")
    
    return model

if __name__ == "__main__":
    train_model()
