#!/usr/bin/env python3
"""
Train identification classifier on precomputed embeddings.

- Loads embeddings .npz (embeddings, labels)
- Encodes labels (LabelEncoder)
- Trains an SVM (with probability) and a simple LogisticRegression baseline
- Evaluates using stratified train/test split and prints metrics
- Saves classifier and label encoder to models/
"""
import argparse
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from utils import load_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, required=True, help='Path to embeddings npz')
    parser.add_argument('--out_dir', type=str, default='models')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X, y, paths = load_embeddings(args.embeddings)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # split on identities ensuring samples per identity are split correctly:
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=args.test_size, random_state=args.random_state)
    print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # SVM with RBF kernel (probability=True for API confidence)
    # --- SMART TRAINING (GRID SEARCH) ---
    print("[INFO] Training SVM (Full Mode - Finding Best Settings)...")
    # Define the grid of settings to test
    param_grid = {'C': [1, 5, 10, 50], 'gamma': ['scale', 0.001, 0.0001]}
    
    # Initialize the classifier
    svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=args.random_state)
    
    # Run the Grid Search (This is the heavy lifting)
    gs = GridSearchCV(svc, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)
    
    # Get the winner
    best_svc = gs.best_estimator_
    print(f"[INFO] Best Params Found: {gs.best_params_}")

    # Evaluate
    preds = best_svc.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"--- SVM Test Accuracy: {acc:.4f}")

    # Save
    joblib.dump(best_svc, os.path.join(args.out_dir, 'classifier.joblib'))
    joblib.dump(le, os.path.join(args.out_dir, 'label_encoder.joblib'))
    print(f"[INFO] Saved classifier and label encoder to {args.out_dir}")

if __name__ == "__main__":
    main()