#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 03_train_models.py
Description: Trains multiple ML models in parallel with hyperparameter optimization.
Outputs:
 - Trained model files (.joblib)
 - Training metrics summary
 - Model complexity metrics
 - Learning curves
"""

import sys
import os
import time
import joblib
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parallel_training import ParallelModelTrainer, prepare_data_for_training

TRAINING_FILE = 'data/processed/Enhanced_Training_Dataset.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(TRAINING_FILE):
    print(f"❌ Training file not found: {TRAINING_FILE}")
    exit(1)

print("📥 Loading training dataset...")
df_train = pd.read_csv(TRAINING_FILE)

# Step 1: Preprocess and split
print("🧹 Preparing data...")
data = prepare_data_for_training(df_train)

# Step 2: Initialize trainer
trainer = ParallelModelTrainer(random_state=42)

# Step 3: Optimize hyperparameters
best_params = trainer.parallel_hyperparameter_optimization(data['X_train'], data['y_train'])

# Step 4: Create models
model_configs = trainer.create_optimized_models(best_params)

# Step 5: Train models
results = trainer.parallel_model_training(
    model_configs,
    data['X_train'], data['y_train'],
    data['X_test'], data['y_test']
)

# Step 6: Create ensemble
ensemble = trainer.create_ensemble_model(results, data['X_train'], data['y_train'])

# Step 7: Save models
trainer.save_models(results, ensemble)
print("✅ All models saved")

# Step 8: Analyze model complexity
print("📏 Measuring model complexity...")
complexity_rows = []
for name, info in results.items():
    if info['status'] != 'success':
        continue
    model_path = os.path.join(RESULTS_DIR, f"{name}.joblib")
    model_size = os.path.getsize(model_path) / 1024  # KB

    start_pred = time.time()
    _ = info['model'].predict(data['X_test'])
    pred_time = time.time() - start_pred

    complexity_rows.append({
        'Model': name,
        'TrainingTime_sec': round(info.get('training_time', 0), 2),
        'PredictionTime_sec': round(pred_time, 4),
        'ModelSize_KB': round(model_size, 1)
    })

df_complexity = pd.DataFrame(complexity_rows)
df_complexity.to_csv(os.path.join(RESULTS_DIR, 'model_complexity.csv'), index=False)
print("✅ Saved model complexity to model_complexity.csv")

# Step 9: Learning curves (only for a few models to limit runtime)
print("📉 Generating learning curves...")
curve_results = {}
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

selected_models = {
    'RandomForest_LC': RandomForestClassifier(**best_params.get('random_forest', {}), class_weight='balanced', n_jobs=2),
    'MLP_LC': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    'XGBoost_LC': xgb.XGBClassifier(**best_params.get('xgboost', {}), use_label_encoder=False, eval_metric='logloss', n_jobs=2)
}

for name, model in selected_models.items():
    print(f"⏳ Computing learning curve for {name}...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, data['X_train'], data['y_train'],
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=3, scoring='f1', n_jobs=2
    )
    curve_results[name] = {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'test_scores': test_scores
    }

with open(os.path.join(RESULTS_DIR, 'learning_curves.pkl'), 'wb') as f:
    pickle.dump(curve_results, f)

print("✅ Learning curves saved to learning_curves.pkl")
