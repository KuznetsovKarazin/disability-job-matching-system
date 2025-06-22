#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 02_visualize_dataset.py
Description: Performs visual analysis of the enhanced training dataset.
Output: PNG plots and statistical summaries.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
from utils.visualization import visualize_distribution, visualize_correlations

TRAINING_DATA_PATH = "data/processed/Enhanced_Training_Dataset.csv"

if not os.path.exists(TRAINING_DATA_PATH):
    print(f"❌ Training dataset not found: {TRAINING_DATA_PATH}")
    exit(1)

print("📊 Loading training data...")
df = pd.read_csv(TRAINING_DATA_PATH)
print(f"✅ Loaded {df.shape[0]} records with {df.shape[1]} features")

print("📈 Generating visualizations...")
visualize_distribution(df)
visualize_correlations(df)
print("✅ Visualizations saved to output files")