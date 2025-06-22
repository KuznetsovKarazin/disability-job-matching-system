#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 01_generate_dataset.py
Description: Generates the extended candidate and company datasets,
and creates the final training dataset with compatibility scores and distances.
Output:
  - data/processed/Dataset_Candidati_Aggiornato_Extended.csv
  - data/processed/Dataset_Aziende_con_Stima_Assunzioni_Extended.csv
  - data/processed/Enhanced_Training_Dataset.csv
"""

import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from utils.feature_engineering import extend_candidates_dataset, extend_companies_dataset
from utils.scoring import EnhancedScoringSystem

# Define paths
RAW_CANDIDATI_PATH = "data/raw/Dataset_Candidati_Aggiornato.csv"
RAW_AZIENDE_PATH = "data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv"
CANDIDATI_EXT_PATH = "data/processed/Dataset_Candidati_Aggiornato_Extended.csv"
AZIENDE_EXT_PATH = "data/processed/Dataset_Aziende_con_Stima_Assunzioni_Extended.csv"
TRAINING_DATA_PATH = "data/processed/Enhanced_Training_Dataset.csv"

# Step 1: Load raw data
print("📥 Loading raw datasets...")
df_cand = pd.read_csv(RAW_CANDIDATI_PATH)
df_az = pd.read_csv(RAW_AZIENDE_PATH)

# Step 2: Extend datasets
print("🔧 Extending candidate and company datasets...")
df_cand_ext = extend_candidates_dataset(df_cand)
df_az_ext = extend_companies_dataset(df_az)

# Save extended versions
df_cand_ext.to_csv(CANDIDATI_EXT_PATH, index=False)
df_az_ext.to_csv(AZIENDE_EXT_PATH, index=False)
print(f"✅ Extended datasets saved to {CANDIDATI_EXT_PATH} and {AZIENDE_EXT_PATH}")

# Step 3: Generate training dataset
print("🧠 Generating training dataset with features and labels...")
scoring_system = EnhancedScoringSystem()
df_train = scoring_system.generate_enhanced_training_data(df_cand_ext, df_az_ext)

# Save training dataset
df_train.to_csv(TRAINING_DATA_PATH, index=False)
print(f"✅ Training dataset saved to {TRAINING_DATA_PATH}")
