# utils/feature_engineering.py
# -*- coding: utf-8 -*-
"""
Feature engineering utilities for:
- Extending candidate dataset
- Extending company dataset
"""

import pandas as pd
import numpy as np
import os

def extend_candidates_dataset(df):
    """Extend candidate dataset with new features"""
    n = len(df)

    # Simulate disability type
    disabilities = [
        'Motoria', 'Intellettiva', 'Sensoriale', 'Psichica',
        "Disturbi Specifici dell'Apprendimento",
        'Disturbi del linguaggio e della comunicazione',
        'Altro non specificato'
    ]
    p_disabilities = [0.20, 0.20, 0.15, 0.15, 0.10, 0.05, 0.15]
    df['Tipo di Disabilità'] = np.random.choice(disabilities, size=n, p=p_disabilities)

    # Simulate first employment year
    df['Data Inizio Prima Occupazione'] = np.random.randint(2000, 2024, size=n)

    # Compute unemployment duration
    possible_years = df['Data Inizio Prima Occupazione'].apply(lambda x: 2025 - x)
    df['Durata Disoccupazione'] = possible_years.apply(lambda y: np.random.randint(0, y + 1))

    # Compute work experience
    exp = possible_years - df['Durata Disoccupazione']
    df['Years_of_Experience'] = exp.clip(lower=0).astype(int)

    # Assign education level
    all_edu = ['Nessuno', 'Licenza Media', 'Diploma', 'Laurea', 'Master']
    p_all = np.array([0.05, 0.25, 0.40, 0.20, 0.10])

    edu_intel = ['Nessuno', 'Licenza Media', 'Diploma']
    p_intel = np.array([0.05, 0.25, 0.40]) / np.sum([0.05, 0.25, 0.40])

    def sample_education(dis):
        if dis == 'Intellettiva':
            return np.random.choice(edu_intel, p=p_intel)
        else:
            return np.random.choice(all_edu, p=p_all)

    df['Titolo di Studio'] = df['Tipo di Disabilità'].apply(sample_education)

    return df

def extend_companies_dataset(df):
    """Extend company dataset with new features"""
    def map_size(n):
        if n < 50:
            return 'small'
        elif n < 250:
            return 'medium'
        else:
            return 'large'

    df['Company_Size'] = df['Numero Dipendenti'].apply(map_size)

    # Simulate binary features
    np.random.seed(42)
    df['Certification'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    df['Remote'] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])

    # Retention rate
    df['Retention_Rate'] = (
        df['Assunti Categoria Protetta'] /
        df['Quota Dipendenti Categoria Protetta'].replace(0, np.nan)
    ).fillna(0)

    return df
