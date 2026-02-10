#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Exploration Notebook

This script demonstrates comprehensive exploratory data analysis for the
customer segmentation K-Means clustering project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import DataPreprocessor

# Load data
df = pd.read_csv('data/raw/customers.csv')

# Basic info
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# Visualization
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Distribution plots
for i, col in enumerate(df.select_dtypes(include=[np.number]).columns[:6], 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.savefig('visualizations/data_exploration.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to visualizations/data_exploration.png")

# Preprocessing
preprocessor = DataPreprocessor(scaling_method='standard')
X_processed = preprocessor.fit_transform(df)

print(f"\nProcessed data shape: {X_processed.shape}")
print(f"Feature names: {preprocessor.feature_names}")
