#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic usage example for HumanML library.
"""

import pandas as pd
from sklearn.datasets import load_iris
from humanml import HumanML

# Load sample dataset
print("Loading dataset...")
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Initialize HumanML
print("Initializing HumanML...")
model = HumanML(
    task_type="classification",  # or "regression" or "auto"
    output_dir="output",
    verbose=True,
    random_state=42
)

# Fit model
print("Fitting model...")
model.fit(X, y)

# Make predictions
print("Making predictions...")
predictions = model.predict(X)
print(f"Made {len(predictions)} predictions")

# Generate plot
print("Generating plot...")
model.plot("feature_importance")

# Get best model
print("Getting best model...")
best_model, model_name = model.get_best_model()
print(f"Best model: {model_name}")

print("Example completed successfully!")
