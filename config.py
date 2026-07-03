"""
Configuration for Credit Card Fraud Detection System
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "credit_card_transactions.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Dataset generation
N_SAMPLES = 20000
FRAUD_RATIO = 0.08  # 8% fraud — realistic for synthetic data
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.2

# SMOTE
SMOTE_K_NEIGHBORS = 5

# Models to train
MODELS_TO_TRAIN = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
BEST_MODEL_NAME = "xgboost"  # Default best model for API

# Fraud detection threshold (optimized during training via PR curve)
FRAUD_THRESHOLD = 0.5

# Flask API
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
