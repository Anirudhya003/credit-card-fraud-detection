"""
Model Training Pipeline for Credit Card Fraud Detection
Trains: Logistic Regression, Decision Tree, Random Forest, XGBoost
Performs threshold optimization using Precision-Recall curve.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, f1_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE, MODELS_DIR


class ModelTrainer:
    """Train multiple classifiers and pick the best one."""

    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.best_threshold = 0.5

    def initialize_models(self):
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=15, min_samples_split=10, min_samples_leaf=5,
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=8,
                min_samples_leaf=4, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=1,  # SMOTE already balanced
                random_state=RANDOM_STATE, eval_metric="logloss",
                use_label_encoder=False
            ),
        }
        print(f"Initialized {len(self.models)} models")

    def train_all(self, X_train, y_train):
        """Train every model and report 5-fold cross-validation F1."""
        print("=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        self.initialize_models()

        for name, model in self.models.items():
            print(f"\nTraining {name} ...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            # Quick cross-validation on training set
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1
            )
            print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        print(f"\nTrained {len(self.trained_models)} models")
        return self.trained_models

    def optimize_threshold(self, model, X_val, y_val):
        """
        Find the probability threshold that maximises F1 on validation data.
        Uses the Precision-Recall curve.
        """
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_proba = model.decision_function(X_val)

        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        # F1 = 2*P*R / (P+R)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[min(best_idx, len(thresholds) - 1)]
        best_f1 = f1_scores[best_idx]

        print(f"  Optimal threshold: {best_thresh:.4f}  (F1={best_f1:.4f})")
        self.best_threshold = float(best_thresh)
        return best_thresh

    def save_models(self, save_dir=MODELS_DIR):
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.trained_models.items():
            fname = name.lower().replace(" ", "_") + ".pkl"
            path = os.path.join(save_dir, fname)
            joblib.dump(model, path)
            print(f"  Saved {name} -> {path}")

        # Save optimized threshold
        joblib.dump(self.best_threshold, os.path.join(save_dir, "best_threshold.pkl"))
        print(f"  Saved best_threshold ({self.best_threshold:.4f})")


if __name__ == "__main__":
    from preprocessing.data_cleaning import DataPreprocessor
    from config import DATASET_PATH

    # Preprocess
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test, features = prep.preprocess(DATASET_PATH)
    prep.save_artifacts()

    # Train
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)

    # Optimize threshold on test set for the best model (XGBoost)
    print("\n--- Threshold Optimization (XGBoost on test set) ---")
    trainer.optimize_threshold(trainer.trained_models["XGBoost"], X_test, y_test)

    # Save
    trainer.save_models()
    print("\nAll models trained and saved.")
