"""
Credit Card Fraud Detection System - Complete Pipeline
Run this single script to: generate data -> preprocess -> train -> evaluate -> save.
After this completes, start the web app with: python api/app.py
"""

import os
import sys
import time

# Ensure we run from project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from config import DATASET_PATH, MODELS_DIR, RESULTS_DIR, N_SAMPLES, FRAUD_RATIO


def main():
    start = time.time()

    print("=" * 70)
    print("  CREDIT CARD FRAUD DETECTION SYSTEM - FULL PIPELINE")
    print("=" * 70)

    # ── Step 1: Generate Dataset ──────────────────────────────────────
    print("\n[STEP 1/4] Generating synthetic dataset ...")
    from dataset.generate_dataset import generate_dataset

    df = generate_dataset(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO)
    df.to_csv(DATASET_PATH, index=False)
    print(f"  Saved {len(df)} transactions ({df['Fraud'].sum()} fraud) to {DATASET_PATH}")

    # ── Step 2: Preprocess ────────────────────────────────────────────
    print("\n[STEP 2/4] Preprocessing data ...")
    from preprocessing.data_cleaning import DataPreprocessor

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess(DATASET_PATH)
    preprocessor.save_artifacts(MODELS_DIR)

    # ── Step 3: Train Models ──────────────────────────────────────────
    print("\n[STEP 3/4] Training models ...")
    from models.train_model import ModelTrainer

    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)

    # Optimize threshold on test set for the best model (XGBoost)
    print("\n--- Threshold Optimization (XGBoost) ---")
    trainer.optimize_threshold(trainer.trained_models["XGBoost"], X_test, y_test)

    trainer.save_models(MODELS_DIR)

    # ── Step 4: Evaluate Models ───────────────────────────────────────
    print("\n[STEP 4/4] Evaluating models ...")
    from models.evaluate_model import ModelEvaluator

    evaluator = ModelEvaluator()
    evaluator.load_models(MODELS_DIR)
    evaluator.evaluate_all(X_test, y_test)
    evaluator.generate_all(y_test, RESULTS_DIR)

    # ── Summary ───────────────────────────────────────────────────────
    best_name, best_res = evaluator.best_model()
    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Best model : {best_name}")
    print(f"  F1-Score   : {best_res['F1-Score']:.4f}")
    print(f"  ROC-AUC    : {best_res['ROC-AUC']:.4f}")
    print(f"  Precision  : {best_res['Precision']:.4f}")
    print(f"  Recall     : {best_res['Recall']:.4f}")
    print(f"  Time       : {elapsed:.1f}s")
    print(f"\n  Models saved in : {MODELS_DIR}/")
    print(f"  Results saved in: {RESULTS_DIR}/")
    print(f"\n  To start the web app:")
    print(f"    python api/app.py")
    print(f"    Then open http://localhost:5000")
    print("=" * 70)


if __name__ == "__main__":
    main()
