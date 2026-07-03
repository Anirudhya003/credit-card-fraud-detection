"""
Model Evaluation Module for Credit Card Fraud Detection
Produces: metrics table, confusion matrices, ROC curves, PR curves, bar chart.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, RESULTS_DIR


class ModelEvaluator:
    """Evaluate all trained models and generate plots/reports."""

    def __init__(self):
        self.results = {}
        self.models = {}

    def load_models(self, model_dir=MODELS_DIR):
        print("Loading trained models ...")
        for fname in os.listdir(model_dir):
            if fname.endswith(".pkl") and fname not in (
                "scaler.pkl", "label_encoders.pkl",
                "categorical_mappings.pkl", "feature_names.pkl",
                "best_threshold.pkl",
            ):
                name = fname.replace(".pkl", "").replace("_", " ").title()
                self.models[name] = joblib.load(os.path.join(model_dir, fname))
                print(f"  Loaded {name}")
        return self.models

    def evaluate_all(self, X_test, y_test):
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                scores = model.decision_function(X_test)
                y_proba = 1 / (1 + np.exp(-scores))

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, y_proba)
            cm   = confusion_matrix(y_test, y_pred)

            self.results[name] = {
                "Accuracy": acc, "Precision": prec, "Recall": rec,
                "F1-Score": f1, "ROC-AUC": auc,
                "y_pred": y_pred, "y_proba": y_proba, "cm": cm,
            }

            print(f"\n  {name}")
            print(f"    Accuracy : {acc:.4f}")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall   : {rec:.4f}")
            print(f"    F1-Score : {f1:.4f}")
            print(f"    ROC-AUC  : {auc:.4f}")

        return self.results

    def best_model(self):
        return max(self.results.items(), key=lambda x: x[1]["F1-Score"])

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_confusion_matrices(self, save_dir=RESULTS_DIR):
        os.makedirs(save_dir, exist_ok=True)
        n = len(self.results)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for idx, (name, res) in enumerate(self.results.items()):
            sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                        ax=axes[idx], cbar=False,
                        xticklabels=["Legit", "Fraud"],
                        yticklabels=["Legit", "Fraud"])
            axes[idx].set_title(f"{name}\nF1={res['F1-Score']:.4f}", fontsize=10)
            axes[idx].set_ylabel("Actual")
            axes[idx].set_xlabel("Predicted")

        for i in range(len(self.results), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        path = os.path.join(save_dir, "confusion_matrices.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    def plot_roc_curves(self, y_test, save_dir=RESULTS_DIR):
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for name, res in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            plt.plot(fpr, tpr, label=f"{name} (AUC={res['ROC-AUC']:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "roc_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    def plot_precision_recall(self, y_test, save_dir=RESULTS_DIR):
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for name, res in self.results.items():
            prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
            plt.plot(rec, prec, label=name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="best", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "precision_recall_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    def plot_metrics_comparison(self, save_dir=RESULTS_DIR):
        os.makedirs(save_dir, exist_ok=True)
        models = list(self.results.keys())
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        data = {m: [self.results[mod][m] for mod in models] for m in metrics}
        df = pd.DataFrame(data, index=models)

        ax = df.plot(kind="bar", figsize=(10, 5))
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "metrics_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    def save_reports(self, y_test, save_dir=RESULTS_DIR):
        os.makedirs(save_dir, exist_ok=True)

        # Classification reports
        lines = ["CLASSIFICATION REPORTS", "=" * 70, ""]
        for name, res in self.results.items():
            lines.append(name)
            lines.append("-" * 70)
            lines.append(classification_report(
                y_test, res["y_pred"], target_names=["Legitimate", "Fraud"]
            ))
            lines.append("")
        path = os.path.join(save_dir, "classification_reports.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Saved {path}")

        # CSV summary
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy": round(res["Accuracy"], 4),
                "Precision": round(res["Precision"], 4),
                "Recall": round(res["Recall"], 4),
                "F1-Score": round(res["F1-Score"], 4),
                "ROC-AUC": round(res["ROC-AUC"], 4),
            })
        df = pd.DataFrame(rows)
        path = os.path.join(save_dir, "evaluation_results.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path}")

    def generate_all(self, y_test, save_dir=RESULTS_DIR):
        """Generate all plots and reports."""
        print("\n--- Generating Plots & Reports ---")
        self.plot_confusion_matrices(save_dir)
        self.plot_roc_curves(y_test, save_dir)
        self.plot_precision_recall(y_test, save_dir)
        self.plot_metrics_comparison(save_dir)
        self.save_reports(y_test, save_dir)

        best_name, best_res = self.best_model()
        print(f"\nBEST MODEL: {best_name}  (F1={best_res['F1-Score']:.4f}, AUC={best_res['ROC-AUC']:.4f})")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing.data_cleaning import DataPreprocessor
    from config import DATASET_PATH

    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test, features = prep.preprocess(DATASET_PATH)

    evaluator = ModelEvaluator()
    evaluator.load_models()
    evaluator.evaluate_all(X_test, y_test)
    evaluator.generate_all(y_test)
