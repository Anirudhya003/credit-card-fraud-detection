"""
Data Preprocessing Pipeline for Credit Card Fraud Detection
Handles: missing values, encoding, feature engineering, scaling, SMOTE.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SIZE, RANDOM_STATE, SMOTE_K_NEIGHBORS, MODELS_DIR


class DataPreprocessor:
    """End-to-end preprocessing: load -> clean -> engineer -> split -> scale -> balance."""

    def __init__(self, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        # Mappings used for encoding (stored so API can reuse them)
        self.categorical_mappings = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_data(self, filepath):
        print(f"Loading data from {filepath} ...")
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    def handle_missing_values(self, df):
        print("\n--- Handling Missing Values ---")
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ("float64", "int64"):
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  Filled {total_missing} missing values")
        else:
            print("  No missing values found")
        return df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_categorical(self, df):
        print("\n--- Encoding Categorical Features ---")
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        # Drop ID column
        if "Transaction_ID" in cat_cols:
            cat_cols.remove("Transaction_ID")
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            self.categorical_mappings[col] = dict(
                zip(le.classes_, le.transform(le.classes_))
            )
            print(f"  Encoded {col}: {self.categorical_mappings[col]}")
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df):
        """Create derived features that help the model detect fraud."""
        print("\n--- Feature Engineering ---")

        # 1. Log of transaction amount (reduces skew from extreme values)
        df["Log_Amount"] = np.log1p(df["Transaction_Amount"])

        # 2. Is the transaction at night? (midnight to 5 AM)
        df["Is_Night"] = (df["Transaction_Hour"].between(0, 5)).astype(int)

        # 3. Is the card new? (less than 30 days)
        df["Is_New_Card"] = (df["Days_Since_Card_Issued"] < 30).astype(int)

        # 4. Is the account dormant? (no transaction in 30+ days)
        df["Is_Dormant"] = (df["Previous_Transaction_Days"] >= 30).astype(int)

        # 5. Amount relative to typical — high deviation signals fraud
        #    (z-score within this dataset)
        mean_amt = df["Transaction_Amount"].mean()
        std_amt = df["Transaction_Amount"].std()
        df["Amount_Zscore"] = (df["Transaction_Amount"] - mean_amt) / std_amt

        # 6. Interaction: night transaction on a new card
        df["Night_NewCard"] = df["Is_Night"] * df["Is_New_Card"]

        # 7. Absolute latitude (distance from equator — foreign location proxy)
        df["Abs_Lat"] = df["Location_Lat"].abs()

        print("  Created 7 engineered features")
        return df

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def preprocess(self, filepath):
        """
        Run the complete pipeline.
        Returns: X_train, X_test, y_train, y_test, feature_names
        """
        print("=" * 60)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        df = self.load_data(filepath)
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)
        df = self.engineer_features(df)

        # Drop ID column
        if "Transaction_ID" in df.columns:
            df.drop(columns=["Transaction_ID"], inplace=True)

        # Separate features / target
        X = df.drop("Fraud", axis=1)
        y = df["Fraud"]
        self.feature_names = X.columns.tolist()

        # Train-test split (stratified)
        print(f"\n--- Train/Test Split ({1-self.test_size:.0%} / {self.test_size:.0%}) ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

        # Feature scaling (fit on train only)
        print("\n--- Feature Scaling (StandardScaler) ---")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=self.feature_names
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=self.feature_names
        )
        print("  Scaling complete")

        # SMOTE on training data only
        print("\n--- SMOTE (Balancing Training Set) ---")
        print(f"  Before: Fraud={int(y_train.sum())}, Legit={(y_train==0).sum()}")
        smote = SMOTE(random_state=self.random_state, k_neighbors=SMOTE_K_NEIGHBORS)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
        X_train_bal = pd.DataFrame(X_train_bal, columns=self.feature_names)
        y_train_bal = pd.Series(y_train_bal)
        print(f"  After : Fraud={int(y_train_bal.sum())}, Legit={(y_train_bal==0).sum()}")

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        return X_train_bal, X_test_scaled, y_train_bal, y_test, self.feature_names

    # ------------------------------------------------------------------
    # Save scaler + encoders for the API
    # ------------------------------------------------------------------
    def save_artifacts(self, save_dir=MODELS_DIR):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(save_dir, "label_encoders.pkl"))
        joblib.dump(self.categorical_mappings, os.path.join(save_dir, "categorical_mappings.pkl"))
        joblib.dump(self.feature_names, os.path.join(save_dir, "feature_names.pkl"))
        print(f"  Saved scaler, encoders, feature_names to {save_dir}")


if __name__ == "__main__":
    from config import DATASET_PATH

    preprocessor = DataPreprocessor()
    X_tr, X_te, y_tr, y_te, feats = preprocessor.preprocess(DATASET_PATH)
    preprocessor.save_artifacts()
    print(f"\nFeatures ({len(feats)}): {feats}")
    print(f"X_train shape: {X_tr.shape}")
    print(f"X_test  shape: {X_te.shape}")
