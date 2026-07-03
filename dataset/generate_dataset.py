"""
Synthetic Credit Card Transaction Dataset Generator (India / INR)
Generates realistic Indian credit card transactions with diverse fraud patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_SAMPLES, FRAUD_RATIO, RANDOM_STATE, DATASET_PATH


def generate_dataset(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO, seed=RANDOM_STATE):
    """
    Generate a synthetic credit card transaction dataset (Indian context, INR).

    Fraud patterns modeled:
      1. High-amount transactions (card-not-present)
      2. Late-night activity on new/dormant cards
      3. Geographic anomaly (foreign location + high amount)
      4. Rapid-fire small transactions (card testing)
      5. ATM cash-out on compromised cards
      6. Combined multi-signal fraud
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions (Indian context) ---
    # Indian geography: Lat 8-35, Lon 68-97
    # Typical Indian card spend: Rs 100 - Rs 15,000
    legit = pd.DataFrame({
        "User_ID":                   rng.integers(1, 5001, n_legit),
        "Merchant_ID":               rng.integers(1, 2001, n_legit),
        "Transaction_Amount":        np.abs(rng.normal(2500, 2000, n_legit)).clip(50, 25000),
        "Transaction_Hour":          rng.choice(np.concatenate([np.arange(7, 23), [0, 23]]), n_legit),
        "Transaction_Day":           rng.integers(1, 29, n_legit),
        "Transaction_Month":         rng.integers(1, 13, n_legit),
        "Location_Lat":              rng.uniform(8.0, 35.0, n_legit),     # India latitude
        "Location_Lon":              rng.uniform(68.0, 97.0, n_legit),    # India longitude
        "Device_Type":               rng.choice(["Mobile", "Desktop", "POS"], n_legit, p=[0.55, 0.25, 0.20]),
        "Transaction_Type":          rng.choice(["Online", "In-Store"], n_legit, p=[0.65, 0.35]),
        "Previous_Transaction_Days": rng.integers(0, 30, n_legit),
        "Days_Since_Card_Issued":    rng.integers(30, 3650, n_legit),
        "Fraud":                     0,
    })

    # --- Fraudulent transactions (six distinct patterns) ---
    fraud_rows = []
    patterns = ["high_amount", "night_new_card", "geo_anomaly",
                "rapid_fire", "atm_cashout", "combined"]

    for i in range(n_fraud):
        p = patterns[i % len(patterns)]
        row = {
            "User_ID":       rng.integers(1, 5001),
            "Merchant_ID":   rng.integers(1, 2001),
            "Fraud":         1,
        }

        if p == "high_amount":
            # Extremely high INR amounts (Rs 2,00,000 - Rs 50,00,000)
            row.update({
                "Transaction_Amount":        rng.uniform(200000, 5000000),
                "Transaction_Hour":          rng.integers(0, 24),
                "Transaction_Day":           rng.integers(1, 29),
                "Transaction_Month":         rng.integers(1, 13),
                "Location_Lat":              rng.uniform(8.0, 35.0),
                "Location_Lon":              rng.uniform(68.0, 97.0),
                "Device_Type":               rng.choice(["Mobile", "Desktop"]),
                "Transaction_Type":          "Online",
                "Previous_Transaction_Days": rng.integers(0, 15),
                "Days_Since_Card_Issued":    rng.integers(60, 1500),
            })

        elif p == "night_new_card":
            # Late night (12 AM - 5 AM) on newly issued card
            row.update({
                "Transaction_Amount":        rng.uniform(25000, 500000),
                "Transaction_Hour":          rng.choice([0, 1, 2, 3, 4, 5]),
                "Transaction_Day":           rng.integers(1, 29),
                "Transaction_Month":         rng.integers(1, 13),
                "Location_Lat":              rng.uniform(-60, 60),        # foreign location
                "Location_Lon":              rng.uniform(-180, 180),
                "Device_Type":               "Mobile",
                "Transaction_Type":          "Online",
                "Previous_Transaction_Days": 0,
                "Days_Since_Card_Issued":    rng.integers(1, 20),
            })

        elif p == "geo_anomaly":
            # Transaction from foreign country (outside India)
            row.update({
                "Transaction_Amount":        rng.uniform(100000, 3000000),
                "Transaction_Hour":          rng.integers(0, 24),
                "Transaction_Day":           rng.integers(1, 29),
                "Transaction_Month":         rng.integers(1, 13),
                "Location_Lat":              rng.choice([rng.uniform(-45, -10), rng.uniform(50, 70)]),
                "Location_Lon":              rng.choice([rng.uniform(-130, -60), rng.uniform(100, 170)]),
                "Device_Type":               rng.choice(["Mobile", "ATM"]),
                "Transaction_Type":          rng.choice(["Online", "ATM_Withdrawal"]),
                "Previous_Transaction_Days": rng.integers(1, 10),
                "Days_Since_Card_Issued":    rng.integers(30, 1000),
            })

        elif p == "rapid_fire":
            # Small test charges (Rs 10 - Rs 500) — card testing pattern
            row.update({
                "Transaction_Amount":        rng.uniform(10, 500),
                "Transaction_Hour":          rng.choice([1, 2, 3, 4]),
                "Transaction_Day":           rng.integers(1, 29),
                "Transaction_Month":         rng.integers(1, 13),
                "Location_Lat":              rng.uniform(-90, 90),
                "Location_Lon":              rng.uniform(-180, 180),
                "Device_Type":               "Mobile",
                "Transaction_Type":          "Online",
                "Previous_Transaction_Days": 0,
                "Days_Since_Card_Issued":    rng.integers(1, 60),
            })

        elif p == "atm_cashout":
            # Large ATM withdrawal (Rs 1,00,000 - Rs 80,00,000)
            row.update({
                "Transaction_Amount":        rng.uniform(100000, 8000000),
                "Transaction_Hour":          rng.choice([0, 1, 2, 3, 4, 5]),
                "Transaction_Day":           rng.integers(1, 29),
                "Transaction_Month":         rng.integers(1, 13),
                "Location_Lat":              rng.uniform(-90, 90),
                "Location_Lon":              rng.uniform(-180, 180),
                "Device_Type":               "ATM",
                "Transaction_Type":          "ATM_Withdrawal",
                "Previous_Transaction_Days": rng.integers(30, 180),
                "Days_Since_Card_Issued":    rng.integers(1, 90),
            })

        else:  # combined — multiple red flags
            row.update({
                "Transaction_Amount":        rng.uniform(500000, 10000000),
                "Transaction_Hour":          rng.choice([2, 3, 4]),
                "Transaction_Day":           rng.integers(1, 29),
                "Transaction_Month":         rng.integers(1, 13),
                "Location_Lat":              rng.uniform(-90, 90),
                "Location_Lon":              rng.uniform(-180, 180),
                "Device_Type":               rng.choice(["ATM", "Mobile"]),
                "Transaction_Type":          rng.choice(["ATM_Withdrawal", "Online"]),
                "Previous_Transaction_Days": rng.integers(40, 180),
                "Days_Since_Card_Issued":    rng.integers(1, 30),
            })

        fraud_rows.append(row)

    fraud_df = pd.DataFrame(fraud_rows)

    # Combine and shuffle
    df = pd.concat([legit, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add sequential Transaction_ID
    df.insert(0, "Transaction_ID", [f"TXN{i:06d}" for i in range(len(df))])

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING SYNTHETIC CREDIT CARD DATASET (INDIA / INR)")
    print("=" * 60)
    print(f"  Samples : {N_SAMPLES}")
    print(f"  Fraud % : {FRAUD_RATIO * 100:.1f}%")

    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)

    print(f"\nDataset saved to: {DATASET_PATH}")
    print(f"  Total transactions : {len(df)}")
    print(f"  Fraudulent         : {df['Fraud'].sum()} ({df['Fraud'].mean()*100:.2f}%)")
    print(f"  Legitimate         : {(df['Fraud']==0).sum()}")
    print(f"\nSample rows:")
    print(df.head(5).to_string(index=False))
