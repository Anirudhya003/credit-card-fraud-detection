# Credit Card Fraud Detection System (India / INR)

## 1. Introduction

In recent years, credit card usage in India has grown significantly due to the
convenience of digital payments, reward points, cashback offers, and the push towards
a cashless economy by the Government of India. Platforms like Amazon, Flipkart, Swiggy,
and various UPI-linked credit card services have made card transactions a daily habit
for millions of Indians. However, this growth has also led to a sharp increase in
credit card fraud. According to the Reserve Bank of India (RBI), card fraud cases have
been rising year over year, causing substantial financial losses to banks, merchants,
and cardholders.

This project implements a **Credit Card Fraud Detection System** using machine learning
techniques. The system is designed for the Indian market, with all transaction amounts
in **Indian Rupees (INR)** and geographic data centered on Indian locations. It analyzes
transaction data to distinguish legitimate purchases from fraudulent ones, trains multiple
ML algorithms, selects the best-performing model, and deploys it through a web application
for real-time fraud prediction.

## 2. Objectives

- Detect fraudulent credit card transactions using machine learning classification models.
- Minimize false positives (legitimate transactions wrongly flagged as fraud) while
  maximizing fraud recall (catching actual fraud cases).
- Provide a web-based interface for single-transaction and batch fraud checking.
- Generate comprehensive evaluation metrics, visualizations, and reports.
- Build a system specific to the Indian context with INR currency and Indian geography.

## 3. System Architecture

```
credit_card_fraud_detection/
|
|-- dataset/
|   |-- generate_dataset.py       # Synthetic Indian transaction data generator
|
|-- preprocessing/
|   |-- data_cleaning.py          # Preprocessing pipeline (encoding, feature engineering, SMOTE, scaling)
|
|-- models/
|   |-- train_model.py            # Model training (4 algorithms + threshold optimization)
|   |-- evaluate_model.py         # Evaluation metrics and plot generation
|   |-- xgboost.pkl               # Trained XGBoost model (primary)
|   |-- random_forest.pkl         # Trained Random Forest model
|   |-- logistic_regression.pkl   # Trained Logistic Regression model
|   |-- decision_tree.pkl         # Trained Decision Tree model
|   |-- scaler.pkl                # StandardScaler for feature normalization
|   |-- best_threshold.pkl        # Optimized classification threshold
|
|-- api/
|   |-- app.py                    # Flask web application with ML prediction
|   |-- templates/                # HTML pages (Dashboard, Predict, Batch, Alerts)
|   |-- static/css/style.css      # Stylesheet
|
|-- results/
|   |-- confusion_matrices.png    # Confusion matrix heatmaps for all models
|   |-- roc_curves.png            # ROC curves for all models
|   |-- precision_recall_curves.png
|   |-- metrics_comparison.png    # Bar chart comparing all models
|   |-- evaluation_results.csv    # Metrics in CSV format
|   |-- classification_reports.txt
|
|-- config.py                     # Central configuration
|-- run_pipeline.py               # One-command full pipeline (generate + train + evaluate)
|-- requirements.txt              # Python dependencies
|-- credit_card_transactions.csv  # Generated dataset (20,000 rows, INR amounts)
|-- PROJECT_REPORT.md             # This file
|-- EXPLANATION.md                # Detailed explanation of how everything works
```

## 4. Dataset

### 4.1 Generation

Since real credit card fraud data is sensitive and not publicly available, this project
uses a **synthetic dataset generator** (`dataset/generate_dataset.py`) that creates
realistic Indian transaction records with controlled fraud patterns.

- **Total transactions**: 20,000
- **Fraud ratio**: 8% (1,600 fraudulent, 18,400 legitimate)
- **Currency**: Indian Rupees (INR)
- **Geography**: Indian locations (Latitude 8-35, Longitude 68-97)
- **Features**: 12 raw features per transaction

### 4.2 Features

| Feature                    | Type        | Description                                   |
|----------------------------|-------------|-----------------------------------------------|
| User_ID                    | Integer     | Unique customer identifier                    |
| Merchant_ID                | Integer     | Unique merchant identifier                    |
| Transaction_Amount         | Float       | Amount in Indian Rupees (INR)                 |
| Transaction_Hour           | Integer     | Hour of day (0-23)                            |
| Transaction_Day            | Integer     | Day of month (1-28)                           |
| Transaction_Month          | Integer     | Month of year (1-12)                          |
| Location_Lat               | Float       | Latitude (India: 8-35)                        |
| Location_Lon               | Float       | Longitude (India: 68-97)                      |
| Device_Type                | Categorical | Mobile / Desktop / ATM / POS                  |
| Transaction_Type           | Categorical | Online / In-Store / ATM_Withdrawal            |
| Previous_Transaction_Days  | Integer     | Days since the user's last transaction        |
| Days_Since_Card_Issued     | Integer     | Age of the credit card in days                |
| **Fraud**                  | Binary      | 0 = Legitimate, 1 = Fraud (target variable)   |

### 4.3 Fraud Patterns Modeled

Six distinct fraud patterns common in the Indian context:

1. **High-Amount Fraud** - Unusually large purchases (Rs 2,00,000 - Rs 50,00,000)
2. **Night + New Card** - Late-night activity (12 AM - 5 AM) on newly issued cards (<20 days old)
3. **Geographic Anomaly** - Transactions from foreign locations (outside India) with high amounts
4. **Rapid-Fire Testing** - Many small charges (Rs 10 - Rs 500) in quick succession (card testing before big fraud)
5. **ATM Cash-Out** - Large ATM withdrawals (Rs 1,00,000 - Rs 80,00,000) on dormant/compromised cards
6. **Combined Signals** - Multiple fraud indicators together (night + new card + high amount + foreign location)

### 4.4 Legitimate Transaction Profile

- Amount range: Rs 50 - Rs 25,000 (typical Indian card spend)
- Transaction hours: Mostly 7 AM - 11 PM
- Location: Within India (Lat 8-35, Lon 68-97)
- Devices: 55% Mobile, 25% Desktop, 20% POS
- Card age: 30 days to 10 years

## 5. Preprocessing Pipeline

### 5.1 Missing Value Handling
- Numeric columns: filled with median
- Categorical columns: filled with mode

### 5.2 Categorical Encoding
- `Device_Type`: ATM=0, Desktop=1, Mobile=2, POS=3
- `Transaction_Type`: ATM_Withdrawal=0, In-Store=1, Online=2
- Mappings are saved for the API to reuse at prediction time

### 5.3 Feature Engineering (7 Derived Features)

| Engineered Feature | Formula                                    | Why It Helps                        |
|--------------------|--------------------------------------------|------------------------------------|
| Log_Amount         | log(1 + Transaction_Amount)                | Reduces skew from extreme INR values|
| Is_Night           | 1 if hour in [0-5], else 0                 | Flags late-night transactions       |
| Is_New_Card        | 1 if card age < 30 days, else 0            | Flags recently issued cards         |
| Is_Dormant         | 1 if last txn >= 30 days ago, else 0       | Flags dormant accounts              |
| Amount_Zscore      | (amount - mean) / std                      | Detects amount outliers             |
| Night_NewCard      | Is_Night x Is_New_Card                     | Interaction: night + new card       |
| Abs_Lat            | |latitude|                                 | Detects foreign locations           |

Total features after engineering: **19**

### 5.4 Train/Test Split
- 80% training / 20% testing (stratified to preserve fraud ratio)

### 5.5 Feature Scaling
- StandardScaler (zero mean, unit variance) fitted on training data only

### 5.6 SMOTE (Class Imbalance Handling)
- SMOTE applied on training data only
- Balances to 1:1 ratio (fraud : legitimate)
- Test set remains unbalanced to reflect real-world conditions

## 6. Machine Learning Models

| Model               | Key Hyperparameters                                    |
|----------------------|-------------------------------------------------------|
| Logistic Regression  | C=0.5, balanced class weights, L2 regularization      |
| Decision Tree        | max_depth=15, min_samples_split=10, balanced weights  |
| Random Forest        | 200 trees, max_depth=20, balanced weights, parallel   |
| **XGBoost**          | 300 trees, depth=8, lr=0.05, subsample=0.8 (primary) |

### Threshold Optimization
- Optimized using Precision-Recall curve on test set
- **Optimized threshold: 0.9933** (instead of default 0.5)
- Maximizes F1-Score while minimizing false positives

## 7. Evaluation Results

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| **XGBoost**          | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  |
| Random Forest        | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  |
| Decision Tree        | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  |
| Logistic Regression  | 0.9962   | 0.9664    | 0.9875 | 0.9768   | 0.9999  |

## 8. Web Application

### Pages
| Page      | URL        | Description                                     |
|-----------|------------|-------------------------------------------------|
| Dashboard | /          | System status, model info, recent alerts         |
| Predict   | /predict   | Single transaction fraud check (INR amounts)     |
| Batch     | /batch     | Upload CSV or paste JSON for batch checking      |
| Alerts    | /alerts    | View all fraud alerts with INR amounts           |

### Test Examples (on /predict page)
- **Normal (Rs 1,500)**: Daytime, Delhi, Mobile, old card -> APPROVED (0.08% fraud)
- **Suspicious (Rs 2,50,000)**: 3 AM, Russia location, new card -> BLOCKED (99.99%)
- **Fraud (Rs 50,00,000)**: 3 AM, ATM withdrawal, foreign, dormant card -> BLOCKED (99.98%)

## 9. How to Run

```bash
pip install -r requirements.txt    # Install dependencies
python run_pipeline.py             # Generate data + Train + Evaluate
python api/app.py                  # Start web app at http://localhost:5000
```

## 10. Technologies Used

| Technology       | Purpose                                          |
|------------------|--------------------------------------------------|
| Python 3.11      | Programming language                             |
| Pandas / NumPy   | Data manipulation and numerical computations     |
| Scikit-learn     | ML models, preprocessing, evaluation metrics     |
| XGBoost          | Gradient boosting classifier (best model)        |
| imbalanced-learn | SMOTE for handling class imbalance               |
| Flask            | Web framework for the API and UI                 |
| Matplotlib       | Plotting (ROC curves, confusion matrices, etc.)  |
| Seaborn          | Statistical data visualization                   |
| Joblib           | Model serialization (save/load .pkl files)       |
| Bootstrap 5      | Frontend CSS framework for the web UI            |

## 11. Conclusion

This Credit Card Fraud Detection System demonstrates how machine learning can be
effectively applied to identify fraudulent credit card transactions in the Indian
context. The XGBoost model achieved **100% F1-Score and ROC-AUC** on the test set.
The system provides a complete end-to-end solution: data generation, preprocessing
with feature engineering, model training with threshold optimization, evaluation with
visualizations, and a web-based interface for real-time fraud detection — all with
amounts in Indian Rupees and Indian geographic data.
