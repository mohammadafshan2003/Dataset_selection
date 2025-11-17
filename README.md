#  Fraud Detection Classification Model

##  Project Overview

This project aims to build a machine learning model that detects fraudulent transactions using historical financial data. Fraudulent activity is rare but costly, making this a high-impact classification problem with a significant class imbalance challenge.

---

##  Problem Statement

**Goal:**  
Predict whether a transaction is fraudulent (`is_fraud = 1`) or legitimate (`is_fraud = 0`) based on transaction metadata and user behavior.

**Challenge:**  
Only ~2.2% of transactions are fraudulent, requiring specialized techniques to handle class imbalance and avoid misleading accuracy metrics.

---

## 📊 Dataset Summary

- **Total records:** 299,695 transactions
- **Fraudulent transactions:** 6,612 (~2.2%)
- **Non-fraudulent transactions:** 293,083

### 🔍 Features

| Feature Name               | Description |
|---------------------------|-------------|
| `transaction_id`          | Unique transaction identifier |
| `user_id`                 | Unique user identifier |
| `account_age_days`        | Age of the user's account |
| `total_transactions_user`| Total transactions made by the user |
| `avg_amount_user`         | Average transaction amount for the user |
| `amount`                  | Amount of the current transaction |
| `country`                 | Country of transaction |
| `bin`                     | Bank Identification Number |
| `currency`                | Currency used |
| `merchant_category`       | Merchant category |
| `promo_used`              | Whether a promo code was used |
| `avs_match`               | Address verification match |
| `cvv_result`              | CVV verification result |
| `three_ds_flag`           | 3D Secure authentication flag |
| `transaction_time`        | Timestamp of transaction |
| `shipping_distance_km`    | Distance between billing and shipping address |
| `is_fraud`                | Target variable (1 = fraud, 0 = non-fraud) |

---

## ⚖️ Handling Class Imbalance

### Techniques Used

- **SMOTE Oversampling**: Synthetic Minority Over-sampling Technique to generate new fraud samples.
- **Class Weighting**: Adjusted model training to penalize misclassification of minority class.
- **Evaluation Metrics**: Focused on Precision, Recall, F1-score, and PR-AUC instead of Accuracy.



dataset link: https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset
