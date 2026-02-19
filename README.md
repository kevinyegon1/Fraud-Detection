# Fraud Detection System: Stacking Ensemble Approach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange)](https://scikit-learn.org/)

## 📌 Project Overview
Financial fraud accounts for billions in annual losses globally. This project develops a robust machine learning pipeline to identify fraudulent transactions in a highly imbalanced dataset. By leveraging **Stacking Ensemble** methods and sophisticated feature analysis, the model achieves high sensitivity to fraudulent patterns, specifically identifying high-risk windows (night-time) and low-trust device signatures.

## 📊 Dataset Characteristics
The dataset consists of **10,000 entries** with a significant class imbalance:
* **Non-Fraud:** > 80%
* **Fraud:** < 20%

### Key Exploratory Insights
* **Temporal Patterns:** Fraud peaks significantly between **12:00 AM and 3:00 AM**.
* **Risk Factors:** High correlation with **Foreign Transactions**, **Location Mismatches** (< 2,000 occurrences), and **Grocery** merchant categories.
* **Technical Metrics:** Transactions are high-risk when **Device Trust Score < 40** and **Transaction Velocity** is between 5 and 8.



## 🛠️ Tech Stack & Methodology

### Data Preprocessing
* **Scaling:** Applied `StandardScaler` to numerical features (Age, Device Score, etc.).
* **Encoding:** Utilized `LabelEncoder` for categorical features (Merchant Category, Location).
* **Imbalance Handling:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set, ensuring the model doesn't overfit the majority class.

### Model Architecture: Stacking Ensemble
To maximize predictive power, a Stacking Classifier was implemented:
1.  **Base Models:** * Random Forest Classifier
    * XGBoost Classifier
    * Logistic Regression
2.  **Meta-Model (Final Estimator):** * Logistic Regression (used to blend the predictions of the base models).

## 📈 Performance Results
The model was evaluated using metrics suited for imbalanced classification:

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | **0.95** |
| **Data Split** | 80/20 Train-Test |

The **0.95 ROC-AUC** indicates an exceptional ability to distinguish between legitimate transactions and fraudulent activity across all threshold levels.

## 📁 Repository Structure
```text
├── data/               # Dataset (if public)
├── exploratory data analysis notebook/          # EDA 
├── model notebook/                # Model Training notebooks
├── requirements.txt    # List of dependencies
└── README.md
