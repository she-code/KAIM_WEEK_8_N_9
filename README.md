# Improved detection of fraud cases for e-commerce and bank transactions

## Project Overview
This project focuses on fraud detection for e-commerce and bank credit transactions using advanced machine learning and data analysis techniques. The goal is to build accurate, explainable, and efficient models that can identify fraudulent activities in real-time while minimizing false positives and false negatives.

The solution leverages geolocation analysis, transaction pattern recognition, and feature engineering to enhance detection capabilities. Key steps include data preprocessing, feature extraction, model training and evaluation, and model interpretability using explainable AI (XAI) tools.

---

## Project Structure
---
```
KAIM_WEEK_8_N_9
├── .dvc/ # DVC Metadata
├── .github/
│ └── workflows/ # GitHub Actions workflows
├── data/
│ ├── raw/ # Raw data (should never be modified)
│ └── processed/ # Processed/cleaned data (gitignored)
├── notebooks/
│ └── README.md # Documentation for notebooks
│ └── 1_Fraud_Data_EDA_FeatureEng.ipynb # EDA, Featuring Eng for fraud data
│ └── 1.1_Credit_card_EDA.ipynb # EDA for credit card data
│ └── 2_model_training.ipynb # model training
├── scripts/
│ └── README.md # Documentation for scripts
├── src/
│ └── utils/ # Utility functions
    │ ├── data_loader.py # Data loading utilities
│ └── README.md # Documentation for source code
├── tests/
│ └── README.md # Testing documentation
├── .gitattributes
├── .gitignore
├── README.md # Main project documentation
└── requirements.txt # Python dependencies
```

---

## ⚙️ Tech Stack

**Languages & Libraries:**  
Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, XGBoost, SHAP, SciPy

**Modeling:**  
Linear Regression, Random Forest, XGBoost  
Logistic Regression and Classifiers for claim frequency  
SHAP for feature importance and interpretability

---

## Key Tasks Completed

### ✅ Task 1 - Data Analysis and Preprocessing

Processed Fraud_Data.csv and credit_card_data.csv for fraud detection. For Fraud_Data, handled missing country values post-IP merge, cleaned duplicates, converted timestamps to datetime, performed EDA (univariate and bivariate analyses), engineered features (transaction frequency, velocity, time-based), addressed class imbalance with SMOTE, scaled numerical features, and encoded categorical variables. For credit_card_data, conducted EDA to visualize severe class imbalance (99.83% non-fraud, 0.17% fraud) using log-scale count plot, confirming need for SMOTE. Data stored in processed format for modeling.

### ✅ Task 2 - Model Building and Training  

Trained and evaluated models on preprocessed Fraud_Data.csv and creditcard.csv for fraud detection, using stratified 80-20 train-test splits on class (Fraud_Data) and Class (creditcard) targets. Preprocessed Fraud_Data by converting datetime columns to numeric features (hour, day, month) and encoding categorical variables with LabelEncoder. Trained Logistic Regression and Random Forest models, evaluated with AUC-PR, F1-Score, and Confusion Matrix, visualized via heatmaps. Random Forest outperformed Logistic Regression (AUC-PR: 0.6207 vs. 0.3527 for Fraud_Data; 0.8042 vs. 0.6297 for creditcard) and was saved as fraud_data_rf_model.pkl and creditcard_rf_model.pkl in ../models/ for future use, with results stored for Task 3 analysis.

### ✅ Task 3 - Model Explainability Analysis
Leveraged SHAP to interpret our fraud detection models, generating global feature importance plots and local prediction explanations. For Fraud_Data, transaction patterns like max_purchase_velocity emerged as key fraud signals, while PCA components (V14, V12) proved most significant for creditcard transactions. Saved SHAP values and visualizations (summary plots, force plots) to ../plots/, demonstrating our models detect fraud through logical, explainable patterns. This analysis validates model reliability by aligning detection logic with known fraud indicators.

---
## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/she-code/KAIM_WEEK_8_N_9.git
cd KAIM_WEEK_8_N_9
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
3. **Install dependencies:**

```bash
pip install -r requirements.txt

```

## Contributors
- Frehiwot Abebie
