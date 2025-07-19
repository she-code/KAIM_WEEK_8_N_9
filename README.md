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
