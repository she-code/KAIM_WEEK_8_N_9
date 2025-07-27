# Notebooks

This folder contains Jupyter notebooks illustrating the process of data ingestion, cleaning, preprocessing, and exploratory data analysis for fraud detection datasets.

## 1_Fraud_Data_EDA_FeatureEng.ipynb

- Loaded and preprocessed Fraud_Data.csv for fraud detection analysis
- Handled missing country values post-IP merge with 'Unknown' imputation
- Cleaned data by removing duplicates and converting signup_time, purchase_time to datetime
- Conducted EDA: univariate analysis (histograms for purchase_value, age; count plots for source, browser, sex, class) and bivariate analysis (boxplots, count plots vs. class)
- Engineered features: transaction_count, avg_time_between_transactions, time_since_last_purchase, hour_of_day, day_of_week, -time_since_signup
- Addressed class imbalance using SMOTE on training data
- Scaled numerical features with StandardScaler and one-hot encoded categorical features (source, browser, sex, country)
- Stored processed data for modeling

## 1.1_Credit_card_EDA.ipynb

- Loaded credit_card_data.csv for credit card fraud detection
- Performed EDA to analyze class distribution, confirming severe imbalance (99.83% non-fraud, 0.17% fraud) with log-scale count plot
- Highlighted need for SMOTE to address class imbalance in subsequent modeling steps

## 2_Model_Building_Summary.ipynb

- Loaded Fraud_Data.csv and creditcard.csv
- Preprocessed Fraud_Data: converted datetime columns to numeric (hour, day, month), encoded categorical features with LabelEncoder
- Ensured numeric features for modeling
- Performed 80-20 stratified train-test split on class (Fraud_Data) and Class (creditcard)
- Trained and compared:
    - Logistic Regression: Interpretable baseline
    - Random Forest: Ensemble model for complex patterns
- Evaluated with AUC-PR, F1-Score, and Confusion Matrix
- Visualized results with confusion matrix heatmaps
- Selected Random Forest as best model for higher AUC-PR, F1-Score, and robust performance
- Saved Random Forest models as fraud_data_rf_model.pkl and creditcard_rf_model.pkl in ../models/
- Stored results for Task 3 analysis