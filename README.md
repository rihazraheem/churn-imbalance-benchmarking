# Customer Churn Prediction: Benchmarking Imbalance Strategies

This repository contains a comprehensive machine learning pipeline to predict customer churn using two distinct datasets: **IBM Telco** and **Bank Churn**. The project focuses on benchmarking different strategies for handling class imbalance—specifically comparing **SMOTE** and **Cost-Sensitive Learning (Class Weights)**—across multiple classification models.

## 📊 Project Overview
The primary goal is to determine which imbalance-handling strategy yields the highest predictive performance and statistical significance across different business domains (Telecommunications vs. Banking).

### Key Features
* **EDA:** Detailed exploratory data analysis of churn drivers and feature distributions.
* **Imbalance Handling:** Implementation and comparison of `None`, `SMOTE`, and `Class Weights`.
* **Model Zoo:** Benchmarking Logistic Regression, Random Forest, and XGBoost.
* **Statistical Rigor:** Performance validation using Paired T-Tests and Wilcoxon Signed-Rank Tests.
* **Experiment Tracking:** Integrated with MLflow for lifecycle management.

## 📂 Project Structure

├── data/                   # Raw datasets (Bank and Telco)
├── notebooks/
│   ├── EDA.ipynb           # Feature distribution and correlation analysis
│   └── experiments.ipynb   # Model training and benchmarking logic
├── src/                    # Modular source code
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   ├── models.py           # Model configurations and hyperparameter grids
│   ├── tuning.py           # Imbalance-aware pipeline construction
│   └── evaluation.py       # Metrics calculation and statistical tests
├── results/                # CSV files containing fold-level and aggregated metrics
├── requirements.txt        # Project dependencies
└── README.md



## 🎯 The Baseline & Business Context

In churn prediction, the "Baseline" (no imbalance handling) typically optimizes for **Accuracy**. However, in a lopsided dataset where only 20% of customers churn, a model can reach 80% accuracy by simply predicting "No Churn" for everyone.

**The Business Problem:** * **False Negatives:** We predict a customer stays, but they leave. This costs the company the entire Customer Lifetime Value (CLV).
* **False Positives:** We predict a customer leaves, but they stay. This costs us a small discount or an unnecessary retention email.

**Our Goal:** Shift the model's focus from global accuracy to **Recall**, ensuring we capture as many "at-risk" customers as possible.

## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)
Before modeling, we analyzed the datasets to understand the underlying patterns of churn:
* **Telco:** Identified that "Month-to-month" contracts and high "Monthly Charges" are high-risk indicators.
* **Bank:** Observed that age and the number of products used significantly influence the exit rate.

### 2. Imbalance Strategies
* **None:** Baseline performance on original class distribution.
* **SMOTE:** Synthetic Minority Over-sampling Technique applied within cross-validation folds to prevent data leakage.
* **Class Weights:** Adjusting the cost function of the models (e.g., `scale_pos_weight` in XGBoost) to penalize minority class errors more heavily.

### 3. Evaluation
Models were evaluated using **Stratified K-Fold Cross-Validation** (5 folds) with a focus on:
* **ROC-AUC:** Primary metric for overall model ranking.
* **F1-Score:** Balance between Precision and Recall.
* **Recall:** Crucial for identifying the maximum number of at-risk customers.

## 📈 Key Findings
Based on the results in `aggregated_results_N.csv`:
* **Performance:** Tree-based models (XGBoost and Random Forest) consistently outperformed Logistic Regression in ROC-AUC across both datasets.
* **Strategy Impact:** Class weighting generally improved Recall significantly compared to the baseline, making it a robust choice for business scenarios where missing a churner is expensive.
* **Statistical Significance:** Statistical tests confirmed that the performance improvements of XGBoost over Logistic Regression are statistically significant ($p < 0.05$).


##📊 Model Performance Comparison


| Dataset | Model | Strategy | ROC-AUC | F1-Score | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bank_Churn | XGB | None (Baseline) | 0.8896 | 0.6386 | 0.5604 |
| Bank_Churn | XGB | Class_Weights | 0.8874 | 0.6436 | 0.7846 |
| Bank_Churn | XGB | SMOTE | 0.8871 | 0.6455 | 0.5880 |
| Bank_Churn | RandomForest | None | 0.8855 | 0.6249 | 0.5364 |
| Bank_Churn | Logistic | Class_Weights | 0.8187 | 0.5588 | 0.7375 |
| IBM_Telco | XGB | Class_Weights | 0.8465 | 0.6302 | 0.7978 |
| IBM_Telco | XGB | SMOTE | 0.8456 | 0.6086 | 0.5950 |
| IBM_Telco | Logistic | None | 0.8450 | 0.6017 | 0.5543 |
| IBM_Telco | RandomForest | Class_Weights | 0.8429 | 0.6258 | 0.7169 |




## 🧠 Strategy Analysis: Why Class Weights Won

While both **SMOTE** and **Class Weights** improved the models, **Cost-Sensitive Learning (Class Weights)** emerged as the most robust strategy for these datasets.

### Why Class Weights outperformed SMOTE:
1.  **Data Integrity:** SMOTE creates synthetic "near-neighbors." In datasets with many categorical features (like Telco), these synthetic points can sometimes create "noise" or unrealistic feature combinations. Class Weights works on the original, real data points.
2.  **Loss Function Modification:** By increasing the penalty for misclassifying the minority class, Class Weights forces the model's decision boundary to prioritize the churners without over-fitting to synthetic noise.
3.  **Algorithmic Efficiency:** In XGBoost, the `scale_pos_weight` parameter directly adjusts the gradient updates, making it more mathematically direct than generating 1,000s of new rows.



### The "Cost" of Higher Recall:
Note that as **Recall** increased (specifically with Class Weights), **Precision** slightly decreased. This represents the "Retention Tax"—the cost of sending marketing offers to some customers who weren't actually planning to leave, in exchange for saving the ones who were.

