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
```text
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

