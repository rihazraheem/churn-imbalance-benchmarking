import pandas as pd

def clean_telco(df):
    df = df.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', pd.NA), errors='coerce')
    tc_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(tc_median)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    return df

def clean_bank(df):
    df = df.copy()
    cols_to_drop = ['Surname', 'id', 'CustomerId']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df['Exited'] = df['Exited'].astype(int)
    return df

def get_feature_lists(X):
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return cat_cols, num_cols
