"""
Kidney Disease Dataset - Data Preprocessing
Member 3's responsibility

⚠️ This is the most complex dataset:
   - 26 columns with many missing values
   - Mix of numeric and categorical columns
   - Target column is text: 'ckd' or 'notckd'

Steps:
1. Load raw data
2. Fix column names and data types
3. Encode categoricals
4. Handle missing values (many!)
5. Encode target variable
6. Apply SMOTE
7. Scale features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

RAW_PATH = "data/raw/kidney.csv"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    """Load raw kidney disease dataset."""
    df = pd.read_csv(RAW_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['classification'].value_counts()}")
    return df

def preprocess(df):
    """
    Clean and preprocess the kidney disease dataset.
    
    Handles:
    - Empty strings (missing values in CSV)
    - Binary yes/no columns: htn, dm, cad, pe, ane
    - Categorical columns: rbc/pc (normal/abnormal), pcc/ba (present/notpresent), appet (good/poor)
    - Sparse columns (drop if >50% missing)
    - Target encoding: ckd→1, notckd→0
    """

    # Step 1: Drop id column if it exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Step 2: Replace empty strings with NaN
    df = df.replace('', np.nan)

    # Step 3: Strip spaces from all string values
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    # Step 4: Drop columns with >50% missing values (too sparse to be useful)
    threshold = 0.5 * len(df)
    cols_before = len(df.columns)
    df = df.dropna(axis=1, thresh=threshold)
    print(f"Dropped {cols_before - len(df.columns)} sparse columns")





    # Step 5: Encode binary yes/no columns
    binary_yes_no = ['htn', 'dm', 'cad', 'pe', 'ane']
    for col in binary_yes_no:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0})

    # Step 6: Encode normal/abnormal columns (rbc, pc)
    for col in ['rbc', 'pc']:
        if col in df.columns:
            df[col] = df[col].map({'normal': 0, 'abnormal': 1})

    # Step 7: Encode present/notpresent columns (pcc, ba)
    for col in ['pcc', 'ba']:
        if col in df.columns:
            df[col] = df[col].map({'notpresent': 0, 'present': 1})

    # Step 8: Encode appetite (good=1, poor=0)
    if 'appet' in df.columns:
        df['appet'] = df['appet'].map({'good': 1, 'poor': 0})

    # Step 9: Encode target variable FIRST (before numeric conversion)
    # Strip extra whitespace/tabs that may exist in the target column
    df['classification'] = df['classification'].str.strip()
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    # Drop any rows where classification is still NaN
    df = df.dropna(subset=['classification'])
    df['classification'] = df['classification'].astype(int)

    # Step 10: Convert any remaining string columns to numeric
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 11: Fill ALL numeric missing values with median
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            if not pd.isna(median_val):
                df[col] = df[col].fillna(median_val)
            else:
                # Column is all NaN, drop it
                df = df.drop(col, axis=1)

    # Safety: drop any remaining rows with NaN
    df = df.dropna()
    df = df.reset_index(drop=True)

    print("\nAfter cleaning:")
    print(f"Shape: {df.shape}")
    print(f"Missing values remain:\n{df.isnull().sum()}")
    print(f"Class distribution:\n{df['classification'].value_counts()}")
    return df

def split_and_smote(df):
    """Split into train/test and apply SMOTE."""
    X = df.drop('classification', axis=1)
    y = df['classification']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nBefore SMOTE:\n{y_train.value_counts()}")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, X_test, y_train_resampled, y_test, X.columns.tolist()

def scale_features(X_train, X_test):
    """Standardize features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/saved_models/scaler_kidney.joblib")
    print("Scaler saved.")
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test, feature_names = split_and_smote(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("\n✅ Kidney disease preprocessing complete.")
