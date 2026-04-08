"""
Heart Disease Dataset - Data Preprocessing
Member 2's responsibility

Dataset: 303 patients, 14 features
Target: AHD (Yes=heart disease, No=no heart disease)

Categorical columns: ChestPain (4 types), Thal (3 types + NA)
Columns with NA: Ca, Thal

Steps:
1. Load raw data
2. Drop row-index column
3. Handle NA values in Ca and Thal
4. Encode categorical columns (ChestPain, Thal)
5. Encode target variable (Yes→1, No→0)
6. Apply SMOTE for class imbalance
7. Scale features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

RAW_PATH = "data/raw/heart.csv"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    """Load raw heart disease dataset."""
    df = pd.read_csv(RAW_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    return df

def preprocess(df):
    """
    Clean and preprocess the heart disease dataset.

    Handles:
    - Row-index column (first unnamed column with row numbers)
    - NA values in Ca and Thal columns
    - Categorical encoding: ChestPain (one-hot), Thal (one-hot)
    - Target encoding: Yes→1, No→0
    """

    # Step 1: Drop the row-index column (first column contains 1,2,3...)
    # The CSV has 15 columns but header only names 14 — first col is row number
    first_col = df.columns[0]
    if first_col == 'Age':
        # Check if there's an unnamed index column that got merged
        # The data rows start with: 1,63,1,typical,...
        # So the actual columns are shifted — the "Age" column contains row numbers
        # We need to fix the column mapping
        print("Detected row-index baked into data. Fixing column alignment...")
        # The real columns should be: [index, Age, Sex, ChestPain, RestBP, Chol, Fbs, RestECG, MaxHR, ExAng, Oldpeak, Slope, Ca, Thal, AHD]
        # But the header only has 14 names, so the first column values (1,2,3...) got read as Age values
        # Let's re-read with proper handling
        pass

    # Check if Age column actually contains row indices (1, 2, 3, ...)
    if df['Age'].iloc[0] == 1 and df['Age'].iloc[1] == 2 and df['Age'].iloc[2] == 3:
        print("Row indices detected in Age column. Re-mapping columns...")
        # The actual data has 15 fields per row but only 14 headers
        # Re-read with the index column
        df = pd.read_csv(RAW_PATH, header=0)
        # Drop the first column (it's just row numbers)
        # Reassign columns: shift everything
        new_cols = ['RowIndex', 'Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs',
                    'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal', 'AHD']
        if len(df.columns) == 14:
            # Header has 14 cols, data has 15 fields — pandas may have auto-assigned index
            # Let's try re-reading
            df = pd.read_csv(RAW_PATH, header=None, skiprows=1,
                             names=new_cols)
        elif len(df.columns) == 15:
            df.columns = new_cols
        df = df.drop('RowIndex', axis=1)
        print(f"After fixing: Shape={df.shape}, Columns={df.columns.tolist()}")
    else:
        # Normal case — just check for any unnamed index columns
        unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
        if unnamed_cols:
            df = df.drop(unnamed_cols, axis=1)

    # Step 2: Strip whitespace from all string columns
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    # Step 3: Handle NA values
    # Replace string 'NA' with actual NaN
    df = df.replace('NA', np.nan)

    # Convert Ca to numeric (it has NA values)
    df['Ca'] = pd.to_numeric(df['Ca'], errors='coerce')

    # Fill Ca NaN with median
    if df['Ca'].isnull().sum() > 0:
        df['Ca'] = df['Ca'].fillna(df['Ca'].median())
        print(f"Filled {df['Ca'].isnull().sum()} remaining NaN in Ca with median")

    # Fill Thal NaN with mode (most frequent value)
    if df['Thal'].isnull().sum() > 0:
        thal_mode = df['Thal'].mode()[0]
        df['Thal'] = df['Thal'].fillna(thal_mode)
        print(f"Filled Thal NaN with mode: {thal_mode}")

    # Step 4: Encode categorical columns
    # ChestPain: typical, nontypical, nonanginal, asymptomatic
    # Thal: normal, fixed, reversable
    df = pd.get_dummies(df, columns=['ChestPain', 'Thal'], drop_first=True, dtype=int)

    # Step 5: Encode target variable: Yes→1, No→0
    df['AHD'] = df['AHD'].map({'Yes': 1, 'No': 0})

    # Drop any remaining rows with NaN
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Ensure AHD is integer
    df['AHD'] = df['AHD'].astype(int)

    print(f"\nAfter cleaning:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Class distribution:\n{df['AHD'].value_counts()}")
    return df

def split_and_smote(df):
    """Split into train/test and apply SMOTE to training data only."""
    X = df.drop('AHD', axis=1)
    y = df['AHD']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nBefore SMOTE — Train class distribution:\n{y_train.value_counts()}")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Train class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, X_test, y_train_resampled, y_test, X.columns.tolist()

def scale_features(X_train, X_test):
    """Standardize features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models/saved_models/", exist_ok=True)
    joblib.dump(scaler, "models/saved_models/scaler_heart.joblib")
    print("Scaler saved.")
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test, feature_names = split_and_smote(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("\n✅ Heart disease preprocessing complete.")
