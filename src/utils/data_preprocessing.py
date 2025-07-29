import pandas as pd
import numpy as np

def load_data(file_path):
    """Load data from a specified file path."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")

def clean_data(df):
    """Clean the DataFrame by handling missing values and duplicates."""
    df = df.drop_duplicates()
    df = df.fillna(method='ffill')  # Forward fill for missing values
    return df

def transform_data(df):
    """Transform the DataFrame as needed for analysis."""
    # Example transformation: converting categorical variables to dummy variables
    df = pd.get_dummies(df, drop_first=True)
    return df

def save_data(df, file_path):
    """Save the DataFrame to a specified file path."""
    if file_path.endswith('.csv'):
        df.to_csv(file_path, index=False)
    elif file_path.endswith('.parquet'):
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError("Unsupported file format for saving")