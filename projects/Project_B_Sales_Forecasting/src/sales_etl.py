import pandas as pd
import numpy as np

def load_data(file_path):
    """Load sales data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the sales data."""
    # Example cleaning steps
    data.dropna(inplace=True)  # Remove missing values
    data['date'] = pd.to_datetime(data['date'])  # Convert date column to datetime
    return data

def transform_data(data):
    """Transform the sales data for analysis."""
    # Example transformation steps
    data['sales'] = data['quantity'] * data['price']  # Calculate total sales
    return data

def save_data(data, output_path):
    """Save the processed sales data to a CSV file."""
    data.to_csv(output_path, index=False)

def main():
    """Main function to execute the ETL process."""
    # Load raw data
    raw_data_path = '../data/raw/sales_data.csv'
    sales_data = load_data(raw_data_path)

    # Clean data
    cleaned_data = clean_data(sales_data)

    # Transform data
    transformed_data = transform_data(cleaned_data)

    # Save processed data
    processed_data_path = '../data/processed/sales_data_processed.csv'
    save_data(transformed_data, processed_data_path)

if __name__ == "__main__":
    main()