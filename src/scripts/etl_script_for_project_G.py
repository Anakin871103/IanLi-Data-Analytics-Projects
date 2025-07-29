# ETL Script for Project G

import pandas as pd

def extract_data(file_path):
    """Extract data from a given file path."""
    data = pd.read_csv(file_path)
    return data

def transform_data(data):
    """Transform the data as needed for analysis."""
    # Example transformation: drop missing values
    transformed_data = data.dropna()
    return transformed_data

def load_data(data, output_path):
    """Load the transformed data to the specified output path."""
    data.to_csv(output_path, index=False)

def main():
    # Define file paths
    input_file_path = 'data/raw/project_g_data.csv'  # Update with the actual path
    output_file_path = 'data/processed/project_g_transformed_data.csv'  # Update with the actual path

    # ETL process
    extracted_data = extract_data(input_file_path)
    transformed_data = transform_data(extracted_data)
    load_data(transformed_data, output_file_path)

if __name__ == "__main__":
    main()