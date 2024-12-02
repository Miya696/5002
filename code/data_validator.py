import pandas as pd
import logging

def validate_cleaned_data(cleaned_data_file):
    """
    Validate the integrity of the cleaned data:
    - Check for duplicate rows
    - Check for missing values
    - Verify data types
    - Ensure uniqueness
    """

    # Read cleaned data
    cleaned_data = pd.read_csv(cleaned_data_file)

    # Checking for duplicate lines
    duplicates = cleaned_data[cleaned_data.duplicated()]
    if not duplicates.empty:
        logging.warning(f"Data cleaned test: Found {len(duplicates)} duplicate rows.")
        print("Duplicate rows:")
        print(duplicates)  
    else:
        logging.info("No duplicate rows found.")

    # Checking for missing values
    missing_values = cleaned_data.isnull().sum().sum()
    if missing_values > 0:
        logging.warning(f"Data cleaned test: Found {missing_values} missing values.")
        print("Rows with missing values:")
        print(cleaned_data[cleaned_data.isnull().any(axis=1)])  
    else:
        logging.info("No missing values found.")

    # Checking data types
    for column in cleaned_data.columns:
        expected_type = get_expected_column_type(column)
        actual_type = cleaned_data[column].dtype
        if expected_type != actual_type:
            logging.warning(f"Column '{column}' has type {actual_type}, expected {expected_type}.")
            print(f"Column '{column}' type mismatch. Example values:")
            print(cleaned_data[[column]].head())  
        else:
            logging.info(f"Column '{column}' has expected type {expected_type}.")


    # Checking uniqueness
    if 'record_number' in cleaned_data.columns:
        unique_count = cleaned_data['record_number'].nunique()
        total_count = len(cleaned_data)
        if unique_count != total_count:
            logging.warning(f"Expected unique 'record_number', found {total_count - unique_count} duplicates.")
            print("Rows with duplicate 'record_number':")
            print(cleaned_data[cleaned_data.duplicated('record_number', keep=False)])  
        else:
            logging.info("All 'record_number' values are unique.")
    else:
        logging.warning("No 'record_number' column found for uniqueness check.")

def get_expected_column_type(column_name):
    """
    Returns the expected data type based on the column name.
    """
    column_types = {
        'resident_type': 'object',  
        'family_composition': 'object',  
        'sex': 'int64', 
        'age': 'int64',  
        'marital_status': 'int64', 
        'country_of_birth': 'int64', 
        'health': 'int64', 
        'ethnic_group': 'int64', 
        'record_number': 'int64',
        'religion': 'int64', 
        'economic_activity': 'object', 
        'occupation': 'object', 
        'hours_worked_per_week': 'object', 
        'student': 'int64'
    }
    return column_types.get(column_name, 'object') 

if __name__ == "__main__":
    # Get the path to the cleaned data file from the command line arguments
    import sys
    if len(sys.argv) != 2:
        print("Usage: python data_validator.py <cleaned_data_file>")
        sys.exit(1)

    cleaned_data_file = sys.argv[1]

    # Call the validation function
    validate_cleaned_data(cleaned_data_file)
