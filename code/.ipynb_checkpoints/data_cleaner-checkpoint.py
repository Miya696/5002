import pandas as pd
import sys
import json
from collections import Counter
from IPython.display import display
import numpy as np
import os
import logging
import subprocess

class DataCleaner:
    """
    A class to handle the cleaning and processing of raw data in batches.
    
    Attributes:
        batch_size (int): Number of rows to process per batch.
        start_row (int): Starting row to resume progress from.
        rawdata_file_path (str): Path to the raw data file.
        variable_file (str): Path to the variable file.
        progress_file (str): Path to the progress file.
        labels (dict): Store labels used for data processing.
        df (DataFrame): The dataframe to hold the raw data.
        validation_errors (dict): Store validation errors encountered during processing.
        category_counts (Counter): Counter to keep track of category counts.
        unique_categories (dict): Dictionary to store unique categories encountered in data.
    """
    def __init__(self, rawdata_file_path, variable_file, batch_size=90000, start_row=0, progress_file="progress.json"):
        """
        Initializes the DataCleaner instance with the provided file paths, batch size, 
        starting row, and progress file path.
        
        Args:
            rawdata_file_path (str): Path to the raw data file.
            variable_file (str): Path to the variable file.
            batch_size (int, optional): Number of rows to process per batch. Defaults to 90000.
            start_row (int, optional): Starting row for data processing. Defaults to 0.
            progress_file (str, optional): Path to the progress file to track processing progress. Defaults to "progress.json".
        """
        self.batch_size = batch_size  # Number of rows to process per batch
        self.start_row = start_row    # Starting row, used to resume progress
        self.rawdata_file_path = rawdata_file_path  # Path to the raw data file
        self.variable_file = variable_file  # Path to the variable file
        self.progress_file = progress_file  # Path to the progress file
        self.labels = {}  # Store labels
        self.df = None
        self.validation_errors = {}  # Store validation errors
        self.category_counts = Counter()  # Counter for category counts
        self.unique_categories = {}  # Store unique categories
        self.setup_logging()  # Set up the logger

        # Load processing progress
        self.load_progress(progress_file)  # Load the progress from the file

        # Read the JSON file containing labels and convert all keys to lowercase
        try:
            with open(variable_file, 'r') as f:
                self.labels = {key.lower(): value for key, value in json.load(f).items()}
        except FileNotFoundError:
            logging.error(f"Variable file {variable_file} not found.")
        except json.JSONDecodeError:
            logging.error(f"Error decoding variable file {variable_file}.")

        # Skip the rows that have already been processed when loading data
        self.load_data()

    def setup_logging(self):
        """Set up the logger to redirect stdout and stderr to the log file"""
        logging.basicConfig(
            filename='../log/data_cleaning.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        sys.stdout = open('../log/stdout.log', 'w')
        sys.stderr = open('../log/stderr.log', 'w')

    def load_data(self):
        """Load data based on progress"""
        try:
            # Read the CSV file, skipping the rows that have already been processed
            self.df = pd.read_csv(self.rawdata_file_path, skiprows=range(1, self.start_row + 1))
            # Convert all column names to lowercase
            self.df.columns = self.df.columns.str.lower()
            logging.info(f"Loaded data from row {self.start_row}.")
            # Print column names for verification
            logging.info(f"Columns in DataFrame: {self.df.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()  # Return an empty DataFrame if loading fails

    # def process_batch(self, start_row):
    def process_batch(self, batch):     
        """Process a batch of data"""
        try:
            # batch = pd.read_csv(self.rawdata_file_path, skiprows=range(1, start_row), nrows=self.batch_size)
            # print(f"Processing batch starting from row {start_row}...")
            logging.info(f"Processing batch starting from row {self.start_row}.")
            
            # Process the data for this batch
            # batch.columns = batch.columns.str.lower()
            batch.columns = self.df.columns
            logging.info(f"Columns in batch: {batch.columns.tolist()}")
        
            self.check_type_conversion(batch)
            self.check_missing_values(batch)
            self.check_duplicate_rows(batch)
            self.verify_record_number_unique(batch)
            self.validate_data_ranges(batch)
            
            # After processing the batch, update the progress
            self.start_row += len(batch)
            return batch
        except Exception as e:
            # print(f"Error processing batch starting from row {start_row}: {e}")
            # return pd.DataFrame()  
            logging.error(f"Error processing batch: {e}")
            
            return pd.DataFrame()

    def process_data_in_batches(self):
        """Process all data in batches"""
        total_rows = len(self.df)
        logging.info(f"总行数: {total_rows}")
        
        # Process data in batch size
        for start_row in range(self.start_row, total_rows, self.batch_size):
            end_row = min(start_row + self.batch_size, total_rows)
            batch = self.df.iloc[start_row:end_row]
            
            self.process_batch(batch)
            
            # Log processing progress
            logging.info(f"Processed {end_row}/{total_rows} 行。")
            print(f"Processed {end_row}/{total_rows} 行。")
        
        print("Cleaning successful")
        logging.info("Cleaning successful")

    
    def check_type_conversion(self, batch):
        """Check the data types of columns in the dataset and assert that each column contains only string or numeric values."""
        print("Checking data types:")
        logging.info("=====Checking data types:=====")
        logging.info(f"Batch info:\n{batch.info()}")
        # print(self.df.info())
        print(batch.info())

        # Check if each value in the column is either a string or numeric
        for column in batch.columns:
            for index, value in batch[column].items():
                try:
                    # Check if the value is not a string, int or float
                    assert isinstance(value, (str, int, float)), (
                        # f"Column '{column}' has a value that is neither a string nor a number. "
                        # f"Invalid value {value} at index {index} (Type: {type(value)})"
                        f"Column '{column}' has invalid value {value} at index {index}"
                    )
                except AssertionError as e:
                    logging.error(f"Failed! {e}")
                    # If assertion fails, print the error and continue
                    print(f"Failed! {e}")
                    # Optionally log or handle the invalid value
                    continue  

    def check_missing_values(self, batch):
        """Check if there are any missing values in the dataset and output detailed information in a table."""
        logging.info("=====Checking for missing value:=====")
        missing_values = batch.isnull().sum()

        rows_to_drop = pd.DataFrame()

        # Ensure the batch column names are lowercase
        batch.columns = batch.columns.str.lower()

        # Log the total count of missing values
        if missing_values.sum() == 0:
            logging.info("No missing values detected.")
            print("No missing values detected.")
        else:
            logging.warning(f"Missing values detected:\n{missing_values}")
            print(f"Missing values detected:\n{missing_values}")
            
            # Output the locations of missing values in each column
            print("\nColumn missing value locations:")
            for col in batch.columns:
                if col in batch.columns and batch[col].isnull().any():
                    indices = batch[batch[col].isnull()].index.tolist()
                    # print(f"{col}: index {indices}")
                    # logging.error(f"error {col}: index {indices}")
                    logging.warning(f"Column '{col}': missing values at indices {indices}")
                    print(f"Column '{col}': missing values at indices {indices}")
    
            # Filter out rows with missing values
            rows_to_drop = batch[batch.isnull().any(axis=1)]
            
            print("\nRows with missing values:")
            logging.info(f"Rows to drop:\n{rows_to_drop.to_string()}")
            
            # notebook
            # display(rows_to_drop) 
        return rows_to_drop

    
    def check_duplicate_rows(self, batch):
        """Check for duplicate rows in the dataset."""
        logging.info("=====Checking for duplicate rows:=====")
        
        # Get the number of duplicate rows in the batch
        duplicate_rows = batch.duplicated().sum()
        batch.columns = batch.columns.str.lower()

        rows_to_drop = pd.DataFrame()
        
        try:
            assert duplicate_rows == 0, f"{duplicate_rows} duplicate rows detected."
            logging.info("No duplicate rows detected.")
            print("No duplicate rows detected.")
        except AssertionError as e:
            logging.error(f"Failed! {e}")
            print(f"Failed! {e}")
    
            # Display duplicate rows in a table format (for logging purposes)
            logging.info("\nDuplicate rows:")
            print("\nDuplicate rows:")
            duplicate_data = batch[batch.duplicated()]
            
            # Log the duplicate data (logging will handle formatting)
            logging.info(duplicate_data.to_string())

            # Collect rows to drop
            rows_to_drop = duplicate_data

            # for notebook
            # display(duplicate_data) 
        return rows_to_drop


    def verify_record_number_unique(self, batch):
        """Ensure the 'record_number' is unique."""
        logging.info("=====Verifying 'record_number' uniqueness:=====")
        # Print column names for debugging purposes
        logging.info(f"Columns in batch: {batch.columns.tolist()}")

        rows_to_drop = pd.DataFrame()
        batch.columns = batch.columns.str.lower()
    
        try:
            # Assert that 'record_number' column exists
            assert 'record_number' in batch.columns, "'record_number' column not found in the dataset."
    
            # Assert that 'record_number' values are unique
            assert batch['record_number'].is_unique, (
                f"'record_number' is not unique. "
                f"Duplicate record numbers: {batch['record_number'][batch['record_number'].duplicated()].tolist()}"
            )
            logging.info("Record_Number is unique.")
            print("Record_Number is unique.")
        
        except AssertionError as e:
            logging.error(f"Failed! {e}")
            print(f"Failed! {e}")
    
            # Display the rows with duplicate 'record_number' values 
            logging.info("\nDuplicate rows:")
            print("Duplicate rows:")
            duplicate_record_number = batch[batch['record_number'].duplicated()]

             # Collect rows to drop
            rows_to_drop = duplicate_record_number
            
            # for notebook
            # display(duplicate_record_number) 
    
            # Log the duplicate data (logging will handle formatting)
            logging.info(duplicate_record_number.to_string())
        return rows_to_drop


    def validate_data_ranges(self, batch):
            """Validate data ranges"""
            logging.info("=====validate_data_ranges:=====")
            rows_to_drop = pd.DataFrame()
            has_errors = False  # Flag variable to track if there are any errors.
        
            for column in batch.columns:
                if column in self.labels:
                    valid_values = list(self.labels[column].keys())
                
                    # Store unique values using a set
                    self.unique_categories[column] = set(batch[column].dropna().unique())
                
                    # Count occurrences of each category using Counter
                    self.category_counts.update(batch[column].dropna().astype(str))
                
                    # Identify invalid values
                    invalid_values = batch[~batch[column].astype(str).str.lower().isin(
                        [v.lower() for v in valid_values])]
                        
                    if not invalid_values.empty:
                        has_errors = True
                        print(f"\nBroken records in column '{column}':")
                        logging.info(f"\nBroken records in column '{column}':")
                        logging.info(invalid_values[[column]].to_string())  # Log invalid records in a table format

                        # Collect rows to drop
                        rows_to_drop = pd.concat([rows_to_drop, invalid_values])
                        
                        # display(invalid_values[[column]])  # Display invalid records in a table format
                
                        # Use try-except to handle assertion
                        try:
                            assert invalid_values.empty, (
                                f"Column '{column}' contains invalid values: "
                                f"{invalid_values[column].unique().tolist()}"
                            )
                        except AssertionError as e:
                            logging.error(f"Failed! {e}")
                
            # Print unique categories and their counts
            logging.info("\nUnique categories and counts per column:")
            print("\nUnique categories and counts per column:")
            for column, unique_values in self.unique_categories.items():    
                unique_values = [int(v) if isinstance(v, np.integer) else v for v in unique_values]
                if all(isinstance(v, (int, float)) for v in unique_values):
                    unique_values = sorted(unique_values)  # Sort integers for cleaner output
                    logging.info(f"{column}: {unique_values} (Total unique: {len(unique_values)})")
                    print(f"{column}: {unique_values} (Total unique: {len(unique_values)})")
                else:
                    logging.info(f"{column}: {unique_values} (Total unique: {len(unique_values)})")
                    print(f"{column}: {unique_values} (Total unique: {len(unique_values)})")

            print("\nCategory counts across columns:")
            for category, count in self.category_counts.items():
                print(f"{category}: {count} occurrences")
            
            # Print validation success message when all columns have been validated and no errors are found.
            if not has_errors:
                logging.info("\nAll data ranges validated.")
                print("\nAll data ranges validated.")
            return rows_to_drop


    def logical_validation(self, batch):
        """Apply logical rules to identify rows that need to be deleted."""
        drop_data = pd.DataFrame()

        # Rule 1: When Student = 1, age must be 1, and Economic_Activity must be '4'
        drop_data1 = batch[(batch['student'] == 1) & ~((batch['age'] == 1) | (batch['economic_activity'] == '4'))]
        logging.info(f"{(drop_data1)}")
        
        if not drop_data1.empty:
            drop_data = pd.concat([drop_data, drop_data1])
            drop_data1.to_csv('../data/rule_1_deleted_data.csv', index=False)
            logging.info(f"Rule 1: Found {len(drop_data1)} rows to delete, saved to data/rule_1_deleted_data.csv")
        else:
            logging.info("Rule 1: No rows to delete")

        # Rule 2: When Family_Composition = 'X', Residence_Type must be 'C'
        drop_data2 = batch[(batch['family_composition'] == 'X') & (batch['residence_type'] != 'C')]
        if not drop_data2.empty:
            drop_data = pd.concat([drop_data, drop_data2])
            drop_data2.to_csv('../data/rule_2_deleted_data.csv', index=False)
            logging.info(f"Rule 2: Found {len(drop_data2)} rows to delete, saved to data/rule_2_deleted_data.csv")
        else:
            logging.info("Rule 2: No rows to delete")

        # Rule 3: When Occupation = 'X', Economic_Activity must be in [6, 7, 8, 9, X]
        drop_data3 = batch[(batch['occupation'] == 'X') & (~batch['economic_activity'].isin(['6', '7', '8', '9', 'X']))]
        if not drop_data3.empty:
            drop_data = pd.concat([drop_data, drop_data3])
            drop_data3.to_csv('../data/rule_3_deleted_data.csv', index=False)
            logging.info(f"Rule 3: Found {len(drop_data3)} rows to delete, saved to data/rule_3_deleted_data.csv")
        else:
            logging.info("Rule 3: No rows to delete")

        # Rule 4: When Hours_Worked_Per_Week = 'X', Economic_Activity must be in [5, 6, 7, 8, 9, X] and age = 1
        drop_data4 = batch[(batch['hours_worked_per_week'] == 'X') &
                             ~(batch['economic_activity'].isin(['5', '6', '7', '8', '9', 'X'])) &
                             ~(batch['age'] == 1)]
        if not drop_data4.empty:
            drop_data = pd.concat([drop_data, drop_data4])
            drop_data4.to_csv('../data/rule_4_deleted_data.csv', index=False)
            logging.info(f"Rule 4: Found {len(drop_data4)} rows to delete, saved to data/rule_4_deleted_data.csv")
        else:
            logging.info("Rule 4: No rows to delete")

        # Rule 5: When Approximate_Social_Grade = 'X', Residence_Type must be 'C' and age = 1
        drop_data5 = batch[(batch['approximate_social_grade'] == 'X') &
                             (batch['residence_type'] == 'P') & ~(batch['age'] == 1)]
        if not drop_data5.empty:
            drop_data = pd.concat([drop_data, drop_data5])
            drop_data5.to_csv('../data/rule_5_deleted_data.csv', index=False)
            logging.info(f"Rule 5: Found {len(drop_data5)} rows to delete, saved to data/rule_5_deleted_data.csv")
        else:
            logging.info("Rule 5: No rows to delete")

        # Rule 6: When Industry = 'X', Economic_Activity must be in [6, 7, 8, 9, X]
        drop_data6 = batch[(batch['industry'] == 'X') & (~batch['economic_activity'].isin(['6', '7', '8', '9', 'X']))]
        if not drop_data6.empty:
            drop_data = pd.concat([drop_data, drop_data6])
            drop_data6.to_csv('../data/rule_6_deleted_data.csv', index=False)
            logging.info(f"Rule 6: Found {len(drop_data6)} rows to delete, saved to data/rule_6_deleted_data.csv")
        else:
            logging.info("Rule 6: No rows to delete")

        # Remove duplicate rows from the collected data
        drop_data = drop_data.drop_duplicates()

        # Return rows that need to be deleted
        return drop_data


    def save_progress(self, progress_file="progress.json"):
        """Save progress"""
        progress = {'start_row': self.start_row}
        
        try:
            # If no path is specified, use the default directory
            if os.path.dirname(progress_file) == '':
                progress_file = os.path.join(os.getcwd(), progress_file) # Use the current working directory
            
            # Ensure the directory exists, create it if not
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
            logging.info(f"Progress saved to {progress_file}.")
        except (OSError, IOError) as e:
            logging.error(f"File operation error when saving progress: {e}")
        except Exception as e:
            logging.error(f"Unexpected error saving progress: {e}")

    def load_progress(self, progress_file):
        """Load progress"""
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.start_row = progress.get('start_row', 0)
                logging.info(f"Progress loaded: start_row = {self.start_row}.")
        except FileNotFoundError:
            logging.warning(f"Progress file {progress_file} not found. Starting from the beginning.")
            self.start_row = 0  # Start from the beginning if no progress file is found
        except json.JSONDecodeError:
            logging.error(f"Error decoding progress file {progress_file}. Starting from the beginning.")
            self.start_row = 0  # Start from the beginning if there is an error decoding the progress file
        except Exception as e:
            logging.error(f"Unexpected error loading progress: {e}. Starting from the beginning.")
            self.start_row = 0  # Start from the beginning for other unexpected errors

    def process_in_batches(self):
        """Process the CSV file in batches to check for missing values and clean the data"""
        batch_number = 1
        rows_to_drop = pd.DataFrame()

        # Get the total number of rows in the file
        total_rows = sum(1 for line in open(self.rawdata_file_path))  # Get the number of rows in the file
        logging.info(f"Total rows in file: {total_rows}")

        # Ensure start_row is initialized, if not initialized in __init__
        if not hasattr(self, 'start_row'):
            self.start_row = 0

        # If start_row is already equal to the total number of rows, skip processing
        if self.start_row >= total_rows:
            logging.info("No new data to process. All data has already been processed.")
            return rows_to_drop  # Return an empty DataFrame indicating no rows to delete

        # Process the data in batches
        # for batch in pd.read_csv(self.rawdata_file_path, chunksize=self.batch_size):
        for batch in pd.read_csv(self.rawdata_file_path, chunksize=self.batch_size, skiprows=self.start_row):
            logging.info(f"\nProcessing batch {batch_number}...")

            # Reset index of the batch before processing
            batch = batch.reset_index(drop=True)
            
            # Call process_batch to handle the current batch
            cleaned_batch = self.process_batch(batch)          

            # Update progress
            self.start_row += len(batch)
            self.save_progress()

            batch_number += 1

        # Merge all rows to be deleted
        # logging.info(f"Total rows dropped: {len(rows_to_drop)}")
        # logging.info(f"Total rows processed: {self.start_row}")
        logging.info(f"Total rows perpared to dropped: {len(rows_to_drop)}")
        logging.info(f"Total rows perpared to processed: {self.start_row}")
        return rows_to_drop


    def clean_data(self, output_file_path):
        """Perform data cleaning and save the results"""
        logging.info("Starting data cleaning process...")
        batch_number = 1  # Batch number
    
        rows_to_drop_indices = set()  # To store indices of all rows to be dropped
    
        # Process the data in batches
        for batch in pd.read_csv(self.rawdata_file_path, chunksize=self.batch_size, skiprows=self.start_row):
            logging.info(f"Processing batch {batch_number}...")
    
            # Check for missing values
            missing_rows = self.check_missing_values(batch)
            rows_to_drop_indices.update(missing_rows.index)
    
            # Check for duplicate rows
            duplicate_rows = self.check_duplicate_rows(batch)
            rows_to_drop_indices.update(duplicate_rows.index)
    
            # Logical validation
            invalid_rows = self.logical_validation(batch)
            rows_to_drop_indices.update(invalid_rows.index)

            # Range validation
            invalid_rows = self.validate_data_ranges(batch)
            rows_to_drop_indices.update(invalid_rows.index)

            # Uniqueness validation
            invalid_rows = self.verify_record_number_unique(batch)
            rows_to_drop_indices.update(invalid_rows.index)

            # Drop rows to be deleted in the current batch
            cleaned_batch = batch.drop(index=rows_to_drop_indices, errors='ignore').reset_index(drop=True)
    
            # If 'record_number' column exists, reset sorting
            if 'record_number' in cleaned_batch.columns:
                cleaned_batch['record_number'] = range(1, len(cleaned_batch) + 1)
                logging.info("'record_number' column reset.")
            else:
                logging.warning("'record_number' column not found. No reset performed.")
    
            # Save the cleaned batch data
            mode = 'w' if batch_number == 1 else 'a'
            header = batch_number == 1
            cleaned_batch.to_csv(output_file_path, index=False, mode=mode, header=header)
            
            logging.info(f"Batch {batch_number} cleaned and saved.")
            self.start_row += len(batch)
            self.save_progress()
            batch_number += 1
    
        logging.info(f"Data cleaning process completed. Total rows processed: {self.start_row}")
        logging.info(f"Rows to drop: {len(rows_to_drop_indices)}")
        logging.info(f"Cleaned data saved to {output_file_path}")
    
        # Clean up the progress file, indicating that the processing is complete
        if os.path.exists("progress.json"):
            os.remove("progress.json")
            logging.info("Progress file removed. Data processing complete.")
        
        # Call validation script to validate the cleaned data
        subprocess.run(['python', 'data_validator.py', output_file_path])
        logging.info("Data clean code test passed. Congratulations! Result deatils in log/data_cleaning.log")


if __name__ == "__main__":
    # Ensure three command-line arguments: input file, variable file, and output file
    if len(sys.argv) != 6:
        print("Usage: python data_cleaner.py <input_csv_file> <variable_json_file> <output_csv_file> <progress_file> <batch_size>")
        sys.exit(1)

    input_file = sys.argv[1]  # Input file
    labels_file = sys.argv[2]  # Labels file
    output_file = sys.argv[3]  # Output file
    progress_file = sys.argv[4]  # Progress file
    batch_size = int(sys.argv[5])  # Batch size

    # Create a DataCleaner instance with the provided input file and labels file
    cleaner = DataCleaner(input_file, labels_file)

    # Initialize progress before processing batches
    cleaner.load_progress(progress_file=progress_file)
    
    # Process data in batches and call the respective check methods within the loop
    for batch in pd.read_csv(input_file, chunksize=cleaner.batch_size, skiprows=cleaner.start_row):
        # Call check methods for each batch
        cleaner.check_type_conversion(batch)
        cleaner.check_missing_values(batch)
        cleaner.check_duplicate_rows(batch)
        cleaner.verify_record_number_unique(batch)
        cleaner.validate_data_ranges(batch)
   
    
        # Clean the data using the batch processing approach and save to output file
        cleaner.clean_data(output_file)


