import pandas as pd
import sys
import json
from collections import Counter
from IPython.display import display
import numpy as np


class DataCleaner:
    def __init__(self, rawdata_file_path, variable_file):
        

        # Read the JSON file containing labels and convert all keys to lowercase
        with open(variable_file, 'r') as f:
            self.labels = {key.lower(): value for key, value in json.load(f).items()}

        self.validation_errors = {}
        self.category_counts = Counter()
        self.unique_categories = {}
        
        # Read the CSV file into a DataFrame
        self.df = pd.read_csv(rawdata_file_path)

        # Convert column names to lowercase
        self.df.columns = self.df.columns.str.lower()



    # def load_checkpoint(self):
    #     """Load the last processed chunk index."""
    #     if os.path.exists(self.checkpoint_path):
    #         with open(self.checkpoint_path, 'r') as f:
    #             return int(f.read().strip())
    #     return 0

    # def save_checkpoint(self, chunk_index):
    #     """Save the current chunk index as a checkpoint."""
    #     with open(self.checkpoint_path, 'w') as f:
    #         f.write(str(chunk_index))

    # def process_chunk(self, chunk):
    #     """Process a single chunk."""
    #     chunk.columns = chunk.columns.str.lower()  # Convert columns to lowercase
    #     self.df = chunk

    #     # Run checks on the chunk
    #     self.check_type_conversion()
    #     self.check_missing_values()
    #     self.check_duplicate_rows()
    #     self.verify_record_number_unique()
    #     self.validate_data_ranges()
        

    # def process_file_in_chunks(self):
    #     """Read and process the CSV file in chunks."""
    #     start_chunk = self.load_checkpoint()
    #     print(f"Resuming from chunk {start_chunk}...")

    #     for chunk_index, chunk in enumerate(pd.read_csv(self.rawdata_file_path, chunksize=self.chunksize)):
    #         if chunk_index < start_chunk:
    #             continue  # Skip already processed chunks

    #         print(f"Processing chunk {chunk_index}...")
    #         self.process_chunk(chunk)
    #         self.save_checkpoint(chunk_index)  # Save progress
    #         print(f"Chunk {chunk_index} processed and checkpoint saved.")
        
    #     print("File processing completed.")
    #     os.remove(self.checkpoint_path)  # Remove checkpoint file after successful completion

    
    def check_type_conversion(self):
        """Check the data types of columns in the dataset and assert that each column contains only string or numeric values."""
        print("Checking data types:")
        print(self.df.info())

        # Check if each value in the column is either a string or numeric
        for column in self.df.columns:
            for index, value in self.df[column].items():
                try:
                    # Check if the value is not a string, int or float
                    assert isinstance(value, (str, int, float)), (
                        f"Column '{column}' has a value that is neither a string nor a number. "
                        f"Invalid value {value} at index {index} (Type: {type(value)})"
                    )
                except AssertionError as e:
                    # If assertion fails, print the error and continue
                    print(f"Failed! {e}")
                    # Optionally log or handle the invalid value
                    continue  


    def check_missing_values(self):
        """Check if there are any missing values in the dataset and output detailed information in a table."""
        missing_values = self.df.isnull().sum()
        
        try:
            assert missing_values.sum() == 0, f"missing values detected:\n{missing_values}"
            print("No missing values detected.")
        except AssertionError as e:
            print(f"Failed! {e}")
            
            # 输出每列的缺失值位置
            print("\nColumn missing value locations:")
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    print(f"{col}: index{self.df[self.df[col].isnull()].index.tolist()}")
    
            # 筛选出包含缺失值的行
            missing_rows = self.df[self.df.isnull().any(axis=1)]
            
            # 使用 display() 确保表格格式正确
            print("\nRows with missing values:")
            display(missing_rows) 


    def check_duplicate_rows(self):
        """Check for duplicate rows in the dataset."""
        duplicate_rows = self.df.duplicated().sum()
        try:
            assert duplicate_rows == 0, f"{duplicate_rows} duplicate rows detected."
            print("No duplicate rows detected.")
        except AssertionError as e:
            print(f"Failed! {e}")

            # Display duplicate rows in a table format
            print("\nDuplicate rows:")
            duplicate_data = self.df[self.df.duplicated()]
            display(duplicate_data)  


    def verify_record_number_unique(self):
        """Ensure the 'record_number' is unique."""
        try:
            # Assert that 'record_number' column exists
            assert 'record_number' in self.df.columns, "'record_number' column not found in the dataset."
            
            # Assert that 'record_number' values are unique
            assert self.df['record_number'].is_unique, (
                f"'record_number' is not unique. "
                f"Duplicate record numbers: {self.df['record_number'][self.df['record_number'].duplicated()].tolist()}"
            )
            print("Record_Number is unique.")
        
        except AssertionError as e:
            print(f"Failed! {e}")
            
            # Display the rows with duplicate 'record_number' values
            print("\nDuplicate rows:")
            dulpicate_record_number = self.df[self.df['record_number'].duplicated()]
            display(dulpicate_record_number)  

    
    def validate_data_ranges(self):
        """Validate each column’s values based on the JSON-defined valid ranges and display broken records."""
        has_errors = False  # 标志变量，用于跟踪是否有错误
    
        for column in self.df.columns:
            if column in self.labels:
                valid_values = list(self.labels[column].keys())
            
                # Store unique values using a set
                self.unique_categories[column] = set(self.df[column].dropna().unique())
            
                # Count occurrences of each category using Counter
                self.category_counts.update(self.df[column].dropna().astype(str))
            
                # Identify invalid values
                invalid_values = self.df[~self.df[column].astype(str).str.lower().isin(
                    [v.lower() for v in valid_values])]
                    
                if not invalid_values.empty:
                    has_errors = True
                    print(f"\nBroken records in column '{column}':")
                    display(invalid_values[[column]])  # Display invalid records in a table format
            
                    # Use try-except to handle assertion
                    try:
                        assert invalid_values.empty, (
                            f"Column '{column}' contains invalid values: "
                            f"{invalid_values[column].unique().tolist()}"
                        )
                    except AssertionError as e:
                        print(f"Failed! {e}")
            
        # Print unique categories and their counts
        print("\nUnique categories and counts per column:")
        for column, unique_values in self.unique_categories.items():    
            unique_values = [int(v) if isinstance(v, np.integer) else v for v in unique_values]
            if all(isinstance(v, (int, float)) for v in unique_values):
                unique_values = sorted(unique_values)  # Sort integers for cleaner output
                print(f"{column}: {unique_values} (Total unique: {len(unique_values)})")
            else:
                print(f"{column}: {unique_values} (Total unique: {len(unique_values)})")
            
        print("\nCategory counts across columns:")
        for category, count in self.category_counts.items():
            print(f"{category}: {count} occurrences")
        
        # 在所有列验证完成且无错误时打印验证成功消息
        if not has_errors:
            print("\nAll data ranges validated.")

    def logical_validation(self):
        """Apply logical rules to identify rows that need to be deleted."""
        drop_data = pd.DataFrame()

        # Rule 1: When Student = 1, age must be 1, and Economic_Activity must be '4'
        drop_data1 = self.df[(self.df['student'] == 1) & ~((self.df['age'] == 1) | (self.df['economic_activity'] == '4'))]
        if not drop_data1.empty:
            drop_data = pd.concat([drop_data, drop_data1])
            drop_data1.to_csv('../data/rule_1_deleted_data.csv', index=False)
            print(f"Rule 1: Found {len(drop_data1)} rows to delete, saved to data/rule_1_deleted_data.csv")
        else:
            print("Rule 1: No rows to delete")

        # Rule 2: When Family_Composition = 'X', Residence_Type must be 'C'
        drop_data2 = self.df[((self.df['family_composition'] == 'X') & (self.df['residence_type'] != 'C'))]
        if not drop_data2.empty:
            drop_data = pd.concat([drop_data, drop_data2])
            drop_data2.to_csv('../data/rule_2_deleted_data.csv', index=False)
            print(f"Rule 2: Found {len(drop_data2)} rows to delete, saved to data/rule_2_deleted_data.csv")
        else:
            print("Rule 2: No rows to delete")

        # Rule 3: When Occupation = 'X', Economic_Activity must be in [6, 7, 8, 9, X]
        drop_data3 = self.df[(self.df['occupation'] == 'X') & (~self.df['economic_activity'].isin(['6', '7', '8', '9', 'X']))]
        if not drop_data3.empty:
            drop_data = pd.concat([drop_data, drop_data3])
            drop_data3.to_csv('../data/rule_3_deleted_data.csv', index=False)
            print(f"Rule 3: Found {len(drop_data3)} rows to delete, saved to data/rule_3_deleted_data.csv")
        else:
            print("Rule 3: No rows to delete")

        # Rule 4: When Hours_Worked_Per_Week = 'X', Economic_Activity must be in [5, 6, 7, 8, 9, X] and age = 1
        drop_data4 = self.df[(self.df['hours_worked_per_week'] == 'X') &
                             ~(self.df['economic_activity'].isin(['5', '6', '7', '8', '9', 'X'])) &
                             ~(self.df['age'] == 1)]
        if not drop_data4.empty:
            drop_data = pd.concat([drop_data, drop_data4])
            drop_data4.to_csv('../data/rule_4_deleted_data.csv', index=False)
            print(f"Rule 4: Found {len(drop_data4)} rows to delete, saved to data/rule_4_deleted_data.csv")
        else:
            print("Rule 4: No rows to delete")

        # Rule 5: When Approximate_Social_Grade = 'X', Residence_Type must be 'C' and age = 1
        drop_data5 = self.df[(self.df['approximate_social_grade'] == 'X') &
                             (self.df['residence_type'] == 'P') & ~(self.df['age'] == 1)]
        if not drop_data5.empty:
            drop_data = pd.concat([drop_data, drop_data5])
            drop_data5.to_csv('../data/rule_5_deleted_data.csv', index=False)
            print(f"Rule 5: Found {len(drop_data5)} rows to delete, saved to data/rule_5_deleted_data.csv")
        else:
            print("Rule 5: No rows to delete")

        # Rule 6: When Industry = 'X', Economic_Activity must be in [6, 7, 8, 9, X]
        drop_data6 = self.df[(self.df['industry'] == 'X') & (~self.df['economic_activity'].isin(['6', '7', '8', '9', 'X']))]
        if not drop_data6.empty:
            drop_data = pd.concat([drop_data, drop_data6])
            drop_data6.to_csv('../data/rule_6_deleted_data.csv', index=False)
            print(f"Rule 6: Found {len(drop_data6)} rows to delete, saved to data/rule_6_deleted_data.csv")
        else:
            print("Rule 6: No rows to delete")

        # Remove duplicate rows from the collected data
        drop_data = drop_data.drop_duplicates()

        # Return rows that need to be deleted
        return drop_data

    def clean_data(self, output_file_path):
        """Perform logical validation and drop rows that don't meet the criteria."""
        rows_to_drop = self.logical_validation()
        print(f"Total rows dropped: {len(rows_to_drop)}")

        # Remove the rows from the original dataset that need to be dropped
        cleaned_df = self.df[~self.df.index.isin(rows_to_drop.index)]

        # Reset index for the cleaned dataset
        cleaned_df = cleaned_df.reset_index(drop=True)

        # Save the cleaned data to a CSV file
        cleaned_df.to_csv(output_file_path, index=False)
        print(f"Cleaned data saved to {output_file_path}")


if __name__ == "__main__":
    # Ensure three command-line arguments: input file, variable file, and output file
    if len(sys.argv) != 4:
        print("Usage: python data_cleaner.py <input_csv_file> <variable_json_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    labels_file = sys.argv[2]
    output_file = sys.argv[3]

    # Create a DataCleaner instance and call methods for data validation and cleaning
    cleaner = DataCleaner(input_file, labels_file)
    cleaner.check_type_conversion()
    cleaner.check_missing_values()
    cleaner.check_duplicate_rows()
    cleaner.verify_record_number_unique()
    cleaner.validate_data_ranges()
    cleaner.clean_data(output_file)

