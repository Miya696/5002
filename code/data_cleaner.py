import pandas as pd
import sys
import json


class DataCleaner:
    def __init__(self, rawdata_file_path, variable_file):
        # Read the CSV file into a DataFrame
        self.df = pd.read_csv(rawdata_file_path)

        # Convert column names to lowercase
        self.df.columns = self.df.columns.str.lower()

        # Read the JSON file containing labels and convert all keys to lowercase
        with open(variable_file, 'r') as f:
            self.labels = {key.lower(): value for key, value in json.load(f).items()}

        self.validation_errors = {}

    def check_type_conversion(self):
        """Check the data types of columns in the DataFrame."""
        print("Checking data types:")
        print(self.df.info())

    def check_missing_values(self):
        """Check if there are any missing values in the DataFrame."""
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            print("Missing values detected:")
            print(missing_values)
        else:
            print("No missing values detected.")

    def check_duplicate_rows(self):
        """Check for duplicate rows in the DataFrame."""
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            print(f"Duplicate rows detected: {duplicate_rows}")
        else:
            print("No duplicate rows detected.")

    def verify_record_number_unique(self):
        """Ensure the 'record_number' column exists and is unique."""
        if 'record_number' not in self.df.columns:
            print("Error: 'record_number' column not found in the dataset.")
            return

        if not self.df['record_number'].is_unique:
            print("Record_Number is not unique.")
            duplicate_records = self.df['record_number'][self.df['record_number'].duplicated()]
            print("Problematic record numbers:", duplicate_records.tolist())
        else:
            print("Record_Number is unique.")

    def validate_data_ranges(self):
        """Validate each column’s values based on the JSON-defined valid ranges."""
        for column in self.df.columns:
            if column in self.labels:
                valid_values = list(self.labels[column].keys())
                invalid_values = self.df[~self.df[column].astype(str).str.lower().isin(
                    [v.lower() for v in valid_values])]
                if not invalid_values.empty:
                    self.validation_errors[column] = invalid_values[column].unique().tolist()

        if self.validation_errors:
            print("Validation errors detected:")
            print(self.validation_errors)
        else:
            print("All data ranges are valid.")

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

