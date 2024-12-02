============Project Overview============
This project is designed to use relative paths to ensure portability across different environments. 
This project is designed to perform data cleaning in a batch-processing manner using Python and shell scripts. The directory structure, execution steps, and requirements are outlined below to help users efficiently clean and validate large datasets.

============Directory Structure============
code/: Contains the scripts for data cleaning, and data validation, including Python and shell scripts.
data/: Stores the raw datasets, cleaned output files, cleaning processing files, test files, and variable json files.
log/: Contains log files generated during the data cleaning process.
notebook/: Includes Jupyter notebooks for data visualization and reporting.

============How to Use============
1/Prerequisites
To run the script directly.
Ensure Python and the required libraries are installed before running the scripts. To install necessary packages, run: 
pip install pandas numpy matplotlib ipywidgets  

2/Running the Data Cleaning Script
Option 1: Direct Python Execution
Navigate to the code/ directory and run the following command:
python data_cleaner.py ../data/Scotland_teaching_file_1PCT.csv ../data/data_dictionary.json ../data/output.csv progress.json 30000  
(30000: Number of rows processed per batch.)

Option 2: Using Shell Script
Run the shell script: 
./run_data_cleaner.sh  
Ensure the input file in run_data_cleaner.sh is set to the correct dataset before execution:
INPUT_FILE="../data/Scotland_teaching_file_1PCT.csv"  # Replace this for testing  

============Key Features============
Batch Processing: Set yourself the number of rows per batch to process large datasets, e.g. 30,000 rows.
Progress tracking: Progress is saved in progress.json to recover from the last processed row in case of interruption.
Output saving: Avoid overwriting completed data. Make sure output.csv is saved before restarting.
Validation Rules: Implement 11 validation rules to clean and verify data integrity for reusability.
Automated Testing: Perform automated functional tests after cleaning.

============Logs============
The entire cleaning process is logged in data_cleaning.log located in the log/ directory for traceability.

============Testing============
To run the script with test data in data/ file, replace the INPUT_FILE in run_data_cleaner.sh with the test file path:
INPUT_FILE="../data/Scotland_teaching_file_1PCT_For_Test.csv"  
 
============Future Enhancements============
Modularization: Further improve modularity of the data processing and testing logic.
Enhanced Logging: Increase the readability and clarity of log messages.
Documentation: Expand and improve this README with more detailed explanations following official standards.

============Requirements============
Python 3.x
Libraries: pandas, numpy, matplotlib, ipywidgets
To install required packages:
pip install pandas numpy matplotlib ipywidgets  

============Execution Commands============
Direct Python:
python refined_dataset.py <input_file> <labels_file> <output_file> <progress_file> <batch_size>  

Shell Script:
./run_data_cleaner.sh  

