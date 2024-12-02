#!/bin/bash
# INPUT_FILE, LABELS_FILE, OUTPUT_FILE, PROGRESS_FILE, and BATCH_SIZE are variables that specify the input file path, labels file path, output file path, the path to the file where progress will be saved, and the number of rows processed per batch.
# python3 data_cleaner.py: Executes a Python script passing the above parameters. This command starts data_cleaner.py and performs the processing of the CSV file according to the path in the parameters.


# Set variables: input file, labels file, output file, progress file
INPUT_FILE="../data/Scotland_teaching_file_1PCT.csv"  # Path to the input CSV file
# INPUT_FILE="../data/Scotland_teaching_file_1PCT_For_Test.csv"  # Test file
LABELS_FILE="../data/data_dictionary.json"  # Path to the labels file
OUTPUT_FILE="../data/output.csv"  # Path to the output CSV file
PROGRESS_FILE="progress.json"  # Path to the file for saving progress

# Set the batch size
BATCH_SIZE=30000

# Start the Python script
echo "Starting data cleaning process..."

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python is not installed. Please install Python first."
    exit 1
fi

# Run the Python script, passing parameters: input file, labels file, output file, and batch size
python3 data_cleaner.py "$INPUT_FILE" "$LABELS_FILE" "$OUTPUT_FILE" "$PROGRESS_FILE" "$BATCH_SIZE"

echo "Data cleaning process completed!"



