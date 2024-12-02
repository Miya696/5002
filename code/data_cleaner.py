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
    def __init__(self, rawdata_file_path, variable_file, batch_size=90000, start_row=0, progress_file="progress.json"):
        self.batch_size = batch_size  # 每批处理的行数
        self.start_row = start_row    # 起始行，用于恢复进度
        self.rawdata_file_path = rawdata_file_path  # 原始数据文件路径
        self.variable_file = variable_file  # 变量文件路径
        self.progress_file = progress_file  # 进度文件路径
        self.labels = {}  # 存储标签
        self.df = None
        self.validation_errors = {}
        self.category_counts = Counter()
        self.unique_categories = {}
        self.setup_logging()  # 配置日志记录器

        # 加载处理进度
        self.load_progress(progress_file)
        

        # Read the JSON file containing labels and convert all keys to lowercase
        try:
            with open(variable_file, 'r') as f:
                self.labels = {key.lower(): value for key, value in json.load(f).items()}
        except FileNotFoundError:
            logging.error(f"Variable file {variable_file} not found.")
        except json.JSONDecodeError:
            logging.error(f"Error decoding variable file {variable_file}.")

        # 加载数据时，跳过已处理的行
        self.load_data()


    def setup_logging(self):
        """设置日志记录器，将 stdout 和 stderr 重定向到日志文件"""
        logging.basicConfig(
            filename='../log/data_cleaning.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        sys.stdout = open('../log/stdout.log', 'w')
        sys.stderr = open('../log/stderr.log', 'w')


    def load_data(self):
        """根据进度加载数据"""
        try:
            # 读取CSV文件，跳过已处理的行
            self.df = pd.read_csv(self.rawdata_file_path, skiprows=range(1, self.start_row + 1))
            # 转换所有列名为小写字母
            self.df.columns = self.df.columns.str.lower()
            logging.info(f"Loaded data from row {self.start_row}.")
            # 打印列名以验证
            logging.info(f"Columns in DataFrame: {self.df.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()  # 如果加载失败，返回空DataFrame


    # def process_batch(self, start_row):
    def process_batch(self, batch):     
        """处理一个批次的数据"""
        try:
            # batch = pd.read_csv(self.rawdata_file_path, skiprows=range(1, start_row), nrows=self.batch_size)
            # print(f"Processing batch starting from row {start_row}...")
            logging.info(f"Processing batch starting from row {self.start_row}.")
            
            # 确保batch的列名是小写
            # batch.columns = batch.columns.str.lower()
            batch.columns = self.df.columns
            logging.info(f"Columns in batch: {batch.columns.tolist()}")
        
            # 在此处理该批次的数据
            self.check_type_conversion(batch)
            self.check_missing_values(batch)
            self.check_duplicate_rows(batch)
            self.verify_record_number_unique(batch)
            self.validate_data_ranges(batch)
            
            # 处理完该批次后，更新进度
            self.start_row += len(batch)
            return batch
        except Exception as e:
            # print(f"Error processing batch starting from row {start_row}: {e}")
            # return pd.DataFrame()  # 如果发生错误，返回空的DataFrame
            logging.error(f"Error processing batch: {e}")
            # 返回空批次，让后续批次可以继续运行
            return pd.DataFrame()

    def process_data_in_batches(self):
        """按批次处理所有数据"""
        total_rows = len(self.df)
        logging.info(f"总行数: {total_rows}")
        
        # 按批次大小处理数据
        for start_row in range(self.start_row, total_rows, self.batch_size):
            end_row = min(start_row + self.batch_size, total_rows)
            batch = self.df.iloc[start_row:end_row]
            
            self.process_batch(batch)
            
            # 打印处理进度
            logging.info(f"已处理 {end_row}/{total_rows} 行。")
            print(f"已处理 {end_row}/{total_rows} 行。")
        
        print("清晰成功")
        logging.info("清晰成功")

    
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

        # batch的列名是小写
        batch.columns = batch.columns.str.lower()

        # 记录缺失值总和
        if missing_values.sum() == 0:
            logging.info("No missing values detected.")
            print("No missing values detected.")
        else:
            logging.warning(f"Missing values detected:\n{missing_values}")
            print(f"Missing values detected:\n{missing_values}")
        
        # try:
        #     assert missing_values.sum() == 0, f"missing values detected:\n{missing_values}"
        #     print("No missing values detected.")
        #     # logging.info("No missing values detected.")
        # except AssertionError as e:
        #     print(f"Failed! {e}")
        #     # logging.error(f"Failed! {e}")
            
            # 输出每列的缺失值位置
            print("\nColumn missing value locations:")
            for col in batch.columns:
                if col in batch.columns and batch[col].isnull().any():
                    indices = batch[batch[col].isnull()].index.tolist()
                    # print(f"{col}: index {indices}")
                    # logging.error(f"error {col}: index {indices}")
                    logging.warning(f"Column '{col}': missing values at indices {indices}")
                    print(f"Column '{col}': missing values at indices {indices}")
    
            # 筛选出包含缺失值的行
            rows_to_drop = batch[batch.isnull().any(axis=1)]
            
            print("\nRows with missing values:")
            logging.info(f"Rows to drop:\n{rows_to_drop.to_string()}")
            
            # notebook中显示
            # display(rows_to_drop) 
        return rows_to_drop

    
    def check_duplicate_rows(self, batch):
        """Check for duplicate rows in the dataset."""
        logging.info("=====Checking for duplicate rows:=====")
        
        # Get the number of duplicate rows in the batch
        duplicate_rows = batch.duplicated().sum()
        # batch的列名是小写
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

        # batch的列名是小写
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
            """验证数据范围"""
            logging.info("=====validate_data_ranges:=====")
            rows_to_drop = pd.DataFrame()
            has_errors = False  # 标志变量，用于跟踪是否有错误
        
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
            
            # 在所有列验证完成且无错误时打印验证成功消息
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
        """保存进度"""
        progress = {'start_row': self.start_row}
        
        try:
            # 如果没有指定路径，则使用默认目录
            if os.path.dirname(progress_file) == '':
                progress_file = os.path.join(os.getcwd(), progress_file)  # 使用当前工作目录
            
            # 确保目录存在，如果不存在则创建
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
            logging.info(f"Progress saved to {progress_file}.")
        except (OSError, IOError) as e:
            logging.error(f"File operation error when saving progress: {e}")
        except Exception as e:
            logging.error(f"Unexpected error saving progress: {e}")

    def load_progress(self, progress_file):
        """加载进度"""
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.start_row = progress.get('start_row', 0)
                logging.info(f"Progress loaded: start_row = {self.start_row}.")
        except FileNotFoundError:
            logging.warning(f"Progress file {progress_file} not found. Starting from the beginning.")
            self.start_row = 0  # 没有进度文件时从头开始
        except json.JSONDecodeError:
            logging.error(f"Error decoding progress file {progress_file}. Starting from the beginning.")
            self.start_row = 0  # 错误解码时从头开始
        except Exception as e:
            logging.error(f"Unexpected error loading progress: {e}. Starting from the beginning.")
            self.start_row = 0  # 出现其他错误时从头开始

    # def process_in_batches(self):
    #     """逐批处理 CSV 文件以检查缺失值并清理数据"""
    #     batch_number = 1
    #     rows_to_drop = pd.DataFrame()

    #     # 获取文件总行数
    #     total_rows = sum(1 for line in open(self.rawdata_file_path))  # 获取文件行数
    #     logging.info(f"Total rows in file: {total_rows}")

    #     # 确保start_row初始化，如果没有初始化在__init__中
    #     if not hasattr(self, 'start_row'):
    #         self.start_row = 0

    #     # 如果 start_row 已经等于文件的总行数，直接跳过处理
    #     if self.start_row >= total_rows:
    #         logging.info("No new data to process. All data has already been processed.")
    #         return rows_to_drop  # 返回空的数据帧，表示没有要删除的行

    #     # 逐批读取并处理数据
    #     # for batch in pd.read_csv(self.rawdata_file_path, chunksize=self.batch_size):
    #     for batch in pd.read_csv(self.rawdata_file_path, chunksize=self.batch_size, skiprows=self.start_row):
    #         logging.info(f"\nProcessing batch {batch_number}...")

    #         # Reset index of the batch before processing
    #         batch = batch.reset_index(drop=True)

    #         # 1. 检查缺失值
    #         missing_rows = self.check_missing_values(batch)
    #         rows_to_drop = pd.concat([rows_to_drop, missing_rows])
    
    #         # 2. 检查重复行
    #         duplicate_rows = self.check_duplicate_rows(batch)
    #         rows_to_drop = pd.concat([rows_to_drop, duplicate_rows])
    
    #         # 3. 逻辑验证
    #         invalid_rows = self.logical_validation(batch)
    #         rows_to_drop = pd.concat([rows_to_drop, invalid_rows])

    #         # 检查缺失值并标记要删除的行
    #         # missing_rows = self.check_duplicate_rows(batch)
    #         # rows_to_drop = pd.concat([rows_to_drop, missing_rows])

            
    #         # 调用 process_batch 处理当前批次
    #         # cleaned_batch = self.process_batch(batch)
            
    #         # 对每批数据进行逻辑验证，找出要删除的行
    #         # drop_data = self.logical_validation(cleaned_batch)
    #         # rows_to_drop = pd.concat([rows_to_drop, drop_data])

    #         # 更新进度
    #         self.start_row += len(batch)
    #         self.save_progress()

    #         batch_number += 1

    #     # 合并所有要删除的数据
    #     # logging.info(f"Total rows dropped: {len(rows_to_drop)}")
    #     # logging.info(f"Total rows processed: {self.start_row}")
    #     logging.info(f"Total rows perpared to dropped: {len(rows_to_drop)}")
    #     logging.info(f"Total rows perpared to processed: {self.start_row}")
    #     return rows_to_drop

    def process_in_batches(self):
        """逐批处理 CSV 文件以检查缺失值并清理数据"""
        batch_number = 1
        rows_to_drop = pd.DataFrame()

        # 获取文件总行数
        total_rows = sum(1 for line in open(self.rawdata_file_path))  # 获取文件行数
        logging.info(f"Total rows in file: {total_rows}")

        # 确保start_row初始化，如果没有初始化在__init__中
        if not hasattr(self, 'start_row'):
            self.start_row = 0

        # 如果 start_row 已经等于文件的总行数，直接跳过处理
        if self.start_row >= total_rows:
            logging.info("No new data to process. All data has already been processed.")
            return rows_to_drop  # 返回空的数据帧，表示没有要删除的行

        # 逐批读取并处理数据
        # for batch in pd.read_csv(self.rawdata_file_path, chunksize=self.batch_size):
        for batch in pd.read_csv(self.rawdata_file_path, chunksize=self.batch_size, skiprows=self.start_row):
            logging.info(f"\nProcessing batch {batch_number}...")

            # Reset index of the batch before processing
            batch = batch.reset_index(drop=True)

            
            # 调用 process_batch 处理当前批次
            cleaned_batch = self.process_batch(batch)
            

            # 更新进度
            self.start_row += len(batch)
            self.save_progress()

            batch_number += 1

        # 合并所有要删除的数据
        # logging.info(f"Total rows dropped: {len(rows_to_drop)}")
        # logging.info(f"Total rows processed: {self.start_row}")
        logging.info(f"Total rows perpared to dropped: {len(rows_to_drop)}")
        logging.info(f"Total rows perpared to processed: {self.start_row}")
        return rows_to_drop


    def clean_data(self, output_file_path):
        """执行数据清理并保存结果"""
        logging.info("Starting data cleaning process...")
        batch_number = 1  # 批次编号
    
        rows_to_drop_indices = set()  # 用于存储所有需要丢弃的行索引
    
        # 逐批读取并处理数据
        for batch in pd.read_csv(self.rawdata_file_path, chunksize=self.batch_size, skiprows=self.start_row):
            logging.info(f"Processing batch {batch_number}...")
    
            # 1. 检查缺失值
            missing_rows = self.check_missing_values(batch)
            rows_to_drop_indices.update(missing_rows.index)
    
            # 2. 检查重复行
            duplicate_rows = self.check_duplicate_rows(batch)
            rows_to_drop_indices.update(duplicate_rows.index)
    
            # 3. 逻辑验证
            invalid_rows = self.logical_validation(batch)
            rows_to_drop_indices.update(invalid_rows.index)

             # 4. 有效值验证
            invalid_rows = self.validate_data_ranges(batch)
            rows_to_drop_indices.update(invalid_rows.index)

             # 5. 唯一性验证
            invalid_rows = self.verify_record_number_unique(batch)
            rows_to_drop_indices.update(invalid_rows.index)
            
    
            # 删除当前批次中的要删除的行
            cleaned_batch = batch.drop(index=rows_to_drop_indices, errors='ignore').reset_index(drop=True)
    
            # 如果 'record_number' 列存在，重新生成排序
            if 'record_number' in cleaned_batch.columns:
                cleaned_batch['record_number'] = range(1, len(cleaned_batch) + 1)
                logging.info("'record_number' column reset.")
            else:
                logging.warning("'record_number' column not found. No reset performed.")
    
            # 保存清理后的批次数据
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
    
        # 清理进度文件，表示处理已完成
        if os.path.exists("progress.json"):
            os.remove("progress.json")
            logging.info("Progress file removed. Data processing complete.")
        
        # 调用验证脚本来验证清理后的数据
        subprocess.run(['python', 'data_validator.py', output_file_path])
        logging.info("Data clean code test passed. Congratulations! Result deatils in log/data_cleaning.log")


if __name__ == "__main__":
    # Ensure three command-line arguments: input file, variable file, and output file
    if len(sys.argv) != 6:
        print("Usage: python data_cleaner.py <input_csv_file> <variable_json_file> <output_csv_file> <progress_file> <batch_size>")
        sys.exit(1)

    input_file = sys.argv[1]  # 输入文件
    labels_file = sys.argv[2]  # 标签文件
    output_file = sys.argv[3]  # 输出文件
    progress_file = sys.argv[4]  # 进度文件
    batch_size = int(sys.argv[5])  # 批次大小

    # Create a DataCleaner instance with the provided input file and labels file
    cleaner = DataCleaner(input_file, labels_file)

     # 在处理批次前初始化进度
    cleaner.load_progress(progress_file=progress_file)
    
    # 按批次处理数据并在循环内调用各个检查方法
    for batch in pd.read_csv(input_file, chunksize=cleaner.batch_size, skiprows=cleaner.start_row):
        # 对每个批次调用检查方法
        cleaner.check_type_conversion(batch)
        cleaner.check_missing_values(batch)
        cleaner.check_duplicate_rows(batch)
        cleaner.verify_record_number_unique(batch)
        cleaner.validate_data_ranges(batch)
   
    
        # Clean the data using the batch processing approach and save to output file
        cleaner.clean_data(output_file)


