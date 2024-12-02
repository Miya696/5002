#!/bin/bash

# 设置变量：输入文件、标签文件、输出文件、进度文件
INPUT_FILE="../data/Scotland_teaching_file_1PCT_For_Test.csv"   # 输入的 CSV 文件路径
LABELS_FILE="../data/data_dictionary.json"   # 标签文件路径
OUTPUT_FILE="../data/output.csv"    # 输出的 CSV 文件路径
PROGRESS_FILE="progress.json"   # 保存进度的文件路径

# 设置每批次的大小
BATCH_SIZE=30000

# 启动 Python 脚本
echo "Starting data cleaning process..."

# 检查 Python 环境是否安装
if ! command -v python3 &> /dev/null
then
    echo "Python is not installed. Please install Python first."
    exit 1
fi

# 运行 Python 脚本，传递参数：输入文件、标签文件、输出文件和批次大小
python3 data_cleaner.py "$INPUT_FILE" "$LABELS_FILE" "$OUTPUT_FILE" "$PROGRESS_FILE" "$BATCH_SIZE"

echo "Data cleaning process completed!"



