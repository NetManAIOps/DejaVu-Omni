import os
import re
import numpy as np
import sys

def parse_log_line(line):
    # 使用正则表达式匹配"command output one-line summary:" 后的结果
    match = re.search(r'command output one-line summary: (.*)', line)
    if match:
        values = match.group(1).split(',')
        # 尝试将每个值转换为浮点数，非数值的内容保留为 None
        parsed_values = []
        for v in values:
            try:
                parsed_values.append(float(v))
            except ValueError:
                parsed_values.append(None)
        return parsed_values
    return None

def average_logs(log_path):
    results = []
    all_logs = []  # 用于存储每个log的解析结果
    
    # 遍历路径中的所有实验目录
    for root, dirs, _ in os.walk(log_path):
        for dir_name in dirs:
            log_file_path = os.path.join(root, dir_name, 'log')
            if os.path.exists(log_file_path):  # 确保log文件存在
                with open(log_file_path, 'r') as f:
                    for line in f:
                        values = parse_log_line(line)
                        if values is not None:
                            results.append(values)
                            all_logs.append(values)
                            break  # 每个log文件只处理一个command output one-line summary
    
    # 转置结果以计算每一列的平均值
    results_array = np.array(results, dtype=object).T  # 使用object以支持None值
    averages = []

    for column in results_array:
        # 忽略None值，计算有效数值的平均值
        valid_numbers = [num for num in column if num is not None]
        if valid_numbers:
            averages.append(np.mean(valid_numbers))
        else:
            averages.append(None)  # 如果没有有效数值，保留None

    return all_logs, averages

def main():
    if len(sys.argv) < 2:
        print("请提供日志文件路径，例如：python script.py /path/to/log/directory")
        sys.exit(1)
    
    log_path = sys.argv[1]
    all_logs, averages = average_logs(log_path)
    
    # 打印每个log的结果
    print("Individual Log Results:")
    for log in all_logs:
        print(','.join(['' if v is None else f'{v:.2f}' for v in log]))
    
    # 插入空行
    print("\n" * 2)
    
    # 打印平均结果
    print("Average Results:")
    print(','.join(['' if v is None else f'{v:.2f}' for v in averages]))

if __name__ == "__main__":
    main()
