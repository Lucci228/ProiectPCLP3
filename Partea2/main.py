import pandas as pd
import numpy as np
import argparse

# Your code goes here

def iqr_finder(iqr_list = None):
    if iqr_list is None:
        return None
    iqr_list.sort()
    q1 = np.percentile(iqr_list, 25)
    q3 = np.percentile(iqr_list, 75)
    iqr = q3 - q1
    return iqr

def outliner_remover(iqr_list = None, iqr = None):
    if iqr_list is None or iqr is None:
        return None
    iqr_filtered = []
    Lower_bound = np.percentile(iqr_list, 25) - 1.5 * iqr
    Upper_bound = np.percentile(iqr_list, 75) + 1.5 * iqr
    for i in iqr_list:
        if i >= Lower_bound and i <= Upper_bound:
            iqr_filtered.append(i)
    return iqr_filtered

def main(file_path, column_name):
    df = pd.read_csv(file_path)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.dropna(subset=[column_name])
    column_data = df[column_name].tolist()
    iqr = iqr_finder(column_data)
    iqr_new_list = outliner_remover(column_data, iqr)
    print(iqr_new_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='Path to the CSV file')
    parser.add_argument('--column_name', type=str, help='Name of the column')
    args = parser.parse_args()
    main(args.file_path, args.column_name)