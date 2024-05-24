import pandas as pd
import numpy as np

# Your code goes here

def iqr_finder(iqr_list = None):
    if iqr_list is None:
        return None
    iqr_list.sort()
    q1 = np.percentile(iqr_list, 25)
    q3 = np.percentile(iqr_list, 75)
    iqr = q3 - q1
    return iqr

def main():
    # Your code goes here
    age = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    iqr = iqr_finder(age)
    pass

if __name__ == "__main__":
    main()