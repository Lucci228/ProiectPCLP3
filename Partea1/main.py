import pandas as pd


def main():
    df = pd.read_csv('./train.csv')
    rows = len(df.axes[0])
    cols = len(df.axes[1])
    data_types = df.dtypes
    total_missing_values = df.isnull().sum()
    duplicated_rows = df.duplicated().sum()
    print("The database has {} rows and {} columns".format(rows, cols))
    print("Missing values from cols: {}".format(total_missing_values))
    print("Nr of duplicated rows: {}".format(duplicated_rows))
    print(data_types)
    pr_survived = df['Survived'].sum() / float(rows) * 100
    pr_survived = round(pr_survived, 2)
    pr_dead = 100 - pr_survived
    print("Percentage of survived {}%\nPercentage of dead {}%".format(pr_survived, pr_dead))
    return 0


if __name__ == '__main__':
    main()
