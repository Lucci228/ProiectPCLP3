import pandas as pd


def do_task_1(df):
    rows = len(df.axes[0])
    cols = len(df.axes[1])
    data_types = df.dtypes
    total_missing_values = df.isnull().sum()
    duplicated_rows = df.duplicated().sum()
    print("The database has {} rows and {} columns".format(rows, cols))
    print("Missing values from cols: {}".format(total_missing_values))
    print("Nr of duplicated rows: {}".format(duplicated_rows))
    print(data_types)


def do_task_2(df):
    survived_percentage = df['Survived'].value_counts(normalize=True) * 100
    survived_percentage = round(survived_percentage, 2)
    print("Survived {}".format(survived_percentage))
    people_per_room = df.loc[:, 'Pclass'].value_counts()
    room_percentage = df['Pclass'].value_counts(normalize=True) * 100
    room_percentage = round(room_percentage, 2)
    room_percentage = room_percentage.sort_values()
    print("Room {}".format(room_percentage))


def main():
    df = pd.read_csv('./train.csv')
    #do_task_1(df)
    do_task_2(df)
    return 0


if __name__ == '__main__':
    main()
