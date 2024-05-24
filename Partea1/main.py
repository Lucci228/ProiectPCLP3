import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

    room_percentage = df['Pclass'].value_counts(normalize=True) * 100
    room_percentage = round(room_percentage, 2)

    gender_percentage = df['Sex'].value_counts(normalize=True) * 100
    gender_percentage = round(gender_percentage, 2)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].pie(survived_percentage, labels=['Survived', 'Dead'], autopct='%1.1f%%',
                  colors=['skyblue', 'salmon'])
    axs[0, 0].set_title('Survived vs Dead')

    axs[0, 1].pie(room_percentage, labels=['First Class', 'Second Class', 'Third Class'], autopct='%1.1f%%', colors=['gold', 'purple', 'cyan'])
    axs[0, 1].set_title('Room Class Dispersion')

    axs[1, 0].pie(gender_percentage, labels=['Male', 'Female'], autopct='%1.1f%%',
                  colors=['lightblue', 'lightcoral'])
    axs[1, 0].set_title('Male vs Female')
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv('./train.csv')
    do_task_1(df)
    do_task_2(df)
    return 0


if __name__ == '__main__':
    main()
