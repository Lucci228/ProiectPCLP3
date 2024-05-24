import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def do_task_1(df):
    rows = len(df.axes[0])
    cols = len(df.axes[1])
    data_types = df.dtypes
    total_missing_values = df.isnull().sum()
    duplicated_rows = df.duplicated().sum()
    print("The database has {} rows and {} columns".format(rows, cols))
    print("Missing values from cols:\n{}".format(total_missing_values))
    print("Nr of duplicated rows:\n{}".format(duplicated_rows))
    print(df)
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


#pentru taskul 3
def generate_histograms(df):
    for i in df.columns:
        if df[i].dtype == 'int64' or df[i].dtype == 'float64':
            fig, ax = plt.subplots()
            if df[i].dtype == 'int64':
                sns.histplot(df[i], color='purple', discrete=True, edgecolor='black', ax=ax)
                ax.locator_params(axis='x', integer=True)
            else:
                sns.histplot(df[i], color='purple', edgecolor='black', ax=ax)
            plt.title(f'Histogram for {i}')
            plt.xlabel(i)
            plt.ylabel('Frequency')
            plt.show()


def do_task_4(df):
    print("====TASK 4====")
    total_rows = len(df.index)
    incomplete_data = df.isnull().sum()
    incomplete_data = incomplete_data[incomplete_data > 0]
    for i in incomplete_data.index:
        proportion = incomplete_data[i] / total_rows * 100
        proportion = round(proportion, 2)
        print("Column {} has {} missing values".format(i, incomplete_data[i]), end=' ')
        print("({}% of the total data)".format(proportion))
    incomplete_data = df[df.isnull().any(axis=1)]
    total_survived = df['Survived'].value_counts()
    incomplete_data_survived = incomplete_data['Survived'].value_counts()
    survived_percentage = incomplete_data_survived / total_survived * 100
    survived_percentage = round(survived_percentage, 2)
    print("{}% of alive people have incomplete data".format(survived_percentage[0]))
    print("{}% of dead people have incomplete data".format(survived_percentage[1]))


def main():
    df = pd.read_csv('./train.csv')
    do_task_1(df)
    do_task_2(df)
    generate_histograms(df)
    do_task_4(df)
    return 0


if __name__ == '__main__':
    main()
