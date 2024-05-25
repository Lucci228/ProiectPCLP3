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


def get_index(value, brackets):
    if value is None:
        return None
    for i in range(len(brackets)):
        if brackets[i][0] <= value <= brackets[i][1]:
            return i


def add_age_brackets(df):
    age_brackets = [(0, 20), (21, 40), (41, 60), (60, df['Age'].max())]
    new_collumn = []
    for i in df['Age']:
        new_collumn.append(get_index(i, age_brackets))
    df.insert(12, 'Age_Bracket', new_collumn, True)
    return df


def do_task_5(df):
    age_brackets = [(0, 20), (21, 40), (41, 60), (60, df['Age'].max())]
    print("====TASK 5====")
    df = add_age_brackets(df)
    age_category_counts = df['Age_Bracket'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    colors = sns.color_palette("tab10", len(age_brackets))
    sns.histplot(data=df, x='Age_Bracket', hue='Age_Bracket', palette=colors, discrete=True, edgecolor='black', ax=ax)
    plt.title('Number of Passengers in Each Age Category')
    plt.xlabel('Age Category')
    ax.set_xticks(range(len(age_brackets)))
    ax.set_xticklabels(['0-20 years', '21-40 years', '41-60 years', '61+ years'], rotation=0)
    plt.ylabel('Number of Passengers')
    plt.show()


def do_task_6(df):
    age_brackets = [(0, 20), (21, 40), (41, 60), (60, df['Age'].max())]
    males_df = df[(df['Sex'] == 'male')]
    male_age_brackets = males_df['Age_Bracket'].value_counts().sort_index()
    male_survived = males_df[(males_df['Survived'] == 1)]
    survived_age_brackets = male_survived['Age_Bracket'].value_counts().sort_index()
    percent_survived = survived_age_brackets / male_age_brackets * 100
    percent_survived = round(percent_survived, 2)

    survival_rate = []
    for i in male_survived['Age']:
        index = get_index(i, age_brackets)
        if index is not None:
            survival_rate.append(percent_survived[index])
        else:
            survival_rate.append(None)
    male_survived.insert(13, 'Survival_Rate', survival_rate, True)

    fig, axs = plt.subplots(2, 1, figsize=(5, 10))

    sns.barplot(x=percent_survived.index, y=percent_survived.values,
                palette='viridis', hue=percent_survived.index, legend=False, ax=axs[0])
    axs[0].set_title('Survival Rate by Age Bracket')
    axs[0].set_xlabel('Age Bracket')
    axs[0].set_ylabel('Survival Rate (%)')
    axs[0].set_xticks(range(len(percent_survived.index)))
    axs[0].set_xticklabels(['0-20 years', '21-40 years', '41-60 years', '61+ years'], rotation=0)

    sns.lineplot(x=male_survived['Age'], y=male_survived['Survival_Rate'], color='blue', ax=axs[1])
    axs[1].set_title('Scatter plot of Survival Rate by Age')
    axs[1].set_ylabel('Survival Rate (%)')

    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv('./train.csv')
    do_task_1(df)
    do_task_2(df)
    generate_histograms(df)
    do_task_4(df)
    do_task_5(df)
    do_task_6(df)
    print(df)
    return 0


if __name__ == '__main__':
    main()
