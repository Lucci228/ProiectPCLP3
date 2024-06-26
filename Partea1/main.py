import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def do_task_1(df):
    rows = len(df.axes[0])
    cols = len(df.axes[1])
    data_types = df.dtypes
    missing_values_cols = df.isnull().sum()
    duplicated_rows = df.duplicated().sum()
    total_missing_values = missing_values_cols.sum()
    print("====TASK 1====")
    print("The database has {} rows and {} columns".format(rows, cols))
    print("\nTotal missing values: {}".format(total_missing_values))
    print("\nMissing values from cols:\n{}".format(missing_values_cols))
    print("\nNr of duplicated rows:\n{}\n".format(duplicated_rows))
    print(df)
    print(data_types)
    print("====END TASK 1====")


def do_task_2(df):

    survived_percentage = df['Survived'].value_counts(normalize=True) * 100
    survived_percentage = round(survived_percentage, 2)

    room_percentage = df['Pclass'].value_counts(normalize=True) * 100
    room_percentage = round(room_percentage, 2)

    gender_percentage = df['Sex'].value_counts(normalize=True) * 100
    gender_percentage = round(gender_percentage, 2)

    fig, axs = plt.subplots(3, 1, figsize=(5, 8))
    axs[0].pie(survived_percentage, labels=['Survived', 'Dead'], autopct='%1.1f%%',
                  colors=['skyblue', 'salmon'])
    axs[0].set_title('Survived vs Dead')

    axs[1].pie(room_percentage, labels=['First Class', 'Second Class', 'Third Class'], autopct='%1.1f%%', colors=['gold', 'purple', 'cyan'])
    axs[1].set_title('Room Class Dispersion')

    axs[2].pie(gender_percentage, labels=['Male', 'Female'], autopct='%1.1f%%',
                  colors=['lightblue', 'lightcoral'])
    axs[2].set_title('Male vs Female')
    plt.tight_layout()
    plt.show()


def do_task_3(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n = len(numeric_cols)
    n_rows = n // 2 + n % 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 10))
    axs = axs.ravel()  # Flatten the array of axes
    idx = 0
    for col in numeric_cols:
        if df[col].dtype == 'int64':
            sns.histplot(df[col], color='blue', discrete=True, edgecolor='black', ax=axs[idx])
            axs[idx].locator_params(axis='x', integer=True)
        else:
            sns.histplot(df[col], color='blue', edgecolor='black', ax=axs[idx])
        axs[idx].set_title(f'Histogram for {col}')
        axs[idx].set_xlabel(col)
        axs[idx].set_ylabel('Frequency')
        idx += 1
    for i in range(idx, len(axs)):
        fig.delaxes(axs[i])
    plt.tight_layout()
    plt.show()
    sns.histplot(df["PassengerId"], color='blue', edgecolor='black')
    plt.title('Histogram for PassengerId')
    plt.xlabel('PassengerId')
    plt.ylabel('Frequency')
    plt.show()
    sns.histplot(df["Fare"], color='lightblue', edgecolor='black')
    plt.title('Histogram for Fare')
    plt.xlabel('Fare')
    plt.ylabel('Frequency')
    plt.show()


def get_index(value, brackets):
    if value is None:
        return None
    for i in range(len(brackets)):
        if brackets[i][0] <= value <= brackets[i][1]:
            return i


def add_age_brackets(df):
    age_brackets = [(0, 20), (21, 40), (41, 60), (60, df['Age'].max())]
    new_col = []
    for i in df['Age']:
        new_col.append(get_index(i, age_brackets))
    df.insert(12, 'Age_Bracket', new_col, True)
    return df


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
    incomplete_data = df[df.isnull().any(axis=1)] #df.isnull().anu(axis=1) returneaza o lista cu true in dreptul liniilor cu null/NaN values
    total_survived = df['Survived'].value_counts()
    incomplete_data_survived = incomplete_data['Survived'].value_counts()
    incomplete_percentage = incomplete_data_survived / total_survived * 100
    incomplete_percentage = round(incomplete_percentage, 2)
    complete_data = 100 - incomplete_percentage
    print("{}% of dead people have incomplete data".format(incomplete_percentage[0]))
    print("{}% of survive people have incomplete data".format(incomplete_percentage[1]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie([incomplete_percentage[0], complete_data[0]], labels=['Incomplete', 'Complete'], autopct='%1.1f%%',
               colors=['red', 'green'])
    axs[0].set_title('Dead')
    axs[1].pie([incomplete_percentage[1], complete_data[1]], labels=['Incomplete', 'Complete'], autopct='%1.1f%%',
               colors=['red', 'green'])
    axs[1].set_title('Alive')
    plt.show()
    print("====END TASK 4====")


def replace_bracket(df):
    age_brackets = [(0, 20), (21, 40), (41, 60), (60, df['Age'].max())]
    new_col = []
    for i in df['Age']:
        new_col.append(get_index(int(i), age_brackets))
    df['Age_Bracket'] = new_col
    return df


def do_task_5(df):
    age_brackets = [(0, 20), (21, 40), (41, 60), (60, df['Age'].max())]
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
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    sns.barplot(x=percent_survived.index, y=percent_survived.values,
                palette='viridis', hue=percent_survived.index, legend=False, ax=axs)
    axs.set_title('Survival Rate by Age Bracket')
    axs.set_xlabel('Age Bracket')
    axs.set_ylabel('Survival Rate (%)')
    axs.set_xticks(range(len(percent_survived.index)))
    axs.set_xticklabels(['0-20 years', '21-40 years', '41-60 years', '61+ years'], rotation=0)
    plt.tight_layout()
    plt.show()


def do_task_7(df):
    children_df = df[(df['Age'] < 18)]
    percent_kids = len(children_df.index) / len(df.index) * 100
    percent_kids = round(percent_kids, 2)
    print("====TASK 7====")
    print("{}% of passengers are children".format(percent_kids))
    children_survived_rate = len(children_df[(children_df['Survived'] == 1)].index) / len(children_df.index) * 100
    children_survived_rate = round(children_survived_rate, 2)
    adults_df = df[(df['Age'] >= 18)]
    adults_survived_rate = len(adults_df[(adults_df['Survived'] == 1)].index) / len(adults_df.index) * 100
    adults_survived_rate = round(adults_survived_rate, 2)
    sns.barplot(x=['Children', 'Adults'], y=[children_survived_rate, adults_survived_rate], palette='Spectral',
                hue=['Children', 'Adults'], legend=False)
    plt.title('Survival Rate for Children and Adults')
    plt.ylabel('Survival Rate (%)')
    plt.show()


def complete_df(df):
    total_rows = len(df.index)
    incomplete_data = df.isnull().sum()
    incomplete_data = incomplete_data[incomplete_data > 0]
    for i in incomplete_data.index:
        if df[i].dtype == 'int64' or df[i].dtype == 'float64':
            df[i] = df[i].fillna(df[i].mean())
        else:
            df[i] = df[i].fillna(df[i].mode()[0])
    df = replace_bracket(df)


def do_task_9(df):
    title_gender = {
        'Mr': 'male',
        'Miss': 'female',
        'Mrs': 'female',
        'Master': 'male',
        'Dr': 'common',
        'Rev': 'male',
        'Mlle': 'female',
        'Major': 'male',
        'Col': 'male',
        'Countess': 'female',
        'Capt': 'male',
        'Ms': 'female',
        'Sir': 'male',
        'Lady': 'female',
        'Mme': 'female',
        'Don': 'male',
        'Jonkheer': 'male'
    }
    new_df = df
    new_df['Title'] = new_df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    new_df['Expected_Sex'] = new_df['Title'].map(title_gender)
    new_df['Match_gender'] = new_df.apply(
        lambda row: True if row['Expected_Sex'] == 'common' else row['Sex'] == row['Expected_Sex'], axis=1)
    matched_titles = new_df['Match_gender'].value_counts().iloc[0]
    print("Task 9: There are {} missmatched titles".format(len(new_df) - matched_titles))
    plt.figure(figsize=(10, 8))
    sns.countplot(data=new_df, x='Title')
    plt.title('Number of Passengers by Title')
    plt.xlabel('Title')
    plt.ylabel('Number of Passengers')
    plt.xticks(rotation=90)
    plt.show()


def investigate_alone_survival(df):
    alone = df[(df['SibSp'] == 0) & (df['Parch'] == 0)]
    not_alone = df[(df['SibSp'] != 0) | (df['Parch'] != 0)]
    not_alone_survived = not_alone['Survived']
    alone_survived = alone['Survived']
    plt.hist([not_alone_survived, alone_survived], bins=2, color=['g', 'r'], label=['Not Alone', 'Alone'], edgecolor='black')
    plt.title('Survival Rates for Alone vs Not Alone Passengers')
    plt.xlabel('Survival Status')
    plt.ylabel('Number of Passengers')
    plt.legend()
    plt.xticks([0.25, 0.75], ['Did not survive', 'Survived'])
    plt.show()


def do_task_10(df):
    dataset = df.head(100)
    plt.figure(figsize=(10, 10))
    sns.catplot(data=dataset, x='Survived', y='Fare', hue='Pclass', kind='swarm', palette='tab10', size=3)
    plt.show()


def main():
    df = pd.read_csv('./train.csv')
    do_task_1(df)
    do_task_2(df)
    do_task_3(df)
    do_task_4(df)
    df = add_age_brackets(df)
    df.to_csv('data1.csv', sep=',', index=False, encoding='utf-8')
    do_task_5(df)
    do_task_6(df)
    do_task_7(df)
    complete_df(df)
    df.to_csv('data2.csv', sep=',', index=False, encoding='utf-8')
    do_task_9(df)
    investigate_alone_survival(df)
    do_task_10(df)
    return 0


if __name__ == '__main__':
    main()
