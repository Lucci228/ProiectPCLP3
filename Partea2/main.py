import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


# mai verifica preciziile:
# - doar cu iqr
# - doar cu z-score
# - fara nicio metoda de eliminare a outlierilor
# - cu ambele metode de eliminare a outlierilor


def plot_metrics(accuracy, log_loss):
    metrics = [accuracy, log_loss]
    metrics_names = ['Accuracy', 'Log Loss']
    plt.figure(figsize=(10, 5))
    plt.bar(metrics_names, metrics, color=['blue', 'red'])
    plt.ylabel('Score')
    plt.title('Model Metrics')
    for i in range(len(metrics)):
        plt.text(i, metrics[i], round(metrics[i], 2), ha = 'center')
    plt.show()


def prediction_survival(df=None):
    if df is None:
        return None
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    features_columns = ['Sex_male', 'Sex_female', 'Age', 'Fare']
    embarked_columns = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
    for column in embarked_columns:
        if column in df.columns:
            features_columns.append(column)
    features = df[features_columns]
    target = df['Survived']
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100}%")
    y_pred_proba = model.predict_proba(x_test)
    loss = log_loss(y_test, y_pred_proba)
    print(f"Log Loss: {loss}")
    plot_metrics(accuracy, loss)
    return accuracy, loss


def outlier_remover_iqr(df):
    possible_outlier_columns = ['Age', 'Fare', 'SibSp', 'Parch']
    for column in possible_outlier_columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


def calculate_z_scores(column):
    return (column - column.mean()) / column.std()


def outlier_remover_z_score(df):
    df = df.dropna()
    possible_outlier_columns = ['Age', 'Fare', 'SibSp', 'Parch']
    for column in possible_outlier_columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            z_scores = calculate_z_scores(df[column])
            median = np.median(z_scores)
            median_absolute_deviation = np.median([np.abs(x - median) for x in z_scores])
            if median_absolute_deviation != 0:
                modified_z_scores = [0.6745 * (x - median) / median_absolute_deviation if not np.isnan(x) else 0 for x in z_scores]
            else:
                modified_z_scores = [0 for x in z_scores]
            df.loc[:, column] = np.array(modified_z_scores, dtype=df[column].dtype)
    return df


def main(file_path, output_file_path):
    df = pd.read_csv(file_path)
    df_new = outlier_remover_iqr(df)
    df_new2 = outlier_remover_z_score(df_new)
    prediction_survival(df_new2)
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to the CSV file')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
