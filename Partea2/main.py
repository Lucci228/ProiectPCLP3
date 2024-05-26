import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


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
    features = df[['Sex_male', 'Sex_female', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
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
    if iqr == 0:
        return iqr_list
    lower_bound = np.percentile(iqr_list, 25) - 1.5 * iqr
    upper_bound = np.percentile(iqr_list, 75) + 1.5 * iqr
    return [i if lower_bound <= i <= upper_bound else 0 for i in iqr_list]


def modified_z_score(column_data):
    median = np.median(column_data)
    median_absolute_deviation = np.median([np.abs(x - median) for x in column_data])
    modified_z_scores = [0.6745 * (x - median) / median_absolute_deviation for x in column_data]
    return modified_z_scores


def main(file_path, output_file_path):
    df = pd.read_csv(file_path)
    prediction_survival(df)
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to the CSV file')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
