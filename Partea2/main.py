import pandas as pd
import numpy as np
import argparse
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def prediction_survival(lst = None):
    # Încărcați datele
    df = pd.DataFrame(lst)
    
    # Înlăturați valorile lipsă
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Check if 'Survived' is a column in the DataFrame
    if 'Survived' not in df.columns:
        print("'Survived' is not a column in the DataFrame")
        return None
    
    # One-hot encoding for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]).toarray())
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, df_encoded], axis=1)
    
    # Convertiți coloanele categorice în valori numerice
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    
    # Normalizați caracteristicile numerice
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    # Împărțiți datele în două părți
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Antrenare model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluare model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

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

def modified_z_score(column_data):
    median = np.median(column_data)
    median_absolute_deviation = np.median([np.abs(x - median) for x in column_data])
    modified_z_scores = [0.6745 * (x - median) / median_absolute_deviation for x in column_data]
    return modified_z_scores

def main(file_path, column_name):
    df = pd.read_csv(file_path)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.dropna(subset=[column_name])
    column_data = df[column_name].tolist()
    print(column_data)
    
    z_scores = modified_z_score(column_data)
    z_scores_abs = np.abs(z_scores)
    filtered_column_data = [x for x, z in zip(column_data, z_scores_abs) if z < 3]

    iqr = iqr_finder(column_data)
    iqr_new_list = outliner_remover(column_data, iqr)
    #print(iqr_new_list)
    #print(filtered_column_data)
    print(prediction_survival(df))

    prep_data, validated_data = prediction_survival(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='Path to the CSV file')
    parser.add_argument('--column_name', type=str, help='Name of the column')
    args = parser.parse_args()
    main(args.file_path, args.column_name)