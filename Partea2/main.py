import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder





# Trebuie sa schimb pe ce se bazeaza modelul de predictie
# deocamdata se bazeaza pe survived column, dar trebuie pe sex, age, fare, embarked
# Probabil trebuie sa schimb urmatoarele:
# 1. mai intai sa transform valorile coloanelor in numere
# 2. sa nu se mai bazeze pe survived







def prediction_survival(lst = None):
    # Încărcați datele
    df = pd.DataFrame(lst)
    
    # Înlăturați valorile lipsă
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Convertiți coloanele categorice în valori numerice
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    
    # One-hot encoding for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]).toarray())
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, df_encoded], axis=1)
    
    # Normalizați caracteristicile numerice
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    # Împărțiți datele în două părți
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Antrenare model
    model = RandomForestClassifier() # model inseamna pe ce se bazeaza survabilitatea
    
    # Convert all feature names to strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    
    # Fit the model
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
    if iqr == 0:
        return iqr_list
    Lower_bound = np.percentile(iqr_list, 25) - 1.5 * iqr
    Upper_bound = np.percentile(iqr_list, 75) + 1.5 * iqr
    return [i if Lower_bound <= i <= Upper_bound else 0 for i in iqr_list]


def modified_z_score(column_data):
    median = np.median(column_data)
    median_absolute_deviation = np.median([np.abs(x - median) for x in column_data])
    modified_z_scores = [0.6745 * (x - median) / median_absolute_deviation for x in column_data]
    return modified_z_scores


def main(file_path, output_file_path):
    # Step 1: Read data from CSV
    df = pd.read_csv(file_path)

    # Step 2: Iterate over each column
    for column_name in df.columns:
        # Check if the column data is numeric and not one of the first two columns
        if pd.api.types.is_numeric_dtype(df[column_name]) and column_name not in df.columns[:2]:
            # Step 3: Remove outliers
            column_data = df[column_name].tolist()
            iqr = iqr_finder(column_data)
            cleaned_data = outliner_remover(column_data, iqr)

            # Step 4: Replace the original column with the cleaned data
            if cleaned_data:
                df[column_name] = cleaned_data

    # Step 5: Write the DataFrame back to a new CSV file
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to the CSV file')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
