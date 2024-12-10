import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from typing import Tuple, List, Dict


def read_csv_to_ndarray(file_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Reads a CSV file and converts it to a NumPy array for features and a list for labels.
    """
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("CSV file is empty or data is invalid")
    
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].astype(str).tolist()
    return features, labels


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encodes string labels into numeric values.
    """
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_map[label] for label in labels])
    return encoded_labels, label_map


def split_data(features: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into training and testing sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    return x_train, y_train, x_test, y_test


def clean_data(features: np.ndarray) -> np.ndarray:
    """
    Cleans the feature matrix by replacing invalid values (NaN, inf, -inf) with finite values.
    """
    # Replace NaN, inf, -inf with 0 or other appropriate values
    cleaned_features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned_features


def export_model(model: DecisionTreeClassifier, label_map: Dict[str, int], filepath: str) -> None:
    """
    Exports the model and metadata using joblib.
    """
    export_data = {'model': model, 'label_map': label_map}
    joblib.dump(export_data, filepath)
    print(f"Model and metadata exported to {filepath}")


def import_model(filepath: str) -> Tuple[DecisionTreeClassifier, Dict[str, int]]:
    """
    Imports the model and metadata using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    data = joblib.load(filepath)
    return data['model'], data['label_map']


def main():
    file_paths = [
        "./data/1.csv",
        "./data/2.csv",
        "./data/3.csv",
        "./data/4.csv"
    ]
    
    all_features = []
    all_labels = []

    print("Reading data from CSV...")
    for file_path in file_paths:
        features, labels = read_csv_to_ndarray(file_path)
        all_features.append(features)
        all_labels.extend(labels)

    features = np.vstack(all_features)
    
    print("Cleaning data...")
    features = clean_data(features)
    
    print("Encoding labels...")
    encoded_labels, label_map = encode_labels(all_labels)
    
    print("Splitting data...")
    x_train, y_train, x_test, y_test = split_data(features, encoded_labels)
    
    print("Training model...")
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(x_train, y_train)
    
    print("Making predictions...")
    predictions = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.2%}")
    
    export_model(model, label_map, "decision_tree_model.pkl")
    
    print("Importing model...")
    imported_model, imported_label_map = import_model("decision_tree_model.pkl")
    print(f"Imported label map: {imported_label_map}")


if __name__ == "__main__":
    main()
