import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import joblib
import os
import json
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.utils import resample

def export_class_labels_json(label_map: Dict[str, int], filepath: str) -> None:
    """
    Exports the class label mapping (from label to numeric encoding) to a JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"Class labels exported to {filepath}")

def balance_classes_equal_samples(features: np.ndarray, labels: np.ndarray, n_samples_per_class: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balances the dataset by downsampling each class to have an equal number of samples (n_samples_per_class),
    and shuffles the resulting dataset.
    """
    unique_labels = np.unique(labels)
    balanced_features = []
    balanced_labels = []
    
    for label in unique_labels:
        # Get all the indices for the current label
        class_indices = np.where(labels == label)[0]
        
        # Downsample to have n_samples_per_class for each class
        sampled_indices = resample(class_indices, n_samples=n_samples_per_class, random_state=42)
        
        balanced_features.append(features[sampled_indices])
        balanced_labels.append(labels[sampled_indices])
    
    # Concatenate all the balanced data
    balanced_features = np.vstack(balanced_features)
    balanced_labels = np.concatenate(balanced_labels)
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(balanced_features.shape[0])
    balanced_features = balanced_features[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]
    
    return balanced_features, balanced_labels


def read_csv_to_ndarray(file_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Reads a CSV file and converts it to a NumPy array for features and a list for labels.
    """
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("CSV file is empty or data is invalid")
    
    feature_names = df.columns[:-1].tolist()
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].astype(str).tolist()
    return features, labels, feature_names


def select_top_features(features: np.ndarray, labels: np.ndarray, feature_names: List[str], n_features: int = 5) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Selects top N features based on importance scores using actual labels.
    """
    # Train a preliminary model with actual labels to get feature importance
    prelim_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    prelim_model.fit(features, labels)
    
    # Get indices of top N features
    importances = prelim_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:n_features]
    
    # Select top features and their names
    selected_features = features[:, top_indices]
    selected_names = [feature_names[i] for i in top_indices]
    
    print("\nTop 5 features selected:")
    for idx, (name, importance) in enumerate(zip(selected_names, importances[top_indices]), 1):
        print(f"{idx}. {name} (importance: {importance:.4f})")
    
    return selected_features, selected_names, top_indices.tolist()


def create_feature_mapping(file_paths: List[str], selected_indices: List[int]) -> Dict[str, Dict[int, str]]:
    """
    Creates a mapping of feature indices to their names for each file.
    """
    feature_mapping = {}
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        _, _, feature_names = read_csv_to_ndarray(file_path)
        
        file_mapping = {idx: feature_names[original_idx] 
                       for idx, original_idx in enumerate(selected_indices)}
        feature_mapping[file_name] = file_mapping
    
    return feature_mapping


def export_feature_mapping(mapping: Dict[str, Dict[int, str]], filepath: str) -> None:
    """
    Exports the feature mapping to a JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Feature mapping exported to {filepath}")


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
    Cleans the feature matrix by replacing invalid values with finite values.
    """
    cleaned_features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned_features


def export_model(model: DecisionTreeClassifier, label_map: Dict[str, int], selected_features: List[str], filepath: str) -> None:
    """
    Exports the model and metadata using joblib.
    """
    export_data = {
        'model': model, 
        'label_map': label_map,
        'selected_features': selected_features
    }
    joblib.dump(export_data, filepath)
    print(f"Model and metadata exported to {filepath}")


def import_model(filepath: str) -> Tuple[DecisionTreeClassifier, Dict[str, int], List[str]]:
    """
    Imports the model and metadata using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    data = joblib.load(filepath)
    return data['model'], data['label_map'], data['selected_features']


def save_feature_importance_chart(model: DecisionTreeClassifier, feature_names: List[str], filepath: str):
    """
    Saves a bar chart of feature importances.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Top 5 Features)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Feature importance chart saved to {filepath}")


def save_confusion_matrix(y_test: np.ndarray, predictions: np.ndarray, filepath: str):
    """
    Saves a confusion matrix chart.
    """
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Confusion matrix saved to {filepath}")

def analyze_class_distribution(labels: List[str], title: str = "Class Distribution") -> None:
    """
    Analyzes and prints the distribution of classes in the dataset.
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print(f"\n{title}:")
    print("-" * 40)
    for label, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"Class '{label}': {count} samples ({percentage:.2f}%)")
    
    imbalance_ratio = max(counts) / min(counts)
    print(f"\nImbalance Ratio (majority:minority): {imbalance_ratio:.2f}:1")


def calculate_class_weights(labels: List[str]) -> Dict[str, float]:
    """
    Calculates balanced class weights inversely proportional to class frequencies.
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique)
    
    weights = {label: n_samples / (n_classes * count) for label, count in zip(unique, counts)}
    return weights


def perform_cross_validation(features: np.ndarray, labels: np.ndarray, class_weights: Dict[str, float], n_splits: int = 5) -> None:
    """
    Performs stratified k-fold cross-validation with SMOTE and class weights.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Create pipeline with SMOTE and weighted classifier
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('classifier', DecisionTreeClassifier(max_depth=10, class_weight=class_weights, random_state=42))
    ])
    
    scores = cross_val_score(pipeline, features, labels, cv=cv, scoring='balanced_accuracy')
    
    print(f"\nCross-validation Results ({n_splits}-fold):")
    print("-" * 40)
    print(f"Mean Balanced Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"Individual Fold Scores: {', '.join(f'{score:.4f}' for score in scores)}")


def balance_dataset(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE to balance the dataset.
    """
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(features, labels)
    
    return X_resampled, y_resampled

def perform_cross_validation(features: np.ndarray, labels: np.ndarray, n_splits: int = 5) -> None:
    """
    Performs stratified k-fold cross-validation and prints results.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    scores = cross_val_score(model, features, labels, cv=cv, scoring='accuracy')
    
    print(f"\nCross-validation Results ({n_splits}-fold):")
    print("-" * 40)
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"Individual Fold Scores: {', '.join(f'{score:.4f}' for score in scores)}")


def analyze_feature_correlations(features: np.ndarray, feature_names: List[str], labels: np.ndarray) -> None:
    """
    Analyzes and plots feature correlations to check for potential data leakage.
    """
    # Create DataFrame with features and labels
    df = pd.DataFrame(features, columns=feature_names)
    df['label'] = labels
    
    # Calculate correlations
    correlations = df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()
    
    # Check for high correlations with label
    label_correlations = correlations['label'].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
    print("\nFeature-Label Correlations:")
    print("-" * 40)
    for feature, corr in label_correlations.items():
        print(f"{feature}: {corr:.4f}")
    
    # Warning for potential data leakage
    high_corr_features = label_correlations[label_correlations > 0.9]
    if not high_corr_features.empty:
        print("\nWARNING: Potential data leakage detected!")
        print("The following features have very high correlation (>0.9) with the label:")
        for feature, corr in high_corr_features.items():
            print(f"- {feature}: {corr:.4f}")


def print_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_map: Dict[str, int]) -> None:
    """
    Prints detailed classification metrics including precision, recall, and F1-score.
    """
    # Inverse label mapping for readable class names
    inv_label_map = {v: k for k, v in label_map.items()}
    
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=[inv_label_map[i] for i in sorted(inv_label_map.keys())])
    
    print("\nDetailed Classification Metrics:")
    print("-" * 40)
    print(report)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    print("\nPer-class Performance:")
    print("-" * 40)
    for i in range(len(precision)):
        class_name = inv_label_map[i]
        print(f"Class '{class_name}':")
        print(f"- Precision: {precision[i]:.4f}")
        print(f"- Recall: {recall[i]:.4f}")
        print(f"- F1-score: {f1[i]:.4f}")
        print(f"- Support: {support[i]}")

def main():
    file_paths = [
        "./data/1.csv",
        "./data/2.csv",
        "./data/3.csv",
        "./data/4.csv"
    ]
    
    all_features = []
    all_labels = []
    all_feature_names = []

    print("Reading data from CSV...")
    for file_path in file_paths:
        features, labels, feature_names = read_csv_to_ndarray(file_path)
        all_features.append(features)
        all_labels.extend(labels)
        if not all_feature_names:
            all_feature_names = feature_names

    features = np.vstack(all_features)
    
    # Analyze class distribution
    print("\nAnalyzing class distribution...")
    analyze_class_distribution(all_labels)
    
    print("\nCleaning data...")
    features = clean_data(features)
    
    print("Encoding labels...")
    encoded_labels, label_map = encode_labels(all_labels)

    export_class_labels_json(label_map, "class_labels.json")
    
    print("\nSelecting top 5 features...")
    selected_features, selected_names, selected_indices = select_top_features(
        features, encoded_labels, all_feature_names
    )

    # Balance dataset by equal sampling and shuffle
    print("\nBalancing dataset by equal sampling of each class and shuffling...")
    balanced_features, balanced_labels = balance_classes_equal_samples(features, encoded_labels, n_samples_per_class=20)
    
    # Save balanced and shuffled dataset to CSV
    balanced_df = pd.DataFrame(balanced_features, columns=all_feature_names)
    balanced_df['label'] = balanced_labels
    balanced_df.to_csv("balanced_shuffled_dataset.csv", index=False)
    print("Balanced and shuffled dataset saved to 'balanced_shuffled_dataset.csv'")
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    perform_cross_validation(selected_features, encoded_labels)
    
    # Analyze feature correlations
    print("\nAnalyzing feature correlations...")
    analyze_feature_correlations(selected_features, selected_names, encoded_labels)
    
    # Create and export feature mapping
    feature_mapping = create_feature_mapping(file_paths, selected_indices)
    export_feature_mapping(feature_mapping, "feature_mapping.json")
    
    # Split data and train final model
    print("\nSplitting data...")
    x_train, y_train, x_test, y_test = split_data(selected_features, encoded_labels)
    
    print("Training final model...")
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(x_train, y_train)
    
    print("Making predictions...")
    predictions = model.predict(x_test)
    
    # Print detailed classification metrics
    print_classification_metrics(y_test, predictions, label_map)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nFinal Model Accuracy: {accuracy:.2%}")
    
    export_model(model, label_map, selected_names, "decision_tree_model.pkl")
    
    # Save feature importance chart using selected feature names
    save_feature_importance_chart(model, selected_names, filepath="feature_importance.png")
    
    # Save confusion matrix
    save_confusion_matrix(y_test, predictions, filepath="confusion_matrix.png")


if __name__ == "__main__":
    main()