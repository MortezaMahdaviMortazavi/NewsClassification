import os
import pandas as pd
import numpy as np
import pickle

def read_pickle(file_path):
    """
    Read and load data from a pickle file.

    Parameters:
        file_path (str): The path to the pickle file.

    Returns:
        object: The object loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data, file_path):
    """
    Write data to a pickle file.

    Parameters:
        data (object): The data to be stored in the pickle file.
        file_path (str): The path to the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)



def stratified_k_fold_split(df, category_column, num_folds, output_folder):
    """
    Split a DataFrame into training and validation sets using stratified k-fold cross-validation,
    and save the sets in separate folders.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - category_column: str
        The column containing the class labels.
    - num_folds: int
        The number of folds for stratified k-fold cross-validation.
    - output_folder: str
        The folder where the sets for each fold will be saved.
    """
    if category_column not in df.columns:
        raise ValueError(f"Column '{category_column}' not found in the DataFrame.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    class_indices = {}
    for class_label in df[category_column].unique():
        class_indices[class_label] = df[df[category_column] == class_label].index.tolist()

    fold_train_sets = []
    fold_val_sets = []

    samples_per_fold_per_class = {class_label: len(indices) // num_folds for class_label, indices in class_indices.items()}
    for fold_idx in range(num_folds):
        fold_train_indices = []
        fold_val_indices = []

        for class_label, indices in class_indices.items():
            start_idx = fold_idx * samples_per_fold_per_class[class_label]
            end_idx = (fold_idx + 1) * samples_per_fold_per_class[class_label]

            fold_val_indices.extend(indices[start_idx:end_idx] if fold_idx == num_folds - 1 else indices[start_idx:end_idx])
            fold_train_indices.extend(indices[:start_idx] + indices[end_idx:] if fold_idx == num_folds - 1 else indices[:start_idx] + indices[end_idx:])

        fold_train_sets.append(fold_train_indices)
        fold_val_sets.append(fold_val_indices)

    for fold_idx, (train_indices, val_indices) in enumerate(zip(fold_train_sets, fold_val_sets)):
        fold_folder = os.path.join(output_folder, f"fold_{fold_idx + 1}")
        if not os.path.exists(fold_folder):
            os.makedirs(fold_folder)

        train_data, val_data = df.iloc[train_indices], df.iloc[val_indices]

        train_data.to_csv(os.path.join(fold_folder, "train.csv"), index=False)
        val_data.to_csv(os.path.join(fold_folder, "validation.csv"), index=False)

# df = pd.read_csv('dataset/cleaned_train_length.csv')
# stratified_k_fold_split(df, 'Category', 5, 'dataset')
