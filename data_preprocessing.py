import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_dataset(file_path):
    """
    Load and clean the dataset.
    
    Parameters:
    file_path (str): Path to the dataset file.
    
    Returns:
    pd.DataFrame: Cleaned dataset.
    """
    dataset = pd.read_csv(file_path)
    
    # Perform any dataset-specific cleaning here, e.g.:
    # dataset.dropna(inplace=True)
    # dataset.drop_duplicates(inplace=True)
    
    return dataset

def normalize_feature_values(dataset, feature_columns):
    """
    Normalize feature values using MinMaxScaler.
    
    Parameters:
    dataset (pd.DataFrame): Dataset containing the features to be normalized.
    feature_columns (list): List of column names to be normalized.
    
    Returns:
    pd.DataFrame: Dataset with normalized feature values.
    """
    scaler = MinMaxScaler()
    dataset[feature_columns] = scaler.fit_transform(dataset[feature_columns])
    return dataset

def divide_data_into_training_and_search_spaces(dataset, training_size):
    """
    Divide dataset into training data and materials search space.
    
    Parameters:
    dataset (pd.DataFrame): The dataset to be divided.
    training_size (float): Proportion of the dataset to be used as training data.
    
    Returns:
    tuple: (training_data, search_space) as pd.DataFrame objects.
    """
    training_data, search_space = train_test_split(dataset, train_size=training_size, random_state=42)
    return training_data, search_space

if __name__ == "__main__":
    # Load and clean dataset
    dataset_file_path = "path/to/dataset.csv"
    dataset = load_and_clean_dataset(dataset_file_path)

    # Normalize feature values
    feature_columns = ["feature_1", "feature_2", "feature_3"]  # Replace with your actual feature column names
    dataset = normalize_feature_values(dataset, feature_columns)

    # Divide data into training and search spaces
    training_size = 0.2  # Adjust the size according to your needs
    training_data, search_space = divide_data_into_training_and_search_spaces(dataset, training_size)

    # Save the training data and search space as separate CSV files
    training_data.to_csv("training_data.csv", index=False)
    search_space.to_csv("search_space.csv", index=False)
