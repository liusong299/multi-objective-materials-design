import numpy as np

def maximin_design(X, model, epsilon=0.01):
    """
    Maximin-based design strategy.
    
    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    model (sklearn estimator): Trained regression model.
    epsilon (float): Small positive number to avoid division by zero.
    
    Returns:
    int: Index of the selected material.
    """
    mean, std = model.predict(X, return_std=True)
    improvement_probability = std / (std + epsilon)
    distance = np.abs(mean - np.mean(mean)) / (np.std(mean) + epsilon)
    maximin_score = improvement_probability * distance
    selected_material_index = np.argmax(maximin_score)
    return selected_material_index

def centroid_design(X, model):
    """
    Centroid-based design strategy.
    
    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    model (sklearn estimator): Trained regression model.
    
    Returns:
    int: Index of the selected material.
    """
    mean, _ = model.predict(X, return_std=True)
    centroid = np.mean(X, axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    selected_material_index = np.argmin(distances)
    return selected_material_index

def random_selection(X):
    """
    Random selection strategy.
    
    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    
    Returns:
    int: Index of the selected material.
    """
    selected_material_index = np.random.randint(0, len(X))
    return selected_material_index

def pure_exploitation(X, model):
    """
    Pure exploitation strategy.
    
    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    model (sklearn estimator): Trained regression model.
    
    Returns:
    int: Index of the selected material.
    """
    mean, _ = model.predict(X, return_std=True)
    selected_material_index = np.argmin(mean)
    return selected_material_index

def pure_exploration(X, model):
    """
    Pure exploration strategy.
    
    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    model (sklearn estimator): Trained regression model.
    
    Returns:
    int: Index of the selected material.
    """
    _, std = model.predict(X, return_std=True)
    selected_material_index = np.argmax(std)
    return selected_material_index
