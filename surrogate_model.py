import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def train_regression_model(X_train, y_train, model_type='GPR'):
    """
    Train the regression model.
    
    Parameters:
    X_train (pd.DataFrame): Training data features.
    y_train (pd.Series): Training data target values.
    model_type (str): Type of regression model to use. Options: 'GPR' (default) or 'RF'.
    
    Returns:
    sklearn estimator: Trained regression model.
    """
    if model_type == 'GPR':
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    elif model_type == 'RF':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'GPR' or 'RF'.")

    model.fit(X_train, y_train)
    return model

def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate the performance of the regression model using cross-validation.
    
    Parameters:
    model (sklearn estimator): Trained regression model.
    X_test (pd.DataFrame): Test data features.
    y_test (pd.Series): Test data target values.
    
    Returns:
    float: Mean squared error of the model.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

if __name__ == "__main__":
    # Load training data
    training_data = pd.read_csv("training_data.csv")
    X_train = training_data.drop("target", axis=1)  # Replace "target" with the name of your target column
    y_train = training_data["target"]  # Replace "target" with the name of your target column

    # Train regression model
    model_type = "GPR"  # Choose the model type: 'GPR' or 'RF'
    model = train_regression_model(X_train, y_train, model_type)

    # Load test data
    test_data = pd.read_csv("test_data.csv")  # Replace with the path to your test data
    X_test = test_data.drop("target", axis=1)  # Replace "target" with the name of your target column
    y_test = test_data["target"]  # Replace "target" with the name of your target column

    # Evaluate model performance
    mse = evaluate_model_performance(model, X_test, y_test)
    print(f"Mean Squared Error: {mse:.2f}")
