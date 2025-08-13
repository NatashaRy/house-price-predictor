import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    Evaluates the performance of a regression model on both training and test datasets.

    Parameters:
    X_train (DataFrame or ndarray): Training feature set.
    y_train (Series or ndarray): Target variable for training set.
    X_test (DataFrame or ndarray): Test feature set.
    y_test (Series or ndarray): Target variable for test set.
    pipeline (sklearn Pipeline): Trained pipeline model.
    """
    st.write("### Model Evaluation")

    # Evaluate training set
    st.info("**Train Set Performance**")
    regression_evaluation(X_train, y_train, pipeline)

    # Evaluate test set
    st.info("**Test Set Performance**")
    regression_evaluation(X_test, y_test, pipeline)

def regression_evaluation(X, y, pipeline):
    """
    Computes and displays various regression evaluation metrics for a given dataset.

    Parameters:
    X (DataFrame or ndarray): Feature set.
    y (Series, DataFrame, or ndarray): Target variable.
    pipeline (sklearn Pipeline): Trained pipeline model.
    """
    # Make predictions
    predictions = pipeline.predict(X)

    # If y is a DataFrame, extract values
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()

    # Calculate metrics
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    # Visualize metrics
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")