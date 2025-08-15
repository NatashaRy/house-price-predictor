"""
Data management utilities for loading datasets and ML models.

This module provides functions for loading house pricing data,
inherited house data, and machine learning model pickle files.
"""

import os
import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_pricing_data():
    """
    Load raw house pricing dataset from CSV file.

    Returns:
        pd.DataFrame: Raw house pricing data
    """
    file_path = os.path.join(
        "inputs", "datasets", "raw", "house_prices_records.csv"
        )
    print(f"Loading house pricing data from: {file_path}")
    df = pd.read_csv(file_path)
    return df


@st.cache_resource
def load_inherited_data():
    """
    Load raw inherited house prices from CSV file.

    Returns:
        pd.DataFrame: Raw inherited house pricing data
    """
    file_path = os.path.join(
        "inputs", "datasets", "raw", "inherited_houses.csv"
        )
    print(f"Loading inherited house data from: {file_path}")
    df_inherited = pd.read_csv(file_path)
    return df_inherited


def load_pkl_file(file_path):
    """
    Load ML model pickle file.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        object: Loaded model object
    """
    print(f"Loading pickle file from: {file_path}")
    return joblib.load(file_path)
