"""
Data management utilities for loading datasets and ML models.

This module provides functions for loading house pricing data,
inherited house data, and machine learning model pickle files.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_pricing_data():
    """
    Load raw house pricing dataset from CSV file.

    Returns:
        pd.DataFrame: Raw house pricing data
    """
    df = pd.read_csv(r"inputs\datasets\raw\house_prices_records.csv")
    return df


@st.cache_resource
def load_inherited_data():
    """
    Load raw inherited house prices from CSV file.

    Returns:
        pd.DataFrame: Raw inherited house pricing data
    """
    df_inherited = pd.read_csv(r"inputs\datasets\raw\inherited_houses.csv")
    return df_inherited


def load_pkl_file(file_path):
    """
    Load ML model pickle file.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        object: Loaded model object
    """
    return joblib.load(file_path)