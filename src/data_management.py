import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Caching for unnecessary reloading of dataset
@st.cache_resource
# Load raw house pricing dataset
def load_pricing_data(file_path="inputs/datasets/raw/house_prices_records.csv"):
    try:
        df = pd.read_csv(file_path)
        return df
    # Error handeling if file is not found
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return None

@st.cache_resource
#Load raw inherited house prices
def load_inherited_data(file_path="inputs/datasets/raw/inherited_houses.csv"):
    try:
        df_i = pd.read_csv(file_path)
        return df_i
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return None
    
# Load ML model pickle file
def load_pkl_file(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None