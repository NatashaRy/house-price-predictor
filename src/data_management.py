import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Caching for unnecessary reloading of dataset
@st.cache_resource
# Load raw house pricing dataset
def load_pricing_data():
    df = pd.read_csv(r"inputs\datasets\raw\house_prices_records.csv")
    return df

@st.cache_resource
#Load raw inherited house prices
def load_inherited_data():
    df_inherited = pd.read_csv(r"inputs\datasets\raw\inherited_houses.csv")
    return df_inherited

# Load ML model pickle file
def load_pkl_file(file_path):
    return joblib.load(file_path)