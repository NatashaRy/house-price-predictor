import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps
from src.data_management import load_pricing_data

# Set theme style
sns.set_theme(style="darkgrid")


def analysis_body():
    """
    Displays content of the data analysis page in the Streamlit app.

    This includes: 
    - Introduction
    - Correlation analysis with possiblity to inspect dataset
    - Visualization of key findings
    """
    # Load house pricing data
    df = load_pricing_data()

    # List of variables to study in the analysis
    vars_to_study = [
        '1stFlrSF',
        'GarageArea',
        'GarageYrBlt',
        'GrLivArea',
        'KitchenQual_Ex',
        'KitchenQual_Gd',
        'KitchenQual_TA',
        'MasVnrArea',
        'OverallQual',
        'TotalBsmtSF',
        'YearBuilt',
        'YearRemodAdd'
    ]

    # Title and introduction
    st.title("Correlation Analysis")
    st.markdown("**This page explores the relationships between house attributes and the target variable, sale price."  
        "The goal is to identify key predictors for modeling and provide insights into how these attributes influence house prices.**"
    )

    # Reminder of business requirement 1
    st.info(
        "***Business Requirement 1**: The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.*"
    )

    # Optional inspection of the first 10 rows of the raw dataset
    if st.checkbox("Would you like to inspect the raw dataset?"):
        st.write("##### Inspection of house prices raw data")
        st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        st.write(df.head(10))
    st.markdown("---")

    # Summary of correlation study and key variables
    st.write("## Correlation Study\n"
            "To understand the relationships between house attributes and sale prices, we performed a correlation study."
            "The variables are correlated to a property's sale price. This addresses the project's first business requirement.\n\n"
            "**By using Pearson and Spearman correlation methods, we found that the most correlated variables are**:\n"
            f"{vars_to_study}"
    )    

    # Conclusions from correlation study
    st.write("### Summary of conclusions from correlation study\n"
            "The correlation study revealed that the following variables have a strong relationship with the sale price:\n"
    )

    st.markdown("---")

    # Visualizations
    st.write("## Visualizations\n"
            "To further illustrate the findings from the correlation study, we provide the following visualizations:\n"
            "- Scatter plots for each of the most correlated variables against sale price.\n"
            "- Heatmap of the correlation matrix.\n"
            "These visualizations will help in understanding the relationships more intuitively."
    )

    # Target variables


    # Show heatmaps



