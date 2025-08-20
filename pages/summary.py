import streamlit as st
import pandas as pd
from src.data_management import load_pricing_data
from utils import create_toc

# This page displays content of the summary page in the Streamlit app.
#
# That includes:
# - Page introduction
# - Project summary
# - Dataset details
#   - With quick dataset summary and dataset preview
# - List of Business requirements
# - Link to README file

# Define page configuration
st.set_page_config(
    page_title="Project Overview",
    page_icon="🗒️",
    initial_sidebar_state="expanded",
)

# Title and page introduction
st.title("🗒️Project Overview")
st.write(
    """
    This page serves as the starting point for th
    House Price Predictor App.
    Here, you'll find a quick summary of the project's
    objectives, the dataset used,
    and the business requirements we aim to address.
    Use this as a guide to navigate
    through the app and explore the insights and predictions it offers.
    """
    )

# Information box with project summary
st.info(
    "### **Project Summary**\n"
    "This project, 'House Price Predictor,' aims to analyze house "
    "attributes and predict sale prices for houses in Ames, Iowa.\n"
    "The goal is to help the client maximize the sales price of their "
    "inherited properties by providing insights into the most "
    "important factors affecting house prices and enabling "
    "accurate price predictions."
)

# Displays dataset details
st.header("🔎Dataset Details")
st.write(
    "The dataset used in this project is the "
    "[Heritage Housing dataset]"
    "(https://www.kaggle.com/codeinstitute/housing-prices-data). "
    "The dataset contains information about houses in Ames, Iowa, "
    "including their attributes and sale prices.\n"
)

# Displays quick summary of dataset
st.subheader("Quick dataset summary:")

try:
    df = load_pricing_data()
    st.write(f"* **Number of rows**: {df.shape[0]}")
    st.write(f"* **Number of columns**: {df.shape[1]}")
    st.write(f"* **Target variable**: `SalePrice`")
    st.write(
        "* **Key variables**: "
        f"`GrLivArea`, `OverallQual`, `YearBuilt`\n"
    )
except FileNotFoundError:
    st.error(
        "Dataset file not found. "
        "Please ensure the dataset is available "
        "in the specified path."
    )

st.markdown(" ")

st.write("\n#### Dataset Preview")
st.dataframe(df.head())

# Horizontal line
st.divider()

# Copied from README file
st.header("🎯Business Requirements")
st.write(
    "In this project our client has stated two business "
    "requirements that we need to address:\n"
    "1. The client is interested in discovering how the house "
    "attributes correlate with the sale price. Therefore, the "
    "client expects data visualizations of the correlated "
    "variables against the sale price to show that. \n"
    "This is addressed in the "
    "[Correlation Analysis](analysis) page.\n"
    "2. The client is interested in predicting the house sale "
    "price from her four inherited houses and any "
    "other houses in Ames, Iowa.\n"
    "This is addressed in the "
    "[Predict House Sale Price](price_predictions) "
    "and [Machine Learning Model](ml_pipeline_prediction) pages."
    )

# Horizontal line
st.divider()

# Link to README file, section "Requirements and Hypotheses"
st.write(
    (
        "#### ⬇️ Wanna know more about this project?\n"
        "Please visit [README file]"
        "(https://github.com/NatashaRy/"
        "house-price-predictor/blob/main/README.md) "
        "for additional information."
    )
)
