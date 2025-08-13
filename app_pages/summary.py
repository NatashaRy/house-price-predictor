import streamlit as st
import pandas as pd
from src.data_management import load_pricing_data


def summary_body():
    """
    Displays content of the quick summary page in the Streamlit app.

    This includes: 
    - Project Overview
    - Dataset Details
    - Business Requirements
    - Link to README file
    """
    # Title and introduction
    st.title("Project Overview")
    st.info("### Summary\n"
             "**This project, 'House Price Predictor,' aims to analyze house attributes and predict sale prices for houses in Ames, Iowa.\n"
             "The goal is to help the client maximize the sales price of their inherited properties by providing insights into the most important factors affecting house prices and enabling accurate price predictions.**"
    )

    # Displays Dataset Details
    st.write("### Dataset Details\n"
             "The dataset used in this project is the [Heritage Housing dataset](https://www.kaggle.com/datasets/your-dataset-link). The dataset contains information about house in Ames, Iowa, including their attributes and sale prices.\n"
             "#### Quick dataset summary:")
    # Displays dataset characteristics
    try:
        df = load_pricing_data()
        st.write(f"* **Number of rows**: {df.shape[0]}")
        st.write(f"* **Number of columns**: {df.shape[1]}")
        st.write(f"* **Target variable**: `SalePrice`")
        st.write(f"* **Key variables**: `GrLivArea`, `OverallQual`, `YearBuilt`")
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure the dataset is available in the specified path.")
    
    # Horizontal line
    st.markdown("---")

    # Displays Business Requirements
    st.write("### Business Requirements\n"
            "In this project our client has stated two business requirements that we need to address:\n"
            "1. The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.\n"
            "2. The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa."
    )
    st.markdown("---")

    # Link to README file
    st.markdown("#### ⬇️ Wanna know more about this project?\n"
                "Please visit [README file](https://github.com/NatashaRy/milestone-project-heritage-housing-issues/blob/main/README.md) for additional information."
    )