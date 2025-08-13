import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.encoding import OneHotEncoder
from src.data_management import load_pricing_data
from feature_engine.discretisation import ArbitraryDiscretiser
import ppscore as pps

# Set theme style
sns.set_theme(style="darkgrid")


def analysis_body():
    """
    Displays content of the data analysis page in the Streamlit app.

    This includes: 
    - Introduction
    - Correlation analysis with possibility to inspect dataset
    - Visualizations:
        - Heatmaps for Pearson, Spearman, and PPS Matrix
        - Distribution of Sale Price
        - Bivariate Analysis
    """
    # Load raw house pricing data
    df = load_pricing_data()

    # Check if data is loaded successfully
    if df is None:
        st.error("Failed to load the dataset. Please check the file path.")
        return

##### Handle missing values
    # Code from Notebook 3, Section 3.1: Convert Categorical Variables
    categorical_vars = df.select_dtypes(include='object').columns
    numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns

    df[categorical_vars] = df[categorical_vars].fillna("Missing")
    for col in numerical_vars:
        df[col] = df[col].fillna(df[col].median())

##### Perform one-hot encoding
    # Code from Notebook 3, Section 3.1: Convert Categorical Variables
    encoder = OneHotEncoder(variables=categorical_vars.to_list(), drop_last=False)
    df_ohe = encoder.fit_transform(df)

    ##### List of variables to study in the analysis
    # Code from Notebook 3, Section 4.1: Combine Top Correlation Features
    vars_to_study = [
        '1stFlrSF', 'GarageArea', 'GarageYrBlt', 'GrLivArea',
        'KitchenQual_Ex', 'KitchenQual_Gd', 'KitchenQual_TA',
        'MasVnrArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd'
    ]

##### Create DataFrame for analysis
    # Code from Notebook 3, Section 4.2: Create New DataFrame for Exploratory Data Analysis
    df_eda = df_ohe.filter(vars_to_study + ['SalePrice'])

    # Check if all variables are present
    missing_vars = [var for var in vars_to_study if var not in df_eda.columns]
    if missing_vars:
        st.error(f"The following variables are missing from the dataset: {missing_vars}")
        return

##### Title and introduction
    # Code from Notebook 3, Section 3: Correlation and PPS Analysis
    st.title("Correlation Analysis")
    st.markdown(
        "**This page explores the relationships between house attributes and the target variable, sale price.**\n"
        "**The goal is to identify key predictors for modeling and provide insights into how these attributes influence house prices.**"
    )

##### Reminder of business requirement 1
    st.info("This page is designed to answer **Business Requirement 1**: "
        "The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualizations of the correlated variables against the sale price to show that."
    )

##### Optional inspection of the first 10 rows of the raw dataset
    # Code from Notebook 3, Section 1: Load Data
    if st.checkbox("Would you like to inspect the raw dataset?"):
        st.write("##### Inspection of house prices raw data")
        st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        st.write(df.head(10))

    st.markdown("---")

##### Conclusions
    st.write("## Conclusions\n"
             "The correlation analysis and visualization provided valuable insights into the relationships between house attributes and sale prices. These findings will guide the next steps in feature engineering and modeling.\n\n"
             "### Key Observations:\n"
             "1. **Size Matters**: Larger properties, as indicated by variables like `1stFlrSF`, `GrLivArea`, `TotalBsmtSF`, and `GarageArea`, are strongly associated with higher sale prices.\n"
             "2. **Time Matters**: Recently built houses (`YearBuilt`) and houses with recent remodels (`YearRemodAdd`) tend to have higher sale prices, reflecting the value of modern features and updates.\n"
             "3. **Quality Matters**: Higher overall quality (`OverallQual`) and kitchen quality (`KitchenQual`) ratings are among the strongest predictors of higher sale prices.\n"
    )

    st.markdown("---")

##### Visualizations
    # Code from Notebook 3, Section 3.2: Calculate and Visualize Relationships in Dataset
    st.write("## Data Visualizations\n"
             "To further illustrate the findings from the correlation study, we provide the following visualizations:\n"
             "- Heatmaps of the correlation and predictive power score (PPS).\n"
             "- Histogram for distribution of target variable.\n"
             "- Scatter plots for each of the most correlated variables against sale price.\n"
             "\nThese visualizations will help in understanding the relationships more intuitively."
    )

#### Heatmap Section
    # Code from Notebook 3, Section 3.4: Generate Correlation Heatmaps
    st.write("\n### Heatmaps\n"
             "Below are visualizations of correlation heatmaps using different methods."
    )

    # Pearson Correlation Heatmap
    if st.checkbox("Show Pearson Correlation Heatmap"):
        pearson_corr = df_eda.corr(method="pearson")
        correlation_heatmap(pearson_corr, "Pearson Correlation Heatmap")

    # Spearman Correlation Heatmap
    if st.checkbox("Show Spearman Correlation Heatmap"):
        spearman_corr = df_eda.corr(method="spearman")
        correlation_heatmap(spearman_corr, "Spearman Correlation Heatmap")
    
    # PPS Heatmap
    if st.checkbox("Show PPS Matrix Heatmap"):
        df_eda_pps = df_eda.select_dtypes(include=[np.number])
        df_eda_pps = df_eda_pps.fillna(0)
        pps_matrix_raw = pps.matrix(df_eda_pps)
        pps_matrix = pps_matrix_raw.pivot(index='y', columns='x', values='ppscore')
        pps_heatmap(pps_matrix, "PPS Heatmap")

#### Distribution of Target Variable
    # Code from Notebook 3, Section 4.3: Visualization of Target Variable Distribution
    st.write("\n### Distribution of Target Variable")
    if st.checkbox("Show Distribution"):
        plot_target_hist(df, "SalePrice")

#### Bivariate Analysis of Key Variables and SalePrice
    # Code from Notebook 3, Section 4.4: Bivariate Analysis of Key Variables and SalePrice
    st.write("\n### Bivariate Analysis of Key Variables and SalePrice")
    st.write("Check the box below to display visualizations for all key variables:")

    if st.checkbox("Show all visualizations for key variables"):
        for var in vars_to_study:
            # Check if the variable is categorical or continuous
            if len(df_eda[var].unique()) <= 10:  # Categorical variables
                plot_box(df_eda, var, 'SalePrice')
            else:  # Continuous variables
                plot_lm(df_eda, var, 'SalePrice')
    

##### Functions
# Code from Notebook 3, Section 3.4: Generate Correlation Heatmaps
def correlation_heatmap(df, title, threshold=0.5):
    """
    Generate a heatmap to visualize correlations between variables.
    """
    if df.shape[1] > 1:  # Check if there is enough columns
        # Filter rows and columns with values below the threshold
        filtered_data = df.loc[(abs(df) >= threshold).any(axis=1), (abs(df) >= threshold).any(axis=0)]

        # Create mask for upper triangle and values <= 0.2
        mask = np.zeros_like(filtered_data, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[(abs(filtered_data) <= 0.2)] = True

        # Draw heatmap
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(
            filtered_data,
            annot=True,
            cmap="Spectral",
            mask=mask,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title(title, fontsize=16)
        st.pyplot(fig)

# Code from Notebook 3, Section 3.6.2: Generate Heatmaps for PPS Matrix
# PPS Heatmap
def pps_heatmap(df, title, threshold=0.2):
    """
    Generate a heatmap for PPS matrices, filtering out weak predictive scores.
    """
    # Create mask for upper triangle and values <= 0.2
    mask = np.zeros_like(df, dtype=bool)
    mask[abs(df) <= threshold] = True

    # Draw heatmap
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        df,
        mask=mask,
        annot=True,
        cmap="Spectral",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)

# Code from Notebook 3, Section 4.3: Visualization of Target Variable Distribution
def plot_target_hist(df, target_var):
    """
    Plot a histogram of the target variable with KDE overlay.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=df,
        x=target_var,
        kde=True,
        color=sns.color_palette("Spectral")[0],
        ax=ax
    )
    ax.set_title(f"Distribution of {target_var}", fontsize=20)
    ax.set_xlabel(target_var, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    st.pyplot(fig)


# Code from Notebook 3, Section 4.4: Bivariate Analysis of Key Variables and SalePrice
def plot_box(df, col, target_var):
    """
    Create a boxplot for a categorical variable against the target variable.
    """
    num_categories = len(df[col].unique())
    palette = sns.color_palette("Spectral", n_colors=num_categories)

    plt.figure(figsize=(8, 4))
    sns.boxplot(
        data=df,
        x=col, 
        y=target_var, 
        palette=palette
    )
    plt.title(f"{col}", fontsize=20)
    plt.xlabel(col, fontsize=9)
    plt.ylabel(target_var, fontsize=9)
    fig = plt.gcf()
    st.pyplot(fig)


# Code from Notebook 3, Section 4.4: Bivariate Analysis of Key Variables and SalePrice
def plot_lm(df, col, target_var):
    """
    Create a linear regression plot for a continuous variable against the target variable.
    """
    plt.figure(figsize=(8, 4))

    # Scatterplot
    scatter = plt.scatter(
        x=df[col],
        y=df[target_var],
        c=df[col],
        cmap='Spectral',
        alpha=0.7,
        edgecolors='k'
    )

    # Regression plot
    sns.regplot(
        data=df,
        x=col,
        y=target_var,
        scatter=False,
        line_kws={"color": "black"}
    )

    # Colors
    cbar= plt.colorbar(scatter)
    cbar.set_label(f"{col}", fontsize=9)

    plt.title(f"{col} vs {target_var}", fontsize=20)
    plt.xlabel(col, fontsize=9)
    plt.ylabel(target_var, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    fig = plt.gcf()
    st.pyplot(fig)