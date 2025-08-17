import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps
from feature_engine.encoding import OneHotEncoder
from src.data_management import load_pricing_data

# Set theme style
sns.set_theme(style="darkgrid")

# Page title
page_title = "Correlation Analysis"


def analysis_body():
    """
    Displays content of the correlation analysis page in the Streamlit app.

    This includes:
    - Introduction of correlation analysis
    - Reminder of business requirement 1
    - Optional: Checkbox to inspect raw dataset
    - Conclusions with key observations
    - Data visualizations:
        - Optional: Dropdown for custom heatmap variables
        - Optional: Checkboxes to display predefined heatmaps for:
            - Pearson
            - Spearman
            - PPS Matrix
        - Optional: Checkbox to display distribution of Sale Price
        - Optional: Checkbox to display bivariate analysis
    """
    # Load raw house pricing data
    df = load_pricing_data()

    # Check if data is loaded successfully
    if df is None:
        st.error("Failed to load the dataset. Please check the file path.")
        return

    """
    Code from Notebook 3,
    Section 3.1: Convert Categorical Variables
    """
    categorical_vars = df.select_dtypes(include='object').columns
    numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns

    df[categorical_vars] = df[categorical_vars].fillna("Missing")
    for col in numerical_vars:
        df[col] = df[col].fillna(df[col].median())

    """
    Code from Notebook 3,
    Section 3.1: Convert Categorical Variables
    """
    encoder = OneHotEncoder(
        variables=categorical_vars.to_list(), drop_last=False
    )
    df_ohe = encoder.fit_transform(df)

    # Code from Notebook 3, Section 4.1: Combine Top Correlation Features
    vars_to_study = [
        '1stFlrSF', 'GarageArea', 'GarageYrBlt', 'GrLivArea',
        'KitchenQual_Ex', 'KitchenQual_Gd', 'KitchenQual_TA',
        'MasVnrArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd'
    ]

    # Code from Notebook 3, Section 4.2: Create New DataFrame for
    df_eda = df_ohe.filter(vars_to_study + ['SalePrice'])

    # Check if all variables are present
    missing_vars = [var for var in vars_to_study if var not in df_eda.columns]
    if missing_vars:
        st.error(
            f"The following variables are missing from the dataset: "
            f"{missing_vars}"
        )
        return

    # Title and introduction
    st.title(page_title)
    st.markdown(
        """
        On this page, we explore how various house attributes
        influence the sale price.
        Through interactive visualizations and correlation
        metrics, you can uncover
        patterns and relationships that are key to understanding the dataset.

        Use the tools below to inspect predefined heatmaps,
        analyze the distribution
        of the target variable, and perform bivariate analysis
        for deeper insights.
        """
        )

    # Reminder of business requirement 1
    st.info(
        "This page is designed to answer **Business Requirement 1**: "
        "The client is interested in discovering how the house attributes "
        "correlate with the sale price. Therefore, the client expects data "
        "visualizations of the correlated variables against the sale price "
        "to show that."
    )

    # Code from Notebook 3, Section 1: Load Data
    if st.checkbox("Would you like to inspect the raw dataset?"):
        st.write("##### Inspection of house prices raw data")
        st.write(
            f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
        )
        st.write(df.head(10))

    st.markdown("---")

    # Conclusions
    st.write(
        "## Conclusions\n"
        "The correlation analysis and visualization provided valuable "
        "insights into the relationships between house attributes and sale "
        "prices. These findings will guide the next steps in feature "
        "engineering and modeling.\n\n"
        "### Key Observations:\n"
        "1. **Size Matters**: Larger properties, as indicated by variables "
        "like `1stFlrSF`, `GrLivArea`, `TotalBsmtSF`, and `GarageArea`, "
        "are strongly associated with higher sale prices.\n"
        "2. **Time Matters**: Recently built houses (`YearBuilt`) and "
        "houses with recent remodels (`YearRemodAdd`) tend to have higher "
        "sale prices, reflecting the value of modern features and updates.\n"
        "3. **Quality Matters**: Higher overall quality (`OverallQual`) and "
        "kitchen quality (`KitchenQual`) ratings are among the strongest "
        "predictors of higher sale prices.\n"
    )

    st.markdown("---")

    # Code from Notebook 3, Section 3.2: Calculate and Visualize
    st.write(
        "## Data Visualizations\n"
        "To further illustrate the findings from the correlation study, "
        "we provide the following visualizations:\n"
        "- Custom heatmaps with chosen variables by user.\n"
        "- Predefined heatmaps correlation and predictive power score (PPS).\n"
        "- Histogram for distribution of target variable.\n"
        "- Scatter plots for each of the most correlated variables "
        "against sale price.\n"
        "\nThese visualizations will help in understanding the "
        "relationships more intuitively."
    )

    # Code from Notebook 3, Section 3.4: Generate Correlation Heatmaps
    st.write(
        "\n### Heatmaps\n"
        "Below are visualizations of correlation heatmaps using "
        "different methods."
    )

    # Dynamic variable selection
    st.write("\n#### Custom Correlation Heatmap")
    st.write(
        """
        Select one or more variables from the list below to
         generate a custom correlation heatmap. This allows you to
          focus on specific relationships of interest and analyze
          their relationship. The heatmap will only include the
          selected variables.
        """
        )
    selected_vars = st.multiselect(
        "Select variables for analysis", df_eda.columns
    )
    if selected_vars:
        st.write("Correlation Heatmap")
        corr = df_eda[selected_vars].corr()
        correlation_heatmap(corr, "Custom Correlation Heatmap")

    st.write("\n#### Predefined Heatmaps")
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
        pps_matrix = pps_matrix_raw.pivot(
            index='y', columns='x', values='ppscore'
        )
        pps_heatmap(pps_matrix, "PPS Heatmap")

    # Code from Notebook 3, Section 4.3: Visualization of Target Variable
    st.write("\n### Distribution of Target Variable")
    if st.checkbox("Show Distribution"):
        plot_target_hist(df, "SalePrice")

    # Code from Notebook 3, Section 4.4: Bivariate Analysis of Key Variables
    st.write("\n### Bivariate Analysis of Key Variables and SalePrice")
    st.write(
        "Check the box below to display visualizations for all key variables:"
    )

    if st.checkbox("Show all visualizations for key variables"):
        for var in vars_to_study:
            # Check if the variable is categorical or continuous
            if len(df_eda[var].unique()) <= 10:  # Categorical variables
                plot_box(df_eda, var, 'SalePrice')
            else:  # Continuous variables
                plot_lm(df_eda, var, 'SalePrice')


# Functions
def correlation_heatmap(df, title, threshold=0.5):
    """
    Code from Notebook 3, Section 3.4:
    Generate Correlation Heatmaps

    Args:
        df (pd.DataFrame): Correlation matrix
        title (str): Title for the heatmap
        threshold (float): Minimum correlation value to display
    """
    if df.shape[1] > 1:  # Check if there are enough columns
        # Filter rows and columns with values below the threshold
        filtered_data = df.loc[
            (abs(df) >= threshold).any(axis=1),
            (abs(df) >= threshold).any(axis=0)
        ]

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


def pps_heatmap(df, title, threshold=0.2):
    """
    Code from Notebook 3, Section 3.6.2:
    Generate Heatmaps for PPS Matrix

    Args:
        df (pd.DataFrame): PPS matrix
        title (str): Title for the heatmap
        threshold (float): Minimum PPS value to display
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
    # Remove axis titles
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_title(title, fontsize=16)
    st.pyplot(fig)


def plot_target_hist(df, target_var):
    """
    Code from Notebook 3, Section 4.3:
    Visualization of Target Variable

    Args:
        df (pd.DataFrame): Dataset containing the target variable
        target_var (str): Name of the target variable column
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


def plot_box(df, col, target_var):
    """
    Code from Notebook 3,
    Section 4.4: Bivariate Analysis of Key Variables and SalePrice
    Create a boxplot for a categorical variable against the target variable.

    Args:
        df (pd.DataFrame): Dataset containing the variables
        col (str): Name of the categorical variable column
        target_var (str): Name of the target variable column
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


def plot_lm(df, col, target_var):
    """
    Code from Notebook 3, Section 4.4:
    Bivariate Analysis of Key Variables and SalePrice

    Args:
        df (pd.DataFrame): Dataset containing the variables
        col (str): Name of the continuous variable column
        target_var (str): Name of the target variable column
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
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{col}", fontsize=9)

    plt.title(f"{col} vs {target_var}", fontsize=20)
    plt.xlabel(col, fontsize=9)
    plt.ylabel(target_var, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)

    fig = plt.gcf()
    st.pyplot(fig)
