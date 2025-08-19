import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps
from feature_engine.encoding import OneHotEncoder
from src.data_management import load_pricing_data
from utils import create_toc

# This page displays content of the
# correlation analysis page in the Streamlit app.
# This includes:
# - Page introduction
# - Optional: Checkbox to review raw dataset,
#   displays raw data table if checked
# - Summary of correlation analysis
#   - Optional: Expander with more details about the analysis
#   - Key observations and insights
# - Tabs to display heatmaps:
#     - Tab 1: Predefined heatmaps
#           - Checkbox to display Pearson Correlation Heatmap
#           - Checkbox to display Spearman Correlation Heatmap
#           - Checkbox to display PPS Matrix Heatmap
#     - Tab 2: Custom heatmap
#           - Users can select their own variables for correlation
#             analysis using the multiselect dropdown.
# - Checkbox to display distribution plot of target variable.
# - Checkbox to display bivariate analysis plots of key variables.

# Define page configuration
st.set_page_config(
    page_title="Correlation Analysis",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Create sidebar table of contents
create_toc([
    ("📑 Summary of Analysis", "summary-of-analysis"),
    ("🔥 Heatmaps", "heatmaps"),
    ("🎯 Target Distribution", "target-variable-distribution"),
    ("📊 Bivariate Analysis", "bivariate-analysis"),
], page_title="Correlation Analysis")

# Set theme style
sns.set_theme(style="darkgrid")

# Title and introduction
st.title("📈Correlation Analysis")
st.write(
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

st.markdown(" ")


# Function to create a correlation heatmap
def correlation_heatmap(df, threshold=0.5, figsize=(12, 8),
                        font_size=8, title="Correlation Heatmap"):
    """
    Generate a heatmap to visualize strong correlations between variables.
    Code copied from Notebook 3:
    Section: 5.2 Calculate and Visualize Relationships in Dataset
    """
    if df.shape[1] > 1:  # Check for enough columns
        # Filter rows and columns with values below the threshold
        filtered_data = df.loc[(abs(df) >= threshold).any(axis=1),
                               (abs(df) >= threshold).any(axis=0)]

        # Create a mask to hide the upper triangle
        # and values below the threshold
        mask = np.zeros_like(filtered_data, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[(abs(filtered_data) <= 0.2)] = True

        # Format data for better readability
        formatted_data = filtered_data.applymap(
                        lambda x: round(x, 2) if abs(x) > 0.2 else 0)

        # Draw heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            formatted_data,
            annot=True,
            cmap=sns.color_palette("Spectral"),
            mask=mask,
            annot_kws={"size": font_size},
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(title, fontsize=14)
        st.pyplot(fig)


# Function to create a PPS heatmap
def pps_heatmap(df, threshold=0.2, figsize=(12, 8),
                font_size=8, title="PPS Heatmap"):
    """
    Generate a heatmap to visualize Predictive Power Score (PPS)
    between variables.
    Code copied from Notebook 3:
    Section: 5.2 Calculate and Visualize Relationships in Dataset
    """
    if df.shape[1] > 1:
        # Filter rows and columns with values under the threshold
        filtered_data = df.loc[(abs(df) >= threshold).any(axis=1),
                               (abs(df) >= threshold).any(axis=0)]

        # Create a mask to hide values <= threshold
        # and values that are exactly 0
        mask = np.zeros_like(filtered_data, dtype=bool)
        mask[abs(filtered_data) <= 0.2] = True
        mask[filtered_data == 0] = True

        # Format data for better readability
        formatted_data = filtered_data.applymap(
                        lambda x: round(x, 2) if abs(x) > 0.2 else 0)

        # Draw heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            formatted_data,
            annot=True,
            cmap=sns.color_palette("Spectral"),
            annot_kws={"size": font_size},
            linewidths=0.5,
            mask=mask,
            ax=ax
        )
        ax.set_title(title, fontsize=14)
        st.pyplot(fig)


# Code copied from Notebook 3
# Section: 6.3 Bivariate Analysis of Key Variables and SalePrice
# Dictionary for variable names
variable_names = {
    'YearBuilt': 'Year Built',
    'YearRemodAdd': 'Year Remodeled/Added',
    'GrLivArea': 'Above Ground Living Area (sq ft)',
    'GarageArea': 'Garage Area (sq ft)',
    'OverallQual': 'Overall Quality (1-10 scale)',
    'SalePrice': 'Sale Price (USD)',
    '1stFlrSF': '1st Floor Area (sq ft)',
    'KitchenQual_Gd': 'Kitchen Quality - Good',
    'KitchenQual_Ex': 'Kitchen Quality - Excellent',
    'KitchenQual_TA': 'Kitchen Quality - Typical/Average',
    'TotalBsmtSF': 'Total Basement Area (sq ft)',
    'MasVnrArea': 'Masonry Veneer Area (sq ft)',
    'GarageYrBlt': 'Garage Year Built',
}

# Time variables
time_vars = ['YearBuilt', 'YearRemodAdd']


# Function to create a scatterplot with regression line
def plot_lm(df, col, target_var):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Scatter plot
    scatter = ax.scatter(
        x=df[col],
        y=df[target_var],
        c=df[col],
        cmap='Spectral',
        alpha=0.7,
        edgecolor='k'
    )

    # Regression line
    sns.regplot(
        data=df,
        x=col,
        y=target_var,
        scatter=False,
        line_kws={'color': 'black'},
        ax=ax
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f"{col}", fontsize=9)

    # Title and labels
    ax.set_title(
        f"{variable_names.get(col, col)} vs {target_var}", fontsize=20
    )
    ax.set_xlabel(variable_names.get(col, col), fontsize=9)
    ax.set_ylabel(variable_names.get(target_var, target_var), fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)


# Function to create a boxplot for categorical variables
def plot_box(df, col, target_var):
    num_categories = len(df[col].unique())
    palette = sns.color_palette("Spectral", n_colors=num_categories)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df,
        x=col,
        y=target_var,
        palette=palette,
        ax=ax
    )

    # Title and labels
    ax.set_title(
        f"{variable_names.get(col, col)} vs {target_var}", fontsize=20
    )
    ax.set_xlabel(variable_names.get(col, col), fontsize=9)
    ax.set_ylabel(variable_names.get(target_var, target_var), fontsize=9)

    st.pyplot(fig)


# Function to create a line plot for time variables
def plot_line(df, col, target_var):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=df,
        x=col,
        y=target_var,
        color=sns.color_palette("Spectral")[1],
        ax=ax
    )

    # Title and labels
    ax.set_title(
        f"{variable_names.get(col, col)} vs {target_var}", fontsize=20
    )
    ax.set_xlabel(variable_names.get(col, col), fontsize=9)
    ax.set_ylabel(variable_names.get(target_var, target_var), fontsize=9)

    st.pyplot(fig)


# Main analysis function
def analysis():
    """
    Main function to display the correlation analysis page.
    """
    # Define target variable
    target_var = 'SalePrice'

    # Load raw house pricing data
    df = load_pricing_data()

    # Check if data is loaded successfully
    if df is None:
        st.error("Failed to load the dataset. Please check the file path.")
        return

    # Process categorical and numerical variables
    categorical_vars = df.select_dtypes(include='object').columns
    numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns

    df[categorical_vars] = df[categorical_vars].fillna("Missing")
    for col in numerical_vars:
        df[col] = df[col].fillna(df[col].median())

    # One-hot encoding
    encoder = OneHotEncoder(
        variables=categorical_vars.to_list(), drop_last=False
    )
    df_ohe = encoder.fit_transform(df)

    # Select variables for analysis
    vars_to_study = [
        '1stFlrSF', 'GarageArea', 'GarageYrBlt', 'GrLivArea',
        'KitchenQual_Ex', 'KitchenQual_Gd', 'KitchenQual_TA',
        'MasVnrArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd'
    ]

    df_eda = df_ohe.filter(vars_to_study + [target_var])

    # Check for missing variables
    missing_vars = [var for var in vars_to_study if var not in df_eda.columns]
    if missing_vars:
        st.error(
            f"The following variables are missing from the dataset: "
            f"{missing_vars}"
        )
        return

    # Optional: Inspect raw dataset
    if st.checkbox("**Would you like to inspect the raw dataset?** 🔍"):
        st.write("##### Inspection of house prices raw data")
        st.write(
            f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
        )
        st.write(df.head(10))

    # Horizontal line
    st.divider()

    # Conclusions of correlation analysis
    # Copied from Notebook 3, section Conclusion
    st.header("📑Summary of Analysis")
    st.write(
        """
        The correlation analysis and visualization process
         was successfully completed, providing valuable
         insights into the relationships between house attributes
         and sale prices.
         These findings will guide the next steps in feature
         engineering and modeling.
        """
    )

    # Expander for detailed analysis
    with st.expander(
            "**Read more details about the analysis**",
            expanded=False):
        st.subheader("Detailed Analysis")
        st.write(
            """
            This section provides a deeper dive into the correlation
             analysis conducted on the dataset. By exploring different
              correlation methods and visualizations,we aim to uncover
              meaningful relationships between house attributes and sale
              prices. These insights are critical for understanding the dataset
              and guiding the feature engineering and modeling processes.
            """
        )
        # Correlations methods explained
        st.write(
            """
            #### **Correlation Methods**:\n
            - **Pearson Correlation**: Measures linear relationships
             between variables.
            - **Spearman Correlation**: Captures monotonic relationships,
             even if they are non-linear.
            - **Predictive Power Score (PPS)**: Quantifies the predictive
             strength of one variable for another.
            """
        )
        # Horizontal divider
        st.divider()
        # Heatmap Insights
        st.write(
            """
            #### **Heatmap Insights**:\n
            - Variables like `GrLivArea` and `OverallQual` show strong
             positive correlations with `SalePrice`.
            - PPS highlights non-linear relationships that traditional
             correlation metrics might miss.
            """
        )
        # Horizontal divider
        st.divider()
        # Bivariate Analysis Insights
        st.write(
            """
            #### **Bivariate Analysis Insights**:\n
            - Scatterplots and regression lines reveal trends between
             numerical variables and `SalePrice`.
            - Boxplots show how categorical variables like
             `KitchenQual` impact prices.
            """
        )
        # Whitespace
        st.markdown(" ")
        # Informational box
        st.info(
            """
            All visualizations were saved in the `docs/plots`
            directory for further use in the Streamlit app and
            to meet **Business Requirement 1**.
            """
        )

    # Key Observations from analysis
    st.info(
        """
        ### **Key Observations**:
        1. **Size Matters**: Larger properties, as indicated by
         variables like `1stFlrSF`, `GrLivArea`,
        `TotalBsmtSF`, and `GarageArea`, are strongly associated
         with higher sale prices.
        2. **Time Matters**: Recently built houses (`YearBuilt`)
         and houses with recent remodels (`YearRemodAdd`) tend
         to have higher sale prices, reflecting the value of
         modern features and updates.
        3. **Quality Matters**: Higher overall quality (`OverallQual`)
         and kitchen quality (`KitchenQual`)
         ratings are among the strongest predictors of higher sale prices.
        """
    )

    # Visualization of Heatmaps
    st.header("🔥Heatmaps")

    # Create tabs for heatmaps
    tab1, tab2 = st.tabs([
        "**Predefined Heatmaps**",
        "**Create Custom Heatmap**"
    ])

    # Tab 1: Predefined heatmap
    with tab1:
        st.subheader("Predefined Heatmaps")
        st.write("Check one or more boxes to display "
                 "one of our predefined heat maps."
                 )
        if st.checkbox("Display Pearson Correlation Heatmap"):
            pearson_corr = df_eda.corr(method="pearson")
            correlation_heatmap(
                pearson_corr,
                threshold=0.5,
                title="Pearson Correlation Heatmap"
            )

        if st.checkbox("Display Spearman Correlation Heatmap"):
            spearman_corr = df_eda.corr(method="spearman")
            correlation_heatmap(
                spearman_corr,
                threshold=0.5,
                title="Spearman Correlation Heatmap"
            )

        if st.checkbox("Display PPS Matrix Heatmap"):
            df_eda_pps = df_eda.select_dtypes(include=[np.number])
            df_eda_pps = df_eda_pps.fillna(0)
            pps_matrix_raw = pps.matrix(df_eda_pps)
            pps_matrix = pps_matrix_raw.pivot(
                index='y', columns='x', values='ppscore'
            )
            pps_heatmap(pps_matrix, threshold=0.2, title="PPS Heatmap")

    # Tab 2: Custom heatmap
    with tab2:
        st.subheader("Custom Heatmap")
        selected_vars = st.multiselect(
            "Select variables for the custom heatmap:",
            options=vars_to_study,
            default=vars_to_study[:3]  # Preselect the first 3 variables
        )
        if selected_vars:
            custom_corr = df_eda[selected_vars].corr()
            correlation_heatmap(
                custom_corr,
                threshold=0.0,
                title="Custom Correlation Heatmap"
            )
        else:
            st.warning("Please select at least one variable to "
                       "generate the custom heatmap."
                       )

    # Horizontal line
    st.divider()

    # Target Variable Distribution
    # Code copied from Notebook 3:
    # Section: 6.2 Visualization of Target Variable Distribution
    st.header("🎯Target Variable Distribution")
    if st.checkbox("Display Distribution of Target Variable"):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            df[target_var],
            kde=True,
            color=sns.color_palette("Spectral")[0],
            ax=ax
        )
        ax.set_title(f"Distribution of {target_var}", fontsize=18)
        st.pyplot(fig)

    # Horizontal line
    st.divider()

    # Bivariate Analysis
    st.header("📊Bivariate Analysis")
    if st.checkbox("Display all visualizations for key variables"):
        for col in vars_to_study:
            if len(df_eda[col].unique()) <= 10:
                plot_box(df_eda, col, target_var)
            elif col in time_vars:
                plot_line(df_eda, col, target_var)
            else:
                plot_lm(df_eda, col, target_var)


# Run function with page content
analysis()
