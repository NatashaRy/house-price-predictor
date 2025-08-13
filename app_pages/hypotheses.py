import streamlit as st


def hypotheses_body():
    """
    Displays content of the hypothesis page in the Streamlit app.

    This includes: 
    - Introduction to the hypothesis
    - Detailed list of hypotheses with rationale, validation and results
    """
    # Title and introduction
    st.title("Hypotheses and Validation")
    st.write(
        """
        **This page outlines the hypotheses that guide the analysis of house prices.
        The hypotheses are based on domain knowledge and aim to identify key factors that influence house prices.
        Each hypothesis is tested using statistical methods and visualizations to validate its relevance.
        These hypotheses are directly tied to the client's business requirements, specifically *Business Requirement 1*,
        which focuses on understanding how house attributes correlate with sale prices.**
        """
    )

    st.markdown("---")

    st.write("## Hypotheses and Results\n")


    ### Hypothesis 1
    st.markdown("\n### Hypothesis 1: Larger houses have higher sale price")
    st.markdown(
        """
        - **Rationale**: It is expected that houses with larger living areas (`GrLivArea`, `GarageArea`, `TotalBsmtSF`) 
          will have a higher price due to their size and usability.\n
        - **Validation**: 
            - Performed correlation analysis between `GrLivArea` and `SalePrice`.
            - Created a scatter plot to visualize the relationship.
            - Calculated the correlation coefficient to determine the strength of the relationship.\n
        - **Result**: **Correct** ✅  
            - The analysis confirmed a strong positive correlation between `GrLivArea` and `SalePrice` (Pearson correlation coefficient: 0.71). 
            - The scatter plot showed a clear upward trend, validating the hypothesis.\n
        """
    )


    ### Hypothesis 2
    st.markdown("\n### Hypothesis 2: Houses with higher overall quality have higher sale price")
    st.markdown(
        """
        - **Rationale**: Houses with better construction quality and finishes (`OverallQual`) are likely to command higher prices 
          due to their durability, aesthetics, and buyer appeal.\n
        - **Validation**: 
            - Performed correlation analysis between `OverallQual` and `SalePrice`.
            - Created boxplots to compare `SalePrice` across different quality levels.\n
        - **Result**: **Correct** ✅
            - The analysis showed a strong positive correlation (Pearson correlation coefficient: 0.79). 
            - Boxplots revealed a clear trend of increasing `SalePrice` with higher `OverallQual` levels.\n
        """
    )


    ### Hypothesis 3
    st.markdown("\n### Hypothesis 3: Newer houses have higher sale price")
    st.markdown(
        """
        - **Rationale**: Newer houses (`YearBuilt`) are expected to have higher prices due to modern designs, better materials, 
          and lower maintenance costs.\n
        - **Validation**: 
            - Performed correlation analysis between `YearBuilt` and `SalePrice`.
            - Created a line plot to visualize the relationship.\n
        - **Result**: **Correct** ✅  
            - The analysis confirmed a moderate positive correlation (Pearson correlation coefficient: 0.52). 
            - The line plot showed that newer houses tend to have higher sale prices.\n
        """
    )


    ### Hypothesis 4
    st.markdown("\n### Hypothesis 4: Houses with garages have higher sale price\n")
    st.markdown(
        """
        - **Rationale**: Houses with garages (`GarageArea`) are more desirable as they provide additional storage and parking space, 
          which adds value.\n
        - **Validation**: 
            - Performed correlation analysis between `GarageArea` and `SalePrice`.
            - Created a scatter plot to compare `SalePrice` for houses with and without garages.\n
        - **Result**: **Correct** ✅ 
            - The analysis showed a strong positive correlation (Pearson correlation coefficient: 0.62). 
            - The scatter plot confirmed that houses with larger garage areas tend to have higher sale prices.\n
        """
    )

    st.markdown("---")

##### Conclusion of hypotheses and results
    st.info("### **Summary of Findings**\n"
        "\nThe hypotheses were validated using statistical methods and visualizations. All four hypotheses were confirmed to be correct:\n"
        "- Larger houses, higher quality, newer construction, and the presence of garages are all strongly associated with higher sale prices.\n"
        "- These findings align with **Business Requirement 1** and provide actionable insights for feature engineering and predictive modeling.\n"
    )
    