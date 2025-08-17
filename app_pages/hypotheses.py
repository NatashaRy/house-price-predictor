import streamlit as st

# Page title
page_title = "Hypotheses and Validation"


def hypotheses_body():
    """
    Displays content of the hypothesis page in the Streamlit app.

    This includes:
    - Page introduction
    - Summary of findings from hypotheses validation
    - Dropdown to select hypothesis to explore
    - Detailed list of four hypotheses, includes:
        - Rationale
        - Validation
        - Plot image
        - Results
    """
    # Title and introduction
    st.title(page_title)
    st.write(
        """
        **This page outlines the hypotheses that guide
        the analysis of house prices. The hypotheses are based
        on domain knowledge and aim to identify key factors that
        influence house prices. Each hypothesis is tested using
        statistical methods and visualizations to validate its relevance.
        These hypotheses are directly tied to the client's business
        requirements, specifically *Business Requirement 1*, which focuses
        on understanding how house attributes correlate with sale prices.**
        """
        )

    st.markdown("---")

    # Conclusion of hypotheses and results
    st.write("## Hypotheses and Results\n")
    st.info("### **Summary of Findings**\n"
            "\nThe hypotheses were validated using statistical "
            "methods and visualizations. All four hypotheses were "
            "confirmed to be correct:\n"
            "- Larger houses, higher quality, newer construction, "
            "and the presence of garages are all strongly associated "
            "with higher sale prices.\n"
            "- These findings align with **Business Requirement 1** and "
            "provide actionable insights for feature engineering and "
            "predictive modeling.\n"
            )

    st.markdown(" ")

    # Dropdown for hypothesis selection
    selected_hypothesis = st.selectbox("Select a hypothesis to explore", [
        "Hypothesis 1: Larger houses have higher sale price",
        "Hypothesis 2: Houses with higher overall "
        "quality have higher sale price",
        "Hypothesis 3: Newer houses have higher sale price",
        "Hypothesis 4: Houses with garages have higher sale price"
    ])

    # Hypothesis 1
    if selected_hypothesis ==
    "Hypothesis 1:"
    "Larger houses have higher sale price":
        st.markdown("### Hypothesis 1: Larger houses have higher sale price")
        st.markdown(
            """
            #### **Rationale**:
            Larger houses (`GrLivArea`, `GarageArea`, `TotalBsmtSF`)
            are expected to have higher prices due to:
            - Increased usability
            - Greater living space
            - Higher market demand
            """
            )
        st.markdown("#### **Validation**:")
        st.markdown(
            """
            - A correlation analysis showed that
            `GrLivArea` has a strong positive
            correlatio with sales price *(Pearson correlation: 0.71)*.
            - A scatterplot showed a clear trend where larger
            living areas resulted in higher prices.
            """
            )
        # Add the scatterplot image
        st.image("docs/plots/lm_plot_price_by_GrLivArea.png",
                 caption="Scatterplot: GrLivArea vs SalePrice",
                 use_container_width=True)

        st.markdown(
            """
            #### **Result**: ✅
            The hypothesis was **validated**.
            `GrLivArea` is one of the most significant variables
            in predicting sales price.
            """
            )

    # Hypothesis 2
    elif selected_hypothesis ==
    "Hypothesis 2: "
    "Houses with higher overall quality have higher sale price":
        st.markdown("### Hypothesis 2: Houses with "
                    "higher overall quality have higher sale price")
        st.markdown(
            """
            #### **Rationale**:
            Houses with better construction quality and
            finish (`OverallQual`) are expected to have
            higher prices due to:
            - Durability
            - Aesthetics
            - Greater buyer appeal
            """
            )
        st.markdown("#### **Validation**:")
        st.markdown(
            """
            - A correlation analysis showed that `OverallQual`
            has a very strong positive correlation with sales
            price *(Pearson correlation: 0.79)*.
            - A boxplot showed that houses with higher
            construction quality consistently had higher prices.
            """
            )
        # Add the boxplot image
        st.image("docs/plots/box_plot_price_by_OverallQual.png",
                 caption="Boxplot: OverallQual vs SalePrice",
                 use_container_width=True)

        st.markdown(
            """
            #### Result: ✅
            The hypothesis was **validated**. `OverallQual`
            is one of the most decisive factors for the sales price.
            """
            )

    # Hypothesis 3
    elif selected_hypothesis ==
    "Hypothesis 3: Newer houses have higher sale price":
        st.markdown("### Hypothesis 3: Newer houses have higher sale price")
        st.markdown(
            """
            #### Rationale:
            Newer houses (`YearBuilt`) are expected to
            have higher prices due to:
            - Modern design
            - Better materials
            - Lower maintenance costs
            """
            )
        st.markdown("#### Validation:")
        st.markdown(
            """
            - A correlation analysis showed a positive
            correlation between `YearBuilt` and sales
            price *(Pearson correlation: 0.52)*.
            - A line plot showed that newer houses generally
            have higher prices.
            """
            )
        # Add the line plot image
        st.image("docs/plots/line_plot_price_by_YearBuilt.png",
                 caption="YearBuilt vs SalePrice",
                 use_container_width=True)

        st.markdown(
            """
            #### Result: ✅
            The hypothesis was **partially validated**.
            Although newer houses have higher prices, the
            correlation is not as strong as for other variables.
            """
            )

    # Hypothesis 4
    elif selected_hypothesis ==
    "Hypothesis 4: Houses with "
    "garages have higher sale price":
        st.markdown("### Hypothesis 4: Houses with "
                    "garages have higher sale price")
        st.markdown(
            """
            #### Rationale:
            Houses with garages (`GarageArea`) are more
            attractive because they offer:
            - Extra storage
            - Parking space
            - Increased property value
            """
            )
        st.markdown("#### Validation:")
        st.markdown(
            """
            - A scatterplot showed a positive trend
            between `GarageArea` and sales price
            *(Pearson correlation: 0.62)*.
            - Houses with larger garage areas generally had higher prices.
            """
            )
        # Add the scatterplot image
        st.image("docs/plots/lm_plot_price_by_GarageArea.png",
                 caption="GarageArea vs SalePrice",
                 use_container_width=True)

        st.markdown(
            """
            #### Result: ✅
            The hypothesis was **validated**.
            `GarageArea` affects sales price, but not as
            strongly as `GrLivArea` or `OverallQual`.
            """
            )
