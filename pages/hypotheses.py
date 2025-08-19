import streamlit as st

# This page displays content of the hypotheses page in the Streamlit app.
#
# This includes:
# - Page introduction
# - Dropdown to select hypothesis to explore
# - Detailed list of four hypotheses, including:
#     - Rationale
#     - Validation
#     - Plot image
#     - Results


# Title and Page Introduction
st.title("🔍 Hypotheses and Validation")
st.write(
    """
    This page presents the hypotheses that guide the analysis
    of house prices.
    Each hypothesis is based on domain knowledge and explores
    key factors that
    influence house prices, such as size, quality, and age of the property.

    Use the dropdown menu below to explore each hypothesis in detail,
    including the rationale, validation process, and results supported by
    visualizations. These findings are directly tied to
     **Business Requirement 1**,
    helping to uncover how house attributes correlate with sale prices.
    """
)

# Horizontal line
st.divider()


# Function to display the hypotheses page content
def hypotheses_body():

    # Conclusion of hypotheses and results
    st.header("🧠 Hypotheses and Results\n")
    st.info(
        """
        ### **Summary of Findings**
        The hypotheses were validated using statistical methods
         and visualizations.
        All four hypotheses were confirmed to be correct:
        - Larger houses, higher quality, newer construction, and
         the presence of garages
          are all strongly associated with higher sale prices.
        - These findings align with **Business Requirement 1** and
         provide actionable
          insights for feature engineering and predictive modeling.
        """
    )

    # Horizontal line
    st.divider()

    st.header("🧪Explore Hypotheses")

    # Dropdown for hypothesis selection
    selected_hypothesis = st.selectbox(
        "Select a hypothesis to explore",
        [
            "Hypothesis 1: Larger houses have higher sale price",
            "Hypothesis 2: Houses with higher overall quality have "
            "higher sale price",
            "Hypothesis 3: Newer houses have higher sale price",
            "Hypothesis 4: Houses with garages have higher sale price",
        ],
    )

    # Hypothesis 1
    if (selected_hypothesis ==
            "Hypothesis 1: Larger houses have higher sale price"):
        st.subheader("Hypothesis 1: Larger houses have higher sale price")
        st.write(
            """
            #### Rationale:
            Larger houses (`GrLivArea`, `GarageArea`, `TotalBsmtSF`)
             are expected to have higher prices due to:
            - Increased usability
            - Greater living space
            - Higher market demand
            """
        )
        st.write(
            """
            #### Validation:\n
            - A correlation analysis showed that `GrLivArea` has a strong
             positive correlation with sales price
              *(Pearson correlation: 0.71)*.
            - A scatterplot showed a clear trend where larger living areas
             resulted in higher prices.
            """
        )
        st.image(
            "docs/plots/lm_plot_price_by_GrLivArea.png",
            caption="Scatterplot: GrLivArea vs SalePrice",
            use_container_width=True,
        )
        st.write(
            """
            #### Result: ✅\n
            The hypothesis was **validated**. `GrLivArea` is one of the
             most significant variables in predicting sales price."
            """
            )

    # Hypothesis 2
    elif (selected_hypothesis ==
          "Hypothesis 2: Houses with higher overall quality have "
          "higher sale price"):
        st.subheader("Hypothesis 2: Houses with higher overall "
                     "quality have higher sale price"
                     )
        st.write(
            """
            #### Rationale:
            Houses with better construction quality and finish (`OverallQual`)
             are expected to have higher prices due to:
            - Durability
            - Aesthetics
            - Greater buyer appeal
            """
        )
        st.write(
            """
            #### Validation: \n
            - A correlation analysis showed that `OverallQual` has a very
             strong positive correlation with sales price
              *(Pearson correlation: 0.79)*.
            - A boxplot showed that houses with higher construction quality
             consistently had higher prices.
            """
        )
        st.image(
            "docs/plots/box_plot_price_by_OverallQual.png",
            caption="Boxplot: OverallQual vs SalePrice",
            use_container_width=True,
        )
        st.write(
            """
            #### Result: ✅
            The hypothesis was **validated**. `OverallQual` is one of the most
             decisive factors for the sales price.
            """
        )

    # Hypothesis 3
    elif (selected_hypothesis ==
          "Hypothesis 3: Newer houses have higher sale price"):
        st.subheader("Hypothesis 3: Newer houses have higher sale price")
        st.write(
            """
            #### Rationale:
            Newer houses (`YearBuilt`) are expected to have higher
             prices due to:
            - Modern design
            - Better materials
            - Lower maintenance costs
            """
        )
        st.write(
            """
            #### Validation:\n
            - A correlation analysis showed a positive correlation between
             `YearBuilt` and sales price *(Pearson correlation: 0.52)*.
            - A line plot showed that newer houses generally have
             higher prices.
            """
        )
        st.image(
            "docs/plots/line_plot_price_by_YearBuilt.png",
            caption="YearBuilt vs SalePrice",
            use_container_width=True,
        )
        st.write(
            """
            #### Result: ✅
            The hypothesis was **partially validated**. Although newer
             houses have higher prices, the correlation is not as strong
             as for other variables.
            """
        )

    # Hypothesis 4
    elif (selected_hypothesis ==
          "Hypothesis 4: Houses with garages have higher sale price"):
        st.subheader("Hypothesis 4: Houses with garages "
                     "have higher sale price"
                     )
        st.write(
            """
            #### Rationale:
            Houses with garages (`GarageArea`) are more attractive
             because they offer:
            - Extra storage
            - Parking space
            - Increased property value
            """
        )
        st.write(
            """
            #### Validation:
            - A scatterplot showed a positive trend between `GarageArea`
             and sales price *(Pearson correlation: 0.62)*.
            - Houses with larger garage areas generally had higher prices.
            """
        )
        st.image(
            "docs/plots/lm_plot_price_by_GarageArea.png",
            caption="GarageArea vs SalePrice",
            use_container_width=True,
        )
        st.write(
            """
            #### Result: ✅
            The hypothesis was **validated**. `GarageArea` affects sales price,
             but not as strongly as `GrLivArea` or `OverallQual`.
            """
        )


# Run the hypotheses body function
hypotheses_body()
