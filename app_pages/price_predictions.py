import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from src.data_management import (
    load_inherited_data,
    load_pricing_data,
    load_pkl_file
)
from src.machine_learning.predictive_analysis import (
    predict_house_price,
    predict_inherited_house_price
)

# Page title
page_title = "Predict House Sale Price"


def predict_price_body():
    """
    Displays the content of the predict house price page in the Streamlit app.

    This includes:
    - Page introduction
    - Reminder of business requirement 2
    - Predicting the sale price of inherited houses
        - Data of inherited houses, filtered with best features
        - Table with data of inherited houses
        - Table with predicted sale prices for inherited houses
        - Total predicted sale price for all four inherited houses
        - Interactive bar chart of predicted sale prices
    - Widget to predict house sale price based on user input, including:
        - Garage Area - `GarageArea`
        - Ground Living Area - `GrLivArea`
        - Kitchen Quality - `KitchenQual` (1-5 scale)
        - Overall Quality - `OverallQual` (1-10 scale)
        - Question mark help at scale options explaining what each value equals
        - Button labeled "Predict Price" to execute the prediction
        - Display area for predicted price output
    """
    # Load the saved pipeline
    version = "v1"
    pipeline_path = (
        f"outputs/ml_pipeline/predict_price/{version}/"
        f"regression_pipeline.pkl"
    )
    regression_pipeline = load_pkl_file(pipeline_path)

    train_data_path = (
        f"outputs/ml_pipeline/predict_price/{version}/X_train.parquet"
    )
    house_features = (pd.read_parquet(train_data_path).columns.tolist())

    # Title and introduction
    st.title(page_title)
    st.markdown(
        "This page allows you to predict the sale price of inherited houses "
        "and other houses based on their features."
    )

    st.info(
        "**Business Requirement 2**: The client is interested in predicting "
        "the house sale price from her four inherited houses and any other "
        "houses in Ames, Iowa."
    )

    # Prediction for inherited houses
    st.write("## Predict the sale price of inherited houses")

    # Load dataset of inherited houses
    X_inherited = load_inherited_data()
    if X_inherited is not None:
        st.markdown(
            "<h3 style='margin-bottom: 0;padding-bottom: 0;'>"
            "Inherited house data</h3>"
            "<p style='margin-top: 0;padding-top: 0;'> "
            "(Filtered for prediction)</p>",
            unsafe_allow_html=True
        )

        # Show inherited house data in full width
        st.dataframe(
            X_inherited[house_features], use_container_width=True
        )

        # Make predictions for inherited houses
        X_inherited_with_predictions = predict_inherited_house_price(
            X_inherited, house_features, regression_pipeline
        )

        if X_inherited_with_predictions is not None:
            # Ensure "House" column exists
            if "House" not in X_inherited_with_predictions.columns:
                X_inherited_with_predictions["House"] = [
                    f"House {i+1}"
                    for i in range(len(X_inherited_with_predictions))
                ]

            # Show results in full width
            st.write("#### Predicted sale prices for inherited houses")
            prediction_columns = house_features + ["PredictedSalePrice"]
            st.dataframe(
                X_inherited_with_predictions[prediction_columns],
                use_container_width=True
            )

            # Sum the total sale price
            total_price = X_inherited_with_predictions[
                "PredictedSalePrice"
            ].sum()
            st.write(
                f"### Total predicted sale price for all inherited houses: "
                f"**💲{round(total_price, 2):,}**"
            )

            st.markdown(" ")

            # Visualization: Bar chart of predicted prices
            st.markdown("#### Chart of sales price of inherited houses")

            # Round to two decimal places
            predicted_prices = X_inherited_with_predictions[
                "PredictedSalePrice"
            ].round(2)
            X_inherited_with_predictions[
                "PredictedSalePrice"
            ] = predicted_prices

            # Create a color scale based on "Pastel2"
            colors = qualitative.Pastel2[
                :len(X_inherited_with_predictions)
            ]  # Adjust length to number of houses

            # Create a bar chart with Plotly
            fig = go.Figure(data=[
                go.Bar(
                    # Custom names for x-axis
                    x=X_inherited_with_predictions["House"],
                    # Predicted sale prices
                    y=X_inherited_with_predictions["PredictedSalePrice"],
                    text=X_inherited_with_predictions["PredictedSalePrice"]
                    .apply(lambda x: f"${x:,.2f}"),  # Show values as text
                    textposition="auto",  # Place text automatically
                    # Show only x and y values on hover
                    hoverinfo="x+y",
                    marker=dict(color=colors)  # Use color scale
                )
            ])

            # Customize the layout
            fig.update_layout(
                xaxis_title="Houses",  # X-axis title
                yaxis_title="Sale Price (USD)",  # Y-axis title
                xaxis=dict(
                    tickmode="array",  # Show custom tick names
                    tickvals=list(range(
                        len(X_inherited_with_predictions["House"])
                    )),
                    # Custom names for x-axis
                    ticktext=X_inherited_with_predictions["House"]
                ),
                yaxis=dict(
                    # Format y-axis as currency with two decimal places
                    tickformat="$,.2f"
                ),
                template="plotly_white",  # White background for the chart
                # Reduce padding above and below the chart
                margin=dict(t=20, b=20)
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig)
        else:
            st.error(
                "An error occurred during the prediction for inherited houses."
            )
    else:
        st.error(
            "Could not load data for inherited houses. Check the file path."
        )

    st.write("---")

    # Prediction for user's own house
    st.write("### Predict the sale price of your own house")
    st.write(
        "Enter the features of your house below and click 'Predict Price'."
    )

    # Create input widgets for user's house
    X_live = draw_input_widgets(house_features)

    # Create a placeholder for the prediction result
    prediction_placeholder = st.empty()

    # Make prediction for user's house
    if st.button("Predict Price"):
        price_prediction = predict_house_price(
            X_live, house_features, regression_pipeline
        )
        if price_prediction is not None:
            with prediction_placeholder.container():
                st.write(
                    f"### The predicted sale price for your house is: "
                    f"**💲{round(price_prediction, 2):,}**"
                )
        else:
            with prediction_placeholder.container():
                st.error(
                    "An error occurred during the prediction. "
                    "Check your input and try again."
                )


def draw_input_widgets(house_features):
    """
    Create interactive widgets to collect user's house data in a structured
    layout.

    Args:
        house_features (list): List of features required by the model

    Returns:
        pd.DataFrame: DataFrame containing user input for house features
    """
    # Load dataset to determine realistic input ranges
    df = load_pricing_data()
    if df is None:
        st.error("Could not load data to create input widgets.")
        return pd.DataFrame()

    # Define range for scaling input values
    percentage_min, percentage_max = 0.4, 2.0

    # Initialize an empty DataFrame to store user's input
    X_live = pd.DataFrame([], index=[0])

    # Create a mapping for better feature display names
    feature_display_names = {
        'GarageArea': 'Garage Area (sq ft)',
        'GrLivArea': 'Ground Living Area (sq ft)',
        'TotalBsmtSF': 'Total Basement Area (sq ft)',
        'OverallQual': 'Overall Quality (1-10 scale)',
        'YearBuilt': 'Year Built',
        'KitchenQual': 'Kitchen Quality (1-5 scale)',
        'BsmtExposure': 'Basement Exposure',
        'BsmtFinType1': 'Basement Finished Type 1',
        'GarageFinish': 'Garage Finish',
        'LotFrontage': 'Lot Frontage (linear feet)',
        'BedroomAbvGr': 'Bedrooms Above Ground',
        'MasVnrArea': 'Masonry Veneer Area (sq ft)',
        '2ndFlrSF': 'Second Floor Area (sq ft)',
        'GarageYrBlt': 'Garage Year Built',
        '1stFlrSF': 'First Floor Area (sq ft)',
        'LotArea': 'Lot Area (sq ft)',
        'YearRemodAdd': 'Year Remodeled/Added'
    }

    # Row 1: Garage Area and Ground Living Area
    col1, col2 = st.columns(2)

    with col1:
        if 'GarageArea' in house_features:
            min_garage = 0
            max_garage = 2000
            default_garage = 500

            if 'GarageArea' in df.columns:
                min_garage = int(df['GarageArea'].min() * percentage_min)
                max_garage = int(df['GarageArea'].max() * percentage_max)
                default_garage = int(df['GarageArea'].median())

            garage_area = st.number_input(
                label=feature_display_names.get('GarageArea', 'GarageArea'),
                min_value=min_garage,
                max_value=max_garage,
                value=default_garage,
                step=50,
                key='GarageArea'
            )
            X_live['GarageArea'] = garage_area

    with col2:
        if 'GrLivArea' in house_features:
            min_area = 500
            max_area = 5000
            default_area = 1500

            if 'GrLivArea' in df.columns:
                min_area = int(df['GrLivArea'].min() * percentage_min)
                max_area = int(df['GrLivArea'].max() * percentage_max)
                default_area = int(df['GrLivArea'].median())

            gr_liv_area = st.number_input(
                label=feature_display_names.get('GrLivArea', 'GrLivArea'),
                min_value=min_area,
                max_value=max_area,
                value=default_area,
                step=50,
                key='GrLivArea'
            )
            X_live['GrLivArea'] = gr_liv_area

    # Row 2: Kitchen Quality and Overall Quality
    col1, col2 = st.columns(2)

    with col1:
        if 'KitchenQual' in house_features:
            kitchen_qual = st.slider(
                label=feature_display_names.get('KitchenQual', 'KitchenQual'),
                min_value=1,
                max_value=5,
                value=3,  # Default to middle value (TA)
                step=1,
                key='KitchenQual',
                help="1=Poor, 2=Fair, 3=Typical, 4=Good, 5=Excellent"
            )
            # Convert slider value to dataset format
            quality_mapping = {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'}
            X_live['KitchenQual'] = quality_mapping[kitchen_qual]

    with col2:
        if 'OverallQual' in house_features:
            overall_qual = st.slider(
                label=feature_display_names.get('OverallQual', 'OverallQual'),
                min_value=1,
                max_value=10,
                value=5,  # Default to middle value
                step=1,
                key='OverallQual',
                help="1=Poor, 2=Fair, 3=Average, 4=Good, 5=Excellent"
            )
            # Map 1-5 scale to 1-10 scale for the model
            X_live['OverallQual'] = overall_qual * 2

    # Handle all other features that might be required by the model
    for feature in house_features:
        if feature not in X_live.columns:
            if feature == 'KitchenQual':
                X_live[feature] = "TA"  # Default to Typical/Average
            elif feature == 'OverallQual':
                # Default to middle value (3 on 1-5 scale = 6 on 1-10 scale)
                X_live[feature] = 6
            elif feature in df.select_dtypes(include="object").columns:
                # For other categorical features, use the most common value
                most_common = df[feature].mode()
                if len(most_common) > 0:
                    default_value = most_common[0]
                else:
                    # Use appropriate defaults based on feature type
                    quality_features = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
                    feature_unique = df[feature].unique()
                    if any(
                        qual in feature_unique
                        for qual in quality_features
                    ):
                        default_value = "TA"  # For quality features
                    else:
                        # For other categorical features
                        default_value = "None"
                X_live[feature] = default_value
            else:
                # For numerical features, use median or 0
                median_value = df[feature].median()
                if not pd.isna(median_value):
                    X_live[feature] = median_value
                else:
                    X_live[feature] = 0

    return X_live
