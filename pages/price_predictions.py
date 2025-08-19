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

# This page displays content of the
# Sale Price Predictor page in the Streamlit app.
# This includes:
# - Page introduction
# - Reminder of business requirement 2
# - Predict the sale price of inherited houses
#   - Inherited house data filtered with best features
#       - Tabs to display table/chart of the data
#   - Predicted Sale Prices for Inherited Houses
#       - Tabs to display table/chart of the data
#   - Sum of total sale price for all inherited houses
# - Predict the sale price of your own house
# - User input widget, including:
#     - Garage Area (`GarageArea`)
#     - Ground Living Area (sq ft) (`GrLivArea`)
#     - Overall Quality (1-10 scale) (`OverallQual`)
#     - Kitchen Quality (dropdown) (`KitchenQual`)
#   - Button "Predict Sale Price" to trigger prediction
#   - Return predicted sale value

# Title and introduction
st.title("💰Predict House Sale Price")
st.markdown(
    """
    This page allows you to predict the sale price of houses in Ames,
    Iowa, including the client's inherited houses and any other house
    based on its features.

    - **Inherited Houses**: View the predicted sale prices for the
    client's four inherited houses, along with a total price summary
    and an interactive bar chart.
    - **Your Own House**: Use the interactive widgets below to input
    your house's features and get an instant prediction of its sale
    price.

    Explore the predictions and gain insights into how different
    features influence house prices.
    """
)

# Reminder of business requirement 2
st.info(
    "**Business Requirement 2**: The client is interested in predicting "
    "the house sale price from her four inherited houses and any other "
    "houses in Ames, Iowa."
)


# Function for user input widget
def draw_input_widgets(house_features):
    """
    Create interactive widgets to collect user's house
     data in a structured layout.

    Args:
        house_features (list): List of features required by the model.

    Returns:
        pd.DataFrame: DataFrame containing user input for house features.
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
        'OverallQual': 'Overall Quality (1-10 scale)',
        'KitchenQual': 'Kitchen Quality (dropdown)',
    }

    # Mapping for KitchenQual values
    kitchen_qual_mapping = {
        'Fa': 'Fair',
        'TA': 'Typical/Average',
        'Gd': 'Good',
        'Ex': 'Excellent',
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
            # Create options list with user-friendly labels
            kitchen_options = list(kitchen_qual_mapping.values())
            kitchen_keys = list(kitchen_qual_mapping.keys())

            kitchen_qual_display = st.selectbox(
                label=feature_display_names.get(
                    'KitchenQual',
                    'KitchenQual'
                ),
                options=kitchen_options,  # Display user-friendly names
                index=1,                  # Default to 'Typical/Average'
                key='KitchenQual',
            )

            # Map the selected display value back to the abbreviated code
            selected_index = kitchen_options.index(kitchen_qual_display)
            kitchen_qual = kitchen_keys[selected_index]
            X_live['KitchenQual'] = kitchen_qual

    with col2:
        if 'OverallQual' in house_features:
            overall_qual = st.slider(
                label=feature_display_names.get('OverallQual', 'OverallQual'),
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key='OverallQual',
                help="1=Poor, 10=Excellent"
            )
            X_live['OverallQual'] = overall_qual

    # Handle all other features that might be required by the model
    for feature in house_features:
        if feature not in X_live.columns:
            if feature == 'KitchenQual':
                X_live[feature] = "TA"  # Default to Typical/Average
            elif feature == 'OverallQual':
                X_live[feature] = 5  # Default to Average
            else:
                X_live[feature] = 0  # Default for numerical features

    return X_live


# Function to handle page content
def predict_price_body():
    """
    Displays the content of the predict house price page in the Streamlit app.

    This includes:
    - Predicting the sale price of inherited houses
    - Widget to predict house sale price based on user input
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
    house_features = pd.read_parquet(train_data_path).columns.tolist()

    # Prediction for inherited houses
    st.header("Predict the sale price of inherited houses")

    # Load dataset of inherited houses
    X_inherited = load_inherited_data()
    if X_inherited is not None:
        st.markdown(
            "<h3 style='margin-bottom: 0;padding-bottom: 0;'>"
            "Inherited House Data</h3>"
            "<p style='margin-top: 3px;padding-top: 0;'> "
            "(Filtered for prediction)</p>",
            unsafe_allow_html=True
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

            # Tabbed interface for inherited house data
            tab1, tab2 = st.tabs(["Table", "Chart"])
            # Show table with filtered data
            with tab1:
                st.dataframe(
                    X_inherited[house_features], use_container_width=True
                )
            with tab2:
                # Interactive scatter plot for inherited house data
                fig = go.Figure()
                for feature in house_features:
                    fig.add_trace(
                        go.Scatter(
                            x=X_inherited.index,
                            y=X_inherited[feature],
                            mode="lines+markers",
                            name=feature
                        )
                    )
                fig.update_layout(
                    xaxis_title="House Index",
                    yaxis_title="Feature Values",
                    template="plotly_white"
                )
                st.plotly_chart(fig)

            # Tabbed interface for predicted sale prices
            st.subheader("Predicted Sale Prices for Inherited Houses")
            tab1, tab2 = st.tabs(["Table", "Chart"])
            with tab1:
                st.dataframe(
                    X_inherited_with_predictions[
                        house_features + ["PredictedSalePrice"]
                    ],
                    use_container_width=True
                )
            with tab2:
                # Visualization: Bar chart of predicted prices
                # # Round to two decimal places
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
                        textposition="auto",        # Place text automatically
                        # Show only x and y values on hover
                        hoverinfo="x+y",
                        marker=dict(color=colors)  # Use color scale
                    )
                ])

                # Customize the layout
                fig.update_layout(
                    xaxis_title="Houses",            # X-axis title
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

            # Sum the total sale price
            total_price = X_inherited_with_predictions[
                'PredictedSalePrice'
            ].sum()
            st.write(
                "### Total predicted sale price for all inherited houses: "
                f"**💲{round(total_price, 2):,}**"
            )
        else:
            st.error(
                "An error occurred during the prediction for inherited houses."
            )
    else:
        st.error(
            "Could not load data for inherited houses. Check the file path."
        )

    # Horizontal line
    st.divider()

    # Prediction for user's own house
    st.header("🔮Predict the sale price of your own house")
    st.write("Enter the features of your house below and "
             "click 'Predict Sale Price'."
             )

    # Create input widgets for user's house
    X_live = draw_input_widgets(house_features)

    # Create a placeholder for the prediction result
    prediction_placeholder = st.empty()

    # Make prediction for user's house
    if st.button("Predict Sale Price"):
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


# Run the prediction page
predict_price_body()
