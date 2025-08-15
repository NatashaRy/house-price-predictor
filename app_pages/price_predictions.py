import joblib
from datetime import date
import streamlit as st
import pandas as pd
from src.data_management import load_inherited_data, load_pricing_data, load_pkl_file
from src.machine_learning.predictive_analysis import predict_house_price, predict_inherited_house_price


def predict_price_body():
    """
    Displays the content of the predict house price page in the Streamlit app.

    This includes:
    - Predicting the sale price of inherited houses.
    - Predicting the sale price of a custom house based on user input.
    """

##### Load the saved pipeline
    version = "v1"
    regression_pipeline = load_pkl_file(f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl")
    house_features = (pd.read_parquet(f"outputs/ml_pipeline/predict_price/{version}/X_train.parquet")
                      .columns.tolist())

##### Title and introduction
    st.title("Predict House Sale Price")
    st.markdown("This page allows you to predict the sale price of inherited houses "
                "and other houses based on their features.")

    st.info("***Business Requirement 2**: The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.*")

##### Prediction for inherited houses
    st.write("## Predict the sale price of inherited houses")

    # Load dataset of inherited houses
    X_inherited = load_inherited_data()
    if X_inherited is not None:
        st.write("### Inherited house data (filtered for prediction)")
        st.write(X_inherited[house_features])

        # Make predictions for inherited houses
        X_inherited_with_predictions = predict_inherited_house_price(
            X_inherited, house_features, regression_pipeline
        )

        if X_inherited_with_predictions is not None:
            # Show results
            st.write("### Predicted sale prices for inherited houses")
            st.dataframe(X_inherited_with_predictions[house_features + ["PredictedSalePrice"]])

            # Sum the total sale price
            total_price = X_inherited_with_predictions["PredictedSalePrice"].sum()
            st.write(f"### Total predicted sale price for all inherited houses: "
                     f"**💲{round(total_price, 2):,}**")

            st.markdown(" ")
            # Visualization: Bar chart of predicted prices
            st.bar_chart(X_inherited_with_predictions[["PredictedSalePrice"]])
        else:
            st.error("An error occurred during the prediction for inherited houses.")
    else:
        st.error("Could not load data for inherited houses. Check the file path.")

    st.write("---")

##### Prediction for user's own house
    st.write("### Predict the sale price of your own house")
    st.write("Enter the features of your house below and click 'Predict Price'.")

    # Create input widgets for user's house
    X_live = draw_input_widgets(house_features)

    # Make prediction for user's house
    if st.button("Predict Price"):
        price_prediction = predict_house_price(X_live, house_features, regression_pipeline)
        if price_prediction is not None:
            st.write(f"### The predicted sale price for your house is: "
                     f"**💲{round(price_prediction, 2):,}**")
        else:
            st.error("An error occurred during the prediction. Check your input and try again.")

##### Widget for user input
def draw_input_widgets(house_features):
    """
    Create interactive widgets to collect user's house data.
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

    # Dynamically create widgets for each feature
    for feature in house_features:
        if feature in df.select_dtypes(include="object").columns:  # Categorical features
            st_widget = st.selectbox(
                label=feature,
                options=["Excellent", "Good", "Typical", "Fair", "Poor"],
                index=2
            )
            X_live[feature] = st_widget
        else:  # Numerical features
            st_widget = st.number_input(
                label=feature,
                min_value=int(df[feature].min() * percentage_min),
                max_value=int(df[feature].max() * percentage_max),
                value=int(df[feature].median()),
                step=50
            )
            X_live[feature] = st_widget

    # Hantera saknade värden i kategoriska funktioner
    for feature in df.select_dtypes(include="object").columns:
        if feature not in X_live.columns:
            X_live[feature] = "Typical"

    return X_live