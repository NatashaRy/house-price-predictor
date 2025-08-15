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

    st.info("***Business Requirement 2**: The client is interested in predicting the house sale price from her four inherited houses and any other houses in Ames, Iowa.*")

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

    # Create a placeholder for the prediction result
    prediction_placeholder = st.empty()

    # Make prediction for user's house
    if st.button("Predict Price"):
        price_prediction = predict_house_price(X_live, house_features, regression_pipeline)
        if price_prediction is not None:
            with prediction_placeholder.container():
                st.write(f"### The predicted sale price for your house is: "
                         f"**💲{round(price_prediction, 2):,}**")
        else:
            with prediction_placeholder.container():
                st.error("An error occurred during the prediction. Check your input and try again.")

##### Widget for user input
def draw_input_widgets(house_features):
    """
    Create interactive widgets to collect user's house data in a structured layout.
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
        'OverallQual': 'Overall Quality (1-5 scale)',
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
            garage_area = st.number_input(
                label=feature_display_names.get('GarageArea', 'GarageArea'),
                min_value=int(df['GarageArea'].min() * percentage_min) if 'GarageArea' in df.columns else 0,
                max_value=int(df['GarageArea'].max() * percentage_max) if 'GarageArea' in df.columns else 2000,
                value=int(df['GarageArea'].median()) if 'GarageArea' in df.columns else 500,
                step=50,
                key='GarageArea'
            )
            X_live['GarageArea'] = garage_area
    
    with col2:
        if 'GrLivArea' in house_features:
            gr_liv_area = st.number_input(
                label=feature_display_names.get('GrLivArea', 'GrLivArea'),
                min_value=int(df['GrLivArea'].min() * percentage_min) if 'GrLivArea' in df.columns else 500,
                max_value=int(df['GrLivArea'].max() * percentage_max) if 'GrLivArea' in df.columns else 5000,
                value=int(df['GrLivArea'].median()) if 'GrLivArea' in df.columns else 1500,
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
                max_value=5,
                value=3,  # Default to middle value
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
                X_live[feature] = 6  # Default to middle value (3 on 1-5 scale = 6 on 1-10 scale)
            elif feature in df.select_dtypes(include="object").columns:
                # For other categorical features, use the most common value from the dataset
                most_common = df[feature].mode()
                if len(most_common) > 0:
                    default_value = most_common[0]
                else:
                    # Use appropriate defaults based on feature type
                    if any(qual in df[feature].unique() for qual in ['Ex', 'Gd', 'TA', 'Fa', 'Po']):
                        default_value = "TA"  # For quality features
                    else:
                        default_value = "None"  # For other categorical features
                X_live[feature] = default_value
            else:
                # For numerical features, use median or 0
                X_live[feature] = df[feature].median() if not pd.isna(df[feature].median()) else 0

    return X_live