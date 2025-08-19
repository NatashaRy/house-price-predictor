import streamlit as st
import pandas as pd


def predict_house_price(X_live, house_features, price_pipeline):
    """
    Make a prediction for the house price based on user input.

    Args:
        X_live (pd.DataFrame): Live input data with house features.
        house_features (list): List of features used in the model.
        price_pipeline (Pipeline): The trained ML pipeline.

    Returns:
        float: Predicted house price or None if error occurs.
    """
    try:
        # Create a copy to avoid modifying the original data
        X_live_copy = X_live.copy()

        # Ensure all required features are present
        for feature in house_features:
            if feature not in X_live_copy.columns:
                # Add missing feature with appropriate default value
                if feature == 'KitchenQual':
                    X_live_copy[feature] = "TA"  # Default to Typical/Average
                elif feature in X_live_copy.select_dtypes(
                        include="object").columns:
                    # Default for other categorical features
                    X_live_copy[feature] = "None"

                else:
                    X_live_copy[feature] = 0  # Default for numerical features

        # Handle missing values
        for feature in X_live_copy.columns:
            if X_live_copy[feature].isnull().any():
                if X_live_copy[feature].dtype == "object":
                    X_live_copy[feature].fillna("None", inplace=True)
                else:
                    X_live_copy[feature].fillna(0, inplace=True)

        # Filter to only include the required features
        X_live_price = X_live_copy[house_features]

        # Make a prediction
        price_prediction = price_pipeline.predict(X_live_price)

        # Return the predicted price
        return price_prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        print(f"Error during prediction: {e}")
        return None


def predict_inherited_house_price(X_inherited, house_features, price_pipeline):
    """
    Make a prediction for the sale price of inherited houses.

    Args:
        X_inherited (pd.DataFrame): Data with features for the
         inherited houses.
        house_features (list): List of features used in the model.
        price_pipeline (Pipeline): The trained ML pipeline.

    Returns:
        pd.DataFrame: DataFrame with the inherited houses and their predicted
                      prices, or None if error occurs.
    """
    try:
        # Create a copy to avoid modifying the original data
        X_inherited_copy = X_inherited.copy()

        # Ensure all required features are present
        for feature in house_features:
            if feature not in X_inherited_copy.columns:
                if feature == 'KitchenQual':
                    X_inherited_copy[feature] = "TA"
                elif feature in X_inherited_copy.select_dtypes(
                        include="object").columns:
                    X_inherited_copy[feature] = "None"
                else:
                    X_inherited_copy[feature] = 0

        # Handle missing values
        for feature in X_inherited_copy.columns:
            if X_inherited_copy[feature].isnull().any():
                if X_inherited_copy[feature].dtype == "object":
                    X_inherited_copy[feature].fillna("None", inplace=True)
                else:
                    X_inherited_copy[feature].fillna(0, inplace=True)

        # Filter relevant features
        X_inherited_price = X_inherited_copy[house_features]

        # Make predictions
        predicted_prices = price_pipeline.predict(X_inherited_price)

        # Add the predicted prices to the original DataFrame
        X_inherited_copy["PredictedSalePrice"] = predicted_prices

        return X_inherited_copy
    except Exception as e:
        st.error(f"Error during prediction for inherited houses: {e}")
        print(f"Error during prediction for inherited houses: {e}")
        return None
