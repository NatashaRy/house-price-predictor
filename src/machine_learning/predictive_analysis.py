import streamlit as st
import pandas as pd


def predict_house_price(X_live, house_features, price_pipeline):
    """
    Make a prediction for the house price based on user input.

    Parameters:
    X_live (DataFrame): Live input data with house features.
    house_features (list): List of features used in the model.
    price_pipeline (Pipeline): The trained ML pipeline.

    Returns:
    float: Predicted house price.
    """
    try:
        # Create a copy to avoid modifying the original data
        X_live_copy = X_live.copy()
        
        # Ensure all required features are present
        for feature in house_features:
            if feature not in X_live_copy.columns:
                # Add missing feature with appropriate default value
                if feature in X_live_copy.select_dtypes(include="object").columns:
                    if feature == 'KitchenQual':
                        X_live_copy[feature] = "TA"  # Typical/Average for kitchen quality
                    else:
                        X_live_copy[feature] = "None"  # For other categorical features
                else:
                    X_live_copy[feature] = 0
        
        # Filter to only include the required features
        X_live_price = X_live_copy[house_features].copy()
        
        # Handle missing values more robustly
        for feature in X_live_price.columns:
            if X_live_price[feature].isnull().any():
                if X_live_price[feature].dtype == "object":
                    # For categorical features, use mode or appropriate default
                    mode_value = X_live_price[feature].mode()
                    if len(mode_value) > 0:
                        X_live_price[feature].fillna(mode_value[0], inplace=True)
                    else:
                        # Use appropriate defaults based on feature type
                        if feature == 'KitchenQual' or any(qual in X_live_price[feature].unique() for qual in ['Ex', 'Gd', 'TA', 'Fa', 'Po'] if not pd.isna(qual)):
                            X_live_price[feature].fillna("TA", inplace=True)  # Typical/Average for quality features
                        else:
                            X_live_price[feature].fillna("None", inplace=True)  # For other categorical features
                else:
                    # For numerical features, use median or 0
                    median_value = X_live_price[feature].median()
                    if pd.notna(median_value):
                        X_live_price[feature].fillna(median_value, inplace=True)
                    else:
                        X_live_price[feature].fillna(0, inplace=True)

        # Final check for NaN values
        if X_live_price.isnull().any().any():
            # If still NaN values, fill with default values by data type
            for col in X_live_price.columns:
                if X_live_price[col].isnull().any():
                    if X_live_price[col].dtype == "object":
                        if col == 'KitchenQual' or any(qual in str(X_live_price[col].unique()) for qual in ['Ex', 'Gd', 'TA', 'Fa', 'Po']):
                            X_live_price[col].fillna("TA", inplace=True)
                        else:
                            X_live_price[col].fillna("None", inplace=True)
                    else:
                        X_live_price[col].fillna(0, inplace=True)

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

    Parameters:
    X_inherited (DataFrame): Data with features for the inherited houses.
    house_features (list): List of features used in the model.
    price_pipeline (Pipeline): The trained ML pipeline.

    Returns:
    DataFrame: DataFrame with the inherited houses and their predicted prices.
    """
    try:
        # Create a copy to avoid modifying the original data
        X_inherited_copy = X_inherited.copy()
        
        # Filter relevant features from inherited house data
        X_inherited_price = X_inherited_copy[house_features].copy()
        
        # Handle missing values
        for feature in X_inherited_price.columns:
            if X_inherited_price[feature].isnull().any():
                if X_inherited_price[feature].dtype == "object":
                    # For categorical features, use mode or appropriate default
                    mode_value = X_inherited_price[feature].mode()
                    if len(mode_value) > 0:
                        X_inherited_price[feature].fillna(mode_value[0], inplace=True)
                    else:
                        # Use appropriate defaults based on feature type
                        if feature == 'KitchenQual' or any(qual in X_inherited_price[feature].unique() for qual in ['Ex', 'Gd', 'TA', 'Fa', 'Po'] if not pd.isna(qual)):
                            X_inherited_price[feature].fillna("TA", inplace=True)  # Typical/Average for quality features
                        else:
                            X_inherited_price[feature].fillna("None", inplace=True)  # For other categorical features
                else:
                    # For numerical features, use median or 0
                    median_value = X_inherited_price[feature].median()
                    if pd.notna(median_value):
                        X_inherited_price[feature].fillna(median_value, inplace=True)
                    else:
                        X_inherited_price[feature].fillna(0, inplace=True)

        # Final check for NaN values
        if X_inherited_price.isnull().any().any():
            for col in X_inherited_price.columns:
                if X_inherited_price[col].isnull().any():
                    if X_inherited_price[col].dtype == "object":
                        if col == 'KitchenQual' or any(qual in str(X_inherited_price[col].unique()) for qual in ['Ex', 'Gd', 'TA', 'Fa', 'Po']):
                            X_inherited_price[col].fillna("TA", inplace=True)
                        else:
                            X_inherited_price[col].fillna("None", inplace=True)
                    else:
                        X_inherited_price[col].fillna(0, inplace=True)

        # Make predictions
        predicted_prices = price_pipeline.predict(X_inherited_price)

        # Add the predicted prices to the original DataFrame
        X_inherited_copy["PredictedSalePrice"] = predicted_prices

        return X_inherited_copy
    except Exception as e:
        st.error(f"Error during prediction for inherited houses: {e}")
        print(f"Error during prediction for inherited houses: {e}")
        return None