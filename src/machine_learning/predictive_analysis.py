import streamlit as st


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
        # Filter relevant features from live data
        X_live_price = X_live.filter(house_features)

        # Make a prediction
        price_prediction = price_pipeline.predict(X_live_price)

        # Return the predicted price
        return price_prediction[0]
    except Exception as e:
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
        # Filter relevant features from inherited house data
        X_inherited_price = X_inherited.filter(house_features)

        # Make predictions
        predicted_prices = price_pipeline.predict(X_inherited_price)

        # Add the predicted prices to the DataFrame
        X_inherited["PredictedSalePrice"] = predicted_prices

        return X_inherited
    except Exception as e:
        print(f"Error during prediction for inherited houses: {e}")
        return None