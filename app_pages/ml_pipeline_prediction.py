import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_reg import regression_performance


def ml_pipeline_prediction_body():
    """
    Displays the content of the ML pipline in the Streamlit app.

    This includes:
    - Introduction to the machine learning pipeline
    - Overview of the model's performance and feature importance
    - Visualizations of regression performance
    """
##### Title and introduction
    st.title("ML Model: House Price Prediction")
    st.markdown(
        "**This page provides an overview of the trained machine learning pipeline used to predict house prices.**\n"
        "**The goal is to evaluate the model's performance, understand the importance of features, and visualize the regression results.**"
    )

    st.info(
        """
        To address **Business Requirement 2 (BR2)**, we aimed to train a Regressor model and tune the pipeline 
        to achieve at least 0.75 accuracy in predicting the sales price of a property based on its attributes. 
        We successfully met this target and trained multiple versions of the model to ensure we explored potential improvements.\n
        * The pipeline performance for the best model:
            * Train set **R² = 0.83**
            * Test set **R² = 0.82*\n
        We present the pipeline steps, the list of best features along with the feature importance plot,
        pipeline performance, and regression performance report below.
        """
    )

    st.markdown("---")


##### Load regression pipeline
    version = 'v1'
    model_path = f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl"
    price_pipe = load_pkl_file(model_path)

    if price_pipe is None:
        st.error("Failed to load model from {model_path}, please check file.")
        return

##### Load training and test data
    try:
        version = 'v1'  # Specify version of the data
        X_train = pd.read_parquet(f"outputs/ml_pipeline/predict_price/{version}/X_train.parquet")
        X_test = pd.read_parquet(f"outputs/ml_pipeline/predict_price/{version}/X_test.parquet")
        y_train = pd.read_parquet(f"outputs/ml_pipeline/predict_price/{version}/y_train.parquet")
        y_test = pd.read_parquet(f"outputs/ml_pipeline/predict_price/{version}/y_test.parquet")

        # Synchronize columns between X_train and X_test
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0  # Or np.nan for numerical columns

        extra_cols = set(X_test.columns) - set(X_train.columns)
        X_test = X_test.drop(columns=extra_cols)

        X_test = X_test[X_train.columns]

    except FileNotFoundError as e:
        st.error(f"Failed to load data: {e}")
        return

##### Show ML pipeline
    st.write("### ML Pipeline")
    st.write("Structure of the trained machine learning pipeline used to predict house prices.")
    st.code(price_pipe)  # Show pipeline structure
    st.markdown("---")

##### Show feature importance
    st.write("### Feature Importance")
    st.write(X_train.columns.tolist())
    st.write("The plot shows the importance of each feature in the model.")

    # Path to feature importance image
    feature_importance_path = r"outputs\ml_pipeline\predict_price\v1\feature_importance.png"

    try:
        # Load and display the image
        feature_importance_img = plt.imread(feature_importance_path)
        st.image(feature_importance_img, caption="Feature Importance", use_container_width=True)
    except FileNotFoundError as e:
        st.error(f"Failed to load feature importance image: {e}")

    st.markdown("---")

##### Evaluate the models performance
    st.write("### Model Performance")
    st.write("Performance metrics of the trained model on both the training and test datasets.")

    # Call regression_performance function
    regression_performance(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        pipeline=price_pipe
    )

    # Call regression evaluation plots
    st.write("### Regression Evaluation Plots\n"
             "Compare actual values and predictions for both training and test datasets."
    )
    regression_evaluation_plots(X_train, y_train, X_test, y_test, price_pipe)


##### Regression plots function
def regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5):
    """
    Create scatterplots to compare actual values and predictions
    for both training and test datasets, with multiple colors from the Spectral palette.
    """
    # Convert y_train and y_test to 1D array if they are DataFrames
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values.ravel()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values.ravel()

    # Predictions for Train and Test data
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Scatter plot for Train data
    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0], hue=pred_train, palette="Spectral", legend=False)
    sns.lineplot(x=y_train, y=y_train, color='black', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("Train Set", fontsize=20)

    # Scatterplot for Test data
    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1], hue=pred_test, palette="Spectral", legend=False)
    sns.lineplot(x=y_test, y=y_test, color='black', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Test Set", fontsize=20)

    # Adjust layout and show plot
    plt.tight_layout()
    st.pyplot(fig)