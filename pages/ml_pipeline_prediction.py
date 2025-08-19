import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_reg import regression_performance


# This page displays content of the content of
# the Machine Learning pipeline page in the Streamlit app.
#
# That includes:
# - Page introduction
# - Results from R² performance metrics for train and test set
# - Pipeline steps
# - Feature importance, listing best features
# - Visualization of feature importance
# - Model performance with evaluation of train and test sets performance
# - Results of metrics score:
#     - Mean Absolute Error (MAE)
#     - Mean Squared Error (MSE)
#     - Root Mean Squared Error (RMSE)
#     - R² Score
# - Regression Evaluation Plots for train and test set

# Title and Introduction
st.title("🤖 Machine Learning Model")
st.write(
    """
    This page provides an overview of the trained machine learning
     pipeline used to predict house prices. The goal is to evaluate
     the model's performance, understand the importance of features,
     and visualize the regression results.
    """
)

# Summary of model performance
st.info(
    """
    To address **Business Requirement 2 (BR2)**, we aimed to train a
    Regressor model and tune the pipeline to achieve at least 0.75
    accuracy in predicting the sales price of a property based on its
    attributes. We successfully met this target and trained multiple
    versions of the model to ensure we explored potential improvements.

    * The pipeline performance for the best model:
        * Train set **R² = 0.80**
        * Test set **R² = 0.79**

    Below, we present the pipeline steps, the list of best features
     along with the feature importance plot, pipeline performance,
     and regression performance report.
    """
)

# Horizontal line
st.divider()


def ml_pipeline_prediction_body():
    # Load regression pipeline
    version = 'v1'
    model_path = (
        f"outputs/ml_pipeline/predict_price/{version}/"
        f"regression_pipeline.pkl"
    )
    price_pipe = load_pkl_file(model_path)

    if price_pipe is None:
        st.error(
            f"Failed to load model from {model_path}. Please check the file."
        )
        return

    # Load training and test data
    try:
        X_train = pd.read_parquet(
            f"outputs/ml_pipeline/predict_price/{version}/X_train.parquet"
        )
        X_test = pd.read_parquet(
            f"outputs/ml_pipeline/predict_price/{version}/X_test.parquet"
        )
        y_train = pd.read_parquet(
            f"outputs/ml_pipeline/predict_price/{version}/y_train.parquet"
        )
        y_test = pd.read_parquet(
            f"outputs/ml_pipeline/predict_price/{version}/y_test.parquet"
        )

        # Synchronize columns between X_train and X_test
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0

        extra_cols = set(X_test.columns) - set(X_train.columns)
        X_test = X_test.drop(columns=extra_cols)

        X_test = X_test[X_train.columns]

    except FileNotFoundError as e:
        st.error(f"Failed to load data: {e}")
        return

    # Show ML pipeline
    st.header("⚒️Machine Learning Pipeline")
    st.write("Structure of the trained machine learning pipeline "
             "used to predict house prices."
             )
    st.code(price_pipe)

    # Horizontal line
    st.divider()

    # Display feature importance
    st.header("🌟Feature Importance")
    st.write("Best features in best model:")
    st.write(X_train.columns.tolist())
    st.write("The plot shows the importance of each feature in the model.")

    feature_importance_path = (
        f"outputs/ml_pipeline/predict_price/{version}/"
        f"feature_importance.png"
    )

    try:
        feature_importance_img = plt.imread(feature_importance_path)
        st.image(
            feature_importance_img,
            caption="Feature Importance",
            use_container_width=True
        )
    except FileNotFoundError as e:
        st.error(f"Failed to load feature importance image: {e}")

    # Horizontal line
    st.divider()

    # Evaluate model performance
    st.header("🚀Model Performance")
    st.write(
        "Performance metrics of the trained model on both the "
        "training and test datasets."
    )

    regression_performance(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        pipeline=price_pipe
    )

    # Regression Evaluation Plots
    st.subheader("Regression Evaluation Plots")
    st.write("Compare actual values and predictions for both training "
             "and test datasets."
             )
    regression_evaluation_plots(X_train, y_train, X_test, y_test, price_pipe)


def regression_evaluation_plots(
    X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5
):
    """
    Create scatterplots to compare actual values and predictions
    for both training and test datasets, with multiple colors from the
    Spectral palette.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.DataFrame): Test target
        pipeline: Trained ML pipeline
        alpha_scatter (float): Transparency for scatter points
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

    # Scatterplot for Train data
    sns.scatterplot(
        x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0],
        hue=pred_train, palette="Spectral", legend=False
    )
    sns.lineplot(x=y_train, y=y_train, color='black', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("Train Set", fontsize=20)

    # Scatterplot for Test data
    sns.scatterplot(
        x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1],
        hue=pred_test, palette="Spectral", legend=False
    )
    sns.lineplot(x=y_test, y=y_test, color='black', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Test Set", fontsize=20)

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)


# Run the ML pipeline prediction body function
ml_pipeline_prediction_body()
