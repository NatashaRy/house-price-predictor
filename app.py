import streamlit as st

# Import site navigation
from app_pages.multipage import MultiPage

# Import pages
from app_pages.summary import summary_body
from app_pages.analysis import analysis_body
from app_pages.hypotheses import hypotheses_body
from app_pages.price_predictions import predict_price_body
from app_pages.ml_pipeline_prediction import ml_pipeline_prediction_body

# Initialize MultiPage app
app = MultiPage(app_name="House Price Predictor")

# Add page to the app
app.add_page("Project Overview", summary_body)
app.add_page("Correlation Analysis", analysis_body)
app.add_page("Hypotheses and Validation", hypotheses_body)
app.add_page("Predict House Sale Price", predict_price_body)
app.add_page("Machine Learning Model", ml_pipeline_prediction_body)

# Run the app
try:
    app.run()
except Exception as e:
    st.error(f"An error occurred: {e}")
