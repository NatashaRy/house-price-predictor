import streamlit as st

# Define site's title and icon
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    initial_sidebar_state="expanded",
)


# Define pages
pg = st.navigation([
    st.Page("pages/summary.py", title="Project Overview", icon=":material/home:", default=True),
    st.Page("pages/analysis.py", title="Correlation Analysis", icon=":material/analytics:"),
    st.Page("pages/hypotheses.py", title="Hypotheses and Validation", icon=":material/lightbulb:"),
    st.Page("pages/price_predictions.py", title="Sale Price Predictor", icon=":material/money_bag:"),
    st.Page("pages/ml_pipeline_prediction.py", title="Machine Learning Model", icon=":material/smart_toy:"),
], position="sidebar")

# Run the selected page
pg.run()
