# Import 
import streamlit as st

# Import site navigation
from app_pages.multipage import MultiPage

# Import pages
from app_pages.summary import summary_body
from app_pages.analysis import analysis_body

# Initialize MultiPage app
app = MultiPage(app_name="House Price Predictor")

# Add page to the app
app.add_page("Project Overview", summary_body)
app.add_page("Correlation Analysis", analysis_body)

# Run the app
app.run()

