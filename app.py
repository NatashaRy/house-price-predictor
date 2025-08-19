import streamlit as st

# Define site's title and icon
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(
    """
    <style>
        /* Change font for sidebar navigation */
        [data-testid="stSidebarNav"] a {
            font-family: monospace; /* Change to monospace font */
            font-size: 0.95rem;     /* Adjust font size */
            color: #666357;         /* Link color */
        }

        /* Add custom text above navigation */
        [data-testid="stSidebarNav"]::before {
            content: "🏠 House Price Predictor";    /* Custom text */
            font-size: 1.05rem;         /* Font size for h3-like appearance */
            padding-left: 0.35rem;      /* Add left padding */
            font-family: monospace;     /* Use monospace font */
            font-weight: bold;          /* Make it bold */
            display: block;             /* Ensure it appears as a block */
            margin-bottom: 1.3rem;      /* Add spacing below the text */
            margin-top: 1rem;           /* Optional: Add spacing above */
            text-transform: uppercase;  /* Transform text to uppercase */
        }

        /* Add custom styles for the sidebar */
        .st-emotion-cache-8atqhb.e4man115 > div {
            gap: 0.4rem;                /* Adjust gap between sidebar items */
        }
        [data-testid="stSidebarUserContent"] {
            padding: 0.8rem 0.6rem;     /* Add padding for sidebar content */
        }
        /* h3 elements */
        [data-testid="stSidebarUserContent"] h3 {
            padding: 0.6rem 0 1.2rem 0;     /* Adjust padding */
        }
        [data-testid="stSidebarUserContent"] ul {
            margin-bottom: 0.5rem;      /* Adjust margin for unordered lists */
        }
        /* Sidebar links */
        [data-testid="stSidebarUserContent"] a {
            font-size: 0.875rem;        /* Custom font size */
        }
        [data-testid="stSidebarUserContent"] a:hover {
            color: #8e47ecff;           /* Custom hover color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
pg = st.navigation([
    st.Page(
        "pages/summary.py",
        title="Project Overview",
        icon=":material/home:",
        default=True
    ),
    st.Page(
        "pages/analysis.py",
        title="Correlation Analysis",
        icon=":material/analytics:"
    ),
    st.Page(
        "pages/hypotheses.py",
        title="Hypotheses and Validation",
        icon=":material/science:"
    ),
    st.Page(
        "pages/price_predictions.py",
        title="Sale Price Predictor",
        icon=":material/money_bag:"
    ),
    st.Page(
        "pages/ml_pipeline_prediction.py",
        title="Machine Learning Model",
        icon=":material/smart_toy:"
    ),
])


# Run the selected page
pg.run()

# Content in sidebar footer
# Quick links
st.sidebar.markdown("### Quick Links")
st.sidebar.markdown(
        "- [GitHub Repository](https://github.com/NatashaRy/"
        "house-price-predictor)")
st.sidebar.markdown("- [README file](https://github.com/NatashaRy/"
                    "house-price-predictor/blob/main/README.md)")
st.sidebar.divider()

# Sidebar footer - about this project and copyright
st.sidebar.markdown(
    """
    <div class="sidebar-footer" style="line-height: 1.35">
        <h3 style="margin-bottom: 0rem; padding-bottom: 0;">
            About This Project
        </h3>
        <p style="font-size: 0.79rem; margin-top: 0.2rem; 
        margin-bottom: 1rem; font-family: monospace; color: #666357;">
            This app was created by <b>
             <a href="mailto:natasha@natasharydell.se" 
             style="font-size: 0.8rem;">
             Natasha Rydell</a>
             </b> as a project for the Full Stack Software Developer course,
              module five <b>Predictive Analytics</b>, at
             <a href="https://codeinstitute.net/se/
             full-stack-software-development-diploma/"
             style="font-size: 0.79rem;">
                Code Institute</a>.
            The app predicts house prices in Ames, Iowa, using
             machine learning. Users can analyze data and make predictions.
        </p>
        <div style="font-size: 0.65rem; text-align: center;
         margin-top: 1rem; border-top: 1px solid #45a348;
         padding-top: 0.4rem; font-family: Arial, sans-serif; color: #666357;">
            © 2025 Natasha Rydell. All rights reserved.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
