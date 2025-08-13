import streamlit as st
import importlib

class MultiPage:
    """
    Explain class.
    """
    def __init__(self, app_name):
        # Initialize the page list to store pages in
        self.pages = []
        # Set app name
        self.app_name = app_name

        # Set page configuration including title, layout, initial sidebar state, and icon
        st.set_page_config(
            page_title=self.app_name,
            initial_sidebar_state="expanded",
            page_icon="🏡"
        )

    # Add a new page to app, shows page title in sidebar, function renders page content
    def add_page(self, title, func):
        self.pages.append({"title": title, "function": func})

    # Run the app by displaying the main title, sidebar and will render page of sidebar choice
    def run(self):
        st.sidebar.title(self.app_name)
        # Radio buttons in sidebar navigation
        page = st.sidebar.radio("Navigation", self.pages, format_func=lambda page: page["title"])
        page["function"]()

        st.sidebar.markdown("\n\n\n\n\n\n---")
        st.sidebar.markdown("For more detailed information, please refer to the [README file](https://github.com/NatashaRy/milestone-project-heritage-housing-issues/blob/main/README.md)  |  Made by Natasha Rydell")
        st.sidebar.markdown("Made by [Natasha Rydell](https://github.com/NatashaRy)")