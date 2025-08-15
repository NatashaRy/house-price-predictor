import streamlit as st
import os
import importlib


class MultiPage:
    """
    Class to create a multi-page Streamlit app.
    Allows users to navigate between different pages through a sidebar.
    """

    def __init__(self, app_name):
        """
        Initialize the MultiPage app.

        Args:
            app_name (str): Name of the Streamlit application
        """
        # Initialize the page list to store pages in
        self.pages = []
        # Set app name
        self.app_name = app_name

        """
        Set page configuration including title,
        layout, initial sidebar state, and icon
        """
        st.set_page_config(
            page_title=self.app_name,
            initial_sidebar_state="expanded",
            page_icon="🏡"
        )

    def add_page(self, title, func):
        """
        Add a new page to app.
        Shows page title in sidebar, function renders page content.

        Args:
            title (str): Title of the page
            func (callable): Function that renders the page content
        """
        self.pages.append({"title": title, "function": func})

    def add_pages_from_folder(self, folder_path):
        """
        Dynamically load and add pages from a specified folder.
        Each Python file in the folder (except __init__.py)
        is treated as a page.

        Args:
            folder_path (str): Path to the folder containing page modules
        """
        for file in os.listdir(folder_path):
            if file.endswith(".py") and file != "__init__.py":
                module_name = file[:-3]  # Remove .py extension
                module = importlib.import_module(
                        f"{folder_path}.{module_name}")
                # Add the page using its title and function
                self.add_page(module.page_title, module.page_function)

    def run(self):
        """
        Run the app by displaying the main title and sidebar.

        Renders page based on sidebar choice.
        """
        st.sidebar.title(self.app_name)
        try:
            page = st.sidebar.radio(
                "Navigation",
                self.pages,
                format_func=lambda page: page["title"]
            )
            page["function"]()
        except Exception as e:
            st.error(f"An error occurred: {e}")
