import streamlit as st


# Define function to generate Table of Contents (TOC)
def create_toc(toc_items, page_title):
    """
    Dynamically create a Table of Contents (TOC) in the sidebar.

    Args:
        toc_items (list of tuples): A list of tuples where each tuple contains:
            - The display text for the TOC item (str)
            - The anchor link for the TOC item (str)
        page_title (str): The title of the current page.
    """
    if toc_items:  # Check if the list is empty
        st.sidebar.subheader(f"Table of Contents for {page_title}")
        for text, anchor in toc_items:
            st.sidebar.markdown(f"[{text}](#{anchor})")
    st.sidebar.divider()  # Add a divider after the TOC
