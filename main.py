import streamlit as st
import os
from config.const import CSV_ALL_TRAINS, CSV_FMI, CSV_MATCHED_DATA, VIEWER_FOLDER_NAME
from src.DataViewer import DataViewer



def ensure_directories():
    """
    Ensure that all required directories for the application exist.
    Creates them if they don't exist.
    """
    directories = [
        "data",
        "data/ai_results",
        "data/ai_results/by_month",
        "data/ai_results/by_region",
        VIEWER_FOLDER_NAME  # data/viewers
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            st.info(f"Created directory: {directory}")


def main():
    # Ensure all required directories exist
    ensure_directories()
    
    st.title("Railway-FMI Data Viewer")

    viewer = DataViewer()
    
    # If no data is available, stop execution
    if not viewer.has_data():
        return

    viewer.check_file_pattern(CSV_ALL_TRAINS)
    viewer.check_file_pattern(CSV_FMI)
    viewer.check_file_pattern(CSV_MATCHED_DATA)
    viewer.get_date_range()


if __name__ == "__main__":
    main()