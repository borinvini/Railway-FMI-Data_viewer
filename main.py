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
        VIEWER_FOLDER_NAME,  # data/viewers
        os.path.join(VIEWER_FOLDER_NAME, "train_data"),
        os.path.join(VIEWER_FOLDER_NAME, "weather_data"),
        os.path.join(VIEWER_FOLDER_NAME, "matched_data"),
        os.path.join(VIEWER_FOLDER_NAME, "metadata")
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

    # Use the updated check_file_pattern method with subfolders
    st.subheader("Data File Status")
    
    # Check train data files
    st.markdown("#### Train Data")
    viewer.check_file_pattern(CSV_ALL_TRAINS, "train_data")
    
    # Check weather data files
    st.markdown("#### Weather Data")
    viewer.check_file_pattern(CSV_FMI, "weather_data")
    
    # Check matched data files
    st.markdown("#### Matched Data")
    viewer.check_file_pattern(CSV_MATCHED_DATA, "matched_data")
    
    # Display date range
    st.subheader("Date Range")
    viewer.get_date_range()


if __name__ == "__main__":
    main()