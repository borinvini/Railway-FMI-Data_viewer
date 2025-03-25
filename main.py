import streamlit as st
from config.const import CSV_ALL_TRAINS, CSV_FMI, CSV_MATCHED_DATA
from src.DataViewer import DataViewer


def main():
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