import streamlit as st
from config.const import CSV_FMI, VIEWER_FOLDER_NAME
from src.DataViewer import DataViewer
import os
import pandas as pd

# Initialize DataViewer instance
viewer = DataViewer()

# Check if data exists
if not viewer.has_data():
    st.stop()  # Stop execution if no data

# Get the date dictionary
viewer.get_date_range()
date_dict = viewer.get_date_dict()

if date_dict:
    # Create two columns for side-by-side select boxes
    col1, col2 = st.columns(2)

    with col1:
        selected_year = st.selectbox("Select Year", sorted(date_dict.keys()))

    with col2:
        selected_month = st.selectbox("Select Month", date_dict[selected_year])

    if selected_year and selected_month:
        file_name = f"{CSV_FMI.replace('.csv', '')}_{selected_year}_{str(selected_month).zfill(2)}.csv"
        file_path = os.path.join(VIEWER_FOLDER_NAME, file_name)
        
        # Load the CSV file
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            if not df.empty:
                # Convert the timestamp column to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                
                # Extract unique available days
                available_days = sorted(df["timestamp"].dt.strftime("%Y-%m-%d").dropna().unique())

                selected_day = st.selectbox("Select Day", available_days)

                if selected_day:
                    # Filter data for the selected day
                    filtered_df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == selected_day]
                    
                    st.write(f"### Showing weather data for **{selected_day}**")
                    st.dataframe(filtered_df)  # Show all rows for the selected day

                    # If data is available, allow user to select a specific station
                    if not filtered_df.empty:
                        st.subheader("Select an EMS Station")
                        unique_stations = filtered_df["station_name"].sort_values().unique()

                        selected_station = st.selectbox(
                            "Select Station", 
                            unique_stations,
                            index=0
                        )

                        # Filter data for the selected station
                        station_data = filtered_df[filtered_df["station_name"] == selected_station]

                        # Display the station's weather data
                        st.subheader(f"Weather Data for **{selected_station}** on **{selected_day}**")
                        st.dataframe(station_data)

        else:
            st.warning(f"⚠️ File `{file_name}` not found in `{VIEWER_FOLDER_NAME}`.")