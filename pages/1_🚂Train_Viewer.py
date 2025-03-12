import streamlit as st
from config.const import CSV_ALL_TRAINS, FOLDER_NAME
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
        file_name = f"{CSV_ALL_TRAINS.replace('.csv', '')}_{selected_year}_{str(selected_month).zfill(2)}.csv"
        file_path = os.path.join(FOLDER_NAME, file_name)
        
        # Load the CSV file
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            if not df.empty:
                # Show available days from departureDate column
                df["departureDate"] = pd.to_datetime(df["departureDate"], errors="coerce")
                available_days = sorted(df["departureDate"].dt.strftime("%Y-%m-%d").dropna().unique())

                selected_day = st.selectbox("Select Day", available_days)

                if selected_day:
                    # Filter data for the selected day
                    filtered_df = df[df["departureDate"] == selected_day]
                    
                    st.write(f"### Showing all trains for **{selected_day}**")
                    st.dataframe(filtered_df)  # Show all rows for the selected day

                    # If data is available, allow user to select a specific train number
                    if not filtered_df.empty:
                        st.subheader("Select a Train Number")
                        unique_train_numbers = filtered_df["trainNumber"].sort_values().unique()

                        selected_train_number = st.selectbox(
                            "Select Train Number", 
                            unique_train_numbers,
                            index=0
                        )

                        # Extract timetable data for the selected train
                        train_timetable = filtered_df.loc[
                            filtered_df["trainNumber"] == selected_train_number, 
                            "timeTableRows"
                        ].values[0]

                        # Convert timetable data to a DataFrame
                        timetable_df = pd.DataFrame(eval(train_timetable))

                        # Define preferred column order
                        preferred_columns = [
                            "stationName",
                            "stationShortCode",
                            "type",
                            "scheduledTime",
                            "actualTime",
                            "differenceInMinutes",
                            "cancelled"
                        ]

                        # Reorder the columns based on preferred order
                        reordered_columns = preferred_columns + [
                            col for col in timetable_df.columns if col not in preferred_columns
                        ]
                        timetable_df = timetable_df[reordered_columns]

                        # Display the timetable
                        st.subheader(f"Time Table for Train **{selected_train_number}** on **{selected_day}**")
                        st.dataframe(timetable_df)

        else:
            st.warning(f"⚠️ File `{file_name}` not found in `{FOLDER_NAME}`.")
