import streamlit as st
from config.const import CSV_MATCHED_DATA, FOLDER_NAME
from src.DataViewer import DataViewer
import os
import pandas as pd
from ast import literal_eval

# Initialize DataViewer instance
viewer = DataViewer()

# Check if data exists
if not viewer.has_data():
    st.stop()

# Get the date dictionary
viewer.get_date_range()
date_dict = viewer.get_date_dict()

if date_dict:
    col1, col2 = st.columns(2)

    with col1:
        selected_year = st.selectbox("Select Year", sorted(date_dict.keys()))

    with col2:
        selected_month = st.selectbox("Select Month", date_dict[selected_year])

    if selected_year and selected_month:
        file_name = f"{CSV_MATCHED_DATA.replace('.csv', '')}_{selected_year}_{str(selected_month).zfill(2)}.csv"
        file_path = os.path.join(FOLDER_NAME, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()  # Clean up whitespace in column names

            if not df.empty:
                df["departureDate"] = pd.to_datetime(df["departureDate"], errors="coerce")
                available_days = sorted(df["departureDate"].dt.strftime("%Y-%m-%d").dropna().unique())

                selected_day = st.selectbox("Select Day", available_days)

                if selected_day:
                    filtered_df = df[df["departureDate"] == selected_day]
                    
                    st.write(f"### Showing all matched data for **{selected_day}**")
                    st.dataframe(filtered_df)

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

                        if isinstance(train_timetable, str):
                            train_timetable = train_timetable.replace('nan', 'None')
                            timetable_df = pd.DataFrame(literal_eval(train_timetable))

                            preferred_columns = [
                                "stationName",
                                "stationShortCode",
                                "type",
                                "scheduledTime",
                                "actualTime",
                                "differenceInMinutes",
                                "cancelled"
                            ]

                            reordered_columns = preferred_columns + [
                                col for col in timetable_df.columns if col not in preferred_columns
                            ]
                            timetable_df = timetable_df[reordered_columns]

                            st.subheader(f"Time Table for Train **{selected_train_number}** on **{selected_day}**")
                            st.dataframe(timetable_df)

                            if not timetable_df.empty:
                                st.subheader("Select an EMS Station ")
                                unique_stations = timetable_df["stationName"].sort_values().unique()

                                selected_station = st.selectbox(
                                    "Select Station", 
                                    unique_stations,
                                    index=0
                                )

                                # Filter data for the selected station
                                station_data = timetable_df[timetable_df["stationName"] == selected_station]

                                if not station_data.empty:
                                    # Extract weather data from the timetable dictionary
                                    weather_data = station_data.iloc[0].get("weather_observations")

                                    if weather_data:
                                        if isinstance(weather_data, str):
                                            weather_data = weather_data.replace('nan', 'None')
                                            weather_data = literal_eval(weather_data)

                                        st.subheader(f"Weather Conditions for **{selected_station}** on **{selected_day}**")
                                        st.json(weather_data)
                                    else:
                                        st.warning(f"No weather data available for station **{selected_station}**.")
        else:
            st.warning(f"⚠️ File `{file_name}` not found in `{FOLDER_NAME}`.")
