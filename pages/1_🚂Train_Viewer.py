import streamlit as st
from config.const import CSV_ALL_TRAINS, VIEWER_FOLDER_NAME
from src.DataViewer import DataViewer
import os
import pandas as pd
from ast import literal_eval

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
        file_path = os.path.join(VIEWER_FOLDER_NAME, file_name)
        
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
                        train_timetable = train_timetable.replace('nan', 'None')  # Replace nan with None for safe parsing
                        timetable_df = pd.DataFrame(literal_eval(train_timetable))

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

                        # Check if 'causes' column exists in the timetable data
                        if 'causes' in timetable_df.columns:
                            # Get only rows with non-empty causes data
                            # Filter out both NaN values and empty arrays/lists
                            causes_data = timetable_df[
                                timetable_df['causes'].notna() & 
                                timetable_df['causes'].apply(lambda x: 
                                    not (isinstance(x, str) and x in ('[]', '{}')) and
                                    not (isinstance(x, (list, dict)) and len(x) == 0)
                                )
                            ]
                            
                            if not causes_data.empty:
                                st.subheader("Delay Causes")
                                st.write("The following delay causes were reported for this train:")
                                
                                # Display each cause with its associated station
                                for index, row in causes_data.iterrows():
                                    st.markdown(f"**Station: {row['stationName']} ({row['type']})**")
                                    
                                    # Handle the causes data based on its type
                                    if isinstance(row['causes'], str):
                                        # If it's a string, try to parse it
                                        try:
                                            from ast import literal_eval
                                            causes_obj = literal_eval(row['causes'])
                                            if causes_obj and len(causes_obj) > 0:  # Only display if not empty
                                                st.json(causes_obj)
                                        except:
                                            # If parsing fails, display as text
                                            st.write(row['causes'])
                                    elif isinstance(row['causes'], (list, dict)) and len(row['causes']) > 0:
                                        # If it's already structured data and not empty, display as JSON
                                        st.json(row['causes'])
                                    else:
                                        # For any other type, display as string if not empty
                                        cause_str = str(row['causes'])
                                        if cause_str and cause_str not in ("[]", "{}"):
                                            st.write(cause_str)
                                    
                                    # Add a separator between entries
                                    st.markdown("---")
                            else:
                                # Optionally show a message if no delay causes were found
                                st.info("No delay causes reported for this train.")

        else:
            st.warning(f"⚠️ File `{file_name}` not found in `{VIEWER_FOLDER_NAME}`.")