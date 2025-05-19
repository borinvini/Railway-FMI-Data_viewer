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
                            causes_data = timetable_df[
                                timetable_df['causes'].notna() & 
                                timetable_df['causes'].apply(lambda x: 
                                    not (isinstance(x, str) and x in ('[]', '{}')) and
                                    not (isinstance(x, (list, dict)) and len(x) == 0)
                                )
                            ]
                            
                            if not causes_data.empty:
                                # Load metadata for all levels of delay causes
                                metadata_path = os.path.join(VIEWER_FOLDER_NAME, "metadata", "metadata_train_causes.csv")
                                detailed_metadata_path = os.path.join(VIEWER_FOLDER_NAME, "metadata", "metadata_train_causes_detailed.csv")
                                third_metadata_path = os.path.join(VIEWER_FOLDER_NAME, "metadata", "metadata_third_train_causes.csv")
                                
                                category_code_map = {}
                                detailed_category_code_map = {}
                                third_category_code_map = {}
                                
                                # Load primary category metadata
                                if os.path.exists(metadata_path):
                                    try:
                                        metadata_df = pd.read_csv(metadata_path)
                                        # Create mapping from code to meaning
                                        category_code_map = dict(zip(metadata_df['categoryCode'], metadata_df['categoryName_en']))
                                    except Exception as e:
                                        st.warning(f"Error loading delay causes metadata: {e}")
                                
                                # Load detailed category metadata
                                if os.path.exists(detailed_metadata_path):
                                    try:
                                        detailed_metadata_df = pd.read_csv(detailed_metadata_path)
                                        # Create mapping from detailed code to meaning
                                        detailed_category_code_map = dict(zip(detailed_metadata_df['detailedCategoryCode'], 
                                                                            detailed_metadata_df['detailedCategoryName_en']))
                                    except Exception as e:
                                        st.warning(f"Error loading detailed delay causes metadata: {e}")
                                
                                # Load third category metadata
                                if os.path.exists(third_metadata_path):
                                    try:
                                        third_metadata_df = pd.read_csv(third_metadata_path)
                                        # Create mapping from third code to meaning
                                        third_category_code_map = dict(zip(third_metadata_df['thirdCategoryCode'], 
                                                                        third_metadata_df['thirdCategoryName_en']))
                                    except Exception as e:
                                        st.warning(f"Error loading third level delay causes metadata: {e}")
                                
                                st.subheader("Delay Causes")
                                st.write("The following delay causes were reported for this train:")
                                
                                # Display each cause with its associated station
                                for index, row in causes_data.iterrows():
                                    st.markdown(f"**Station: {row['stationName']} ({row['type']})**")
                                    
                                    # Handle the causes data based on its type
                                    try:
                                        # Get the causes data
                                        causes = row['causes']
                                        if isinstance(causes, str):
                                            causes = literal_eval(causes)
                                        
                                        # Display the JSON
                                        st.json(causes)
                                        
                                        # Add explanations for all category codes
                                        if isinstance(causes, list):
                                            for cause in causes:
                                                if isinstance(cause, dict):
                                                    # Display primary category code meaning
                                                    if 'categoryCode' in cause and category_code_map:
                                                        code = cause['categoryCode']
                                                        if code in category_code_map:
                                                            meaning = category_code_map[code]
                                                            st.markdown(f"**Category code '{code}'**: {meaning}")
                                                    
                                                    # Display detailed category code meaning
                                                    if 'detailedCategoryCode' in cause and detailed_category_code_map:
                                                        detailed_code = cause['detailedCategoryCode']
                                                        if detailed_code in detailed_category_code_map:
                                                            detailed_meaning = detailed_category_code_map[detailed_code]
                                                            st.markdown(f"**Detailed category code '{detailed_code}'**: {detailed_meaning}")
                                                    
                                                    # Display third category code meaning
                                                    if 'thirdCategoryCode' in cause and third_category_code_map:
                                                        third_code = cause['thirdCategoryCode']
                                                        if third_code in third_category_code_map:
                                                            third_meaning = third_category_code_map[third_code]
                                                            st.markdown(f"**Third category code '{third_code}'**: {third_meaning}")
                                    except Exception as e:
                                        # If parsing fails, display the raw data
                                        st.write(row['causes'])
                                        st.warning(f"Error parsing causes data: {e}")
                                    
                                    # Add a separator between entries
                                    st.markdown("---")
                            else:
                                # Optionally show a message if no delay causes were found
                                st.info("No delay causes reported for this train.")

        else:
            st.warning(f"⚠️ File `{file_name}` not found in `{VIEWER_FOLDER_NAME}`.")