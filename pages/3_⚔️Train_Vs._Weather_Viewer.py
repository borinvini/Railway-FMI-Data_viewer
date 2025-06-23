import streamlit as st
from config.const import CSV_MATCHED_DATA, VIEWER_FOLDER_NAME
from src.DataViewer import DataViewer
import os
import pandas as pd
from ast import literal_eval
import folium
from streamlit_folium import st_folium

# Initialize DataViewer instance
viewer = DataViewer()

# Check if data exists
if not viewer.has_data():
    st.stop()

# Function to get station coordinates from various possible metadata files
def get_station_coordinates(station_short_codes):
    """
    Get latitude and longitude for a list of station short codes.
    
    Args:
        station_short_codes: List of station short codes
        
    Returns:
        DataFrame with stationShortCode, latitude, and longitude columns
    """
    # Try the closest_ems_to_train_stations file first (it has train coordinates)
    mapping_file = os.path.join(VIEWER_FOLDER_NAME, "metadata", "metadata_closest_ems_to_train_stations.csv")
    if not os.path.exists(mapping_file):
        mapping_file = os.path.join(VIEWER_FOLDER_NAME, "metadata", "closest_ems_to_train_stations.csv")
    
    if os.path.exists(mapping_file):
        try:
            mapping_df = pd.read_csv(mapping_file)
            
            # Check for and standardize column names
            if 'train_station_short_code' in mapping_df.columns:
                mapping_df['stationShortCode'] = mapping_df['train_station_short_code']
            
            if 'train_lat' in mapping_df.columns and 'train_long' in mapping_df.columns:
                mapping_df['latitude'] = mapping_df['train_lat']
                mapping_df['longitude'] = mapping_df['train_long']
                
                # Filter for the stations we need
                filtered_stations = mapping_df[mapping_df['stationShortCode'].isin(station_short_codes)]
                
                if not filtered_stations.empty:
                    # Also save the EMS information
                    cols_to_keep = ['stationShortCode', 'latitude', 'longitude']
                    ems_cols = [col for col in mapping_df.columns if col.startswith('ems_') or col == 'closest_ems_station']
                    cols_to_keep.extend(ems_cols)
                    return filtered_stations[cols_to_keep]
        except Exception as e:
            st.warning(f"⚠️ Error reading train-EMS mapping file: {e}")
    
    # Next try train_stations metadata
    stations_file = os.path.join(VIEWER_FOLDER_NAME, "metadata", "train_stations.csv")
    if not os.path.exists(stations_file):
        stations_file = os.path.join(VIEWER_FOLDER_NAME, "metadata", "metadata_train_stations.csv")
        
    if os.path.exists(stations_file):
        try:
            stations_df = pd.read_csv(stations_file)
            
            # Check and standardize column names
            col_map = {
                'stationShortCode': ['stationShortCode', 'short_code', 'shortCode'],
                'latitude': ['latitude', 'lat', 'y'],
                'longitude': ['longitude', 'long', 'lon', 'x']
            }
            
            # Find the actual column names in the DataFrame
            for std_col, possible_cols in col_map.items():
                for col in possible_cols:
                    if col in stations_df.columns:
                        stations_df[std_col] = stations_df[col]
                        break
            
            # Check if we have all required columns after mapping
            if all(col in stations_df.columns for col in ['stationShortCode', 'latitude', 'longitude']):
                # Filter for the stations we need
                filtered_stations = stations_df[stations_df['stationShortCode'].isin(station_short_codes)]
                
                if not filtered_stations.empty:
                    return filtered_stations[['stationShortCode', 'latitude', 'longitude']]
        except Exception as e:
            st.warning(f"⚠️ Error reading station metadata file: {e}")
    
    # If we still don't have coordinates, return empty DataFrame
    st.warning("⚠️ Could not find station coordinates in any of the expected files.")
    return pd.DataFrame()

# Helper function to format weather data for popup
def format_weather_data_for_popup(weather_data):
    """
    Format weather data for display in a popup.
    
    Args:
        weather_data: Dictionary or string containing weather data
        
    Returns:
        HTML formatted string with weather data
    """
    if not weather_data:
        return "<p><i>No weather data available</i></p>"
    
    # If weather_data is a string, convert it to a dictionary
    if isinstance(weather_data, str):
        try:
            weather_data = weather_data.replace('nan', 'None')
            weather_data = literal_eval(weather_data)
        except Exception as e:
            return f"<p><i>Error parsing weather data: {str(e)}</i></p>"
    
    # Create HTML for the weather data
    html = "<div style='max-height: 300px; overflow-y: auto;'>"
    html += "<h4>Weather Measurements</h4>"
    html += "<table style='width: 100%; border-collapse: collapse;'>"
    
    # Priority weather conditions to show first
    priority_keys = [
        'Air temperature', 
        'Wind speed', 
        'Gust speed', 
        'Wind direction', 
        'Relative humidity',
        'Precipitation amount', 
        'Snow depth'
    ]
    
    # Show priority keys first
    for key in priority_keys:
        if key in weather_data and weather_data[key] is not None:
            value = weather_data[key]
            # Format the value based on its type
            if isinstance(value, (int, float)):
                if abs(value) < 0.01:
                    value_str = "0"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
                
            html += f"<tr><td><b>{key}</b></td><td>{value_str}</td></tr>"
    
    # Show remaining keys
    for key, value in weather_data.items():
        if key not in priority_keys and value is not None:
            # Format the value based on its type
            if isinstance(value, (int, float)):
                if abs(value) < 0.01:
                    value_str = "0"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
                
            html += f"<tr><td><b>{key}</b></td><td>{value_str}</td></tr>"
    
    html += "</table></div>"
    return html

# Function to create a train route map
def create_train_route_map(timetable_df):
    """
    Create a map showing the route of the train through its stations.
    Only departures will be shown as markers, along with the closest EMS stations
    and lines connecting train stations to their EMS stations.
    
    Args:
        timetable_df: DataFrame containing the train's timetable
        
    Returns:
        Folium map object or None if coordinates can't be found
    """
    # Extract unique stations from timetable
    unique_stations = timetable_df['stationShortCode'].unique()
    
    # Get coordinates for each station and EMS information
    station_coords = get_station_coordinates(unique_stations)
    
    if station_coords.empty:
        st.warning("⚠️ Could not get station coordinates. Map cannot be displayed.")
        return None
    
    # Check if we have EMS information
    has_ems_info = all(col in station_coords.columns for col in ['closest_ems_station', 'ems_latitude', 'ems_longitude'])
    if not has_ems_info:
        st.warning("⚠️ EMS station information not available in the metadata. Only train stations will be shown.")
    
    # Merge coordinates with timetable to get the full sequence with time info
    timetable_with_coords = pd.merge(timetable_df, station_coords, on='stationShortCode', how='left')
    
    # Check if we have all coordinates
    if timetable_with_coords['latitude'].isna().any() or timetable_with_coords['longitude'].isna().any():
        st.warning("⚠️ Some stations are missing coordinates. Map will be incomplete.")
        # Filter out stations without coordinates
        timetable_with_coords = timetable_with_coords.dropna(subset=['latitude', 'longitude'])
    
    if timetable_with_coords.empty:
        return None
        
    # Create a base map centered on the average of coordinates
    center_lat = timetable_with_coords['latitude'].mean()
    center_long = timetable_with_coords['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_long],
        zoom_start=7,
        tiles="OpenStreetMap",
        attr="© OpenStreetMap contributors"
    )
    
    # Create feature groups for different elements
    departures_group = folium.FeatureGroup(name="Departures", show=True)
    route_group = folium.FeatureGroup(name="Train Route", show=True)
    ems_group = folium.FeatureGroup(name="EMS Stations", show=True)
    ems_connections_group = folium.FeatureGroup(name="EMS Connections", show=True)
    
    # Process the timetable to get only departures
    departures = timetable_with_coords[timetable_with_coords['type'] == 'DEPARTURE']
    
    # Create a sequence of all stops for the route line
    # Use all stations but make sure we don't have duplicates in sequence
    all_stations = timetable_with_coords.drop_duplicates('stationShortCode')
    
    # Add markers for departures (green)
    for idx, row in departures.iterrows():
        delay_text = ""
        if 'differenceInMinutes' in row and not pd.isna(row['differenceInMinutes']):
            delay = row['differenceInMinutes']
            delay_color = "green" if delay <= 0 else "red"
            delay_text = f"<br><span style='color:{delay_color}'>Delay: {delay} min</span>"
            
        popup_text = f"""
        <div style='min-width: 180px'>
            <b>{row['stationName']}</b> ({row['stationShortCode']})<br>
            <b>Departure:</b> {row.get('scheduledTime', 'N/A')}{delay_text}
        </div>
        """
        
        # For departures, use a green marker
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"🟢 Departure: {row['stationName']}",
            icon=folium.Icon(color="green", icon="train", prefix="fa")
        ).add_to(departures_group)
    
    # Add a PolyLine connecting all stations in sequence
    # We need to make sure the stations are in the right order based on the timetable
    # Get the sequence by sorting by scheduled time
    timetable_with_coords['scheduledTime'] = pd.to_datetime(timetable_with_coords['scheduledTime'], errors='coerce')
    sequence = timetable_with_coords.sort_values('scheduledTime')
    
    # Get unique stations in sequence (to avoid going back and forth between the same stations)
    unique_sequence = sequence.drop_duplicates('stationShortCode', keep='first')
    
    route_coords = unique_sequence[['latitude', 'longitude']].values.tolist()
    
    if len(route_coords) > 1:  # Need at least 2 points for a line
        folium.PolyLine(
            locations=route_coords,
            color="blue",
            weight=3,
            opacity=0.7,
            tooltip="Train Route"
        ).add_to(route_group)
    
    # Add EMS stations and connections if we have the information
    if has_ems_info:
        # Dictionary to store EMS weather data (to avoid duplicating work for shared EMS stations)
        ems_weather_data = {}
        
        # First pass: collect weather data for each EMS station
        for idx, row in timetable_with_coords.iterrows():
            if pd.notna(row['closest_ems_station']) and 'weather_observations' in row:
                ems_station = row['closest_ems_station']
                
                # Only process each EMS station once
                if ems_station not in ems_weather_data:
                    # Extract weather observations
                    weather_data = row.get('weather_observations')
                    ems_weather_data[ems_station] = weather_data
        
        # Get unique EMS stations to avoid duplicates
        unique_ems = all_stations.drop_duplicates('closest_ems_station')
        
        # Add EMS stations as markers
        for idx, row in unique_ems.iterrows():
            if pd.notna(row['closest_ems_station']) and pd.notna(row['ems_latitude']) and pd.notna(row['ems_longitude']):
                ems_station = row['closest_ems_station']
                
                # Get weather data for this EMS station if available
                weather_data = ems_weather_data.get(ems_station)
                weather_html = format_weather_data_for_popup(weather_data)
                
                ems_popup_text = f"""
                <div style='min-width: 250px; max-width: 400px;'>
                    <h3>☁️ EMS Station: {ems_station}</h3>
                    <b>Coordinates:</b> {row['ems_latitude']:.6f}, {row['ems_longitude']:.6f}<br>
                    <hr>
                    {weather_html}
                </div>
                """
                
                # Add EMS station marker (blue cloud)
                folium.Marker(
                    location=[row['ems_latitude'], row['ems_longitude']],
                    popup=folium.Popup(ems_popup_text, max_width=400),
                    tooltip=f"☁️ EMS: {ems_station}",
                    icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
                ).add_to(ems_group)
        
        # Add lines connecting train stations to their closest EMS
        for idx, row in all_stations.iterrows():
            if pd.notna(row['closest_ems_station']) and pd.notna(row['ems_latitude']) and pd.notna(row['ems_longitude']):
                # Calculate distance between train station and EMS
                distance_km = None
                if 'distance_km' in row:
                    distance_km = row['distance_km']
                
                # Create the connection line
                folium.PolyLine(
                    locations=[
                        [row['latitude'], row['longitude']],
                        [row['ems_latitude'], row['ems_longitude']]
                    ],
                    color="purple",
                    weight=2,
                    opacity=0.6,
                    tooltip=f"Distance to EMS: {distance_km:.2f} km" if distance_km else "Train-EMS Connection",
                    dash_array="5, 10"  # Dashed line
                ).add_to(ems_connections_group)
    
    # Add all feature groups to the map
    departures_group.add_to(m)
    route_group.add_to(m)
    if has_ems_info:
        ems_group.add_to(m)
        ems_connections_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

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
        file_path = os.path.join(VIEWER_FOLDER_NAME, "matched_data", file_name)
        
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
                                "differenceInMinutes_offset",
                                "differenceInMinutes_eachStation_offset",
                                "cancelled",
                                "weather_observations"  # Include weather observations for EMS display
                            ]

                            # Make sure we only include columns that exist
                            reordered_columns = [col for col in preferred_columns if col in timetable_df.columns]
                            # Add any remaining columns
                            reordered_columns += [col for col in timetable_df.columns if col not in reordered_columns]
                            
                            timetable_df = timetable_df[reordered_columns]

                            st.subheader(f"Time Table for Train **{selected_train_number}** on **{selected_day}**")
                            st.dataframe(timetable_df)

                            # ADD MAP VISUALIZATION HERE - Show train route on map
                            st.subheader(f"Route Map for Train **{selected_train_number}**")
                            
                            # Create and display the map
                            if not timetable_df.empty:
                                train_route_map = create_train_route_map(timetable_df)
                                if train_route_map:
                                    st_folium(train_route_map, width=None, height=600, returned_objects=[])
                                else:
                                    st.error("Could not create map due to missing station coordinate data.")

                            if not timetable_df.empty:
                                st.subheader("Select a Train Track ")
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
            st.warning(f"⚠️ File `{file_name}` not found in `{VIEWER_FOLDER_NAME}`.")