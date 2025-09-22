import streamlit as st
import pandas as pd
import os
import glob
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from streamlit_folium import st_folium
from config.const import FMI_BBOX, VIEWER_FOLDER_NAME

# Page configuration
st.set_page_config(
    page_title="Station Delay Statistics",
    page_icon="📊",
    layout="wide"
)

def load_routes_data():
    """Load routes data from metadata_routes.csv if it exists"""
    routes_file = os.path.join(VIEWER_FOLDER_NAME, "metadata", "metadata_routes.csv")
    
    if not os.path.exists(routes_file):
        return None
    
    try:
        df_routes = pd.read_csv(routes_file)
        return df_routes
    except Exception as e:
        st.error(f"❌ Error loading routes file: {e}")
        return None

def parse_route_string(route_string):
    """Parse a route string back into a list of station names"""
    try:
        # Remove quotes and convert string representation of list back to actual list
        route_list = literal_eval(route_string.strip())
        return route_list if isinstance(route_list, list) else []
    except Exception as e:
        st.warning(f"⚠️ Error parsing route: {e}")
        return []

def get_station_coordinates_for_route(station_names):
    """Get coordinates and delay statistics for stations in a route"""
    metadata_path = os.path.join(VIEWER_FOLDER_NAME, "metadata", "metadata_train_stations.csv")
    delay_stats_path = "data/viewers/delay_maps/stations_x_delays.csv"
    
    if not os.path.exists(metadata_path):
        return pd.DataFrame()
    
    try:
        df_stations = pd.read_csv(metadata_path)
        
        # Load delay statistics if available
        df_delays = None
        if os.path.exists(delay_stats_path):
            df_delays = pd.read_csv(delay_stats_path)
            # Calculate delay percentage
            df_delays['delay_percentage'] = 0.0
            mask = df_delays['total_of_trains'] > 0
            df_delays.loc[mask, 'delay_percentage'] = (
                df_delays.loc[mask, 'total_of_delays'] / df_delays.loc[mask, 'total_of_trains'] * 100
            )
        
        # Filter for stations in the route and preserve order
        route_coords = []
        for station_name in station_names:
            station_match = df_stations[df_stations['stationName'] == station_name]
            if not station_match.empty:
                station_data = station_match.iloc[0]
                if pd.notna(station_data['latitude']) and pd.notna(station_data['longitude']):
                    route_station = {
                        'stationName': station_name,
                        'stationShortCode': station_data.get('stationShortCode', ''),
                        'latitude': station_data['latitude'],
                        'longitude': station_data['longitude'],
                        'route_order': len(route_coords),  # Preserve order in route
                        'total_of_delays': 0,
                        'total_of_trains': 0,
                        'delay_percentage': 0.0
                    }
                    
                    # Add delay statistics if available
                    if df_delays is not None:
                        delay_match = df_delays[df_delays['stationShortCode'] == station_data.get('stationShortCode', '')]
                        if not delay_match.empty:
                            delay_data = delay_match.iloc[0]
                            route_station.update({
                                'total_of_delays': delay_data.get('total_of_delays', 0),
                                'total_of_trains': delay_data.get('total_of_trains', 0),
                                'delay_percentage': delay_data.get('delay_percentage', 0.0)
                            })
                    
                    route_coords.append(route_station)
        
        return pd.DataFrame(route_coords)
        
    except Exception as e:
        st.error(f"❌ Error loading station coordinates: {e}")
        return pd.DataFrame()

def create_route_map(route_coords, route_name):
    """Create an interactive map showing the selected train route with delay rate classification"""
    if route_coords.empty:
        st.warning("⚠️ No coordinates found for stations in this route.")
        return None
    
    # Calculate map center
    center_lat = route_coords['latitude'].mean()
    center_lon = route_coords['longitude'].mean()
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap"
    )
    
    # Create feature groups based on delay rate classification
    route_line_group = folium.FeatureGroup(name="Route Line", show=True)
    high_delay_group = folium.FeatureGroup(name="High Priority Stations (≥15%)", show=True)
    medium_delay_group = folium.FeatureGroup(name="Medium Priority Stations (5-15%)", show=True)
    low_delay_group = folium.FeatureGroup(name="Low Priority Stations (<5%)", show=True)
    low_traffic_group = folium.FeatureGroup(name="Low Traffic Stations (<100 trains)", show=True)
    
    # Add route line connecting all stations
    route_coordinates = route_coords[['latitude', 'longitude']].values.tolist()
    
    if len(route_coordinates) > 1:
        folium.PolyLine(
            locations=route_coordinates,
            color="blue",
            weight=4,
            opacity=0.8,
            tooltip=f"Route: {route_name}"
        ).add_to(route_line_group)
    
    # Add markers for each station based on delay rate classification
    for idx, row in route_coords.iterrows():
        order = row['route_order'] + 1  # Human-readable order (1-based)
        delay_pct = row['delay_percentage']
        total_trains = row['total_of_trains']
        
        # Determine marker style based on delay rate classification
        if total_trains < 100:
            color = 'blue'
            group = low_traffic_group
            priority = "🔵 Low Traffic"
            delay_status = f"Low traffic ({total_trains} trains)"
        elif delay_pct >= 15:
            color = 'red'
            group = high_delay_group
            priority = "🔴 High Priority"
            delay_status = f"High delay rate ({delay_pct:.1f}%)"
        elif delay_pct >= 5:
            color = 'orange'
            group = medium_delay_group
            priority = "🟠 Medium Priority"
            delay_status = f"Medium delay rate ({delay_pct:.1f}%)"
        else:
            color = 'green'
            group = low_delay_group
            priority = "🟢 Low Priority"
            delay_status = f"Low delay rate ({delay_pct:.1f}%)"
        
        # Determine if this is start/end station for additional context
        route_position = ""
        if order == 1:
            route_position = " (Route Start)"
        elif order == len(route_coords):
            route_position = " (Route End)"
        
        # Create popup with station and delay information
        popup_text = f"""
        <div style="min-width: 250px">
            <h3>🚂 {row['stationName']}</h3>
            <hr>
            <b>Route Position:</b> Stop {order} of {len(route_coords)}{route_position}<br>
            <b>Station Code:</b> {row['stationShortCode']}<br>
            <b>Category:</b> {priority}<br>
            <b>Delay Rate:</b> {delay_pct:.1f}%<br>
            <b>Total Delays:</b> {row['total_of_delays']:,}<br>
            <b>Total Trains:</b> {row['total_of_trains']:,}<br>
            <b>Coordinates:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>
            <hr>
            <small>{delay_status}</small>
        </div>
        """
        
        # Create tooltip
        tooltip_text = f"{row['stationName']} ({order}/{len(route_coords)}) - {delay_status}"
        
        # Add marker to appropriate delay category group
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=tooltip_text,
            icon=folium.Icon(color=color, icon="train", prefix="fa")
        ).add_to(group)
    
    # Add direction arrows along the route
    if len(route_coordinates) > 1:
        # Add arrows at regular intervals to show direction
        for i in range(len(route_coordinates) - 1):
            start_point = route_coordinates[i]
            end_point = route_coordinates[i + 1]
            
            # Calculate midpoint for arrow placement
            mid_lat = (start_point[0] + end_point[0]) / 2
            mid_lon = (start_point[1] + end_point[1]) / 2
            
            # Add small arrow marker to show direction
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 16px; color: red;">→</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                ),
                tooltip=f"Direction: Station {i+1} → Station {i+2}"
            ).add_to(route_line_group)
    
    # Add all feature groups to map
    route_line_group.add_to(m)
    high_delay_group.add_to(m)
    medium_delay_group.add_to(m)
    low_delay_group.add_to(m)
    low_traffic_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_route_map_plot(df_routes):
    """Create route map visualization with route selection"""
    st.subheader("🛤️ Train Route Map")
    
    if df_routes is None or df_routes.empty:
        st.warning("⚠️ No routes data available. Please ensure the `metadata_routes.csv` file exists.")
        return
    
    # Create route options for selectbox
    route_options = []
    route_summaries = []
    
    for idx, row in df_routes.iterrows():
        route_stations = parse_route_string(row['route'])
        if len(route_stations) >= 2:
            # Create a readable route summary
            start_station = route_stations[0]
            end_station = route_stations[-1]
            num_stations = len(route_stations)
            
            # Create a route summary for display
            if num_stations <= 5:
                # Show all stations for short routes
                route_summary = f"{' → '.join(route_stations)}"
            else:
                # Show start, end, and total count for long routes
                route_summary = f"{start_station} → ... → {end_station} ({num_stations} stations)"
            
            route_options.append(route_summary)
            route_summaries.append({
                'summary': route_summary,
                'route': row['route'],
                'start': start_station,
                'end': end_station,
                'stations': route_stations,
                'count': num_stations
            })
    
    if not route_options:
        st.warning("⚠️ No valid routes found in the dataset.")
        return
    
    # Sort routes by number of stations (longest first)
    route_summaries.sort(key=lambda x: x['count'], reverse=True)
    route_options = [r['summary'] for r in route_summaries]
    
    # Route selection
    st.markdown("### 🚂 Select a Train Route")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_route_summary = st.selectbox(
            "Choose a route to visualize:",
            options=route_options,
            help="Routes are sorted by length (number of stations), longest first"
        )
    
    with col2:
        st.metric(
            label="📊 Total Routes Available",
            value=f"{len(route_options)}"
        )
    
    if selected_route_summary:
        # Find the selected route data
        selected_route_data = next(r for r in route_summaries if r['summary'] == selected_route_summary)
        
        # Display route information
        st.markdown("### 📋 Route Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 Start Station", selected_route_data['start'])
        
        with col2:
            st.metric("🔴 End Station", selected_route_data['end'])
        
        with col3:
            st.metric("🚂 Total Stations", f"{selected_route_data['count']}")
        
        with col4:
            # Calculate approximate route length (this is rough estimation)
            route_coords = get_station_coordinates_for_route(selected_route_data['stations'])
            if not route_coords.empty and len(route_coords) > 1:
                # Simple distance calculation (Euclidean, not actual rail distance)
                total_distance = 0
                for i in range(len(route_coords) - 1):
                    lat1, lon1 = route_coords.iloc[i][['latitude', 'longitude']]
                    lat2, lon2 = route_coords.iloc[i + 1][['latitude', 'longitude']]
                    # Rough distance calculation (1 degree ≈ 111 km)
                    distance = ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5 * 111
                    total_distance += distance
                st.metric("📏 Approx. Distance", f"{total_distance:.0f} km")
            else:
                st.metric("📏 Distance", "N/A")
        
        # Add information about delay data classification
        delay_stats_available = os.path.exists("data/viewers/delay_maps/stations_x_delays.csv")
        if delay_stats_available:
            st.info("ℹ️ **Route stations are color-coded by delay rate** using the same classification as the Interactive Station Map. Data from previously calculated station delay statistics.")
        else:
            st.warning("⚠️ **Delay statistics not available** - Route stations will be shown without delay rate classification. Run 'Calculate Station Delays' first to enable delay-based coloring.")
        
        # Get coordinates for the route
        route_coords = get_station_coordinates_for_route(selected_route_data['stations'])
        
        if not route_coords.empty:
            # Calculate statistics about stations with/without coordinates
            total_stations_in_route = len(selected_route_data['stations'])
            stations_with_coords = len(route_coords)
            missing_coords = total_stations_in_route - stations_with_coords
            
            # Display route stations list
            with st.expander(f"🔍 View All {len(selected_route_data['stations'])} Stations in Route", expanded=False):
                # Display stations in a nicely formatted way
                stations_per_row = 4
                station_rows = [selected_route_data['stations'][i:i + stations_per_row] 
                              for i in range(0, len(selected_route_data['stations']), stations_per_row)]
                
                for i, row_stations in enumerate(station_rows):
                    cols = st.columns(stations_per_row)
                    for j, station in enumerate(row_stations):
                        with cols[j]:
                            order_num = i * stations_per_row + j + 1
                            if order_num == 1:
                                st.markdown(f"**{order_num}.** 🟢 {station}")
                            elif order_num == len(selected_route_data['stations']):
                                st.markdown(f"**{order_num}.** 🔴 {station}")
                            else:
                                st.markdown(f"**{order_num}.** {station}")
            
            # Create and display the map
            st.markdown("### 🗺️ Route Map")
            
            route_map = create_route_map(route_coords, selected_route_summary)
            
            if route_map:
                # Display the map
                st_folium(route_map, width=None, height=800, returned_objects=[])
                
                # Add map legend and information (same as Interactive Station Map)
                st.markdown("### 🎨 Interactive Map Legend")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Delay Rate Categories:**")
                    st.markdown("🔴 **High Priority**: ≥ 15% delay rate")
                    st.markdown("🟠 **Medium Priority**: 5-15% delay rate") 
                    st.markdown("🟢 **Low Priority**: < 5% delay rate")
                    st.markdown("🔵 **Low Traffic**: < 100 trains (any delay rate)")
                
                with col2:
                    # Calculate category statistics for this route
                    if 'delay_percentage' in route_coords.columns and 'total_of_trains' in route_coords.columns:
                        # Check if we actually have delay data
                        has_delay_data = route_coords['total_of_trains'].sum() > 0
                        
                        if has_delay_data:
                            low_traffic_count = (route_coords['total_of_trains'] < 100).sum()
                            high_traffic = route_coords[route_coords['total_of_trains'] >= 100]
                            high_delay_count = (high_traffic['delay_percentage'] >= 15).sum()
                            medium_delay_count = ((high_traffic['delay_percentage'] >= 5) & (high_traffic['delay_percentage'] < 15)).sum()
                            low_delay_count = (high_traffic['delay_percentage'] < 5).sum()
                            
                            st.markdown("**Station Count by Category:**")
                            st.markdown(f"🔴 High Priority: {high_delay_count} stations")
                            st.markdown(f"🟠 Medium Priority: {medium_delay_count} stations")
                            st.markdown(f"🟢 Low Priority: {low_delay_count} stations")
                            st.markdown(f"🔵 Low Traffic: {low_traffic_count} stations")
                        else:
                            st.markdown("**Route Information:**")
                            st.markdown(f"📍 **Stations Mapped**: {stations_with_coords}/{total_stations_in_route}")
                            st.markdown(f"🔵 **Blue Line**: Complete route path")
                            st.markdown(f"➡️ **Direction Arrows**: Show travel direction")
                            st.markdown(f"⚠️ **Delay data**: Not available")
                    else:
                        st.markdown("**Route Information:**")
                        st.markdown(f"📍 **Stations Mapped**: {stations_with_coords}/{total_stations_in_route}")
                        if missing_coords > 0:
                            st.markdown(f"⚠️ **Missing Coordinates**: {missing_coords} stations")
                        st.markdown(f"🔵 **Blue Line**: Complete route path")
                        st.markdown(f"➡️ **Direction Arrows**: Show travel direction")
                        st.markdown(f"⚠️ **Delay data**: Not available for classification")
                
                # Route insights section
                if 'delay_percentage' in route_coords.columns and 'total_of_trains' in route_coords.columns:
                    st.markdown("### 🔍 Route Delay Insights")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Filter out stations with no train data for insights
                    stations_with_data = route_coords[route_coords['total_of_trains'] > 0]
                    
                    if not stations_with_data.empty:
                        with col1:
                            worst_station = stations_with_data.loc[stations_with_data['delay_percentage'].idxmax()]
                            st.metric(
                                label="🔴 Highest Delay Rate on Route",
                                value=f"{worst_station['stationName']}",
                                delta=f"{worst_station['delay_percentage']:.1f}%"
                            )
                        
                        with col2:
                            most_delays_station = stations_with_data.loc[stations_with_data['total_of_delays'].idxmax()]
                            st.metric(
                                label="📊 Most Total Delays on Route",
                                value=f"{most_delays_station['stationName']}",
                                delta=f"{most_delays_station['total_of_delays']:,} delays"
                            )
                        
                        with col3:
                            busiest_station = stations_with_data.loc[stations_with_data['total_of_trains'].idxmax()]
                            st.metric(
                                label="🚂 Busiest Station on Route",
                                value=f"{busiest_station['stationName']}",
                                delta=f"{busiest_station['total_of_trains']:,} trains"
                            )
                        
                        # Route delay summary
                        avg_delay_rate = stations_with_data['delay_percentage'].mean()
                        total_route_delays = stations_with_data['total_of_delays'].sum()
                        total_route_trains = stations_with_data['total_of_trains'].sum()
                        
                        st.info(f"📊 **Route Summary**: Average delay rate: {avg_delay_rate:.1f}% | Total delays: {total_route_delays:,} | Total trains: {total_route_trains:,}")
                    else:
                        st.warning("⚠️ No delay statistics available for stations on this route.")
                
                # Show any missing stations
                if missing_coords > 0:
                    missing_stations = [station for station in selected_route_data['stations'] 
                                      if station not in route_coords['stationName'].values]
                    
                    with st.expander(f"⚠️ Stations Missing Coordinates ({missing_coords})", expanded=False):
                        for station in missing_stations:
                            st.markdown(f"• {station}")
                        st.info("These stations are not shown on the map due to missing coordinate data.")
                
                # Check if delay data is available for classification
                has_delay_data = 'delay_percentage' in route_coords.columns and route_coords['total_of_trains'].sum() > 0
                
                if has_delay_data:
                    st.success(f"🗺️ Successfully mapped {stations_with_coords} stations with delay rate classification.")
                else:
                    st.info(f"🗺️ Successfully mapped {stations_with_coords} stations. Delay statistics not available - stations shown with route position only.")
                
            else:
                st.error("❌ Could not create route map. Please check station coordinate data.")
        else:
            st.error("❌ No coordinate data found for stations in this route.")

def load_station_metadata():
    """Load train station metadata and prepare the base dataframe"""
    metadata_path = "data/viewers/metadata/metadata_train_stations.csv"
    
    if not os.path.exists(metadata_path):
        st.error(f"⚠️ Station metadata file not found at: {metadata_path}")
        return None
    
    try:
        # Load metadata
        df_metadata = pd.read_csv(metadata_path)
        
        # Create new dataframe with required columns
        df_stations = pd.DataFrame()
        df_stations['stationShortCode'] = df_metadata['stationShortCode']
        df_stations['stationName'] = df_metadata['stationName'] if 'stationName' in df_metadata.columns else ''
        df_stations['total_of_delays'] = 0
        df_stations['total_of_trains'] = 0
        
        # Add coordinates if available for mapping later
        if 'latitude' in df_metadata.columns and 'longitude' in df_metadata.columns:
            df_stations['latitude'] = df_metadata['latitude']
            df_stations['longitude'] = df_metadata['longitude']
        
        return df_stations
        
    except Exception as e:
        st.error(f"❌ Error loading station metadata: {e}")
        return None
    
def extract_and_save_routes(matched_data_path):
    """Extract all unique train routes from matched data files and save to routes.csv"""
    
    if not os.path.exists(matched_data_path):
        st.error(f"⚠️ Matched data directory not found at: {matched_data_path}")
        return False
    
    # Get all CSV files in the matched_data directory
    pattern = os.path.join(matched_data_path, "*.csv")
    matched_files = glob.glob(pattern)
    
    if not matched_files:
        st.warning("⚠️ No matched data files found in the directory.")
        return False
    
    # Set to store unique routes (using tuple for hashability)
    unique_routes = set()
    
    # Create a progress bar for route extraction
    route_progress_bar = st.progress(0)
    route_status_text = st.empty()
    
    total_files = len(matched_files)
    processed_routes = 0
    
    for file_idx, file_path in enumerate(matched_files):
        file_name = os.path.basename(file_path)
        route_status_text.text(f"Extracting routes from file {file_idx + 1}/{total_files}: {file_name}")
        
        try:
            # Load the matched data file
            df_matched = pd.read_csv(file_path)
            
            # Process each train (row) in the file
            for idx, row in df_matched.iterrows():
                if 'timeTableRows' in row and pd.notna(row['timeTableRows']):
                    try:
                        # Parse the timeTableRows
                        timetable_str = str(row['timeTableRows']).replace('nan', 'None')
                        timetable_data = literal_eval(timetable_str)
                        
                        # Extract station sequence for this train
                        stations_with_time = []
                        
                        for station_entry in timetable_data:
                            if isinstance(station_entry, dict):
                                station_code = station_entry.get('stationShortCode')
                                station_name = station_entry.get('stationName')
                                scheduled_time = station_entry.get('scheduledTime')
                                
                                if station_code and station_name and scheduled_time:
                                    stations_with_time.append({
                                        'stationName': station_name,
                                        'scheduledTime': scheduled_time
                                    })
                        
                        # Sort by scheduled time to get correct route order
                        if stations_with_time:
                            try:
                                # Sort by scheduled time
                                stations_with_time.sort(key=lambda x: pd.to_datetime(x['scheduledTime']))
                            except:
                                # If parsing fails, keep original order
                                pass
                            
                            # Extract the route as a sequence of station names
                            # Remove duplicates while preserving order
                            route_stations = []
                            seen = set()
                            for station in stations_with_time:
                                station_name = station['stationName']
                                if station_name not in seen:
                                    route_stations.append(station_name)
                                    seen.add(station_name)
                            
                            # Only add routes with at least 2 stations
                            if len(route_stations) >= 2:
                                route_tuple = tuple(route_stations)
                                unique_routes.add(route_tuple)
                                processed_routes += 1
                        
                    except Exception as e:
                        # Skip trains with parsing errors
                        continue
                        
        except Exception as e:
            st.warning(f"⚠️ Error processing file {file_name} for routes: {e}")
            continue
        
        # Update progress
        route_progress_bar.progress((file_idx + 1) / total_files)
    
    # Clear progress indicators
    route_progress_bar.empty()
    route_status_text.empty()
    
    # Convert routes to DataFrame and save
    if unique_routes:
        # Convert set of tuples to list of lists for CSV saving
        routes_list = [list(route) for route in unique_routes]
        
        # Create DataFrame with route as a string representation of list
        df_routes = pd.DataFrame({
            'route': [str(route) for route in routes_list]
        })
        
        # Create directory if it doesn't exist
        output_dir = os.path.join(VIEWER_FOLDER_NAME, "metadata")  # This resolves to "data/viewers/metadata"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save routes to CSV - USE CORRECT FILENAME
        routes_file = os.path.join(output_dir, "metadata_routes.csv")
        
        try:
            df_routes.to_csv(routes_file, index=False)
            st.success(f"✅ Extracted and saved {len(unique_routes):,} unique routes to: `{routes_file}`")
            st.info(f"📈 Processed {processed_routes:,} individual train journeys to find unique routes")
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving routes: {e}")
            st.error(f"🔍 Debug: Attempted to save to: `{routes_file}`")
            return False
    else:
        st.warning("⚠️ No valid routes found in the dataset.")
        return False

def process_matched_data_files(df_stations):
    """Process all matched data files and count delays/trains per station AND extract routes in a single pass"""
    matched_data_path = "data/viewers/matched_data"
    
    if not os.path.exists(matched_data_path):
        st.error(f"⚠️ Matched data directory not found at: {matched_data_path}")
        return df_stations
    
    # Get all CSV files in the matched_data directory
    pattern = os.path.join(matched_data_path, "*.csv")
    matched_files = glob.glob(pattern)
    
    if not matched_files:
        st.warning("⚠️ No matched data files found in the directory.")
        return df_stations
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(matched_files)
    processed_trains = 0
    
    # Create a dictionary for faster lookups (station delays/trains)
    station_stats = {code: {'delays': 0, 'trains': 0} for code in df_stations['stationShortCode']}
    
    # Set to store unique routes (using tuple for hashability)
    unique_routes = set()
    processed_routes = 0
    
    for file_idx, file_path in enumerate(matched_files):
        file_name = os.path.basename(file_path)
        status_text.text(f"Processing file {file_idx + 1}/{total_files}: {file_name} (delays + routes)")
        
        try:
            # Load the matched data file
            df_matched = pd.read_csv(file_path)
            
            # Process each train (row) in the file
            for idx, row in df_matched.iterrows():
                if 'timeTableRows' in row and pd.notna(row['timeTableRows']):
                    try:
                        # Parse the timeTableRows
                        timetable_str = str(row['timeTableRows']).replace('nan', 'None')
                        timetable_data = literal_eval(timetable_str)
                        
                        # Track which stations this train visited (to avoid double counting delays)
                        visited_stations = set()
                        
                        # Extract station sequence for route (with scheduled times)
                        stations_with_time = []
                        
                        # Process each station in the timetable (for both delays and routes)
                        for station_entry in timetable_data:
                            if isinstance(station_entry, dict):
                                station_code = station_entry.get('stationShortCode')
                                station_name = station_entry.get('stationName')
                                scheduled_time = station_entry.get('scheduledTime')
                                
                                # === DELAY COUNTING LOGIC ===
                                if station_code and station_code in station_stats:
                                    # Count this station only once per train
                                    if station_code not in visited_stations:
                                        station_stats[station_code]['trains'] += 1
                                        visited_stations.add(station_code)
                                    
                                    # Check for delays (>= 5 minutes)
                                    delay_offset = station_entry.get('differenceInMinutes_eachStation_offset')
                                    if delay_offset is not None and delay_offset >= 5:
                                        station_stats[station_code]['delays'] += 1
                                
                                # === ROUTE EXTRACTION LOGIC ===
                                if station_code and station_name and scheduled_time:
                                    stations_with_time.append({
                                        'stationName': station_name,
                                        'scheduledTime': scheduled_time
                                    })
                        
                        # === PROCESS ROUTE FOR THIS TRAIN ===
                        if stations_with_time:
                            try:
                                # Sort by scheduled time to get correct route order
                                stations_with_time.sort(key=lambda x: pd.to_datetime(x['scheduledTime']))
                            except:
                                # If parsing fails, keep original order
                                pass
                            
                            # Extract the route as a sequence of station names
                            # Remove duplicates while preserving order
                            route_stations = []
                            seen = set()
                            for station in stations_with_time:
                                station_name = station['stationName']
                                if station_name not in seen:
                                    route_stations.append(station_name)
                                    seen.add(station_name)
                            
                            # Only add routes with at least 2 stations
                            if len(route_stations) >= 2:
                                route_tuple = tuple(route_stations)
                                unique_routes.add(route_tuple)
                                processed_routes += 1
                        
                        processed_trains += 1
                        
                    except Exception as e:
                        # Skip trains with parsing errors
                        continue
                        
        except Exception as e:
            st.warning(f"⚠️ Error processing file {file_name}: {e}")
            continue
        
        # Update progress
        progress_bar.progress((file_idx + 1) / total_files)
    
    # Update the dataframe with the collected station statistics
    for station_code, stats in station_stats.items():
        mask = df_stations['stationShortCode'] == station_code
        df_stations.loc[mask, 'total_of_delays'] = stats['delays']
        df_stations.loc[mask, 'total_of_trains'] = stats['trains']
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show completion message for station processing
    st.success(f"✅ Processed {total_files} files containing {processed_trains:,} train journeys")
    
    # === SAVE ROUTES ===
    if unique_routes:
        # Convert set of tuples to list of lists for CSV saving
        routes_list = [list(route) for route in unique_routes]
        
        # Create DataFrame with route as a string representation of list
        df_routes = pd.DataFrame({
            'route': [str(route) for route in routes_list]
        })
        
        # Use the correct directory path (VIEWER_FOLDER_NAME/metadata)
        output_dir = os.path.join(VIEWER_FOLDER_NAME, "metadata")  # This resolves to "data/viewers/metadata"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the correct filename with metadata_ prefix
        routes_file = os.path.join(output_dir, "metadata_routes.csv")
        
        try:
            df_routes.to_csv(routes_file, index=False)
            st.success(f"✅ Extracted and saved {len(unique_routes):,} unique routes to: `{routes_file}`")
            st.info(f"📈 Processed {processed_routes:,} individual train journeys to find unique routes")
            
            # Debug info to help troubleshoot
            st.info(f"🔍 Debug: Directory `{output_dir}` exists: {os.path.exists(output_dir)}")
            st.info(f"🔍 Debug: File saved at: `{os.path.abspath(routes_file)}`")
            
        except Exception as e:
            st.error(f"❌ Error saving routes: {e}")
            st.error(f"🔍 Debug: Attempted to save to: `{routes_file}`")
            st.error(f"🔍 Debug: Output directory: `{output_dir}`")
            
    else:
        st.warning("⚠️ No valid routes found in the dataset.")
    
    return df_stations

def calculate_delay_statistics(df_stations):
    """Calculate additional statistics from the delay data"""
    # Calculate delay percentage
    df_stations['delay_percentage'] = 0.0
    mask = df_stations['total_of_trains'] > 0
    df_stations.loc[mask, 'delay_percentage'] = (
        df_stations.loc[mask, 'total_of_delays'] / df_stations.loc[mask, 'total_of_trains'] * 100
    )
    
    # Sort by total delays (descending)
    df_stations = df_stations.sort_values('total_of_delays', ascending=False)
    
    return df_stations

def save_results(df_stations):
    """Save the results to CSV file"""
    output_dir = "data/viewers/delay_maps"
    output_file = os.path.join(output_dir, "stations_x_delays.csv")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Select only the required columns for saving
        df_save = df_stations[['stationShortCode', 'total_of_delays', 'total_of_trains']].copy()
        df_save.to_csv(output_file, index=False)
        
        st.success(f"✅ Results saved to: `{output_file}`")
        return True
        
    except Exception as e:
        st.error(f"❌ Error saving results: {e}")
        return False

def create_single_visualization(df_stations, plot_key):
    """Create a single visualization based on the selected plot key"""
    # For route map, we don't need station delay data
    if plot_key == "route_map":
        df_routes = load_routes_data()
        create_route_map_plot(df_routes)
        return
    
    # Filter to stations with at least some traffic for other visualizations
    df_viz = df_stations[df_stations['total_of_trains'] > 0].copy()
    
    if df_viz.empty:
        st.warning("No stations with train traffic found.")
        return
    
    if plot_key == "top_stations":
        create_top_stations_plot(df_viz)
    elif plot_key == "station_map":
        create_station_map_plot(df_viz)

def create_top_stations_plot(df_viz):
    """Create top stations by normalized delay rate visualization"""
    st.subheader("Top 20 Stations by Delay Rate (Normalized)")
    
    # Sort by delay percentage instead of total delays
    top_20 = df_viz.nlargest(20, 'delay_percentage')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_20)), top_20['delay_percentage'])
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"{row['stationName']} ({row['stationShortCode']})" 
                       if row['stationName'] else row['stationShortCode']
                       for _, row in top_20.iterrows()])
    ax.set_xlabel('Delay Rate (% of trains delayed ≥5 minutes)')
    ax.set_title('Top 20 Stations by Delay Rate (Normalized)')
    
    # Color bars based on delay percentage
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top_20)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels showing percentage and context
    for i, (_, row) in enumerate(top_20.iterrows()):
        # Show percentage with context about total trains
        label = f"{row['delay_percentage']:.1f}%"
        if row['total_of_trains'] >= 100:  # Show train count for stations with significant traffic
            label += f"\n({row['total_of_delays']:,}/{row['total_of_trains']:,})"
        ax.text(row['delay_percentage'] + 0.5, i, label, 
               va='center', fontweight='bold', fontsize=9)
    
    # Add a note about data interpretation
    ax.text(0.02, 0.98, 
           f"Note: Shows delay rate as percentage of total trains\nBased on {len(df_viz)} stations with train traffic", 
           transform=ax.transAxes, 
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
           fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add summary statistics below the chart
    st.markdown("### 📊 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        highest_rate_station = top_20.iloc[0]
        st.metric(
            label="🔴 Highest Delay Rate", 
            value=f"{highest_rate_station['stationName']} ({highest_rate_station['stationShortCode']})",
            delta=f"{highest_rate_station['delay_percentage']:.1f}%"
        )
    
    with col2:
        # Find stations with significant traffic (>= 1000 trains) and highest delay rate
        significant_traffic = top_20[top_20['total_of_trains'] >= 1000]
        if not significant_traffic.empty:
            high_traffic_station = significant_traffic.iloc[0]
            st.metric(
                label="🚂 High Traffic + High Delays",
                value=f"{high_traffic_station['stationName']} ({high_traffic_station['stationShortCode']})",
                delta=f"{high_traffic_station['delay_percentage']:.1f}% ({high_traffic_station['total_of_trains']:,} trains)"
            )
        else:
            st.metric(
                label="🚂 High Traffic Threshold",
                value="No stations found",
                delta="with ≥1000 trains"
            )
    
    with col3:
        average_rate = top_20['delay_percentage'].mean()
        st.metric(
            label="📈 Average Rate (Top 20)",
            value=f"{average_rate:.1f}%",
            delta=f"Range: {top_20['delay_percentage'].min():.1f}% - {top_20['delay_percentage'].max():.1f}%"
        )

def create_station_map_plot(df_viz):
    """Create interactive station map using streamlit_folium"""
    st.subheader("Station Delay Map")
    
    # Check if we have coordinate data
    if 'latitude' in df_viz.columns and 'longitude' in df_viz.columns:
        # Filter out stations without coordinates
        df_map = df_viz.dropna(subset=['latitude', 'longitude'])
        
        if not df_map.empty:
            # Parse Finland bounding box from const.py
            # FMI_BBOX = "18,55,35,75" -> [west, south, east, north]
            bbox_parts = [float(x) for x in FMI_BBOX.split(',')]
            west, south, east, north = bbox_parts
            
            # Calculate center of Finland
            center_lat = (south + north) / 2
            center_lon = (west + east) / 2
            
            # Create folium map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=5,
                tiles="OpenStreetMap"
            )
            
            # Create different feature groups for better organization
            high_delay_group = folium.FeatureGroup(name="High Delay Stations (≥15%)", show=True)
            medium_delay_group = folium.FeatureGroup(name="Medium Delay Stations (5-15%)", show=True)
            low_delay_group = folium.FeatureGroup(name="Low Delay Stations (<5%)", show=True)
            low_traffic_group = folium.FeatureGroup(name="Low Traffic Stations (<100 trains)", show=True)
            
            # Add markers for each station
            for _, row in df_map.iterrows():
                # Determine marker color and group based on train traffic first, then delay percentage
                delay_pct = row['delay_percentage']
                total_trains = row['total_of_trains']
                
                if total_trains < 100:
                    color = 'blue'
                    group = low_traffic_group
                    priority = "🔵 Low Traffic"
                elif delay_pct >= 15:
                    color = 'red'
                    group = high_delay_group
                    priority = "🔴 High Priority"
                elif delay_pct >= 5:
                    color = 'orange'
                    group = medium_delay_group
                    priority = "🟠 Medium Priority"
                else:
                    color = 'green'
                    group = low_delay_group
                    priority = "🟢 Low Priority"
                
                # Create popup text with comprehensive information
                popup_text = f"""
                <div style="min-width: 250px">
                    <h3>🚂 {row['stationName']}</h3>
                    <hr>
                    <b>Station Code:</b> {row['stationShortCode']}<br>
                    <b>Category:</b> {priority}<br>
                    <b>Delay Rate:</b> {delay_pct:.1f}%<br>
                    <b>Total Delays:</b> {row['total_of_delays']:,}<br>
                    <b>Total Trains:</b> {row['total_of_trains']:,}<br>
                    <b>Location:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>
                    <hr>
                    <small>{'Low traffic station - delay rate may not be statistically significant' if total_trains < 100 else 'Delays ≥5 minutes are counted'}</small>
                </div>
                """
                
                # Create tooltip
                if total_trains < 100:
                    tooltip_text = f"{row['stationName']} ({row['stationShortCode']}) - Low traffic ({total_trains} trains)"
                else:
                    tooltip_text = f"{row['stationName']} ({row['stationShortCode']}) - {delay_pct:.1f}% delays"
                
                # Add marker to appropriate group
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_text, max_width=350),
                    tooltip=tooltip_text,
                    icon=folium.Icon(color=color, icon="train", prefix="fa")
                ).add_to(group)
            
            # Add all feature groups to map
            high_delay_group.add_to(m)
            medium_delay_group.add_to(m)
            low_delay_group.add_to(m)
            low_traffic_group.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Display the map
            st_folium(m, width=None, height=1200, returned_objects=[])
            
            # Add legend and statistics
            st.markdown("### 🎨 Interactive Map Legend")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Delay Rate Categories:**")
                st.markdown("🔴 **High Priority**: ≥ 15% delay rate")
                st.markdown("🟠 **Medium Priority**: 5-15% delay rate") 
                st.markdown("🟢 **Low Priority**: < 5% delay rate")
                st.markdown("🔵 **Low Traffic**: < 100 trains (any delay rate)")
            
            with col2:
                # Calculate category statistics
                low_traffic_count = (df_map['total_of_trains'] < 100).sum()
                high_traffic = df_map[df_map['total_of_trains'] >= 100]
                high_delay_count = (high_traffic['delay_percentage'] >= 15).sum()
                medium_delay_count = ((high_traffic['delay_percentage'] >= 5) & (high_traffic['delay_percentage'] < 15)).sum()
                low_delay_count = (high_traffic['delay_percentage'] < 5).sum()
                
                st.markdown("**Station Count by Category:**")
                st.markdown(f"🔴 High Priority: {high_delay_count} stations")
                st.markdown(f"🟠 Medium Priority: {medium_delay_count} stations")
                st.markdown(f"🟢 Low Priority: {low_delay_count} stations")
                st.markdown(f"🔵 Low Traffic: {low_traffic_count} stations")
            
            # Additional insights
            st.markdown("### 🔍 Map Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                worst_station = df_map.loc[df_map['delay_percentage'].idxmax()]
                st.metric(
                    label="🔴 Worst Delay Rate",
                    value=f"{worst_station['stationName']}",
                    delta=f"{worst_station['delay_percentage']:.1f}%"
                )
            
            with col2:
                most_delays_station = df_map.loc[df_map['total_of_delays'].idxmax()]
                st.metric(
                    label="📊 Most Total Delays",
                    value=f"{most_delays_station['stationName']}",
                    delta=f"{most_delays_station['total_of_delays']:,} delays"
                )
            
            with col3:
                busiest_station = df_map.loc[df_map['total_of_trains'].idxmax()]
                st.metric(
                    label="🚂 Busiest Station",
                    value=f"{busiest_station['stationName']}",
                    delta=f"{busiest_station['total_of_trains']:,} trains"
                )
            
            st.info(f"🗺️ Interactive map showing {len(df_map)} stations across Finland. Click markers for detailed information, use layer control to filter by category. **Note**: Low traffic stations (<100 trains) are marked in blue regardless of delay rate.")
            
        else:
            st.warning("No stations with valid coordinates found.")
    else:
        st.info("📍 Coordinate data not available for map visualization.")

def create_visualizations(df_stations):
    """Create visualizations for the delay statistics - legacy function for backwards compatibility"""
    # This function is kept for any legacy calls, but now redirects to sidebar-based selection
    # Display info about available visualizations
    st.info("Select a visualization from the sidebar to view station delay analysis charts.")
    return

def display_summary_stats(df_stations):
    """Display summary statistics in a formatted layout"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_delays = df_stations['total_of_delays'].sum()
        st.metric("Total Delays", f"{total_delays:,}")
    
    with col2:
        total_trains = df_stations['total_of_trains'].sum()
        st.metric("Total Trains", f"{total_trains:,}")
    
    with col3:
        overall_delay_rate = (total_delays / total_trains * 100) if total_trains > 0 else 0
        st.metric("Overall Delay Rate", f"{overall_delay_rate:.2f}%")
    
    with col4:
        stations_with_delays = (df_stations['total_of_delays'] > 0).sum()
        st.metric("Stations with Delays", f"{stations_with_delays}")

def display_data_table(df_stations):
    """Display the station delay data table"""
    st.subheader("📋 Station Delay Data")
    
    # Add station name to display if available
    display_columns = ['stationShortCode', 'stationName', 'total_of_delays', 
                      'total_of_trains', 'delay_percentage']
    
    # Filter columns that exist
    display_columns = [col for col in display_columns if col in df_stations.columns]
    
    st.dataframe(
        df_stations[display_columns].style.format({
            'total_of_delays': '{:,}',
            'total_of_trains': '{:,}',
            'delay_percentage': '{:.2f}%'
        }),
        use_container_width=True,
        height=500
    )

def create_download_button(df_stations):
    """Create download button for the results"""
    st.subheader("💾 Download Results")
    
    csv = df_stations.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Station Delay Statistics CSV",
        data=csv,
        file_name="station_delay_statistics_full.csv",
        mime="text/csv"
    )

# Main application
def main():
    st.title("📊 Station Delay Statistics Calculator")
    st.markdown("""
    This tool analyzes all matched train data to calculate the total number of delays 
    and trains for each station in the Finnish railway network.
    
    **Delay Definition**: A delay is counted when `differenceInMinutes_eachStation_offset ≥ 5 minutes`
    """)
    
    # Initialize variables for data
    df_data = None
    show_visualizations = False
    
    # Check if results already exist
    existing_results_path = "data/viewers/delay_maps/stations_x_delays.csv"
    
    if os.path.exists(existing_results_path):
        st.success(f"✅ Previous results found at: `{existing_results_path}`")
        
        if st.checkbox("📈 Load existing results", help="Load and display previously calculated station delay statistics"):
            try:
                with st.spinner("Loading existing results..."):
                    df_existing = pd.read_csv(existing_results_path)
                    
                    # Try to load station names and coordinates from metadata
                    metadata_path = "data/viewers/metadata/metadata_train_stations.csv"
                    if os.path.exists(metadata_path):
                        df_metadata = pd.read_csv(metadata_path)
                        # Merge with metadata to get station names and coordinates
                        df_existing = pd.merge(
                            df_existing, 
                            df_metadata[['stationShortCode', 'stationName', 'latitude', 'longitude']], 
                            on='stationShortCode', 
                            how='left'
                        )
                
                # Calculate percentage and sort
                df_existing = calculate_delay_statistics(df_existing)
                df_data = df_existing
                show_visualizations = True
                
                # Display summary statistics
                st.subheader("📈 Summary Statistics")
                display_summary_stats(df_existing)
                
                # Display the data table
                display_data_table(df_existing)
                
            except Exception as e:
                st.error(f"Error loading existing results: {e}")
        
        # Add separator
        st.markdown("---")
    
    # Add a button to start the calculation
    if st.button("🚀 Calculate Station Delays", type="primary"):
        
        with st.spinner("Loading station metadata..."):
            df_stations = load_station_metadata()
        
        if df_stations is not None:
            st.info(f"📍 Found {len(df_stations)} stations in metadata")
            
            with st.spinner("Processing matched data files... This may take a few minutes."):
                df_stations = process_matched_data_files(df_stations)
            
            # Calculate additional statistics
            df_stations = calculate_delay_statistics(df_stations)
            
            # Save results
            save_results(df_stations)
            
            # Set data for visualization
            df_data = df_stations
            show_visualizations = True
            
            # Display summary statistics
            st.subheader("📈 Summary Statistics")
            display_summary_stats(df_stations)
            
            # Display the data table
            display_data_table(df_stations)
            
        else:
            st.error("Failed to load station metadata. Please check the file path and format.")
    
    # SIDEBAR CONFIGURATION - only show if we have data OR if routes are available
    routes_available = load_routes_data() is not None
    
    if show_visualizations and df_data is not None:
        # Calculate summary metrics for sidebar
        total_stations = len(df_data)
        total_delays = df_data['total_of_delays'].sum()
        total_trains = df_data['total_of_trains'].sum()
        overall_delay_rate = (total_delays / total_trains * 100) if total_trains > 0 else 0
        stations_with_traffic = (df_data['total_of_trains'] > 0).sum()
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Dataset Summary")
        st.sidebar.markdown(f"**Total Stations**: {total_stations:,}")
        st.sidebar.markdown(f"**Stations with Traffic**: {stations_with_traffic:,}")
        st.sidebar.markdown(f"**Overall Delay Rate**: {overall_delay_rate:.2f}%")
        st.sidebar.markdown("---")
        
        # SIDEBAR PLOT SELECTION
        st.sidebar.subheader("📈 Chart Selection")
        
        # Define plot options - add route map if routes are available
        plot_options = {
            "📊 Top Stations by Delays": "top_stations",
            "📋 Data Table Only": "data_only",
            "🗺️ Interactive Station Map": "station_map"
        }
        
        # Add route map option if routes data is available
        if routes_available:
            plot_options["🛤️ Route Map"] = "route_map"
        
        selected_plot = st.sidebar.radio(
            "Choose visualization:",
            options=list(plot_options.keys()),
            index=2  # Default to the interactive station map
        )
        
        # Get the plot key
        plot_key = plot_options[selected_plot]
        
        # Show current selection info
        st.sidebar.markdown(f"**Current View**: {selected_plot}")
        
        # MAIN CONTENT AREA - Display selected plot
        if plot_key != "data_only":
            st.subheader("📊 Station Delay Analysis")
            create_single_visualization(df_data, plot_key)
        
        # Download button (not shown for route map since it doesn't use delay data)
        if plot_key != "route_map":
            st.markdown("---")
            create_download_button(df_data)
    
    elif routes_available and not show_visualizations:
        # Show route map option even if delay data isn't loaded
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Available Features")
        st.sidebar.markdown("**🛤️ Route Map**: Available")
        st.sidebar.markdown("**📊 Delay Analysis**: Load data first")
        
        if st.sidebar.button("🛤️ View Route Map"):
            st.subheader("📊 Train Route Analysis")
            df_routes = load_routes_data()
            create_route_map_plot(df_routes)
    
    elif not show_visualizations:
        # Show information when button hasn't been clicked
        st.info("""
        👆 Click the **Calculate Station Delays** button to:
        1. Load station metadata
        2. Process all matched data files
        3. Count delays and trains for each station
        4. Generate statistics and visualizations
        5. Save results to `data/viewers/delay_maps/stations_x_delays.csv`
        
        ⏱️ **Note**: This process may take several minutes depending on the amount of data.
        """)
        
        # Show route map availability
        if routes_available:
            st.info("🛤️ **Bonus**: Route map visualization is available in the sidebar!")

if __name__ == "__main__":
    main()