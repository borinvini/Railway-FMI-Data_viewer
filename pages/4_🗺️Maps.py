import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
from src.DataViewer import DataViewer
from config.const import CSV_FMI_EMS, CSV_CLOSEST_EMS_TRAIN, VIEWER_FOLDER_NAME

# Page configuration
st.set_page_config(
    page_title="Train & EMS Map Viewer",
    page_icon="🗺️",
    layout="wide"
)

# Initialize DataViewer instance
viewer = DataViewer()

# Title and introduction
st.title("🗺️ Train Stations & Environmental Monitoring Stations Map")
st.markdown("""
This map shows train stations in Finland and their corresponding Environmental Monitoring Stations (EMS) from 
the Finnish Meteorological Institute (FMI). The weather data from these EMS stations is used for analyzing 
train operations in different weather conditions.

Each train station (🚂) is connected to its closest EMS station (☁️) with a line showing the geographical relationship.
""")

# Check if data exists
if not viewer.has_data():
    st.stop()  # Stop execution if no data

# Function to load train-EMS mapping data
def load_train_ems_mapping():
    """Load the mapping between train stations and their closest EMS stations"""
    file_path = os.path.join(VIEWER_FOLDER_NAME, CSV_CLOSEST_EMS_TRAIN)
    
    if not os.path.exists(file_path):
        st.error(f"⚠️ File `{CSV_CLOSEST_EMS_TRAIN}` not found in `{VIEWER_FOLDER_NAME}`.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"❌ Error loading train-EMS mapping data: {e}")
        st.exception(e)
        return None

# Function to load EMS data (can be used as a fallback)
def load_ems_data():
    """Load the EMS stations data from CSV file"""
    file_path = os.path.join(VIEWER_FOLDER_NAME, CSV_FMI_EMS)
    
    if not os.path.exists(file_path):
        st.error(f"⚠️ File `{CSV_FMI_EMS}` not found in `{VIEWER_FOLDER_NAME}`.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"❌ Error loading EMS data: {e}")
        st.exception(e)
        return None

# Load train-EMS mapping data
mapping_data = load_train_ems_mapping()

if mapping_data is not None and not mapping_data.empty:
    # Display dataset info
    st.sidebar.subheader("Dataset Information")
    st.sidebar.write(f"Total train stations: {len(mapping_data)}")
    
    # Get unique EMS stations from mapping
    unique_ems_stations = mapping_data.drop_duplicates(subset=['closest_ems_station', 'ems_latitude', 'ems_longitude'])
    st.sidebar.write(f"Unique EMS stations: {len(unique_ems_stations)}")
    
    # Add filters
    st.sidebar.subheader("Map Filters")
    
    # Filter by maximum distance between train station and EMS
    max_distance = min(50.0, mapping_data['distance_km'].max())
    selected_max_distance = st.sidebar.slider(
        "Max Distance (km)",
        min_value=0.0,
        max_value=50.0,
        value=max_distance,
        step=1.0
    )
    
    # Filter the data
    filtered_data = mapping_data[mapping_data['distance_km'] <= selected_max_distance]
    
    # Search functionality
    search_term = st.sidebar.text_input("Search by Train Station or EMS Name")
    if search_term:
        filtered_data = filtered_data[
            filtered_data['train_station_name'].str.contains(search_term, case=False) | 
            filtered_data['closest_ems_station'].str.contains(search_term, case=False)
        ]
    
    # Create a base map centered on Finland
    m = folium.Map(
        location=[64.9, 25.7],  # Center of Finland (approximate)
        zoom_start=5,
        tiles="CartoDB positron",  # Use a light map as default
        attr="© CartoDB, © OpenStreetMap contributors"
    )
    
    # Add different tile layers
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Standard Map",
        attr="© OpenStreetMap contributors",
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Dark Map",
        attr="© CartoDB, © OpenStreetMap contributors",
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles="Stamen Terrain",
        name="Terrain Map",
        attr="© Stamen Design, © OpenStreetMap contributors",
        control=True
    ).add_to(m)
    
    # Create different feature groups
    train_stations_group = folium.FeatureGroup(name="Train Stations", show=True)
    ems_stations_group = folium.FeatureGroup(name="EMS Stations", show=True)
    connections_group = folium.FeatureGroup(name="Station Connections", show=True)
    
    # Dictionary to keep track of already added EMS stations
    added_ems_stations = {}
    
    # Add markers and connections for each pair
    for idx, row in filtered_data.iterrows():
        train_lat, train_long = row['train_lat'], row['train_long']
        ems_lat, ems_long = row['ems_latitude'], row['ems_longitude']
        distance = row['distance_km']
        
        # Create train station marker
        train_popup_text = f"""
        <div style="width: 300px">
            <h4>🚂 Train Station: {row['train_station_name']}</h4>
            <b>Short Code:</b> {row['train_station_short_code']}<br>
            <b>Coordinates:</b> {train_lat:.6f}, {train_long:.6f}<br>
            <b>Closest EMS:</b> {row['closest_ems_station']}<br>
            <b>Distance to EMS:</b> {distance:.2f} km<br>
        </div>
        """
        
        folium.Marker(
            location=[train_lat, train_long],
            popup=folium.Popup(train_popup_text, max_width=300),
            tooltip=f"🚂 {row['train_station_name']} ({row['train_station_short_code']})",
            icon=folium.Icon(color="red", icon="train", prefix="fa")
        ).add_to(train_stations_group)
        
        # Add EMS station marker if not already added
        ems_key = f"{row['closest_ems_station']}_{ems_lat}_{ems_long}"
        
        if ems_key not in added_ems_stations:
            ems_popup_text = f"""
            <div style="width: 300px">
                <h4>☁️ EMS Station: {row['closest_ems_station']}</h4>
                <b>Coordinates:</b> {ems_lat:.6f}, {ems_long:.6f}<br>
                <b>Connected Train Stations:</b> {added_ems_stations.get(ems_key, 0) + 1}<br>
            </div>
            """
            
            folium.Marker(
                location=[ems_lat, ems_long],
                popup=folium.Popup(ems_popup_text, max_width=300),
                tooltip=f"☁️ {row['closest_ems_station']}",
                icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
            ).add_to(ems_stations_group)
            
            # Mark as added
            added_ems_stations[ems_key] = 1
        else:
            # Increment counter for this EMS station
            added_ems_stations[ems_key] += 1
        
        # Add connection line between train station and EMS station
        folium.PolyLine(
            locations=[[train_lat, train_long], [ems_lat, ems_long]],
            color="gray",
            weight=2,
            opacity=0.7,
            tooltip=f"Distance: {distance:.2f} km"
        ).add_to(connections_group)
    
    # Add all feature groups to the map
    train_stations_group.add_to(m)
    ems_stations_group.add_to(m)
    connections_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display the count of stations being shown
    st.write(f"Displaying {len(filtered_data)} train stations and {len(added_ems_stations)} EMS stations")
    
    # Display the map using streamlit-folium (full width)
    st_data = st_folium(m, width=None, height=600, returned_objects=[])
    
    # Statistics and additional information (below the map)
    st.subheader("Map Statistics")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Calculate average distance
    avg_distance = filtered_data['distance_km'].mean()
    min_distance = filtered_data['distance_km'].min()
    max_distance = filtered_data['distance_km'].max()
    
    with col1:
        st.metric("Average Distance to EMS", f"{avg_distance:.2f} km")
    with col2:
        st.metric("Closest EMS Proximity", f"{min_distance:.2f} km")
    with col3:
        st.metric("Furthest EMS Proximity", f"{max_distance:.2f} km")
    
    # Create two columns for closest and furthest pairs
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Show train stations with the closest EMS stations
        st.subheader("Top 5 Closest Train-EMS Pairs")
        closest_pairs = filtered_data.nsmallest(5, 'distance_km')
        for _, row in closest_pairs.iterrows():
            st.write(f"**{row['train_station_name']}** → **{row['closest_ems_station']}**: {row['distance_km']:.2f} km")
    
    with col_right:
        # Show train stations with the furthest EMS stations
        st.subheader("Top 5 Furthest Train-EMS Pairs")
        furthest_pairs = filtered_data.nlargest(5, 'distance_km')
        for _, row in furthest_pairs.iterrows():
            st.write(f"**{row['train_station_name']}** → **{row['closest_ems_station']}**: {row['distance_km']:.2f} km")
    
    # Map interaction info
    st.subheader("Map Interaction")
    if st_data.get("last_clicked"):
        lat, lon = st_data["last_clicked"]["lat"], st_data["last_clicked"]["lng"]
        st.write(f"Last clicked position: {lat:.6f}, {lon:.6f}")
        
        col_train, col_ems = st.columns(2)
        
        # Calculate closest train station to click
        filtered_data['click_distance'] = ((filtered_data['train_lat'] - lat)**2 + 
                                       (filtered_data['train_long'] - lon)**2)**0.5 * 111
        
        closest_train = filtered_data.loc[filtered_data['click_distance'].idxmin()]
        
        with col_train:
            st.write(f"Closest train station to click: **{closest_train['train_station_name']}** ({closest_train['click_distance']:.2f} km)")
        
        # Calculate closest EMS station to click
        filtered_data['click_distance_ems'] = ((filtered_data['ems_latitude'] - lat)**2 + 
                                           (filtered_data['ems_longitude'] - lon)**2)**0.5 * 111
        
        closest_ems = filtered_data.loc[filtered_data['click_distance_ems'].idxmin()]
        
        with col_ems:
            st.write(f"Closest EMS station to click: **{closest_ems['closest_ems_station']}** ({closest_ems['click_distance_ems']:.2f} km)")
    else:
        st.info("Click on the map to find the closest stations to that point.")
    
    # EMS stations with the most train connections
    st.subheader("EMS Stations By Number of Train Connections")
    
    # Count train stations per EMS
    ems_counts = filtered_data.groupby('closest_ems_station').size().reset_index(name='train_count')
    ems_counts = ems_counts.sort_values('train_count', ascending=False)
    
    # Display as a bar chart
    st.bar_chart(ems_counts.set_index('closest_ems_station')['train_count'])
    
    # Display the data in an expandable section
    with st.expander("View Data Table", expanded=False):
        # Remove computational columns
        display_data = filtered_data.copy()
        if 'click_distance' in display_data.columns:
            display_data = display_data.drop(columns=['click_distance'])
        if 'click_distance_ems' in display_data.columns:
            display_data = display_data.drop(columns=['click_distance_ems'])
            
        st.dataframe(display_data)
        
else:
    st.warning("⚠️ No train-EMS mapping data available. Please check your data files.")
    
    # Try to load just the EMS data as a fallback
    st.subheader("Fallback: EMS Stations Only")
    ems_data = load_ems_data()
    
    if ems_data is not None and not ems_data.empty:
        st.info("Displaying only EMS stations since train-EMS mapping is not available.")
        
        # Create a simple map with EMS stations
        m = folium.Map(
            location=[64.9, 25.7],  # Center of Finland (approximate)
            zoom_start=5,
            tiles="OpenStreetMap",
            attr="© OpenStreetMap contributors"
        )
        
        # Try to determine the latitude and longitude columns
        lat_options = ['latitude', 'lat', 'y']
        lon_options = ['longitude', 'lon', 'long', 'x']
        name_options = ['station_name', 'name', 'stationName']
        
        lat_col = next((col for col in lat_options if col in ems_data.columns), None)
        lon_col = next((col for col in lon_options if col in ems_data.columns), None)
        name_col = next((col for col in name_options if col in ems_data.columns), None)
        
        if lat_col and lon_col and name_col:
            for idx, row in ems_data.iterrows():
                folium.Marker(
                    location=[row[lat_col], row[lon_col]],
                    popup=f"EMS Station: {row[name_col]}",
                    tooltip=row[name_col],
                    icon=folium.Icon(color="blue", icon="cloud")
                ).add_to(m)
            
            st_folium(m, width=800, height=600, returned_objects=[])
        else:
            st.error("Could not determine the coordinate columns in the EMS data.")
    else:
        st.error("No data available to display.")