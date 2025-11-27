from datetime import datetime
import io
import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import os
from ast import literal_eval

# Page configuration
st.set_page_config(
    page_title="Finland Train Stations Map",
    page_icon="🗺️",
    layout="wide"
)

st.title("🇫🇮 Finland Train Stations Map")

st.markdown("""
### Delay-Based Station Categorization
This map visualizes Finnish railway passenger stations color-coded by their delay performance:
- 🔴 **High Delay** (≥15% delay rate) - Stations with significant delay issues (labeled with station name and delay %)
- 🟡 **Medium Delay** (5-15% delay rate) - Stations with moderate delays
- 🟢 **Low Delay** (<5% delay rate) - Stations with good performance

**Railway Connection Colors:**
- 🟢 **Green**: Low traffic (≤200 trains total) OR both stations have <5% delay
- 🟡 **Yellow**: At least one station has 5-15% delay (with >200 trains total)
- 🔴 **Red**: At least one station has ≥15% delay (with >200 trains total)

**Note**: Only passenger stations with ≥100 trains are shown. Delay rate = (total_of_delays / total_of_trains) × 100
""")

# Load Finland map with high resolution
@st.cache_data
def load_map():
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    world = gpd.read_file(url)
    finland = world[world['NAME'] == 'Finland']
    return finland

# Load train stations data
@st.cache_data
def load_train_stations():
    stations_file = "data/viewers/metadata/metadata_train_stations.csv"
    
    if not os.path.exists(stations_file):
        st.error(f"⚠️ Train stations file not found at: {stations_file}")
        return None
    
    try:
        df_stations = pd.read_csv(stations_file)
        
        # Check if required columns exist
        if 'latitude' not in df_stations.columns or 'longitude' not in df_stations.columns:
            st.error("⚠️ Missing 'latitude' or 'longitude' columns in the dataset")
            return None
        
        # Filter to show only Finnish stations
        if 'countryCode' in df_stations.columns:
            initial_count = len(df_stations)
            df_stations = df_stations[df_stations['countryCode'] == 'FI']
            filtered_count = initial_count - len(df_stations)
            if filtered_count > 0:
                st.info(f"ℹ️ Filtered out {filtered_count} non-Finnish stations. Showing only stations with countryCode='FI'")
        else:
            st.warning("⚠️ 'countryCode' column not found. Showing all stations.")
        
        # Remove stations with missing coordinates
        df_stations = df_stations.dropna(subset=['latitude', 'longitude'])
        
        return df_stations
        
    except Exception as e:
        st.error(f"❌ Error loading train stations data: {e}")
        return None

# Load delay statistics
@st.cache_data
def load_delay_statistics():
    """Load delay statistics for determining line colors"""
    delay_stats_file = "data/viewers/delay_maps/stations_x_delays.csv"
    
    if not os.path.exists(delay_stats_file):
        st.warning(f"⚠️ Delay statistics file not found at: {delay_stats_file}")
        return None
    
    try:
        df_delays = pd.read_csv(delay_stats_file)
        
        # Calculate delay percentage
        df_delays['delay_percentage'] = 0.0
        mask = df_delays['total_of_trains'] > 0
        df_delays.loc[mask, 'delay_percentage'] = (
            df_delays.loc[mask, 'total_of_delays'] / df_delays.loc[mask, 'total_of_trains'] * 100
        )
        
        return df_delays
        
    except Exception as e:
        st.error(f"❌ Error loading delay statistics: {e}")
        return None

# Load routes data
@st.cache_data
def load_routes_data():
    routes_file = "data/viewers/metadata/metadata_routes.csv"
    
    if not os.path.exists(routes_file):
        st.warning(f"⚠️ Routes file not found at: {routes_file}")
        return None
    
    try:
        df_routes = pd.read_csv(routes_file)
        return df_routes
    except Exception as e:
        st.error(f"❌ Error loading routes data: {e}")
        return None

# Create graph from routes (unique connections only)
def create_routes_graph(df_routes):
    """
    Create a graph from routes where each connection appears only once.
    
    Args:
        df_routes: DataFrame with 'route' column containing string representations of station lists
        
    Returns:
        DataFrame with columns: station_1, station_2
    """
    if df_routes is None or df_routes.empty:
        return None
    
    # Use a set to store unique connections
    # Store as sorted tuples to avoid duplicates (A,B) == (B,A)
    unique_connections = set()
    
    # Process each route
    for idx, row in df_routes.iterrows():
        route_str = row['route']
        
        try:
            # Parse the route string to get list of stations
            route_stations = literal_eval(route_str)
            
            # Create connections between consecutive stations
            for i in range(len(route_stations) - 1):
                station_1 = route_stations[i]
                station_2 = route_stations[i + 1]
                
                # Create a sorted tuple to ensure uniqueness regardless of order
                connection = tuple(sorted([station_1, station_2]))
                unique_connections.add(connection)
                
        except Exception as e:
            st.warning(f"⚠️ Error parsing route at index {idx}: {e}")
            continue
    
    # Convert to DataFrame
    if unique_connections:
        connections_list = [{'station_1': conn[0], 'station_2': conn[1]} 
                          for conn in unique_connections]
        df_graph = pd.DataFrame(connections_list)
        
        # Sort by station_1, then station_2 for consistency
        df_graph = df_graph.sort_values(['station_1', 'station_2']).reset_index(drop=True)
        
        return df_graph
    
    return None

# Save graph to CSV
def save_routes_graph(df_graph):
    """Save the routes graph to CSV file"""
    if df_graph is None or df_graph.empty:
        st.error("❌ Cannot save empty graph")
        return False
    
    # Create directory if it doesn't exist
    output_dir = "data/metadata"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, "metadata_routes_graph.csv")
    
    try:
        df_graph.to_csv(output_file, index=False)
        st.success(f"✅ Routes graph saved successfully to: `{output_file}`")
        st.info(f"📊 Total unique connections: {len(df_graph):,}")
        return True
    except Exception as e:
        st.error(f"❌ Error saving routes graph: {e}")
        return False

def save_figure_to_pdf(fig, filename):
    """
    Save a matplotlib figure to PDF format and return as bytes for download
    
    Args:
        fig: Matplotlib figure object
        filename: Name for the PDF file
        
    Returns:
        BytesIO object containing the PDF data
    """
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format='pdf', dpi=300, bbox_inches='tight')
    pdf_buffer.seek(0)
    return pdf_buffer

def create_pdf_download_button(fig, button_label, file_name):
    """
    Create a download button for PDF figure
    
    Args:
        fig: Matplotlib figure object
        button_label: Label for the download button
        file_name: Name of the PDF file to download
    """
    pdf_buffer = save_figure_to_pdf(fig, file_name)
    
    st.download_button(
        label=button_label,
        data=pdf_buffer,
        file_name=file_name,
        mime="application/pdf",
        key=f"pdf_{file_name}"
    )

    
def get_connection_color(station_1_name, station_2_name, df_stations, df_delays):
    """
    Determine the color of a railway connection based on traffic and delay rates.
    
    Args:
        station_1_name: Name of first station
        station_2_name: Name of second station
        df_stations: DataFrame with station metadata
        df_delays: DataFrame with delay statistics
        
    Returns:
        tuple: (color_hex, color_name) or None if data not available
    """
    # Create mapping from station name to short code
    station_name_to_code = dict(zip(df_stations['stationName'], df_stations['stationShortCode']))
    
    # Get short codes for both stations
    code_1 = station_name_to_code.get(station_1_name)
    code_2 = station_name_to_code.get(station_2_name)
    
    if code_1 is None or code_2 is None:
        return None
    
    # Get delay statistics for both stations
    station_1_data = df_delays[df_delays['stationShortCode'] == code_1]
    station_2_data = df_delays[df_delays['stationShortCode'] == code_2]
    
    if station_1_data.empty or station_2_data.empty:
        return None
    
    # Extract statistics
    trains_1 = station_1_data['total_of_trains'].iloc[0]
    trains_2 = station_2_data['total_of_trains'].iloc[0]
    delay_pct_1 = station_1_data['delay_percentage'].iloc[0]
    delay_pct_2 = station_2_data['delay_percentage'].iloc[0]
    
    total_trains = trains_1 + trains_2
    max_delay_pct = max(delay_pct_1, delay_pct_2)
    
    # Apply color logic - if at least one station exceeds threshold, use that color
    if total_trains <= 200:
        return ('#2E8B57', 'green')  # Green - Low traffic
    elif max_delay_pct < 5:
        return ('#2E8B57', 'green')  # Green - Both stations low delay
    elif max_delay_pct < 15:
        return ('#FFD700', 'yellow')  # Yellow - At least one station medium delay
    else:
        return ('#DC143C', 'red')  # Red - At least one station high delay

# Load data
with st.spinner("Loading map and station data..."):
    finland = load_map()
    df_stations = load_train_stations()
    df_routes = load_routes_data()
    df_delays = load_delay_statistics()

# Automatically generate and save routes graph when page loads
df_graph = None
if df_routes is not None:
    # Create routes graph
    df_graph = create_routes_graph(df_routes)
    
    if df_graph is not None:
        # Save the graph automatically
        save_routes_graph(df_graph)
        
        # Display statistics
        unique_stations = set(df_graph['station_1'].tolist() + df_graph['station_2'].tolist())
        
        st.markdown("---")
        st.markdown("### 🛤️ Routes Graph Information")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Unique Connections", f"{len(df_graph):,}")
        with col_stat2:
            st.metric("Unique Stations in Graph", f"{len(unique_stations):,}")
        with col_stat3:
            st.metric("Total Routes Processed", f"{len(df_routes):,}")
        
        st.markdown("---")

if df_stations is not None:
    # Display statistics
    st.markdown("### 📊 Dataset Information")
    
    if 'passengerTraffic' in df_stations.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Train Stations", f"{len(df_stations):,}")
        
        with col2:
            passenger_count = df_stations['passengerTraffic'].sum()
            st.metric("🔵 Passenger Stations", f"{passenger_count:,}")
        
        with col3:
            non_passenger_count = len(df_stations) - passenger_count
            st.metric("🔴 Non-Passenger Stations", f"{non_passenger_count:,}")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Train Stations inside Finland", f"{len(df_stations):,}")
    
    
    # Create IEEE-compliant figure (one-column width)
    fig, ax = plt.subplots(figsize=(7.16, 16), facecolor='white')
    ax.set_facecolor('#ffffff')
    
    # Configure IEEE font settings
    plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['font.size'] = 16

    # Plot Finland with IEEE styling - white borders
    finland.plot(
        ax=ax, 
        color='#d3d3d3',
        edgecolor='black',  # White borders
        linewidth=1.0,  # 1.0pt line width
        alpha=0.9
    )
    
    # Draw graph connections if available
    if df_graph is not None and df_delays is not None:
        # Create a dictionary for quick station lookup by name
        station_coords = {}
        for _, station in df_stations.iterrows():
            station_name = station.get('stationName', station.get('stationShortCode', ''))
            if station_name:
                station_coords[station_name] = (station['longitude'], station['latitude'])
        
        # Draw connections with color coding
        connections_drawn = 0
        connections_skipped = 0
        color_counts = {'green': 0, 'yellow': 0, 'red': 0, 'unknown': 0}
        
        for _, row in df_graph.iterrows():
            station_1_name = row['station_1']
            station_2_name = row['station_2']
            
            # Get coordinates for both stations
            if station_1_name in station_coords and station_2_name in station_coords:
                lon1, lat1 = station_coords[station_1_name]
                lon2, lat2 = station_coords[station_2_name]
                
                # Determine color based on delay statistics
                color_result = get_connection_color(station_1_name, station_2_name, df_stations, df_delays)
                
                if color_result:
                    color_hex, color_name = color_result
                    color_counts[color_name] += 1
                else:
                    # Default to gray if no delay data
                    color_hex = '#808080'
                    color_counts['unknown'] += 1
                
                # Draw line between stations
                ax.plot(
                    [lon1, lon2], 
                    [lat1, lat2],
                    color=color_hex,
                    linewidth=4.0,
                    alpha=0.7,
                    zorder=3
                )
                connections_drawn += 1
            else:
                connections_skipped += 1
        
        # Show connection statistics with color breakdown
        if connections_drawn > 0:
            st.info(f"✅ Drew {connections_drawn:,} railway connections | "
                   f"🟢 Green: {color_counts['green']:,} | "
                   f"🟡 Yellow: {color_counts['yellow']:,} | "
                   f"🔴 Red: {color_counts['red']:,} | "
                   f"⚪ Unknown: {color_counts['unknown']:,}"
                   f"{' | ⚠️ ' + str(connections_skipped) + ' connections skipped' if connections_skipped > 0 else ''}")
    elif df_graph is not None:
        # Fallback to single color if no delay data
        station_coords = {}
        for _, station in df_stations.iterrows():
            station_name = station.get('stationName', station.get('stationShortCode', ''))
            if station_name:
                station_coords[station_name] = (station['longitude'], station['latitude'])
        
        connections_drawn = 0
        connections_skipped = 0
        
        for _, row in df_graph.iterrows():
            station_1_name = row['station_1']
            station_2_name = row['station_2']
            
            if station_1_name in station_coords and station_2_name in station_coords:
                lon1, lat1 = station_coords[station_1_name]
                lon2, lat2 = station_coords[station_2_name]
                
                ax.plot(
                    [lon1, lon2], 
                    [lat1, lat2],
                    color='#4169E1',
                    linewidth=4.0,
                    alpha=0.6,
                    zorder=3
                )
                connections_drawn += 1
            else:
                connections_skipped += 1
        
        if connections_drawn > 0:
            st.warning(f"⚠️ Drew {connections_drawn:,} connections in blue (delay data not available). "
                      f"{'⚠️ ' + str(connections_skipped) + ' connections skipped.' if connections_skipped > 0 else ''}")
    
    # Plot train stations - color-code based on delay percentage
    if 'passengerTraffic' in df_stations.columns:
        # Filter to only show passenger stations
        passenger_stations = df_stations[df_stations['passengerTraffic'] == True].copy()
        
        # Load delay statistics to get delay data
        delay_stats_file = "data/viewers/delay_maps/stations_x_delays.csv"
        if os.path.exists(delay_stats_file):
            try:
                df_delays_stations = pd.read_csv(delay_stats_file)
                
                # Filter for stations with at least 100 trains
                df_delays_stations = df_delays_stations[df_delays_stations['total_of_trains'] >= 100].copy()
                
                # Calculate delay percentage    
                df_delays_stations['delay_percentage'] = (df_delays_stations['total_of_delays'] / df_delays_stations['total_of_trains']) * 100
                
                # Merge with passenger stations
                passenger_stations = passenger_stations.merge(
                    df_delays_stations[['stationShortCode', 'total_of_trains', 'total_of_delays', 'delay_percentage']], 
                    on='stationShortCode', 
                    how='left'
                )
                
                # Split stations by delay percentage categories
                high_delay = passenger_stations[passenger_stations['delay_percentage'] >= 15]
                medium_delay = passenger_stations[(passenger_stations['delay_percentage'] >= 5) & 
                                                 (passenger_stations['delay_percentage'] < 15)]
                low_delay = passenger_stations[(passenger_stations['delay_percentage'] < 5) & 
                                              (passenger_stations['delay_percentage'].notna())]
                
                # Plot high delay stations in red
                if not high_delay.empty:
                    ax.scatter(
                        high_delay['longitude'], 
                        high_delay['latitude'],
                        c='#DC143C',
                        s=35,
                        alpha=0.8,
                        edgecolors='#8B0000',
                        linewidths=0.8,
                        marker='o',
                        zorder=5
                    )
                    
                    # Add station name and delay percentage labels
                    for _, station in high_delay.iterrows():
                        label_text = f"{station['stationName']} ({station['delay_percentage']:.1f}%)"
                        ax.annotate(
                            label_text,
                            xy=(station['longitude'], station['latitude']),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=10,
                            fontweight='bold',
                            color='#8B0000',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#DC143C', alpha=0.7),
                            zorder=6
                        )
                
                # Plot medium delay stations in yellow
                if not medium_delay.empty:
                    ax.scatter(
                        medium_delay['longitude'], 
                        medium_delay['latitude'],
                        c='#FFD700',
                        s=35,
                        alpha=0.8,
                        edgecolors='#DAA520',
                        linewidths=0.8,
                        marker='o',
                        zorder=5
                    )
                
                # Plot low delay stations in green
                if not low_delay.empty:
                    ax.scatter(
                        low_delay['longitude'], 
                        low_delay['latitude'],
                        c='#2E8B57',
                        s=35,
                        alpha=0.8,
                        edgecolors='#006400',
                        linewidths=0.8,
                        marker='o',
                        zorder=5
                    )
                
                # Show statistics
                stations_shown = len(high_delay) + len(medium_delay) + len(low_delay)
                st.info(f"🔴 High Delay (≥15%): {len(high_delay):,} | "
                       f"🟡 Medium Delay (5-15%): {len(medium_delay):,} | "
                       f"🟢 Low Delay (<5%): {len(low_delay):,} | "
                       f"Total Shown: {stations_shown:,} stations")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load delay statistics: {e}. Showing all stations in gray.")
                ax.scatter(
                    passenger_stations['longitude'], 
                    passenger_stations['latitude'],
                    c='#808080',
                    s=35,
                    alpha=0.6,
                    edgecolors='#696969',
                    linewidths=0.8,
                    marker='o',
                    zorder=4
                )
        else:
            st.warning(f"⚠️ Delay statistics file not found. Showing all stations in gray.")
            ax.scatter(
                passenger_stations['longitude'], 
                passenger_stations['latitude'],
                c='#808080',
                s=35,
                alpha=0.6,
                edgecolors='#696969',
                linewidths=0.8,
                marker='o',
                zorder=4
            )
    else:
        ax.scatter(
            df_stations['longitude'], 
            df_stations['latitude'],
            c='#808080',
            s=35,
            alpha=0.6,
            edgecolors='#696969',
            linewidths=0.8,
            marker='o',
            label='Train Stations',
            zorder=4
        )
    
    # Add axis labels with IEEE formatting
    ax.set_xlabel('Longitude', fontsize=16, family='serif')
    ax.set_ylabel('Latitude', fontsize=16, family='serif')

    # No title - IEEE uses captions below figures

    # Style the tick labels
    ax.tick_params(axis='both', labelsize=16, width=0.5)

    # IEEE spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor('white')  # White borders
        spine.set_linewidth=1.0  # 1.0pt
    
    # Add legend
    legend_elements = []
    
    # Station legend
    if 'passengerTraffic' in df_stations.columns and os.path.exists("data/viewers/delay_maps/stations_x_delays.csv"):
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w', label='High Delay Station (≥15%)',
                      markerfacecolor='#DC143C', markersize=10, markeredgecolor='#8B0000'),
            plt.Line2D([0], [0], marker='o', color='w', label='Medium Delay Station (5-15%)',
                      markerfacecolor='#FFD700', markersize=10, markeredgecolor='#DAA520'),
            plt.Line2D([0], [0], marker='o', color='w', label='Low Traffic (<100 Trains) / Low Delay Station (<5%)',
                      markerfacecolor='#2E8B57', markersize=10, markeredgecolor='#006400')
                    
        ])
    
    # Connection legend (if we have delay data for connections)
    if df_graph is not None and df_delays is not None:
        legend_elements.extend([
        ])
    elif df_graph is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color='#4169E1', linewidth=4, alpha=0.6, label='Railway Connections')
        )
    
    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=16, framealpha=0.9)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#cccccc')
    
    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#666666')
    
    plt.tight_layout()
    st.pyplot(fig)

    # PDF Download Button for Map 1
    st.markdown("### 💾 Download Map as PDF")
    col_pdf1, col_info1 = st.columns([1, 3])
    with col_pdf1:
        create_pdf_download_button(fig, "📥 Download Map 1 (PDF)", f"finland_delay_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    with col_info1:
        st.info("📄 High-resolution PDF (300 DPI) suitable for publications")
    
    
    # Optional: Show station details
    with st.expander("🔍 View Station Details", expanded=False):
        display_columns = ['stationShortCode', 'stationName', 'latitude', 'longitude']
        
        if 'passengerTraffic' in df_stations.columns:
            display_columns.append('passengerTraffic')
        
        available_columns = [col for col in display_columns if col in df_stations.columns]
        
        if available_columns:
            st.dataframe(
                df_stations[available_columns].sort_values('stationName' if 'stationName' in available_columns else 'stationShortCode'),
                use_container_width=True,
                height=400
            )
    
    # Download option
    st.markdown("### 💾 Download Data")
    csv = df_stations.to_csv(index=False)
    st.download_button(
        label="📥 Download Station Coordinates CSV",
        data=csv,
        file_name="finland_train_stations.csv",
        mime="text/csv"
    )
    
else:
    st.warning("⚠️ Train station data not available. Showing only Finland map.")
    
    fig, ax = plt.subplots(figsize=(12, 14), facecolor='#f5f5f5')
    ax.set_facecolor('#ffffff')
    
    finland.plot(
        ax=ax, 
        color='#d3d3d3',
        edgecolor='#808080',
        linewidth=1.5,
        alpha=0.9
    )
    
    finland.plot(
        ax=ax,
        color='none',
        edgecolor='#000000',
        linewidth=2,
        alpha=0.15,
        linestyle='-'
    )
    
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold', color='#333333')
    ax.tick_params(axis='both', labelsize=10, colors='#333333')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    ax.grid(False)
    
    plt.tight_layout()
    st.pyplot(fig)

# ========== NEW SECTION: TRAIN STATIONS AND EMS MAP ==========
st.markdown("---")
st.title("🌦️ Train Stations & Environmental Monitoring Stations")

st.markdown("""
### Train-EMS Station Connections
This map visualizes the connection between Finnish railway stations and their closest Environmental Monitoring Stations (EMS) 
from the Finnish Meteorological Institute (FMI). Weather data from these EMS stations is used for analyzing train operations 
in different weather conditions.

- 🚂 **Red circles**: Train stations
- ☁️ **Blue triangles**: EMS (Environmental Monitoring Stations)
- **Purple dashed lines**: Connections showing which EMS station is closest to each train station
""")

# Load train-EMS mapping data
@st.cache_data
def load_train_ems_mapping():
    """Load the mapping between train stations and their closest EMS stations, filtered for Finnish passenger stations only"""
    mapping_file = os.path.join("data/viewers/metadata", "metadata_closest_ems_to_train_stations.csv")
    stations_file = os.path.join("data/viewers/metadata", "metadata_train_stations.csv")
    
    if not os.path.exists(mapping_file):
        st.error(f"⚠️ Train-EMS mapping file not found at: {mapping_file}")
        return None
    
    try:
        df_mapping = pd.read_csv(mapping_file)
        
        # Load train stations metadata to get countryCode and passengerTraffic
        if os.path.exists(stations_file):
            df_stations_meta = pd.read_csv(stations_file)
            
            # Filter for Finnish passenger stations only
            if 'countryCode' in df_stations_meta.columns and 'stationShortCode' in df_stations_meta.columns and 'passengerTraffic' in df_stations_meta.columns:
                # Get Finnish passenger stations
                finnish_passenger_stations = df_stations_meta[
                    (df_stations_meta['countryCode'] == 'FI') & 
                    (df_stations_meta['passengerTraffic'] == True)
                ]['stationShortCode'].tolist()
                
                initial_count = len(df_mapping)
                # Filter mapping data to only include Finnish passenger train stations
                df_mapping = df_mapping[df_mapping['train_station_short_code'].isin(finnish_passenger_stations)]
                filtered_count = initial_count - len(df_mapping)
                
                if filtered_count > 0:
                    st.info(f"ℹ️ Filtered out {filtered_count} stations. Showing only Finnish passenger stations (countryCode='FI' and passengerTraffic=True)")
            else:
                st.warning("⚠️ Required columns not found in train stations metadata. Showing all stations.")
        else:
            st.warning(f"⚠️ Train stations metadata file not found at: {stations_file}. Cannot filter by country code or passenger traffic.")
        
        return df_mapping
    except Exception as e:
        st.error(f"❌ Error loading train-EMS mapping data: {e}")
        return None

# Load the mapping data
df_mapping = load_train_ems_mapping()

if df_mapping is not None and not df_mapping.empty:
    # Display statistics
    st.markdown("### 📊 Connection Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Train Stations", f"{len(df_mapping):,}")
    
    with col2:
        unique_ems = df_mapping['closest_ems_station'].nunique()
        st.metric("Unique EMS Stations", f"{unique_ems:,}")
    
    with col3:
        avg_distance = df_mapping['distance_km'].mean()
        st.metric("Average Distance", f"{avg_distance:.1f} km")
    
    # Create the EMS map
    fig_ems, ax_ems = plt.subplots(figsize=(7.16, 16), facecolor='white')
    ax_ems.set_facecolor('#ffffff')

    # Configure IEEE font settings
    plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['font.size'] = 16

    # Plot Finland base map with IEEE styling - white borders
    finland.plot(
        ax=ax_ems, 
        color='#d3d3d3',
        edgecolor='black',  # White borders
        linewidth=1.0,  # 1.0pt line width
        alpha=0.9
    )
    
    # Draw connection lines between train stations and EMS
    connections_drawn = 0
    for _, row in df_mapping.iterrows():
        train_lat = row['train_lat']
        train_long = row['train_long']
        ems_lat = row['ems_latitude']
        ems_long = row['ems_longitude']
        
        # Check if all coordinates are valid
        if pd.notna(train_lat) and pd.notna(train_long) and pd.notna(ems_lat) and pd.notna(ems_long):
            ax_ems.plot(
                [train_long, ems_long],
                [train_lat, ems_lat],
                color='purple',
                linewidth=1.0,
                alpha=0.3,
                linestyle='--',
                zorder=3
            )
            connections_drawn += 1
    
    # Plot train stations (red circles)
    ax_ems.scatter(
        df_mapping['train_long'],
        df_mapping['train_lat'],
        c='#DC143C',
        s=40,
        alpha=0.8,
        edgecolors='#8B0000',
        linewidths=1,
        marker='o',
        label='Train Stations',
        zorder=5
    )
    
    # Plot unique EMS stations (blue triangles)
    unique_ems_stations = df_mapping.drop_duplicates(subset=['closest_ems_station', 'ems_latitude', 'ems_longitude'])
    
    ax_ems.scatter(
        unique_ems_stations['ems_longitude'],
        unique_ems_stations['ems_latitude'],
        c='#4169E1',
        s=80,
        alpha=0.8,
        edgecolors='#00008B',
        linewidths=1,
        marker='^',
        label='EMS Stations',
        zorder=6
    )
    
    # IEEE axis labels
    ax_ems.set_xlabel('Longitude', fontsize=16, family='serif')
    ax_ems.set_ylabel('Latitude', fontsize=16, family='serif')

    # No title - IEEE uses captions below figures

    # IEEE tick styling
    ax_ems.tick_params(axis='both', labelsize=16, width=0.5)

    # IEEE spine styling
    for spine in ax_ems.spines.values():
        spine.set_edgecolor('white')  # White borders
        spine.set_linewidth = 1.0  # 1.0pt
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Train Stations',
                  markerfacecolor='#DC143C', markersize=10, markeredgecolor='#8B0000'),
        plt.Line2D([0], [0], marker='^', color='w', label='EMS Stations',
                  markerfacecolor='#4169E1', markersize=12, markeredgecolor='#00008B'),
        plt.Line2D([0], [0], color='purple', linewidth=2, alpha=0.5, 
                  linestyle='--', label='Station Connections')
    ]
    
    legend = ax_ems.legend(handles=legend_elements, loc='upper left', fontsize=16, framealpha=0.9)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#cccccc')
    
    # Add grid
    ax_ems.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#666666')
    
    plt.tight_layout()
    st.pyplot(fig_ems)

    # PDF Download Button for Map 2
    st.markdown("### 💾 Download Map as PDF")
    col_pdf2, col_info2 = st.columns([1, 3])
    with col_pdf2:
        create_pdf_download_button(fig_ems, "📥 Download Map 2 (PDF)", f"finland_ems_connections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    with col_info2:
        st.info("📄 High-resolution PDF (300 DPI) suitable for publications")
    
    
    # Show connection statistics
    st.info(f"✅ Drew {connections_drawn:,} connection lines between train stations and their closest EMS stations")
    
    # Show additional insights
    st.markdown("### 🔍 Connection Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        closest_pair = df_mapping.loc[df_mapping['distance_km'].idxmin()]
        st.metric(
            label="Closest Connection",
            value=f"{closest_pair['distance_km']:.2f} km",
            delta=f"{closest_pair['train_station_name']} → {closest_pair['closest_ems_station']}"
        )
    
    with col2:
        furthest_pair = df_mapping.loc[df_mapping['distance_km'].idxmax()]
        st.metric(
            label="Furthest Connection",
            value=f"{furthest_pair['distance_km']:.2f} km",
            delta=f"{furthest_pair['train_station_name']} → {furthest_pair['closest_ems_station']}"
        )
    
    with col3:
        median_distance = df_mapping['distance_km'].median()
        st.metric(
            label="Median Distance",
            value=f"{median_distance:.2f} km"
        )
    
    # Optional: Show detailed mapping data
    with st.expander("🔍 View Train-EMS Mapping Details", expanded=False):
        display_cols = [
            'train_station_name', 'train_station_short_code',
            'closest_ems_station', 'distance_km',
            'train_lat', 'train_long', 'ems_latitude', 'ems_longitude'
        ]
        
        available_cols = [col for col in display_cols if col in df_mapping.columns]
        
        st.dataframe(
            df_mapping[available_cols].sort_values('distance_km'),
            use_container_width=True,
            height=400
        )
    
    # Download option
    st.markdown("### 💾 Download Train-EMS Mapping Data")
    csv_mapping = df_mapping.to_csv(index=False)
    st.download_button(
        label="📥 Download Train-EMS Connections CSV",
        data=csv_mapping,
        file_name="train_ems_connections.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ Train-EMS mapping data not available. Please ensure the metadata file exists.")

# ========== END OF NEW SECTION ==========
