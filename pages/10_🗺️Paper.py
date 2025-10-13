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

# Load data
with st.spinner("Loading map and station data..."):
    finland = load_map()
    df_stations = load_train_stations()
    df_routes = load_routes_data()

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
    
    
    # Create a beautiful figure
    fig, ax = plt.subplots(figsize=(14, 16), facecolor='#f5f5f5')
    ax.set_facecolor('#ffffff')
    
    # Plot Finland with beautiful styling
    finland.plot(
        ax=ax, 
        color='#d3d3d3',
        edgecolor='#808080',
        linewidth=1.5,
        alpha=0.9
    )
    
    # Add a subtle shadow effect
    finland.plot(
        ax=ax,
        color='none',
        edgecolor='#000000',
        linewidth=2,
        alpha=0.15,
        linestyle='-'
    )
    
    # Draw graph connections if available
    if df_graph is not None:
        # Create a dictionary for quick station lookup by name
        station_coords = {}
        for _, station in df_stations.iterrows():
            station_name = station.get('stationName', station.get('stationShortCode', ''))
            if station_name:
                station_coords[station_name] = (station['longitude'], station['latitude'])
        
        # Draw connections
        connections_drawn = 0
        connections_skipped = 0
        
        for _, row in df_graph.iterrows():
            station_1_name = row['station_1']
            station_2_name = row['station_2']
            
            # Get coordinates for both stations
            if station_1_name in station_coords and station_2_name in station_coords:
                lon1, lat1 = station_coords[station_1_name]
                lon2, lat2 = station_coords[station_2_name]
                
                # Draw line between stations
                ax.plot(
                    [lon1, lon2], 
                    [lat1, lat2],
                    color='#4169E1',      # Royal blue
                    linewidth=4.0,        # Much thicker lines
                    alpha=0.6,            # Slightly increased opacity for better visibility
                    zorder=3
                )
                connections_drawn += 1
            else:
                connections_skipped += 1
        
        # Show connection statistics
        if connections_drawn > 0:
            st.info(f"✅ Drew {connections_drawn:,} railway connections on the map. "
                   f"{'⚠️ ' + str(connections_skipped) + ' connections skipped due to missing coordinates.' if connections_skipped > 0 else ''}")
    
    # Plot train stations - color-code based on delay percentage
    if 'passengerTraffic' in df_stations.columns:
        # Filter to only show passenger stations
        passenger_stations = df_stations[df_stations['passengerTraffic'] == True].copy()
        
        # Load delay statistics to get delay data
        delay_stats_file = "data/viewers/delay_maps/stations_x_delays.csv"
        if os.path.exists(delay_stats_file):
            try:
                df_delays = pd.read_csv(delay_stats_file)
                
                # Filter for stations with at least 100 trains
                df_delays = df_delays[df_delays['total_of_trains'] >= 100].copy()
                
                # Calculate delay percentage
                df_delays['delay_percentage'] = (df_delays['total_of_delays'] / df_delays['total_of_trains']) * 100
                
                # Merge with passenger stations
                passenger_stations = passenger_stations.merge(
                    df_delays[['stationShortCode', 'total_of_trains', 'total_of_delays', 'delay_percentage']], 
                    on='stationShortCode', 
                    how='left'
                )
                
                # Split stations by delay percentage categories
                # Only categorize stations with traffic data (>=100 trains)
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
                        c='#DC143C',  # Crimson red
                        s=35,
                        alpha=0.8,
                        edgecolors='#8B0000',  # Dark red border
                        linewidths=0.8,
                        marker='o',
                        zorder=5
                    )
                    
                    # Add station name and delay percentage labels for high delay stations
                    for _, station in high_delay.iterrows():
                        label_text = f"{station['stationName']} ({station['delay_percentage']:.1f}%)"
                        ax.annotate(
                            label_text,
                            xy=(station['longitude'], station['latitude']),
                            xytext=(5, 5),  # Offset the text slightly
                            textcoords='offset points',
                            fontsize=8,
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
                        c='#FFD700',  # Gold yellow
                        s=35,
                        alpha=0.8,
                        edgecolors='#DAA520',  # Goldenrod border
                        linewidths=0.8,
                        marker='o',
                        zorder=5
                    )
                
                # Plot low delay stations in green
                if not low_delay.empty:
                    ax.scatter(
                        low_delay['longitude'], 
                        low_delay['latitude'],
                        c='#2E8B57',  # Sea green
                        s=35,
                        alpha=0.8,
                        edgecolors='#006400',  # Dark green border
                        linewidths=0.8,
                        marker='o',
                        zorder=5
                    )
                
                # Show statistics (excluding insufficient data stations)
                passenger_count = df_stations['passengerTraffic'].sum()
                non_passenger_count = len(df_stations) - passenger_count
                stations_shown = len(high_delay) + len(medium_delay) + len(low_delay)
                st.info(f"🔴 High Delay (≥15%): {len(high_delay):,} | "
                       f"🟡 Medium Delay (5-15%): {len(medium_delay):,} | "
                       f"🟢 Low Delay (<5%): {len(low_delay):,} | "
                       f"Total Shown: {stations_shown:,} stations")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load delay statistics: {e}. Showing all stations in gray.")
                # Fallback to gray for all passenger stations
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
                passenger_count = df_stations['passengerTraffic'].sum()
                non_passenger_count = len(df_stations) - passenger_count
                st.info(f"⚪ Passenger Stations Shown: {passenger_count:,} | Non-Passenger Stations Hidden: {non_passenger_count:,}")
        else:
            st.warning(f"⚠️ Delay statistics file not found at: {delay_stats_file}. Showing all stations in gray.")
            # Fallback to gray for all passenger stations
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
            passenger_count = df_stations['passengerTraffic'].sum()
            non_passenger_count = len(df_stations) - passenger_count
            st.info(f"⚪ Passenger Stations Shown: {passenger_count:,} | Non-Passenger Stations Hidden: {non_passenger_count:,}")
    else:
        # Fallback to original single color if passengerTraffic column doesn't exist
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
        st.warning("⚠️ 'passengerTraffic' column not found - showing all stations in gray")
    
    # Add axis labels
    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold', color='#333333')
    
    # Update title to reflect delay-based categorization
    if 'passengerTraffic' in df_stations.columns:
        passenger_count = df_stations['passengerTraffic'].sum()
        title = f'Finnish Railway Network'
    else:
        title = f'Finnish Railway Network - {len(df_stations)} Train Stations'
    
  
    ax.set_title(title, fontsize=16, fontweight='bold', color='#333333', pad=20)
    
    # Style the tick labels
    ax.tick_params(axis='both', labelsize=11, colors='#333333')
    
    # Keep spines visible but styled
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    # Add legend
    legend_elements = []
    
    # Add station legend items based on whether passengerTraffic column exists and delay data
    if 'passengerTraffic' in df_stations.columns:
        # Check if delay statistics were loaded successfully
        delay_stats_file = "data/viewers/delay_maps/stations_x_delays.csv"
        if os.path.exists(delay_stats_file):
            try:
                # If delay data exists, show delay percentage categories (without insufficient data)
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', label='High Delay (≥15%)',
                              markerfacecolor='#DC143C', markersize=10, markeredgecolor='#8B0000')
                )
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', label='Medium Delay (5-15%)',
                              markerfacecolor='#FFD700', markersize=10, markeredgecolor='#DAA520')
                )
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', label='Low Delay (<5%)',
                              markerfacecolor='#2E8B57', markersize=10, markeredgecolor='#006400')
                )
            except:
                # Fallback to single color
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', label='Passenger Stations',
                              markerfacecolor='#808080', markersize=10, markeredgecolor='#696969')
                )
        else:
            # Fallback to single color if no delay data
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label='Passenger Stations',
                          markerfacecolor='#808080', markersize=10, markeredgecolor='#696969')
            )
    else:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label='Train Stations',
                      markerfacecolor='#808080', markersize=10, markeredgecolor='#696969')
        )
    
    if df_graph is not None and connections_drawn > 0:
        legend_elements.append(
            plt.Line2D([0], [0], color='#4169E1', linewidth=4, alpha=0.6, label='Railway Connections')
        )
    
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#cccccc')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#666666')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Optional: Show station details
    with st.expander("🔍 View Station Details", expanded=False):
        # Display columns for the table
        display_columns = ['stationShortCode', 'stationName', 'latitude', 'longitude']
        
        # Add passengerTraffic if it exists
        if 'passengerTraffic' in df_stations.columns:
            display_columns.append('passengerTraffic')
        
        # Filter for columns that exist
        available_columns = [col for col in display_columns if col in df_stations.columns]
        
        if available_columns:
            st.dataframe(
                df_stations[available_columns].sort_values('stationName' if 'stationName' in available_columns else 'stationShortCode'),
                use_container_width=True,
                height=400
            )
        else:
            st.warning("⚠️ Station details columns not found")
    
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
    # If no station data, just show the map
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