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
from config.const import FMI_BBOX

# Page configuration
st.set_page_config(
    page_title="Station Delay Statistics",
    page_icon="📊",
    layout="wide"
)

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

def process_matched_data_files(df_stations):
    """Process all matched data files and count delays and trains per station"""
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
    
    # Create a dictionary for faster lookups
    station_stats = {code: {'delays': 0, 'trains': 0} for code in df_stations['stationShortCode']}
    
    for file_idx, file_path in enumerate(matched_files):
        file_name = os.path.basename(file_path)
        status_text.text(f"Processing file {file_idx + 1}/{total_files}: {file_name}")
        
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
                        
                        # Track which stations this train visited (to avoid double counting)
                        visited_stations = set()
                        
                        # Process each station in the timetable
                        for station_entry in timetable_data:
                            if isinstance(station_entry, dict):
                                station_code = station_entry.get('stationShortCode')
                                
                                if station_code and station_code in station_stats:
                                    # Count this station only once per train
                                    if station_code not in visited_stations:
                                        station_stats[station_code]['trains'] += 1
                                        visited_stations.add(station_code)
                                    
                                    # Check for delays (>= 5 minutes)
                                    delay_offset = station_entry.get('differenceInMinutes_eachStation_offset')
                                    if delay_offset is not None and delay_offset >= 5:
                                        station_stats[station_code]['delays'] += 1
                        
                        processed_trains += 1
                        
                    except Exception as e:
                        # Skip trains with parsing errors
                        continue
                        
        except Exception as e:
            st.warning(f"⚠️ Error processing file {file_name}: {e}")
            continue
        
        # Update progress
        progress_bar.progress((file_idx + 1) / total_files)
    
    # Update the dataframe with the collected statistics
    for station_code, stats in station_stats.items():
        mask = df_stations['stationShortCode'] == station_code
        df_stations.loc[mask, 'total_of_delays'] = stats['delays']
        df_stations.loc[mask, 'total_of_trains'] = stats['trains']
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show completion message
    st.success(f"✅ Processed {total_files} files containing {processed_trains:,} train journeys")
    
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
    # Filter to stations with at least some traffic
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
    
    # SIDEBAR CONFIGURATION - only show if we have data
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
        
        # Define plot options
        plot_options = {
            "📊 Top Stations by Delays": "top_stations",
            "📋 Data Table Only": "data_only",
            "🗺️ Interactive Station Map": "station_map"
        }
        
        selected_plot = st.sidebar.radio(
            "Choose visualization:",
            options=list(plot_options.keys()),
            index=2  # Default to the new interactive map
        )
        
        # Get the plot key
        plot_key = plot_options[selected_plot]
        
        # Show current selection info
        st.sidebar.markdown(f"**Current View**: {selected_plot}")
        
        # MAIN CONTENT AREA - Display selected plot
        if plot_key != "data_only":
            st.subheader("📊 Station Delay Analysis")
            create_single_visualization(df_data, plot_key)
        
        # Download button
        st.markdown("---")
        create_download_button(df_data)
    
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

if __name__ == "__main__":
    main()