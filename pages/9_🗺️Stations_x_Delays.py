import streamlit as st
import pandas as pd
import os
import glob
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def create_visualizations(df_stations):
    """Create visualizations for the delay statistics"""
    # Filter to stations with at least some traffic
    df_viz = df_stations[df_stations['total_of_trains'] > 0].copy()
    
    if df_viz.empty:
        st.warning("No stations with train traffic found.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Top Stations", "Delay Distribution", "Percentage Analysis", "Station Map"])
    
    with tab1:
        st.subheader("Top 20 Stations by Total Delays")
        
        top_20 = df_viz.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(top_20)), top_20['total_of_delays'])
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels([f"{row['stationName']} ({row['stationShortCode']})" 
                           if row['stationName'] else row['stationShortCode']
                           for _, row in top_20.iterrows()])
        ax.set_xlabel('Total Number of Delays (≥5 minutes)')
        ax.set_title('Top 20 Stations by Total Delays')
        
        # Color bars based on delay count
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top_20)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (_, row) in enumerate(top_20.iterrows()):
            ax.text(row['total_of_delays'] + 10, i, f"{row['total_of_delays']:,}", 
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Delay Distribution Analysis")
        
        # Create histogram of delays
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of total delays
        ax1.hist(df_viz['total_of_delays'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Total Delays per Station')
        ax1.set_ylabel('Number of Stations')
        ax1.set_title('Distribution of Total Delays')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of delay percentage
        ax2.hist(df_viz['delay_percentage'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Delay Percentage (%)')
        ax2.set_ylabel('Number of Stations')
        ax2.set_title('Distribution of Delay Percentages')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show distribution statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Delays per Station", f"{df_viz['total_of_delays'].mean():.1f}")
        with col2:
            st.metric("Median Delays per Station", f"{df_viz['total_of_delays'].median():.0f}")
        with col3:
            st.metric("Mean Delay %", f"{df_viz['delay_percentage'].mean():.1f}%")
        with col4:
            st.metric("Median Delay %", f"{df_viz['delay_percentage'].median():.1f}%")
    
    with tab3:
        st.subheader("Stations with Highest Delay Percentages")
        
        # Filter stations with significant traffic (at least 100 trains)
        df_significant = df_viz[df_viz['total_of_trains'] >= 100].copy()
        
        if not df_significant.empty:
            top_20_pct = df_significant.nlargest(20, 'delay_percentage')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bars = ax.barh(range(len(top_20_pct)), top_20_pct['delay_percentage'])
            ax.set_yticks(range(len(top_20_pct)))
            ax.set_yticklabels([f"{row['stationName']} ({row['stationShortCode']})" 
                               if row['stationName'] else row['stationShortCode']
                               for _, row in top_20_pct.iterrows()])
            ax.set_xlabel('Delay Percentage (%)')
            ax.set_title('Top 20 Stations by Delay Percentage (min. 100 trains)')
            
            # Color bars
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top_20_pct)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add value labels
            for i, (_, row) in enumerate(top_20_pct.iterrows()):
                ax.text(row['delay_percentage'] + 0.2, i, 
                       f"{row['delay_percentage']:.1f}%", 
                       va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No stations with at least 100 trains found.")
    
    with tab4:
        st.subheader("Station Delay Map")
        
        # Check if we have coordinate data
        if 'latitude' in df_viz.columns and 'longitude' in df_viz.columns:
            # Create scatter plot on map-like visualization
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Filter out stations without coordinates
            df_map = df_viz.dropna(subset=['latitude', 'longitude'])
            
            if not df_map.empty:
                # Create scatter plot with size based on total trains and color based on delay percentage
                scatter = ax.scatter(
                    df_map['longitude'], 
                    df_map['latitude'],
                    s=df_map['total_of_trains'] / 10,  # Size based on traffic
                    c=df_map['delay_percentage'],       # Color based on delay %
                    cmap='RdYlGn_r',
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Delay Percentage (%)', rotation=270, labelpad=20)
                
                # Add labels for top delay stations
                top_delay_stations = df_map.nlargest(10, 'total_of_delays')
                for _, row in top_delay_stations.iterrows():
                    ax.annotate(
                        row['stationShortCode'],
                        xy=(row['longitude'], row['latitude']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        fontweight='bold'
                    )
                
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('Station Delays Across Finland\n(Size = Traffic Volume, Color = Delay %)')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("💡 Bubble size represents total train traffic, color represents delay percentage")
            else:
                st.warning("No stations with valid coordinates found.")
        else:
            st.info("📍 Coordinate data not available for map visualization.")

# Main application
def main():
    st.title("📊 Station Delay Statistics Calculator")
    st.markdown("""
    This tool analyzes all matched train data to calculate the total number of delays 
    and trains for each station in the Finnish railway network.
    
    **Delay Definition**: A delay is counted when `differenceInMinutes_eachStation_offset ≥ 5 minutes`
    """)
    
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
            
            # Display summary statistics
            st.subheader("📈 Summary Statistics")
            
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
            
            # Display the full table
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
            
            # Create visualizations
            st.subheader("📊 Visualizations")
            create_visualizations(df_stations)
            
            # Download button for the full results
            st.subheader("💾 Download Results")
            
            csv = df_stations.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Station Delay Statistics CSV",
                data=csv,
                file_name="station_delay_statistics_full.csv",
                mime="text/csv"
            )
            
        else:
            st.error("Failed to load station metadata. Please check the file path and format.")
    
    else:
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
        
        # Check if results already exist
        existing_results_path = "data/viewers/delay_maps/stations_x_delays.csv"
        if os.path.exists(existing_results_path):
            st.success(f"✅ Previous results found at: `{existing_results_path}`")
            
            if st.checkbox("Load existing results"):
                try:
                    df_existing = pd.read_csv(existing_results_path)
                    
                    # Try to load station names from metadata
                    metadata_path = "data/viewers/metadata/metadata_train_stations.csv"
                    if os.path.exists(metadata_path):
                        df_metadata = pd.read_csv(metadata_path)
                        df_existing = pd.merge(
                            df_existing, 
                            df_metadata[['stationShortCode', 'stationName']], 
                            on='stationShortCode', 
                            how='left'
                        )
                    
                    # Calculate percentage
                    df_existing = calculate_delay_statistics(df_existing)
                    
                    st.subheader("📋 Existing Station Delay Data")
                    st.dataframe(
                        df_existing.style.format({
                            'total_of_delays': '{:,}',
                            'total_of_trains': '{:,}',
                            'delay_percentage': '{:.2f}%'
                        }),
                        use_container_width=True,
                        height=500
                    )
                    
                    # Show summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Stations", len(df_existing))
                    with col2:
                        st.metric("Total Delays", f"{df_existing['total_of_delays'].sum():,}")
                    with col3:
                        st.metric("Total Trains", f"{df_existing['total_of_trains'].sum():,}")
                        
                except Exception as e:
                    st.error(f"Error loading existing results: {e}")

if __name__ == "__main__":
    main()