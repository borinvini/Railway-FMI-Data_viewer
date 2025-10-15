import streamlit as st
import pandas as pd
import os
from config.const import VIEWER_FOLDER_NAME

# Page configuration
st.set_page_config(
    page_title="Weather Station Interval Analysis",
    page_icon="⏱️",
    layout="wide"
)

st.title("⏱️ Weather Station Measurement Interval Analysis")

st.markdown("""
This page analyzes the measurement intervals (1 minute vs 10 minutes) for each EMS (Environmental Monitoring Station).
""")

# File path
file_path = os.path.join("data", "viewers", "weather_data", "fmi_weather_observations_2024_01.csv")

def analyze_station_intervals(df):
    """
    Analyze the measurement intervals for each station.
    
    Args:
        df: DataFrame with 'station_name' and 'timestamp' columns
    
    Returns:
        Dictionary with station analysis results
    """
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Sort by station and timestamp
    df = df.sort_values(['station_name', 'timestamp'])
    
    station_results = {}
    
    # Group by station_name
    for station_name, group in df.groupby('station_name'):
        # Calculate time differences between consecutive measurements
        group = group.sort_values('timestamp')
        time_diffs = group['timestamp'].diff()
        
        # Remove NaN (first row has no previous measurement)
        time_diffs = time_diffs.dropna()
        
        if len(time_diffs) == 0:
            # Only one measurement for this station
            station_results[station_name] = {
                'interval_type': 'Unknown (single measurement)',
                'avg_interval_minutes': None,
                'measurement_count': len(group)
            }
            continue
        
        # Convert to minutes
        intervals_minutes = time_diffs.dt.total_seconds() / 60
        
        # Calculate average interval
        avg_interval = intervals_minutes.mean()
        
        # Determine interval type based on average
        # Allow some tolerance (e.g., 1 min could be 0.5-2 min, 10 min could be 8-12 min)
        if avg_interval < 3:
            interval_type = '1 minute'
        elif 3 <= avg_interval < 15:
            interval_type = '10 minutes'
        else:
            interval_type = f'Other ({avg_interval:.1f} min)'
        
        station_results[station_name] = {
            'interval_type': interval_type,
            'avg_interval_minutes': avg_interval,
            'measurement_count': len(group)
        }
    
    return station_results

# Load data
st.subheader("📂 Loading Data")

try:
    # Check if file exists
    if not os.path.exists(file_path):
        st.error(f"❌ File not found at: `{file_path}`")
        st.info("💡 Make sure the file path is correct and the file exists.")
        st.stop()
    
    # Load the CSV file
    with st.spinner("Loading weather data..."):
        df = pd.read_csv(file_path)
    
    st.success(f"✅ Successfully loaded {len(df):,} records from the file")
    
    # Display basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        unique_stations = df['station_name'].nunique()
        st.metric("Unique Stations", f"{unique_stations:,}")
    
    with col3:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            date_range = df['timestamp'].max() - df['timestamp'].min()
            st.metric("Date Range", f"{date_range.days} days")
    
    # Show sample data
    with st.expander("🔍 View Sample Data (first 10 rows)", expanded=False):
        st.dataframe(df.head(10))
    
    st.markdown("---")
    
    # Analyze intervals
    st.subheader("📊 Interval Analysis")
    
    with st.spinner("Analyzing measurement intervals for each station..."):
        # Extract only needed columns
        df_analysis = df[['station_name', 'timestamp']].copy()
        
        # Run analysis
        station_results = analyze_station_intervals(df_analysis)
    
    # Count stations by interval type
    interval_counts = {
        '1 minute': 0,
        '10 minutes': 0,
        'Other': 0,
        'Unknown': 0
    }
    
    for station, data in station_results.items():
        interval_type = data['interval_type']
        if '1 minute' in interval_type:
            interval_counts['1 minute'] += 1
        elif '10 minutes' in interval_type:
            interval_counts['10 minutes'] += 1
        elif 'Unknown' in interval_type:
            interval_counts['Unknown'] += 1
        else:
            interval_counts['Other'] += 1
    
    # Display summary table
    st.subheader("📈 Summary Statistics")
    
    summary_data = {
        'Metric': [
            'Total EMS Stations',
            'Stations with 1-minute interval',
            'Stations with 10-minute interval',
            'Stations with other intervals',
            'Stations with unknown interval'
        ],
        'Count': [
            len(station_results),
            interval_counts['1 minute'],
            interval_counts['10 minutes'],
            interval_counts['Other'],
            interval_counts['Unknown']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display as a nice table with styling
    st.dataframe(
        summary_df.style.set_properties(**{
            'text-align': 'left',
            'font-size': '14px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('font-size', '16px'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('padding', '10px')]}
        ]),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Display detailed results
    st.subheader("📋 Detailed Station Information")
    
    # Create DataFrame from results
    detailed_data = []
    for station, data in station_results.items():
        detailed_data.append({
            'Station Name': station,
            'Interval Type': data['interval_type'],
            'Average Interval (minutes)': f"{data['avg_interval_minutes']:.2f}" if data['avg_interval_minutes'] else 'N/A',
            'Total Measurements': data['measurement_count']
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Sort by interval type for better viewing
    detailed_df = detailed_df.sort_values(['Interval Type', 'Station Name'])
    
    # Add filters
    col1, col2 = st.columns(2)
    
    with col1:
        interval_filter = st.multiselect(
            "Filter by Interval Type",
            options=detailed_df['Interval Type'].unique(),
            default=detailed_df['Interval Type'].unique()
        )
    
    with col2:
        search_station = st.text_input("Search Station Name", "")
    
    # Apply filters
    filtered_detailed = detailed_df[detailed_df['Interval Type'].isin(interval_filter)]
    
    if search_station:
        filtered_detailed = filtered_detailed[
            filtered_detailed['Station Name'].str.contains(search_station, case=False, na=False)
        ]
    
    st.dataframe(
        filtered_detailed,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button for detailed results
    csv = filtered_detailed.to_csv(index=False)
    st.download_button(
        label="📥 Download Detailed Results as CSV",
        data=csv,
        file_name="station_intervals_analysis.csv",
        mime="text/csv"
    )
    
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.info("💡 Please check if the file path is correct.")
except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    st.exception(e)