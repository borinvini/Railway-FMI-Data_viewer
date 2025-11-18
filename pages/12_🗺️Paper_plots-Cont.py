import streamlit as st
import pandas as pd
import os
from config.const import VIEWER_FOLDER_NAME

# Page configuration
st.set_page_config(
    page_title="Weather Station Analysis",
    page_icon="⏱️",
    layout="wide"
)

st.title("⏱️ Weather Station Analysis")

st.markdown("""
This page provides comprehensive analysis of Environmental Monitoring Stations (EMS), including:
- Measurement intervals (1 minute vs 10 minutes)
- Weather features availability by station
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

def analyze_weather_features_by_station(df):
    """
    Analyze which weather features each station measures.
    
    Args:
        df: DataFrame with weather observations
    
    Returns:
        DataFrame with stations as rows and weather features as columns (Yes/No values)
    """
    # Identify weather feature columns (exclude metadata columns)
    metadata_columns = {'timestamp', 'station_name', 'station_id', 'latitude', 'longitude', 
                       'fmisid', 'stationName', 'index', 'Unnamed: 0'}
    
    # Get all columns that are not metadata
    weather_features = [col for col in df.columns if col not in metadata_columns]
    
    # Create results dictionary
    results = {}
    
    # Group by station_name
    for station_name, group in df.groupby('station_name'):
        station_measures = {}
        
        for feature in weather_features:
            # Check if the feature column has any non-null values for this station
            has_measurements = group[feature].notna().any()
            station_measures[feature] = 'Yes' if has_measurements else 'No'
        
        results[station_name] = station_measures
    
    # Convert to DataFrame for easy display
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'Station Name'
    
    return results_df

# ========== SECTION 1: MEASUREMENT INTERVAL ANALYSIS ==========
st.header("📊 Section 1: Measurement Interval Analysis")
st.markdown("Analyzing measurement intervals (1 minute vs 10 minutes) for each EMS station.")

try:
    # Check if file exists
    if not os.path.exists(file_path):
        st.error(f"❌ File not found at: `{file_path}`")
        st.info("💡 Make sure the file path is correct and the file exists.")
    else:
        # Load the data
        with st.spinner("Loading weather data..."):
            df_weather = pd.read_csv(file_path)
        
        st.success(f"✅ Successfully loaded {len(df_weather):,} records from the weather file!")
        
        # Display basic info
        st.subheader("📋 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(df_weather):,}")
        
        with col2:
            if 'station_name' in df_weather.columns:
                unique_stations = df_weather['station_name'].nunique()
                st.metric("Number of Stations", unique_stations)
        
        with col3:
            if 'timestamp' in df_weather.columns:
                df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'], errors='coerce')
                time_span = df_weather['timestamp'].max() - df_weather['timestamp'].min()
                st.metric("Time Span", f"{time_span.days} days")
        
        # Perform interval analysis
        st.markdown("---")
        st.subheader("⏱️ Measurement Intervals by Station")
        
        with st.spinner("Analyzing measurement intervals..."):
            interval_results = analyze_station_intervals(df_weather)
        
        # Convert results to DataFrame for display
        interval_df = pd.DataFrame.from_dict(interval_results, orient='index')
        interval_df.index.name = 'Station Name'
        interval_df = interval_df.reset_index()
        
        # Display the results
        st.dataframe(interval_df, use_container_width=True)
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            one_min_count = sum(1 for v in interval_results.values() if v['interval_type'] == '1 minute')
            st.metric("1-Minute Interval Stations", one_min_count)
        
        with col2:
            ten_min_count = sum(1 for v in interval_results.values() if v['interval_type'] == '10 minutes')
            st.metric("10-Minute Interval Stations", ten_min_count)
        
        with col3:
            other_count = len(interval_results) - one_min_count - ten_min_count
            st.metric("Other Interval Stations", other_count)

except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    st.exception(e)

# ========== SECTION 2: WEATHER FEATURES BY STATION ==========
st.markdown("---")
st.header("🌡️ Section 2: Weather Features Measurement by Station")
st.markdown("""
This section shows which weather features each station measures. 
- **Yes**: Station has at least one non-empty measurement for this feature
- **No**: Station has no measurements for this feature (all values are empty/null)
""")

try:
    if os.path.exists(file_path):
        # Load the data if not already loaded
        if 'df_weather' not in locals():
            with st.spinner("Loading weather data..."):
                df_weather = pd.read_csv(file_path)
        
        st.subheader("🔍 Weather Features Availability")
        
        with st.spinner("Analyzing weather features by station..."):
            features_df = analyze_weather_features_by_station(df_weather)
        
        # Display the table
        st.dataframe(features_df, use_container_width=True)
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📊 Feature Coverage Summary")
        
        # Calculate summary statistics
        total_stations = len(features_df)
        
        # Count how many stations measure each feature
        feature_coverage = {}
        for col in features_df.columns:
            yes_count = (features_df[col] == 'Yes').sum()
            feature_coverage[col] = {
                'Stations Measuring': yes_count,
                'Coverage %': f"{(yes_count / total_stations * 100):.1f}%"
            }
        
        coverage_df = pd.DataFrame.from_dict(feature_coverage, orient='index')
        coverage_df.index.name = 'Weather Feature'
        coverage_df = coverage_df.reset_index()
        
        st.dataframe(coverage_df, use_container_width=True)
        
        # Download options
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_features = features_df.to_csv()
            st.download_button(
                label="📥 Download Weather Features Table (CSV)",
                data=csv_features,
                file_name="weather_features_by_station.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_coverage = coverage_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feature Coverage Summary (CSV)",
                data=csv_coverage,
                file_name="feature_coverage_summary.csv",
                mime="text/csv"
            )
        
    else:
        st.info("ℹ️ Please load the weather data file first to analyze weather features by station.")

except Exception as e:
    st.error(f"❌ An error occurred while analyzing weather features: {e}")
    st.exception(e)