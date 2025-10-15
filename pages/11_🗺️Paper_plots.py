import numpy as np
from sklearn.preprocessing import RobustScaler
import streamlit as st
import pandas as pd
import os
from config.const import VIEWER_FOLDER_NAME

# Page configuration
st.set_page_config(
    page_title="Train Station Statistics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Finnish Railway Station Statistics")

st.markdown("""
This page provides comprehensive statistics about train stations in the Finnish railway system,
including geographical distribution and passenger traffic analysis.
""")

# Load the train stations metadata
@st.cache_data
def load_station_metadata():
    """Load train station metadata from CSV file"""
    metadata_path = os.path.join(VIEWER_FOLDER_NAME, "metadata", "metadata_train_stations.csv")
    
    if not os.path.exists(metadata_path):
        st.error(f"⚠️ Station metadata file not found at: {metadata_path}")
        return None
    
    try:
        df = pd.read_csv(metadata_path)
        return df
    except Exception as e:
        st.error(f"❌ Error loading station metadata: {e}")
        return None

# Load data
with st.spinner("Loading station metadata..."):
    df_stations = load_station_metadata()

if df_stations is not None:
    # Display basic info
    st.markdown("---")
    st.subheader("📋 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df_stations):,}")
    
    with col2:
        st.metric("Total Columns", f"{len(df_stations.columns)}")
    
    with col3:
        missing_values = df_stations.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_values:,}")
    
    # Show column names
    with st.expander("🔍 View Column Names", expanded=False):
        st.write("**Available columns:**")
        for col in df_stations.columns:
            st.write(f"- {col}")
    
    st.markdown("---")
    
    # ANALYSIS 1: Total Stations Count
    st.subheader("🚉 Total Stations in Finnish Railway System")
    
    total_stations = len(df_stations)
    
    st.metric(
        label="Total Number of Stations",
        value=f"{total_stations:,}",
        help="Total count of all stations in the dataset"
    )
    
    st.markdown("---")
    
    # ANALYSIS 2: Stations by Country Code
    st.subheader("🌍 Stations Distribution by Country")
    
    if 'countryCode' in df_stations.columns:
        # Count stations by country
        country_counts = df_stations['countryCode'].value_counts().reset_index()
        country_counts.columns = ['Country Code', 'Number of Stations']
        
        # Calculate percentages
        country_counts['Percentage (%)'] = (country_counts['Number of Stations'] / total_stations * 100).round(2)
        
        # Sort by number of stations descending
        country_counts = country_counts.sort_values('Number of Stations', ascending=False)
        
        # Display in a nice table
        st.dataframe(
            country_counts.style.format({
                'Number of Stations': '{:,}',
                'Percentage (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Display key insights
        col1, col2 = st.columns(2)
        
        with col1:
            finnish_stations = country_counts[country_counts['Country Code'] == 'FI']['Number of Stations'].iloc[0] if 'FI' in country_counts['Country Code'].values else 0
            finnish_percentage = country_counts[country_counts['Country Code'] == 'FI']['Percentage (%)'].iloc[0] if 'FI' in country_counts['Country Code'].values else 0
            
            st.info(f"🇫🇮 **Finnish Stations (FI)**: {finnish_stations:,} stations ({finnish_percentage:.2f}%)")
        
        with col2:
            num_countries = len(country_counts)
            st.info(f"🌐 **Total Countries**: {num_countries}")
        
    else:
        st.warning("⚠️ 'countryCode' column not found in the dataset")
    
    st.markdown("---")
    
    # ANALYSIS 3: Passenger Traffic Stations
    st.subheader("🚂 Passenger Traffic Analysis")
    
    if 'passengerTraffic' in df_stations.columns:
        # Count passenger traffic stations
        passenger_counts = df_stations['passengerTraffic'].value_counts().reset_index()
        passenger_counts.columns = ['Passenger Traffic', 'Number of Stations']
        
        # Calculate percentages
        passenger_counts['Percentage (%)'] = (passenger_counts['Number of Stations'] / total_stations * 100).round(2)
        
        # Sort by passenger traffic (True first)
        passenger_counts = passenger_counts.sort_values('Passenger Traffic', ascending=False)
        
        # Replace True/False with more descriptive text
        passenger_counts['Passenger Traffic'] = passenger_counts['Passenger Traffic'].map({
            True: '✅ Yes (Passenger Station)',
            False: '❌ No (Non-Passenger Station)'
        })
        
        # Display in a nice table
        st.dataframe(
            passenger_counts.style.format({
                'Number of Stations': '{:,}',
                'Percentage (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Display key insights
        passenger_stations = df_stations[df_stations['passengerTraffic'] == True].shape[0]
        non_passenger_stations = df_stations[df_stations['passengerTraffic'] == False].shape[0]
        passenger_percentage = (passenger_stations / total_stations * 100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"✅ **Passenger Stations**: {passenger_stations:,} ({passenger_percentage:.2f}%)")
        
        with col2:
            st.info(f"❌ **Non-Passenger Stations**: {non_passenger_stations:,} ({100-passenger_percentage:.2f}%)")
        
    else:
        st.warning("⚠️ 'passengerTraffic' column not found in the dataset")
    
    st.markdown("---")
    
    # ANALYSIS 4: Station Types Distribution
    st.subheader("🏢 Station Types Distribution")
    
    if 'type' in df_stations.columns:
        # Count station types
        type_counts = df_stations['type'].value_counts().reset_index()
        type_counts.columns = ['Station Type', 'Number of Stations']
        
        # Calculate percentages
        type_counts['Percentage (%)'] = (type_counts['Number of Stations'] / total_stations * 100).round(2)
        
        # Sort by number of stations descending
        type_counts = type_counts.sort_values('Number of Stations', ascending=False)
        
        # Display in a nice table
        st.dataframe(
            type_counts.style.format({
                'Number of Stations': '{:,}',
                'Percentage (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Display key insights
        most_common_type = type_counts.iloc[0]
        st.info(f"📊 **Most Common Type**: {most_common_type['Station Type']} with {most_common_type['Number of Stations']:,} stations ({most_common_type['Percentage (%)']:.2f}%)")
        
    else:
        st.warning("⚠️ 'type' column not found in the dataset")
    
    st.markdown("---")
    
    # ANALYSIS 5: Combined Analysis - Passenger Traffic by Country
    st.subheader("🌍🚂 Passenger Traffic by Country")
    
    if 'countryCode' in df_stations.columns and 'passengerTraffic' in df_stations.columns:
        # Create cross-tabulation
        country_passenger_crosstab = pd.crosstab(
            df_stations['countryCode'], 
            df_stations['passengerTraffic'],
            margins=True,
            margins_name='Total'
        )
        
        # Rename columns for clarity
        country_passenger_crosstab.columns = ['Non-Passenger', 'Passenger', 'Total']
        
        # Calculate percentages for each country
        country_passenger_crosstab['Passenger %'] = (
            country_passenger_crosstab['Passenger'] / country_passenger_crosstab['Total'] * 100
        ).round(2)
        
        country_passenger_crosstab['Non-Passenger %'] = (
            country_passenger_crosstab['Non-Passenger'] / country_passenger_crosstab['Total'] * 100
        ).round(2)
        
        # Reset index to make country code a column
        country_passenger_crosstab = country_passenger_crosstab.reset_index()
        country_passenger_crosstab.columns.name = None
        country_passenger_crosstab = country_passenger_crosstab.rename(columns={'countryCode': 'Country Code'})
        
        # Sort by total descending (excluding the Total row)
        country_passenger_crosstab_display = country_passenger_crosstab[country_passenger_crosstab['Country Code'] != 'Total']
        country_passenger_crosstab_display = country_passenger_crosstab_display.sort_values('Total', ascending=False)
        
        # Add the Total row at the end
        total_row = country_passenger_crosstab[country_passenger_crosstab['Country Code'] == 'Total']
        country_passenger_crosstab_display = pd.concat([country_passenger_crosstab_display, total_row], ignore_index=True)
        
        # Display the table
        st.dataframe(
            country_passenger_crosstab_display.style.format({
                'Non-Passenger': '{:,}',
                'Passenger': '{:,}',
                'Total': '{:,}',
                'Passenger %': '{:.2f}%',
                'Non-Passenger %': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
    else:
        st.warning("⚠️ Either 'countryCode' or 'passengerTraffic' column not found in the dataset")
    
    st.markdown("---")
    
    # ANALYSIS 6: Data Quality Check
    st.subheader("🔍 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Values by Column")
        missing_by_column = df_stations.isnull().sum().reset_index()
        missing_by_column.columns = ['Column', 'Missing Count']
        missing_by_column['Missing %'] = (missing_by_column['Missing Count'] / len(df_stations) * 100).round(2)
        missing_by_column = missing_by_column[missing_by_column['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if not missing_by_column.empty:
            st.dataframe(
                missing_by_column.style.format({
                    'Missing Count': '{:,}',
                    'Missing %': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("✅ No missing values found in any column!")
    
    with col2:
        st.markdown("#### Coordinate Completeness")
        
        has_lat = df_stations['latitude'].notna().sum() if 'latitude' in df_stations.columns else 0
        has_lon = df_stations['longitude'].notna().sum() if 'longitude' in df_stations.columns else 0
        has_both = ((df_stations['latitude'].notna()) & (df_stations['longitude'].notna())).sum() if 'latitude' in df_stations.columns and 'longitude' in df_stations.columns else 0
        
        coord_data = pd.DataFrame({
            'Coordinate Status': [
                'Has Latitude',
                'Has Longitude',
                'Has Both Coordinates',
                'Missing Coordinates'
            ],
            'Count': [
                has_lat,
                has_lon,
                has_both,
                total_stations - has_both
            ]
        })
        
        coord_data['Percentage (%)'] = (coord_data['Count'] / total_stations * 100).round(2)
        
        st.dataframe(
            coord_data.style.format({
                'Count': '{:,}',
                'Percentage (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Sample Data Preview
    st.subheader("📄 Sample Data Preview")
    
    st.markdown("**First 10 stations:**")
    st.dataframe(df_stations.head(10), use_container_width=True)
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Data")
    
    csv = df_stations.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Station Metadata CSV",
        data=csv,
        file_name="train_station_statistics.csv",
        mime="text/csv"
    )
    
else:
    st.error("❌ Could not load station metadata. Please ensure the file exists at the correct location.")
    st.info(f"Expected location: `data/viewers/metadata/metadata_train_stations.csv`")

# ========== NEW SECTION: TRAIN ROUTE DELAY TIME SERIES ==========
st.markdown("---")
st.title("📈 Train Route Delay Analysis")

st.markdown("""
### Delay Time Series Visualization
This section shows how delays evolve over time along a specific train route.
The plot displays the delay (in minutes) at each station stop throughout the journey.
""")

# Load train route data
@st.cache_data
def load_route_data():
    """Load train route data from CSV file"""
    route_path = "pages/route.csv"
    
    if not os.path.exists(route_path):
        st.error(f"⚠️ Route data file not found at: {route_path}")
        return None
    
    try:
        df = pd.read_csv(route_path)
        return df
    except Exception as e:
        st.error(f"❌ Error loading route data: {e}")
        return None

# Load the route data
with st.spinner("Loading train route data..."):
    df_route = load_route_data()

if df_route is not None:
    # Display basic route information
    st.markdown("---")
    st.subheader("📋 Route Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stops", f"{len(df_route):,}")
    
    with col2:
        if 'stationName' in df_route.columns:
            first_station = df_route['stationName'].iloc[0]
            st.metric("Origin", first_station)
    
    with col3:
        if 'stationName' in df_route.columns:
            last_station = df_route['stationName'].iloc[-1]
            st.metric("Destination", last_station)
    
    with col4:
        if 'differenceInMinutes' in df_route.columns:
            avg_delay = df_route['differenceInMinutes'].mean()
            st.metric("Avg Delay", f"{avg_delay:.1f} min")
    
    st.markdown("---")
    
    # Process the data for visualization
    if 'scheduledTime' in df_route.columns and 'differenceInMinutes' in df_route.columns:
        # Parse scheduledTime and extract time component
        df_route['scheduledTime_parsed'] = pd.to_datetime(df_route['scheduledTime'])
        df_route['time_only'] = df_route['scheduledTime_parsed'].dt.strftime('%H:%M')
        
        # Create the time series plot
        st.subheader("⏱️ Delay Evolution Along Route")
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot the delay line
        ax.plot(
            df_route['scheduledTime_parsed'],
            df_route['differenceInMinutes'],
            marker='o',
            linewidth=2,
            markersize=8,
            color='#DC143C',
            label='Delay (minutes)'
        )
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='#2E8B57', linestyle='--', linewidth=1.5, alpha=0.7, label='On Time')
        
        # Customize the plot
        ax.set_xlabel('Scheduled Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Delay (minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Train Delay Over Time Along Route', fontsize=14, fontweight='bold', pad=20)
        
        # Format x-axis to show only time
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add legend
        ax.legend(loc='best', fontsize=10)
        
        # Color code the background based on delay severity
        y_min, y_max = ax.get_ylim()
        
        # Green background for on-time (y < 5)
        ax.axhspan(y_min, 5, alpha=0.1, color='green', zorder=0)
        
        # Yellow background for medium delay (5 <= y < 15)
        if y_max > 5:
            ax.axhspan(5, min(15, y_max), alpha=0.1, color='yellow', zorder=0)
        
        # Red background for high delay (y >= 15)
        if y_max > 15:
            ax.axhspan(15, y_max, alpha=0.1, color='red', zorder=0)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display delay statistics
        st.markdown("### 📊 Delay Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_delay = df_route['differenceInMinutes'].max()
            max_delay_station = df_route.loc[df_route['differenceInMinutes'].idxmax(), 'stationName'] if 'stationName' in df_route.columns else 'Unknown'
            st.metric(
                label="🔴 Maximum Delay",
                value=f"{max_delay} min",
                delta=f"at {max_delay_station}"
            )
        
        with col2:
            min_delay = df_route['differenceInMinutes'].min()
            min_delay_station = df_route.loc[df_route['differenceInMinutes'].idxmin(), 'stationName'] if 'stationName' in df_route.columns else 'Unknown'
            st.metric(
                label="🟢 Minimum Delay",
                value=f"{min_delay} min",
                delta=f"at {min_delay_station}"
            )
        
        with col3:
            median_delay = df_route['differenceInMinutes'].median()
            st.metric(
                label="📊 Median Delay",
                value=f"{median_delay:.1f} min"
            )
        
        with col4:
            delayed_stops = (df_route['differenceInMinutes'] >= 5).sum()
            delayed_pct = (delayed_stops / len(df_route) * 100)
            st.metric(
                label="⚠️ Delayed Stops (≥5 min)",
                value=f"{delayed_stops}",
                delta=f"{delayed_pct:.1f}%"
            )
        
        # Show detailed data table
        st.markdown("---")
        st.subheader("📋 Detailed Route Data")
        
        with st.expander("🔍 View All Stops and Delays", expanded=False):
            # Prepare display columns
            display_cols = ['stationName', 'stationShortCode', 'type', 'time_only', 
                          'differenceInMinutes', 'cancelled']
            
            # Filter to available columns
            display_cols = [col for col in display_cols if col in df_route.columns]
            
            # Style the dataframe
            st.dataframe(
                df_route[display_cols].style.format({
                    'differenceInMinutes': '{:,}',
                    'time_only': '{}',
                    'cancelled': '{}'
                }).background_gradient(
                    subset=['differenceInMinutes'],
                    cmap='RdYlGn_r',
                    vmin=0,
                    vmax=df_route['differenceInMinutes'].max()
                ),
                use_container_width=True,
                height=400
            )
        
        # Download option
        st.markdown("---")
        st.subheader("💾 Download Route Data")
        
        csv_route = df_route.to_csv(index=False)
        st.download_button(
            label="📥 Download Route Data CSV",
            data=csv_route,
            file_name="train_route_delays.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ Required columns 'scheduledTime' or 'differenceInMinutes' not found in the dataset")
    
else:
    st.warning("⚠️ Could not load route data. Please ensure the file exists at: `pages/route.csv`")

# ========== END OF NEW SECTION ==========

# ========== NEW SECTION: WEATHER TIME SERIES ==========
st.markdown("---")
st.title("🌦️ Weather Station Time Series Analysis")

st.markdown("""
This section displays time series data from an Environmental Monitoring Station (EMS).
All weather features are normalized using RobustScaler to enable comparison across different scales.
""")

# Load weather station data
@st.cache_data
def load_weather_data():
    """Load weather station time series data"""
    weather_path = "pages/EMS_example.csv"
    
    if not os.path.exists(weather_path):
        st.warning(f"⚠️ Weather data file not found at: {weather_path}")
        return None
    
    try:
        df = pd.read_csv(weather_path)
        
        # Remove unnamed columns (typically index columns from saved CSV)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"❌ Error loading weather data: {e}")
        return None

def plot_weather_time_series(df_weather):
    """Create time series plot for weather features"""
    
    # Get weather feature columns (exclude timestamp and station_name)
    weather_features = [col for col in df_weather.columns 
                       if col not in ['timestamp', 'station_name'] 
                       and df_weather[col].dtype in ['float64', 'int64']]
    
    if not weather_features:
        st.warning("No numeric weather features found in the dataset")
        return
    
    # Display available features
    st.markdown(f"**Weather Features ({len(weather_features)}):** {', '.join(weather_features)}")
    
    # Prepare data for scaling
    # Remove rows where ALL weather features are NaN
    df_clean = df_weather.dropna(subset=weather_features, how='all').copy()
    
    if df_clean.empty:
        st.warning("No valid weather data found after removing NaN values")
        return
    
    # Extract features for scaling
    X = df_clean[weather_features].values
    
    # Replace any remaining NaN with 0 for scaling
    X = np.nan_to_num(X, nan=0.0)
    
    # Apply RobustScaler to normalize all features to similar range
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a DataFrame with scaled values
    df_scaled = pd.DataFrame(X_scaled, columns=weather_features, index=df_clean.index)
    df_scaled['timestamp'] = df_clean['timestamp'].values
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot each weather feature
    colors = plt.cm.tab20(np.linspace(0, 1, len(weather_features)))
    
    for i, feature in enumerate(weather_features):
        ax.plot(df_scaled['timestamp'], df_scaled[feature], 
               label=feature, color=colors[i], linewidth=1.5, alpha=0.8)
    
    # Get the date from the first timestamp
    first_date = df_clean['timestamp'].iloc[0].date()
    
    # Set x-axis limits to show full 24-hour period (00:00 to 00:00 next day)
    start_time = pd.Timestamp(first_date)  # 00:00 of the first date
    end_time = start_time + pd.Timedelta(days=1)  # 00:00 of the next day
    ax.set_xlim(start_time, end_time)
    
    # Format x-axis to show only hours and minutes
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Show every 2 hours
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Labels and title
    ax.set_xlabel('Time (Hour:Minute)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scaled Values (RobustScaler)', fontsize=12, fontweight='bold')
    ax.set_title('Weather Features Time Series - Scaled for Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend INSIDE the plot area (upper right corner)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, df_clean, df_scaled

# Load weather data
with st.spinner("Loading weather station data..."):
    df_weather = load_weather_data()

if df_weather is not None:
    # Display weather data info
    st.subheader("📊 Weather Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df_weather):,}")
    
    with col2:
        if 'station_name' in df_weather.columns:
            station_name = df_weather['station_name'].iloc[0] if not df_weather.empty else "Unknown"
            st.metric("Station Name", station_name)
    
    with col3:
        if 'timestamp' in df_weather.columns:
            time_range = (df_weather['timestamp'].max() - df_weather['timestamp'].min())
            hours = time_range.total_seconds() / 3600
            st.metric("Time Range", f"{hours:.1f} hours")
    
    st.markdown("---")
    
    # Create and display the time series plot
    st.subheader("📈 Weather Features Time Series")
    
    with st.spinner("Generating time series plot with RobustScaler normalization..."):
        result = plot_weather_time_series(df_weather)
        
        if result:
            fig, df_clean, df_scaled = result
            
            # Display the plot
            st.pyplot(fig)
            
            # Add explanation
            st.info("""
            📊 **About this plot:**
            - All weather features are normalized using **RobustScaler** to bring them to a comparable range
            - RobustScaler uses median and interquartile range, making it robust to outliers
            - X-axis shows time in HH:MM format
            - Different colors represent different weather parameters
            """)
            
            # Display statistics about the data
            st.markdown("---")
            st.subheader("📋 Weather Data Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Data Summary**")
                weather_features = [col for col in df_weather.columns 
                                   if col not in ['timestamp', 'station_name'] 
                                   and df_weather[col].dtype in ['float64', 'int64']]
                st.dataframe(
                    df_clean[weather_features].describe().round(2),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Scaled Data Summary**")
                st.dataframe(
                    df_scaled[weather_features].describe().round(2),
                    use_container_width=True
                )
            
            # Sample data preview
            st.markdown("---")
            st.subheader("📄 Raw Data Preview")
            
            with st.expander("🔍 View First 20 Records", expanded=False):
                st.dataframe(df_weather.head(20), use_container_width=True)
            
            # Download option
            st.markdown("---")
            st.subheader("💾 Download Weather Data")
            
            csv_weather = df_weather.to_csv(index=False)
            st.download_button(
                label="📥 Download Weather Time Series CSV",
                data=csv_weather,
                file_name="weather_time_series.csv",
                mime="text/csv"
            )
            
else:
    st.info("""
    ℹ️ Weather time series data not available. 
    
    To view weather station analysis, please ensure the file `pages/EMS_example.csv` exists.
    """)