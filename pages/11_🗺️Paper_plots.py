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