import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import os

# Page configuration
st.set_page_config(
    page_title="Finland Train Stations Map",
    page_icon="🗺️",
    layout="wide"
)

st.title("🇫🇮 Finland Train Stations Map")

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

# Load data
with st.spinner("Loading map and station data..."):
    finland = load_map()
    df_stations = load_train_stations()

if df_stations is not None:
    # Display statistics
    st.markdown("### 📊 Dataset Information")
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
    
    # Plot train stations
    ax.scatter(
        df_stations['longitude'], 
        df_stations['latitude'],
        c='#FF4B4B',              # Streamlit red color
        s=20,                      # Size of markers (reduced from 50)
        alpha=0.7,                 # Transparency
        edgecolors='#8B0000',      # Dark red border
        linewidths=0.5,            # Border width (reduced from 1)
        marker='o',
        label='Train Stations',
        zorder=5                   # Ensure stations are on top
    )
    
    # Add axis labels
    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold', color='#333333')
    ax.set_title(f'Finnish Railway Network - {len(df_stations)} Train Stations', 
                 fontsize=16, fontweight='bold', color='#333333', pad=20)
    
    # Style the tick labels
    ax.tick_params(axis='both', labelsize=11, colors='#333333')
    
    # Keep spines visible but styled
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    # Add legend
    legend = ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
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