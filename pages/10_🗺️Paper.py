import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Page configuration
st.set_page_config(
    page_title="Finland Map",
    page_icon="🗺️",
    layout="wide"
)

st.title("🇫🇮 Finland")

# Load Finland map with high resolution
@st.cache_data
def load_map():
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    world = gpd.read_file(url)
    finland = world[world['NAME'] == 'Finland']
    return finland

# Load and plot
with st.spinner("Loading map..."):
    finland = load_map()

# Create a beautiful figure
fig, ax = plt.subplots(figsize=(12, 14), facecolor='#f5f5f5')  # Light gray background
ax.set_facecolor('#ffffff')  # White for the map area

# Plot Finland with beautiful styling
finland.plot(
    ax=ax, 
    color='#d3d3d3',           # Light gray for land
    edgecolor='#808080',        # Medium gray for borders
    linewidth=1.5,
    alpha=0.9
)

# Add a subtle shadow effect by plotting a slightly offset version
finland.plot(
    ax=ax,
    color='none',
    edgecolor='#000000',
    linewidth=2,
    alpha=0.15,
    linestyle='-'
)

# Remove axes and grid
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
st.pyplot(fig)