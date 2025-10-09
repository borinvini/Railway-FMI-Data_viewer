import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Finland Map",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Finland Map")

# Load Finland map
@st.cache_data
def load_map():
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    finland = world[world['NAME'] == 'Finland']
    return finland

# Load and plot
with st.spinner("Loading map..."):
    finland = load_map()

fig, ax = plt.subplots(figsize=(10, 12))
finland.plot(ax=ax, color='lightgreen', edgecolor='black')
ax.set_title('Finland')
ax.grid(True, alpha=0.3)

st.pyplot(fig)