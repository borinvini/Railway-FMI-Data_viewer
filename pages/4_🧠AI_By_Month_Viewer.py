import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Page configuration
st.set_page_config(page_title="AI Feature Importance Viewer", layout="wide")

# Function to search for feature importance files
def find_feature_importance_files():
    base_path = "data/ai_results/by_month"
    result = {}
    
    # Check if the base directory exists
    if not os.path.exists(base_path):
        st.error(f"Directory '{base_path}' not found. Please check the path.")
        return {}
    
    # Get all subdirectories (model types)
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not model_dirs:
        st.warning(f"No model directories found in '{base_path}'.")
        return {}
    
    # For each model directory, find all feature_importance_*.csv files
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        feature_files = glob.glob(os.path.join(model_path, "feature_importance_*.csv"))
        
        if feature_files:
            result[model_dir] = sorted(feature_files)
    
    return result

# Function to extract month-year info from filename
def extract_date_from_filename(filename):
    # Expected pattern: feature_importance_YYYY-YYYY_MM.csv
    match = re.search(r'feature_importance_(\d{4}-\d{4})_(\d{2})\.csv', os.path.basename(filename))
    if match:
        year_range, month = match.groups()
        return f"{year_range}_{month}"
    return os.path.basename(filename)

# Function to read and combine all feature importance data
def load_feature_importance_data(file_paths):
    all_data = []
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            date_label = extract_date_from_filename(file_path)
            df['Period'] = date_label
            all_data.append(df)
        except Exception as e:
            st.warning(f"Error reading {file_path}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# Function to create a top features bar chart
def plot_feature_importance(df, period=None, top_n=10):
    if period:
        df_plot = df[df['Period'] == period]
    else:
        # If no period specified, take average across all periods
        df_plot = df.groupby('Feature')['Importance'].mean().reset_index()
    
    # Sort by importance and get top N
    df_plot = df_plot.sort_values('Importance', ascending=False).head(top_n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_plot['Feature'], df_plot['Importance'])
    
    # Add a color gradient
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i/len(bars)))
    
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Important Features' + (f' - {period}' if period else ' (Average)'))
    
    # Add values on bars
    for i, v in enumerate(df_plot['Importance']):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    return fig

# Function to create a heatmap of feature importance across time periods
def plot_feature_importance_heatmap(df, selected_features=None):
    # Pivot the data to get features as rows and periods as columns
    pivot_df = df.pivot_table(index='Feature', columns='Period', values='Importance')
    
    # Filter for selected features if specified
    if selected_features and len(selected_features) > 0:
        pivot_df = pivot_df.loc[selected_features]
    
    # Sort features by average importance
    avg_importance = pivot_df.mean(axis=1)
    pivot_df = pivot_df.loc[avg_importance.sort_values(ascending=False).index]
    
    # Create a custom colormap
    colors = ["#f7fbff", "#08306b"]  # Light blue to dark blue
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=100)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_df) * 0.4)))
    sns.heatmap(pivot_df, cmap=cmap, annot=True, fmt=".3f", linewidths=.5, ax=ax)
    ax.set_title('Feature Importance Over Time')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Time Period')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# Function to create a line chart showing how feature importance changes over time
def plot_feature_importance_trends(df, selected_features):
    if not selected_features:
        st.warning("Please select at least one feature to view trends.")
        return None
    
    # Filter for selected features
    df_plot = df[df['Feature'].isin(selected_features)]
    
    # Create a wide-format DataFrame for plotting
    pivot_df = df_plot.pivot(index='Period', columns='Feature', values='Importance')
    
    # Sort the periods chronologically
    periods = sorted(pivot_df.index.tolist())
    pivot_df = pivot_df.loc[periods]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for feature in selected_features:
        if feature in pivot_df.columns:
            ax.plot(pivot_df.index, pivot_df[feature], marker='o', label=feature)
    
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance Trends Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# Main application
def main():
    st.title("🧠 AI Feature Importance Analysis")
    st.write("Explore feature importance across different AI models and time periods")
    
    # Find all feature importance files
    model_files = find_feature_importance_files()
    
    if not model_files:
        st.error("No feature importance files found. Please check the data directory structure.")
        return
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    available_models = list(model_files.keys())
    selected_model = st.sidebar.selectbox("Select AI Model", available_models)
    
    if selected_model:
        # Load data for the selected model
        file_paths = model_files[selected_model]
        df = load_feature_importance_data(file_paths)
        
        if df.empty:
            st.warning(f"No valid data found for model: {selected_model}")
            return
        
        # Show basic information
        st.header(f"Model: {selected_model}")
        st.write(f"Found {len(file_paths)} feature importance files")
        
        # Filter options
        st.sidebar.header("Filter Options")
        
        # Period selection
        available_periods = sorted(df['Period'].unique())
        selected_period = st.sidebar.selectbox(
            "Select Time Period", 
            ["All Periods"] + available_periods
        )
        
        # Feature selection for trend analysis
        all_features = sorted(df['Feature'].unique())
        selected_features = st.sidebar.multiselect(
            "Select Features for Trend Analysis",
            all_features,
            default=all_features[:5] if len(all_features) >= 5 else all_features
        )
        
        # Top N features selection
        top_n = st.sidebar.slider("Top N Features", min_value=3, max_value=20, value=10)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Bar Chart", "Heatmap", "Trends"])
        
        with tab1:
            st.subheader("Feature Importance Bar Chart")
            period_for_chart = None if selected_period == "All Periods" else selected_period
            fig1 = plot_feature_importance(df, period_for_chart, top_n)
            st.pyplot(fig1)
        
        with tab2:
            st.subheader("Feature Importance Heatmap")
            fig2 = plot_feature_importance_heatmap(df, selected_features if selected_features else None)
            st.pyplot(fig2)
        
        with tab3:
            st.subheader("Feature Importance Trends")
            if selected_features:
                fig3 = plot_feature_importance_trends(df, selected_features)
                if fig3:
                    st.pyplot(fig3)
            else:
                st.warning("Please select at least one feature in the sidebar to view trends.")

        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(df)

if __name__ == "__main__":
    main()