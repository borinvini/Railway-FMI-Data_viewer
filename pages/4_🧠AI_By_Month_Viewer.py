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
st.set_page_config(page_title="AI Analysis Viewer", layout="wide")

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

# Function to search for model metrics files
def find_model_metrics_files():
    base_path = "data/ai_results/by_month"
    result = {}
    
    # Check if the base directory exists
    if not os.path.exists(base_path):
        return {}
    
    # Get all subdirectories (model types)
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # For each model directory, find all model_metrics_*.csv files
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        metrics_files = glob.glob(os.path.join(model_path, "model_metrics_*.csv"))
        
        if metrics_files:
            result[model_dir] = sorted(metrics_files)
    
    return result

# Function to extract month-year info from filename
def extract_date_from_filename(filename):
    # Generalized pattern to extract year-month, ignoring any suffixes
    # This matches: prefix_YYYY-YYYY_MM_any_suffix.csv and returns just the YYYY-YYYY_MM part
    match = re.search(r'(?:feature_importance|model_metrics)_(\d{4}-\d{4})_(\d{2})', os.path.basename(filename))
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

# Function to read and combine all model metrics data
def load_model_metrics_data(file_paths):
    all_data = []
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            # For model metrics, we need to reshape to long format
            date_label = extract_date_from_filename(file_path)
            
            # Convert wide format to long format for easier plotting
            df_long = pd.melt(df, var_name='Metric', value_name='Value')
            df_long['Period'] = date_label
            
            all_data.append(df_long)
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

# Function to plot model metrics comparison for a specific period
def plot_model_metrics_comparison(df, period=None, metrics=None):
    if period:
        df_plot = df[df['Period'] == period]
    else:
        # If no period specified, take average across all periods
        df_plot = df.groupby('Metric')['Value'].mean().reset_index()
    
    # Filter for selected metrics if specified
    if metrics and len(metrics) > 0:
        df_plot = df_plot[df_plot['Metric'].isin(metrics)]
    
    # Sort by metric name for consistent display
    df_plot = df_plot.sort_values('Value', ascending=False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df_plot['Metric'], df_plot['Value'])
    
    # Add a color gradient - use a different color scheme than feature importance
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.plasma(i/len(bars)))
    
    ax.set_xlabel('Value')
    ax.set_title(f'Model Metrics' + (f' - {period}' if period else ' (Average across all periods)'))
    
    # Add values on bars
    for i, v in enumerate(df_plot['Value']):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # Set x-axis limits for better visualization
    ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    return fig

# Function to create a heatmap of model metrics across time periods
def plot_model_metrics_heatmap(df, selected_metrics=None):
    # Filter for selected metrics if specified
    if selected_metrics and len(selected_metrics) > 0:
        df_plot = df[df['Metric'].isin(selected_metrics)]
    else:
        df_plot = df
    
    # Pivot the data to get metrics as rows and periods as columns
    pivot_df = df_plot.pivot_table(index='Metric', columns='Period', values='Value')
    
    # Create a custom colormap - use a different color scheme
    colors = ["#fff7fb", "#8e0152"]  # Light purple to dark purple
    cmap = LinearSegmentedColormap.from_list("custom_purple", colors, N=100)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_df) * 0.4)))
    sns.heatmap(pivot_df, cmap=cmap, annot=True, fmt=".3f", linewidths=.5, ax=ax, vmin=0, vmax=1)
    ax.set_title('Model Metrics Over Time')
    ax.set_ylabel('Metric')
    ax.set_xlabel('Time Period')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# Function to create a line chart showing how model metrics change over time
def plot_model_metrics_trends(df, selected_metrics):
    if not selected_metrics:
        st.warning("Please select at least one metric to view trends.")
        return None
    
    # Filter for selected metrics
    df_plot = df[df['Metric'].isin(selected_metrics)]
    
    # Create a wide-format DataFrame for plotting
    pivot_df = df_plot.pivot_table(index='Period', columns='Metric', values='Value')
    
    # Sort the periods chronologically
    periods = sorted(pivot_df.index.tolist())
    pivot_df = pivot_df.loc[periods]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for metric in selected_metrics:
        if metric in pivot_df.columns:
            ax.plot(pivot_df.index, pivot_df[metric], marker='o', label=metric)
    
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Value')
    ax.set_title('Model Metrics Trends Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits for better visualization
    ax.set_ylim(0, 1.1)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# Main application
def main():
    st.title("🧠 AI Analysis Dashboard")
    
    # Create tabs for different analysis types
    analysis_type = st.radio(
        "Select Analysis Type:",
        ["Feature Importance", "Model Metrics"],
        horizontal=True
    )
    
    # Find all relevant files
    feature_files = find_feature_importance_files()
    metrics_files = find_model_metrics_files()
    
    if analysis_type == "Feature Importance":
        if not feature_files:
            st.error("No feature importance files found. Please check the data directory structure.")
            return
        
        # Sidebar for model selection
        st.sidebar.header("Model Selection")
        available_models = list(feature_files.keys())
        selected_model = st.sidebar.selectbox("Select AI Model", available_models)
        
        if selected_model:
            # Load data for the selected model
            file_paths = feature_files[selected_model]
            df = load_feature_importance_data(file_paths)
            
            if df.empty:
                st.warning(f"No valid data found for model: {selected_model}")
                return
            
            # Show basic information
            st.header(f"Feature Importance Analysis: {selected_model}")
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
            
            # Calculate the average importance across all periods for each feature
            feature_avg_importance = df.groupby('Feature')['Importance'].mean().reset_index()
            
            # Sort by importance and get top 5 features (or all if less than 5)
            top_features = feature_avg_importance.sort_values('Importance', ascending=False)['Feature'].tolist()
            default_features = top_features[:5] if len(top_features) >= 5 else top_features
            
            selected_features = st.sidebar.multiselect(
                "Select Features for Trend Analysis",
                all_features,
                default=default_features
            )
            
            # Top N features selection
            total_features = len(all_features)
            default_value = min(10, total_features)
            min_value = min(3, total_features)
            
            top_n = st.sidebar.slider(
                "Top N Features", 
                min_value=min_value, 
                max_value=total_features,
                value=default_value
            )
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Heatmap", "Trends"])
            
            with tab1:
                st.subheader("Feature Importance Bar Chart")
                period_for_chart = None if selected_period == "All Periods" else selected_period
                
                # Add informational text when "All Periods" is selected
                if selected_period == "All Periods":
                    st.info("📊 **Note**: When 'All Periods' is selected, the values shown represent the **mean importance** of each feature calculated across all time periods.")
                    
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
                
    elif analysis_type == "Model Metrics":
        if not metrics_files:
            st.error("No model metrics files found. Please check the data directory structure.")
            return
        
        # Sidebar for model selection
        st.sidebar.header("Model Selection")
        available_models = list(metrics_files.keys())
        selected_model = st.sidebar.selectbox("Select AI Model", available_models)
        
        if selected_model:
            # Load data for the selected model
            file_paths = metrics_files[selected_model]
            df = load_model_metrics_data(file_paths)
            
            if df.empty:
                st.warning(f"No valid metrics data found for model: {selected_model}")
                return
            
            # Show basic information
            st.header(f"Model Metrics Analysis: {selected_model}")
            st.write(f"Found {len(file_paths)} model metrics files")
            
            # Filter options
            st.sidebar.header("Filter Options")
            
            # Period selection
            available_periods = sorted(df['Period'].unique())
            selected_period = st.sidebar.selectbox(
                "Select Time Period", 
                ["All Periods"] + available_periods
            )
            
            # Metric selection for analysis
            all_metrics = sorted(df['Metric'].unique())
            
            # Group metrics by type for easier selection
            accuracy_metrics = [m for m in all_metrics if 'accuracy' in m.lower()]
            f1_metrics = [m for m in all_metrics if 'f1' in m.lower()]
            precision_metrics = [m for m in all_metrics if 'precision' in m.lower()]
            recall_metrics = [m for m in all_metrics if 'recall' in m.lower()]
            
            # Default metrics for selection
            main_metrics = ['accuracy', 'weighted_avg_f1', 'macro_avg_f1']
            default_metrics = [m for m in main_metrics if m in all_metrics]
            
            # Metric selection widget
            metric_categories = {
                "Main Metrics": [m for m in main_metrics if m in all_metrics],
                "Accuracy Metrics": accuracy_metrics,
                "F1 Metrics": f1_metrics,
                "Precision Metrics": precision_metrics,
                "Recall Metrics": recall_metrics
            }
            
            # Create a multi-select for each category
            selected_metrics = []
            for category, metrics in metric_categories.items():
                if metrics:
                    category_metrics = st.sidebar.multiselect(
                        f"Select {category}",
                        metrics,
                        default=metrics if category == "Main Metrics" else []
                    )
                    selected_metrics.extend(category_metrics)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Heatmap", "Trends"])
            
            with tab1:
                st.subheader("Model Metrics Comparison")
                period_for_chart = None if selected_period == "All Periods" else selected_period
                
                # Add informational text when "All Periods" is selected
                if selected_period == "All Periods":
                    st.info("📊 **Note**: When 'All Periods' is selected, the values shown represent the **mean value** of each metric calculated across all time periods.")
                
                fig1 = plot_model_metrics_comparison(df, period_for_chart, selected_metrics if selected_metrics else None)
                st.pyplot(fig1)
            
            with tab2:
                st.subheader("Model Metrics Heatmap")
                fig2 = plot_model_metrics_heatmap(df, selected_metrics if selected_metrics else None)
                st.pyplot(fig2)
            
            with tab3:
                st.subheader("Model Metrics Trends")
                if selected_metrics:
                    fig3 = plot_model_metrics_trends(df, selected_metrics)
                    if fig3:
                        st.pyplot(fig3)
                else:
                    st.warning("Please select at least one metric in the sidebar to view trends.")
            
            # Show raw data
            with st.expander("View Raw Data"):
                wide_df = df.pivot_table(index='Period', columns='Metric', values='Value').reset_index()
                st.dataframe(wide_df)

if __name__ == "__main__":
    main()