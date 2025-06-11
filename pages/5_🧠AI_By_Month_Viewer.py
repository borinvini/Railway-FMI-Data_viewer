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

# Function to search for SHAP importance files
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
    
    # For each model directory, find all SHAP_feature_importance*.csv files
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        feature_files = glob.glob(os.path.join(model_path, "SHAP_feature_importance*.csv"))
        
        if feature_files:
            result[model_dir] = sorted(feature_files)
    
    return result

# Function to search for model metrics files - UPDATED to handle model_metrics*.csv
def find_model_metrics_files():
    base_path = "data/ai_results/by_month"
    result = {}
    
    # Check if the base directory exists
    if not os.path.exists(base_path):
        return {}
    
    # Get all subdirectories (model types)
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # For each model directory, find all model_metrics*.csv files (removed underscore to be more flexible)
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        metrics_files = glob.glob(os.path.join(model_path, "model_metrics*.csv"))
        
        if metrics_files:
            result[model_dir] = sorted(metrics_files)
    
    return result

# Function to extract month-year info from filename
def extract_date_from_filename(filename):
    # Generalized pattern to extract year-month, ignoring any suffixes
    # This matches: SHAP_feature_importance_YYYY-YYYY_MM.csv, feature_importance_YYYY-YYYY_MM.csv, model_metrics_YYYY-YYYY_MM.csv
    match = re.search(r'(?:SHAP_feature_importance|feature_importance|model_metrics)_?(\d{4}-\d{4})_(\d{2})', os.path.basename(filename))
    if match:
        year_range, month = match.groups()
        return f"{year_range}_{month}"
    return os.path.basename(filename)

# Function to read and combine all SHAP feature importance data
def load_feature_importance_data(file_paths):
    all_data = []
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            date_label = extract_date_from_filename(file_path)
            df['Period'] = date_label
            
            # Rename SHAP_Importance_Abs to Importance for consistency with existing plotting functions
            if 'SHAP_Importance_Abs' in df.columns:
                df['Importance'] = df['SHAP_Importance_Abs']
            elif 'Importance' not in df.columns:
                st.warning(f"Neither 'SHAP_Importance_Abs' nor 'Importance' column found in {file_path}")
                continue
                
            all_data.append(df)
        except Exception as e:
            st.warning(f"Error reading {file_path}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# UPDATED: Function to read and combine all model metrics data - Check for accuracy or r2
def load_model_metrics_data(file_paths):
    all_data = []
    available_metric = None
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            date_label = extract_date_from_filename(file_path)
            
            # Check for accuracy first, then r2
            metric_found = None
            metric_value = None
            
            if 'accuracy' in df.columns:
                metric_found = 'accuracy'
                metric_value = df['accuracy'].iloc[0]
            elif 'r2' in df.columns:
                metric_found = 'r2'
                metric_value = df['r2'].iloc[0]
            
            if metric_found:
                # Store the first metric type we find to ensure consistency
                if available_metric is None:
                    available_metric = metric_found
                elif available_metric != metric_found:
                    st.warning(f"Inconsistent metrics across files. Expected {available_metric}, found {metric_found} in {file_path}")
                    continue
                
                # Create a dataframe with the found metric
                metric_df = pd.DataFrame({
                    'Metric': [metric_found],
                    'Value': [metric_value],
                    'Period': [date_label]
                })
                all_data.append(metric_df)
            else:
                st.warning(f"No accuracy or r2 metric found in {file_path}")
        except Exception as e:
            st.warning(f"Error reading {file_path}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Add the available metric type as metadata
        combined_df.attrs['metric_type'] = available_metric
        return combined_df
    return pd.DataFrame()

# Function to create a top features bar chart
def plot_feature_importance(df, period=None, top_n=10):
    if period:
        df_plot = df[df['Period'] == period]
    else:
        # If no period specified, take average across all periods
        df_plot = df.groupby('Feature')['Importance'].mean().reset_index()
    
    # Sort by importance (already absolute values)
    df_plot = df_plot.sort_values('Importance', ascending=False).head(top_n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a consistent color scheme for all bars
    bars = ax.barh(df_plot['Feature'], df_plot['Importance'])
    
    # Add a color gradient
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i/len(bars)))
    
    ax.set_xlabel('SHAP Importance (Absolute)')
    ax.set_title(f'Top {top_n} Most Important Features' + (f' - {period}' if period else ' (Average)'))
    
    # Add values on bars
    for i, v in enumerate(df_plot['Importance']):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center', ha='left', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Function to create a heatmap of feature importance across time periods
def plot_feature_importance_heatmap(df, selected_features=None):
    # Pivot the data to get features as rows and periods as columns
    pivot_df = df.pivot_table(index='Feature', columns='Period', values='Importance')
    
    # Filter for selected features if specified
    if selected_features and len(selected_features) > 0:
        pivot_df = pivot_df.loc[selected_features]
    
    # Sort features by average importance (already absolute values)
    avg_importance = pivot_df.mean(axis=1)
    pivot_df = pivot_df.loc[avg_importance.sort_values(ascending=False).index]
    
    # Create a custom colormap
    colors = ["#f7fbff", "#08306b"]  # Light blue to dark blue
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=100)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_df) * 0.4)))
    sns.heatmap(pivot_df, cmap=cmap, annot=True, fmt=".3f", linewidths=.5, ax=ax)
    ax.set_title('SHAP Feature Importance Over Time (Absolute Values)')
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
    
    # Use different colors for different features
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_features)))
    
    for i, feature in enumerate(selected_features):
        if feature in pivot_df.columns:
            values = pivot_df[feature]
            ax.plot(pivot_df.index, values, marker='o', label=feature, 
                   color=colors[i], linewidth=2, markersize=6)
    
    ax.set_xlabel('Time Period')
    ax.set_ylabel('SHAP Importance (Absolute)')
    ax.set_title('SHAP Feature Importance Trends Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# UPDATED: Function to plot model metrics comparison for a specific period - Handle accuracy or r2
def plot_model_metrics_comparison(df, period=None):
    if df.empty:
        return None
    
    # Get the metric type from dataframe attributes
    metric_type = getattr(df, 'attrs', {}).get('metric_type', 'unknown')
    
    if period:
        df_plot = df[df['Period'] == period]
    else:
        # If no period specified, take average across all periods
        df_plot = df.groupby('Metric')['Value'].mean().reset_index()
    
    # Filter for the available metric
    df_plot = df_plot[df_plot['Metric'] == metric_type]
    
    if df_plot.empty:
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df_plot['Metric'], df_plot['Value'])
    
    # Add color to bars
    color = '#1f77b4' if metric_type == 'r2' else '#ff7f0e'  # Blue for r2, orange for accuracy
    for bar in bars:
        bar.set_color(color)
    
    # Set labels based on metric type
    metric_label = 'R² Value' if metric_type == 'r2' else 'Accuracy'
    metric_title = 'Model R² Score' if metric_type == 'r2' else 'Model Accuracy Score'
    
    ax.set_xlabel(metric_label)
    ax.set_title(f'{metric_title}' + (f' - {period}' if period else ' (Average across all periods)'))
    
    # Add values on bars
    for i, v in enumerate(df_plot['Value']):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # Set x-axis limits for better visualization
    ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    return fig

# UPDATED: Function to create a heatmap of model metrics across time periods - Handle accuracy or r2
def plot_model_metrics_heatmap(df):
    if df.empty:
        return None
    
    # Get the metric type from dataframe attributes
    metric_type = getattr(df, 'attrs', {}).get('metric_type', 'unknown')
    
    # Filter to include only the available metric
    df_plot = df[df['Metric'] == metric_type]
    
    if df_plot.empty:
        return None
    
    # Pivot the data to get metrics as rows and periods as columns
    pivot_df = df_plot.pivot_table(index='Metric', columns='Period', values='Value')
    
    # Create a custom colormap
    colors = ["#fff7fb", "#8e0152"]  # Light purple to dark purple
    cmap = LinearSegmentedColormap.from_list("custom_purple", colors, N=100)
    
    # Set appropriate title and height based on metric type
    metric_title = 'R² Score Over Time' if metric_type == 'r2' else 'Accuracy Score Over Time'
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 3))  # Reduced height since we only have one metric
    sns.heatmap(pivot_df, cmap=cmap, annot=True, fmt=".3f", linewidths=.5, ax=ax, vmin=0, vmax=1)
    ax.set_title(metric_title)
    ax.set_ylabel('Metric')
    ax.set_xlabel('Time Period')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# UPDATED: Function to create a line chart showing how model metrics change over time - Handle accuracy or r2
def plot_model_metrics_trends(df):
    if df.empty:
        return None
    
    # Get the metric type from dataframe attributes
    metric_type = getattr(df, 'attrs', {}).get('metric_type', 'unknown')
    
    # Filter to include only the available metric
    df_plot = df[df['Metric'] == metric_type]
    
    if df_plot.empty:
        return None
    
    # Sort the periods chronologically
    pivot_df = df_plot.pivot_table(index='Period', columns='Metric', values='Value')
    periods = sorted(pivot_df.index.tolist())
    pivot_df = pivot_df.loc[periods]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set labels and colors based on metric type
    if metric_type == 'r2':
        metric_label = 'R² Score'
        metric_ylabel = 'R² Value'
        metric_title = 'R² Score Trend Over Time'
        color = '#1f77b4'  # Blue
    else:  # accuracy
        metric_label = 'Accuracy Score'
        metric_ylabel = 'Accuracy Value'
        metric_title = 'Accuracy Score Trend Over Time'
        color = '#ff7f0e'  # Orange
    
    ax.plot(pivot_df.index, pivot_df[metric_type], marker='o', label=metric_label, color=color, linewidth=2)
    
    ax.set_xlabel('Time Period')
    ax.set_ylabel(metric_ylabel)
    ax.set_title(metric_title)
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
            st.error("No SHAP feature importance files found. Please check the data directory structure.")
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
            st.header(f"SHAP Feature Importance Analysis: {selected_model}")
            st.write(f"Found {len(file_paths)} SHAP feature importance files")
            
            # Add explanation about SHAP values
            st.info("""
            📊 **SHAP Importance Analysis**: This analysis shows SHAP (SHapley Additive exPlanations) absolute feature importance values 
            specifically for **delayed trains only**. These values represent the magnitude of each feature's impact on delays, 
            regardless of whether they increase or decrease delays.
            - **Higher values**: Features that have a stronger impact on delay predictions
            - **Lower values**: Features that have less influence on delay predictions
            """)
            
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
            
            # Calculate the average importance across all periods for each feature (already absolute values)
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
                st.subheader("SHAP Feature Importance Bar Chart")
                period_for_chart = None if selected_period == "All Periods" else selected_period
                
                # Add informational text when "All Periods" is selected
                if selected_period == "All Periods":
                    st.info("📊 **Note**: When 'All Periods' is selected, the values shown represent the **mean SHAP absolute importance** of each feature calculated across all time periods.")
                    
                fig1 = plot_feature_importance(df, period_for_chart, top_n)
                st.pyplot(fig1)
            
            with tab2:
                st.subheader("SHAP Feature Importance Heatmap")
                fig2 = plot_feature_importance_heatmap(df, selected_features if selected_features else None)
                st.pyplot(fig2)
            
            with tab3:
                st.subheader("SHAP Feature Importance Trends")
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
            
            # Get the metric type from the loaded data
            metric_type = getattr(df, 'attrs', {}).get('metric_type', 'unknown')
            
            # Show basic information
            st.header(f"Model Metrics Analysis: {selected_model}")
            st.write(f"Found {len(file_paths)} model metrics files")
            
            # Display metric-specific information
            if metric_type == 'r2':
                st.info("📊 **Note**: This view is showing the R² Score (r2) metric which measures how well the model fits the data (0-1, where 1 is perfect).")
            elif metric_type == 'accuracy':
                st.info("📊 **Note**: This view is showing the Accuracy metric which measures the percentage of correct predictions (0-1, where 1 is 100% accurate).")
            else:
                st.warning("⚠️ **Note**: Unknown metric type detected. Please check your data files.")
            
            # Filter options
            st.sidebar.header("Filter Options")
            
            # Period selection
            available_periods = sorted(df['Period'].unique())
            selected_period = st.sidebar.selectbox(
                "Select Time Period", 
                ["All Periods"] + available_periods
            )
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Heatmap", "Trends"])
            
            with tab1:
                metric_display_name = "R² Score" if metric_type == 'r2' else "Accuracy Score"
                st.subheader(f"{metric_display_name} Comparison")
                period_for_chart = None if selected_period == "All Periods" else selected_period
                
                # Add informational text when "All Periods" is selected
                if selected_period == "All Periods":
                    st.info(f"📊 **Note**: When 'All Periods' is selected, the value shown represents the **mean {metric_display_name.lower()}** calculated across all time periods.")
                
                fig1 = plot_model_metrics_comparison(df, period_for_chart)
                if fig1:
                    st.pyplot(fig1)
                else:
                    st.error("Could not generate comparison chart.")
            
            with tab2:
                metric_display_name = "R² Score" if metric_type == 'r2' else "Accuracy Score"
                st.subheader(f"{metric_display_name} Heatmap")
                fig2 = plot_model_metrics_heatmap(df)
                if fig2:
                    st.pyplot(fig2)
                else:
                    st.error("Could not generate heatmap.")
            
            with tab3:
                metric_display_name = "R² Score" if metric_type == 'r2' else "Accuracy Score"
                st.subheader(f"{metric_display_name} Trends")
                fig3 = plot_model_metrics_trends(df)
                if fig3:
                    st.pyplot(fig3)
                else:
                    st.error("Could not generate trends chart.")
            
            # Show raw data
            with st.expander("View Raw Data"):
                wide_df = df.pivot_table(index='Period', columns='Metric', values='Value').reset_index()
                st.dataframe(wide_df)

if __name__ == "__main__":
    main()