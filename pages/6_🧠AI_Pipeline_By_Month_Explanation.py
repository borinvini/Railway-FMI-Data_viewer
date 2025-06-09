import streamlit as st
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config.const import VIEWER_FOLDER_NAME

def count_preprocessed_data_lines():
    """
    Count the number of rows (train schedules) in all preprocessed CSV files.
    Returns:
        tuple: (total_count, file_counts) where file_counts is a dictionary with filename as key and row count as value
    """
    preprocessed_dir = "data/ai_results/by_month/preprocessed"
    
    # Check if directory exists
    if not os.path.exists(preprocessed_dir):
        st.warning(f"Directory '{preprocessed_dir}' not found.")
        return 0, {}
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.csv')]
    
    if not csv_files:
        st.warning(f"No CSV files found in '{preprocessed_dir}'.")
        return 0, {}
    
    total_count = 0
    file_counts = {}
    
    # Count rows in each file
    for csv_file in csv_files:
        file_path = os.path.join(preprocessed_dir, csv_file)
        try:
            # Use pandas to read and count rows
            df = pd.read_csv(file_path)
            row_count = len(df)
            
            if row_count > 0:
                file_counts[csv_file] = row_count
                total_count += row_count
        except Exception as e:
            # If there's an error reading the file, log it but continue
            st.error(f"Error reading {file_path}: {e}")
    
    return total_count, file_counts

def load_and_display_log_file(log_filename, step_title):
    """
    Load and display a log file in Streamlit using an expandable section.
    
    Args:
        log_filename: Name of the log file to load
        step_title: Title of the step for the expander label
    """
    log_path = os.path.join("data", "ai_results", "by_month", "log", log_filename)
    
    if not os.path.exists(log_path):
        with st.expander(f"📋 View {step_title} Log Details", expanded=False):
            st.warning(f"⚠️ Log file `{log_filename}` not found at `{log_path}`.")
        return
    
    try:
        with open(log_path, 'r', encoding='utf-8') as file:
            log_content = file.read()
        
        if log_content.strip():
            # Use expander for collapsible log content
            with st.expander(f"📋 View {step_title} Log Details", expanded=False):
                # Add a brief summary at the top
                st.markdown(f"**Log File:** `{log_filename}`")
                
                # Display log content in a code block for better formatting
                st.code(log_content, language='text')
                
                # Add a download button for the log file
                st.download_button(
                    label=f"📥 Download {log_filename}",
                    data=log_content,
                    file_name=log_filename,
                    mime="text/plain",
                    key=f"download_{log_filename}"
                )
        else:
            with st.expander(f"📋 View {step_title} Log Details", expanded=False):
                st.info(f"📄 Log file `{log_filename}` is empty.")
            
    except Exception as e:
        with st.expander(f"📋 View {step_title} Log Details", expanded=False):
            st.error(f"❌ Error reading log file `{log_filename}`: {e}")

def classify_stations_by_region(df):
    """
    Classify train stations into 4 regions using K-means clustering.
    
    Args:
        df: DataFrame containing train station data with latitude and longitude columns
    
    Returns:
        DataFrame with region and region_name columns added
    """
    # Extract the coordinates for clustering
    coords = df[['latitude', 'longitude']].copy()
    
    # Standardize the coordinates to give equal weight to latitude and longitude
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Apply K-means clustering with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['region'] = kmeans.fit_predict(coords_scaled)
    
    # Map the numeric clusters to named regions for better interpretability
    # Let's assign names based on general geographic position
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_info = pd.DataFrame(cluster_centers, columns=['center_lat', 'center_long'])
    
    # Determine directional position of clusters for naming
    mean_lat = cluster_info['center_lat'].mean()
    mean_long = cluster_info['center_long'].mean()
    
    # Create region names based on position relative to center
    region_names = []
    for _, row in cluster_info.iterrows():
        if row['center_lat'] >= mean_lat:
            if row['center_long'] >= mean_long:
                region_names.append("Northeast")
            else:
                region_names.append("Northwest")
        else:
            if row['center_long'] >= mean_long:
                region_names.append("Southeast")
            else:
                region_names.append("Southwest")
    
    # Create mapping dictionary
    region_mapping = {i: name for i, name in enumerate(region_names)}
    
    # Add named region column
    df['region_name'] = df['region'].map(region_mapping)
    
    return df, cluster_centers, region_mapping

def plot_station_regions(df, cluster_centers, region_mapping):
    """
    Create a visualization of train stations colored by region.
    
    Args:
        df: DataFrame with region assignments
        cluster_centers: Array of cluster center coordinates
        region_mapping: Dictionary mapping region indices to region names
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'purple']
    markers = ['o', 's', '^', 'D']
    
    for region in range(4):
        region_data = df[df['region'] == region]
        ax.scatter(
            region_data['longitude'],
            region_data['latitude'],
            c=colors[region],
            marker=markers[region],
            label=f"{region_mapping[region]} ({len(region_data)} stations)",
            alpha=0.7,
            s=100
        )
    
    # Plot the cluster centers
    for i, (lat, long) in enumerate(cluster_centers):
        ax.scatter(
            long, lat, 
            c='black', 
            marker='X', 
            s=200, 
            edgecolor='w', 
            linewidth=2,
            label=f"Center {region_mapping[i]}" if i == 0 else None
        )
    
    ax.set_title('Finnish Train Stations Divided into 4 Regions')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Pipeline Viewer",
        page_icon="🤖",
        layout="wide"
    )

    # Title and introduction
    st.title("🤖 AI Pipeline Viewer")
    st.markdown("### Workflow for Train Delay Prediction")
    
    # Monthly data division highlight
    st.markdown("""
    <div style="
        background-color: #f0f7fb;
        border-left: 5px solid #2196F3;
        padding: 10px 15px;
        margin-bottom: 20px;
        border-radius: 4px;
        color: #333333;
    ">
        <h3 style="color: #0d47a1; margin-top: 0;">
            📅 Month-by-Month Data Processing
        </h3>
        <p style="color: #333333;">
            This pipeline <b>divides all train and weather data by calendar months</b> before processing. 
            Each month is treated as a separate dataset that goes through the entire pipeline independently.
        </p>
        <p style="color: #333333;">
            <b>Why monthly processing?</b>
        </p>
        <ul style="color: #333333;">
            <li>Captures seasonal weather patterns and their impact on train delays</li>
            <li>Allows for more targeted model training</li>
            <li>Handles large datasets more efficiently</li>
            <li>Enables comparative analysis between different months</li>
        </ul>
    </div>
    """
    , unsafe_allow_html=True)
    
    # Display pipeline flow image 
    st.markdown("## Pipeline Flow")
    st.image("assets/pipeline_flow.png", caption="Pipeline Flow Diagram")
    
    # Create steps with detailed descriptions and log file mappings
    steps = [
        {
            "title": "1 - Preprocess Files",
            "description": [
                "Loads the CSV file",
                "Extracts nested data from \"timeTableRows\" column",
                "Keeps relevant columns (differenceInMinutes, differenceInMinutes_offset, cancelled, weather_conditions, trainStopping, commercialStop)",
                "Expands weather_conditions into separate columns",
                "Converts boolean columns to numeric (0/1)",
                "Optionally filters trains by required stations"
            ]
        },
        {
            "title": "2 - Snow Depth Data Handling",
            "description": [
                "We have 2 snow columns: 'Snow depth' (snow data from closest EMS) and 'Snow depth Other' (snow data from other EMS that measures snow)",
                "Fills missing values in 'Snow depth' using data from 'Snow depth Other' when available",
                "Drops redundant columns ('Snow depth Other' and 'Snow depth Other Distance')",
            ],
            "log_file": "merge_snow_depth_columns.log",
            "log_title": "Snow Depth Data Handling"
        },
        {
            "title": "3 - Clean Missing Values", 
            "description": [
                "Fills missing values in trainStopping and commercialStop columns with 0",
                "Drops rows with missing values in any required columns (differenceInMinutes, differenceInMinutes_offset, trainDelayed, cancelled)",
                "Drops rows only if ALL 7 important weather conditions are missing: Air temperature, Relative humidity, Dew-point temperature, Precipitation amount, Precipitation intensity, Snow depth, Horizontal visibility", 
                "Keeps rows with at least one important weather condition present",
                "Uses zero imputation for precipitation and snow metrics (Precipitation amount, Precipitation intensity, Snow depth)",
                "Uses median imputation for temperature and continuous variables (Air temperature, Relative humidity, Dew-point temperature, Horizontal visibility)"
            ],
            "log_file": "handle_missing_values.log",
            "log_title": "Clean Missing Values"
        },
        {
            "title": "4 - Remove Duplicates",
            "description": [
                "Removes identical duplicate rows",
                "Improves dataset quality by eliminating redundant entries",
                "Helps ensure model training uses unique examples"
            ],
            "log_file": "remove_duplicates.log",
            "log_title": "Remove Duplicates"
        },
        {
            "title": "5 - Scale Numeric Columns",
            "description": [
                "Identifies all numeric columns in the dataframe",
                "Excludes target variables (differenceInMinutes, differenceInMinutes_offset), boolean features (trainStopping, commercialStop), and missing indicator columns",
                "Uses StandardScaler to standardize the remaining numeric columns (removes mean, scales to unit variance)"
            ],
            "log_file": "scale_numeric_columns.log",
            "log_title": "Scale Numeric Columns"
        },
        {
            "title": "6 - Add Train Delayed Feature",
            "description": [
                "Creates a new binary column 'trainDelayed' based on differenceInMinutes",
                "Sets trainDelayed to True when differenceInMinutes > 0 (train is delayed)",
                "Sets trainDelayed to False when differenceInMinutes <= 0 (train is on time or early)",
                "Positions the new column after differenceInMinutes for logical ordering"
            ]
        },
        {
            "title": "7 - Select Target Variable",
            "description": [
                "Accepts a target_feature parameter (one of 'differenceInMinutes', 'differenceInMinutes_offset', 'trainDelayed', or 'cancelled')",
                "Validates that the specified target feature exists in the dataframe",
                "Drops the other target options while keeping the selected one",
                "Ensures only one target variable remains for model training"
            ]
        },
        {
            "title": "8 - Save Processed Data",
            "description": [
                "Creates the output filename using the month_id parameter and save the dataframe in CSV file"
            ],
            "has_stats": True,  # Flag to include the statistics here
        },
        {
            "title": "9 - Split Month Dataset",
            "description": [
                "Loads the processed CSV file for the given month_id",
                "Identifies the target column (differenceInMinutes, differenceInMinutes_offset, trainDelayed, or cancelled)",
                "Splits features (X) and target (y) variables",
                "Uses stratified splitting for categorical targets (trainDelayed, cancelled), regular splitting for continuous targets",
                "Creates train and test datasets with the specified test_size (default 30%)",
                "Saves train and test sets as separate CSV files with _train.csv and _test.csv suffixes",
                "Reports distribution statistics for both datasets"
            ]
        },
        {
            "title": "10 - Training",
            "description": [
                "Train the dataset with the target variable for multiple model training approaches",
                "For regression: uses custom weighted loss functions to focus more on larger magnitude values"
            ]
        }
    ]

    # Try to locate and load the dataset file for step 7
    dataset_file = None
    possible_locations = [
        "data/ai_results/by_month/preprocessed/preprocessed_data_2020-2024_12.csv",
        "data/preprocessed_data_2020-2024_12.csv",
        "data/viewers/preprocessed_data_2020-2024_12.csv",
        "data/ai_results/preprocessed_data_2020-2024_12.csv",
        "data/ai_results/by_month/preprocessed_data_2020-2024_12.csv"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            try:
                dataset_file = pd.read_csv(location)
                dataset_path = location
                break
            except Exception as e:
                pass  # Silently continue trying other locations

    # Display the pipeline steps
    st.markdown("## AI Pipeline Steps")
    
    # Add explanation about log indicators
    st.info("📋 Steps marked with 📋 have log files available. Click on the expandable sections to view detailed processing logs.")
    
    # Count train schedules in preprocessed files before displaying steps
    total_train_schedules, file_counts = count_preprocessed_data_lines()
    
    for i, step in enumerate(steps):
        # Add log indicator to step title if log file is available
        step_title = step["title"]
        if "log_file" in step:
            step_title += " 📋"
        
        st.subheader(step_title)
        
        # Option 1: Custom HTML with reduced spacing
        html_list = '<div style="line-height: 1.2; margin-top: -10px;">'
        for item in step["description"]:
            html_list += f'<div style="margin-bottom: 2px;">• {item}</div>'
        html_list += '</div>'
        st.markdown(html_list, unsafe_allow_html=True)
        
        # Add statistics to Step 8 (Save Processed Data)
        if "has_stats" in step and step["has_stats"]:
            st.subheader("📊 Training Data Statistics")
            
            # Display results
            if total_train_schedules > 0:
                st.markdown(f"""
                <div style="
                    padding: 10px;
                    border-radius: 5px;
                    background-color: #e6f7ff;
                    color: #000000;
                    border: 1px solid #b8daff;
                    font-size: 16px;
                    ">
                    ✅ <b>Total train schedules used in training: {total_train_schedules:,}</b>
                </div>
                """, unsafe_allow_html=True)
                
                # Display breakdown per file in an expander
                with st.expander("🔍 View schedule count breakdown by month"):
                    for file, count in sorted(file_counts.items()):
                        # Extract month/year info from filename for better display
                        match = re.search(r'(\d{4}-\d{4})_(\d{2})', file)
                        if match:
                            year_range, month = match.groups()
                            display_name = f"{year_range}, Month {month}"
                        else:
                            display_name = file
                        
                        st.write(f"- **{display_name}**: {count:,} schedules")
                        
                    # Add explanation of what a schedule represents
                    st.info("""
                    Each record represents a train schedule entry with associated weather conditions and 
                    delay information. These are used as training examples for the AI models.
                    """)
            else:
                st.warning("⚠️ No preprocessed data files found or accessible. Make sure the preprocessing step has been completed.")
        
        # If this is step 7, display the dataset
        if i == 6:  # Index 6 is step 7
            if dataset_file is not None:
                st.success(f"Sample processed dataset from: {dataset_path}")
                
                # Define target columns
                target_cols = [col for col in dataset_file.columns if col in ['differenceInMinutes', 'cancelled', 'trainDelayed', 'relative_differenceInMinutes']]
                
                # Define feature columns (all non-target columns)
                feature_cols = [col for col in dataset_file.columns if col not in target_cols]
                
                # Reorder columns to show target columns first
                reordered_cols = target_cols + feature_cols
                
                # Display the sample with reordered columns
                st.dataframe(dataset_file[reordered_cols].head(10))
            else:
                st.warning("⚠️ Dataset file 'preprocessed_data_2023-2024_12.csv' not found")
        
        # Load and display log file if this step has one
        if "log_file" in step:
            # Use custom log title if available, otherwise extract from step title
            if "log_title" in step:
                step_name = step["log_title"]
            else:
                step_name = step["title"].split(" - ", 1)[1] if " - " in step["title"] else step["title"]
            load_and_display_log_file(step["log_file"], step_name)
        
        st.markdown("---")  # Add separator line between steps
    
    # Display AI scenarios image at the end
    st.image("assets/ai_scenarios.png", caption="AI Model Training Scenarios")

    # FUTURE WORK SECTION
    st.markdown("## 🗺️ Future Work: Train Station Regional Classification")
    st.markdown("""
    
    ### Planned Features:
    
    - Understand the impact of the missing measurements like snow deph in the models
    - Understand the impact of o commercialStop and trainStopping column in the models
    - Fine tune the features to improve performance. More data, more models.
    - Region-specific delay analysis and prediction.
    - Paper
    

    """)
    
    # Try to load the train stations data for mockup visualization
    train_stations_file = os.path.join(VIEWER_FOLDER_NAME, "metadata", "train_stations.csv")
    mockup_created = False
    
    if os.path.exists(train_stations_file):
        try:
            # Load the train stations data
            df_stations = pd.read_csv(train_stations_file)
            
            # Clean up column names in case there's whitespace
            df_stations.columns = df_stations.columns.str.strip()
            
            # Check if the necessary columns exist
            if 'latitude' in df_stations.columns and 'longitude' in df_stations.columns:
                # Run the clustering algorithm
                df_stations, cluster_centers, region_mapping = classify_stations_by_region(df_stations)
                
                # Create visualization mockup
                st.subheader("Mockup Visualization of Regional Classification")
                fig = plot_station_regions(df_stations, cluster_centers, region_mapping)
                st.pyplot(fig)
                
                mockup_created = True
                
                # Add explanation
                st.caption("Example K-means clustering visualization showing how stations will be grouped into regions")
        except Exception as e:
            pass  # Silently continue if there's an error
    
    # Display info message
    st.info("⏳ This feature is currently under development and will be available in a future update.")
    
    # If mockup wasn't created, explain why
    if not mockup_created:
        st.write("The visualization mockup will show train stations clustered into four geographic regions across Finland.")

if __name__ == "__main__":
    main()