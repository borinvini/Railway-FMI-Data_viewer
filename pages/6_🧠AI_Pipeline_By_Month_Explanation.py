import streamlit as st
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    
    st.info("""
    This pipeline processes train and weather data on a month-by-month basis. 
    Each monthly dataset goes through the entire pipeline independently.
    """)
    
    # NEW SECTION: Count and display train schedules
    st.subheader("📊 Training Data Statistics")
    
    # Count train schedules in preprocessed files
    total_train_schedules, file_counts = count_preprocessed_data_lines()
    
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
    
    # Display pipeline flow image 
    st.markdown("## Pipeline Flow")
    st.image("assets/pipeline_flow.png", caption="Pipeline Flow Diagram")
    
    # Create steps with detailed descriptions
    steps = [
        {
            "title": "1 - Preprocess Files",
            "description": [
                "Loads the CSV file",
                "Extracts nested data from \"timeTableRows\" column",
                "Calculates relative_differenceInMinutes based on delay differences between stops",
                "Keeps relevant columns (differenceInMinutes, relative_differenceInMinutes, cancelled, weather_conditions, trainStopping, commercialStop)",
                "Expands weather_conditions into separate columns"
            ]
        },
        {
            "title": "2 - Clean Missing Values",
            "description": [
                "Drops rows with missing values in required columns (differenceInMinutes, cancelled)",
                "Drops rows only if ALL important weather conditions are missing",
                "Keeps rows with at least one weather condition present",
                "Uses zero imputation for precipitation and snow metrics",
                "Uses median imputation for temperature, humidity, and other continuous variables"
            ]
        },
        {
            "title": "3 - Remove Duplicates",
            "description": [
                "Removes identical duplicate rows"
            ]
        },
        {
            "title": "4 - Scale Numeric Columns",
            "description": [
                "Identifies all numeric columns in the dataframe",
                "Excludes target variables (differenceInMinutes, relative_differenceInMinutes) and boolean features (trainStopping, commercialStop)",
                "Uses StandardScaler to standardize the remaining numeric columns (removes mean, scales to unit variance)"
            ]
        },
        {
            "title": "5 - (OPTIONAL) Add Train Delayed Feature",
            "description": [
                "Creates a new binary column 'trainDelayed' based on differenceInMinutes",
                "Sets trainDelayed to True when differenceInMinutes > 0 (train is delayed)"
            ]
        },
        {
            "title": "6 - Select Target Variable",
            "description": [
                "Accepts a target_feature parameter (one of 'differenceInMinutes', 'relative_differenceInMinutes', 'trainDelayed', or 'cancelled')",
                "Identifies if the specified target feature exists in the dataframe",
                "Drops the other target options while keeping the selected one"
            ]
        },
        {
            "title": "7 - Save Processed Data",
            "description": [
                "Creates the output filename using the month_id parameter and save the dataframe in CSV file"
            ]
        },
        {
            "title": "8 - Split Month Dataset",
            "description": [
                "Loads the processed CSV file for the given month_id",
                "Identifies the target column (differenceInMinutes, trainDelayed, or cancelled)",
                "Splits features (X) and target (y) variables",
                "Uses stratified splitting for categorical targets, regular splitting for continuous targets",
                "Creates train and test datasets with the specified test_size (default 30%)",
                "Saves train and test sets as separate CSV files"
            ]
        },
        {
            "title": "9 - Training",
            "description": [
                "Train the dataset with the target variable for multiple model training approaches",
                "For regression: uses custom weighted loss functions to focus more on larger magnitude values"
            ]
        }
    ]

    # Try to locate and load the dataset file for step 7
    dataset_file = None
    possible_locations = [
        "data/ai_results/by_month/preprocessed/preprocessed_data_2023-2024_12.csv",
        "data/preprocessed_data_2023-2024_12.csv",
        "data/viewers/preprocessed_data_2023-2024_12.csv",
        "data/ai_results/preprocessed_data_2023-2024_12.csv",
        "data/ai_results/by_month/preprocessed_data_2023-2024_12.csv"
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
    
    for i, step in enumerate(steps):
        st.subheader(step["title"])
        
        # Option 1: Custom HTML with reduced spacing
        html_list = '<div style="line-height: 1.2; margin-top: -10px;">'
        for item in step["description"]:
            html_list += f'<div style="margin-bottom: 2px;">• {item}</div>'
        html_list += '</div>'
        st.markdown(html_list, unsafe_allow_html=True)
        
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
        
        st.markdown("---")  # Add separator line between steps
    
    # Display AI scenarios image at the end
    st.image("assets/ai_scenarios.png", caption="AI Model Training Scenarios")

    # Add information about model evaluation and results
    st.subheader("Pipeline Outputs")
    st.markdown("""
    The pipeline outputs several files for each month:
    - Processed data CSV files
    - Train/test split datasets
    - Model evaluation metrics
    - Feature importance rankings
    """)
    
    # NEW SECTION: Train Station Regional Classification
    st.markdown("## 🗺️ Train Station Regional Classification")
    st.markdown("""
    To improve our model's performance, we've implemented a regional classification of train stations 
    across Finland. This allows us to analyze weather impacts and train delays by geographic region.
    """)
    
    # Try to load the train stations data
    train_stations_file = "data/viewers/train_stations.csv"
    
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
                
                # Display the results in a two-column layout
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Create and display the visualization
                    fig = plot_station_regions(df_stations, cluster_centers, region_mapping)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Region Statistics")
                    
                    # Create region statistics
                    region_stats = df_stations.groupby('region_name').agg(
                        station_count=('stationName', 'count')
                    ).reset_index()
                    
                    # Add passenger traffic stats if available
                    if 'passengerTraffic' in df_stations.columns:
                        try:
                            # Convert to boolean if it's not already
                            if df_stations['passengerTraffic'].dtype != bool:
                                df_stations['passengerTraffic'] = df_stations['passengerTraffic'].astype(str).str.lower() == 'true'
                            
                            passenger_stats = df_stations.groupby('region_name').agg(
                                passenger_stations=('passengerTraffic', lambda x: x.sum())
                            ).reset_index()
                            
                            region_stats = region_stats.merge(passenger_stats, on='region_name')
                        except Exception as e:
                            st.warning(f"Could not calculate passenger statistics: {e}")
                    
                    # Display the statistics
                    st.dataframe(region_stats)
                    
                    st.markdown("""
                    ### Classification Method
                    
                    The stations are classified using **K-means clustering** on their geographic coordinates.
                    This approach:
                    
                    - Groups stations based on proximity
                    - Creates natural regional boundaries
                    - Automatically adapts to the geographic distribution
                    - Names regions based on their position relative to Finland's center
                    """)
                
                # Display sample of classified data in an expander
                with st.expander("View Sample of Classified Stations"):
                    st.dataframe(
                        df_stations[['stationName', 'stationShortCode', 'latitude', 'longitude', 'region_name']]
                        .sort_values('region_name')
                        .head(10)
                    )
                
                # Save the classified data
                output_path = "data/viewers/train_stations_with_regions.csv"
                df_stations.to_csv(output_path, index=False)
                st.success(f"Classification results saved to: {output_path}")
                
            else:
                st.error("Could not find latitude and longitude columns in the train stations CSV file.")
        
        except Exception as e:
            st.error(f"Error processing train stations data: {e}")
            st.exception(e)
    else:
        # Try loading from the exact path specified
        train_stations_file = "data/viewers/train_stations.csv"
        st.info(f"Loading train stations data directly from {train_stations_file}")
        
        try:
            # Load the train stations data from the specified path
            sample_df = pd.read_csv(train_stations_file)
            
            # Clean up column names
            sample_df.columns = sample_df.columns.str.strip()
            
            st.success(f"Successfully loaded train stations data from {train_stations_file}")
        except Exception as e:
            st.error(f"Error loading train stations file from {train_stations_file}: {e}")
            st.exception(e)
            return
        
        # Clean up column names
        sample_df.columns = sample_df.columns.str.strip()
        
        # Process the sample data
        sample_df, sample_centers, sample_mapping = classify_stations_by_region(sample_df)
        
        # Create and display the visualization for sample data
        sample_fig = plot_station_regions(sample_df, sample_centers, sample_mapping)
        st.pyplot(sample_fig)
        
        # Display sample processed data
        st.dataframe(
            sample_df[['stationName', 'stationShortCode', 'latitude', 'longitude', 'region_name']]
            .sort_values('region_name')
        )

if __name__ == "__main__":
    main()