import streamlit as st
import pandas as pd
import os

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
    
if __name__ == "__main__":
    main()