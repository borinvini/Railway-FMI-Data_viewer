import streamlit as st

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Pipeline Viewer",
        page_icon="🤖",
        layout="wide"
    )

    # Title and introduction
    st.title("🤖 AI Pipeline Viewer")
    st.markdown("### State Machine Workflow for Train Delay and Cancellation Prediction")
    
    st.info("""
    This pipeline processes train and weather data on a month-by-month basis. 
    Each monthly dataset goes through the entire pipeline independently.
    """)
    
    # Display pipeline flow image at the beginning
    st.image("assets/pipeline_flow.png", caption="Pipeline Flow Diagram")
    
    # Create steps with detailed descriptions
    steps = [
        {
            "title": "1 - Preprocess Files",
            "description": [
                "Loads the CSV file",
                "Extracts nested data from \"timeTableRows\" column",
                "Keeps only relevant columns (differenceInMinutes, cancelled, weather_conditions)",
                "Expands weather_conditions into separate columns"
            ]
        },
        {
            "title": "2 - Clean Missing Values",
            "description": [
                "Drops rows with missing values in required columns (differenceInMinutes, cancelled)",
                "Drops rows only if ALL important weather conditions are missing",
                "Keeps rows with at least one weather condition present"
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
                "Excludes \"differenceInMinutes\" from scaling (a target variable)",
                "Uses StandardScaler to standardize the remaining numeric columns (removes mean, scales to unit variance)"
            ]
        },
        {
            "title": "5 - Add Train Delayed Feature",
            "description": [
                "Creates a new binary column 'trainDelayed' based on differenceInMinutes",
                "Sets trainDelayed to True when differenceInMinutes > 0 (train is delayed)"
            ]
        },
        {
            "title": "6 - Select Target Variable",
            "description": [
                "Accepts a target_feature parameter (one of 'differenceInMinutes', 'trainDelayed', or 'cancelled')",
                "Identifies if the specified target feature exists in the dataframe",
                "Drops the other two target options while keeping the selected one"
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
                "Train the dataset with the target variable for different scenarios:",
                "• Decision tree",
                "• Decision tree with important features",
                "• RandomizedSearchCV",
                "• RandomizedSearchCV with important features",
                "• XGBoost",
                "• XGBoost with RandomizedSearchCV"
            ]
        }
    ]

   
    # Display the pipeline steps as simple text items
    st.markdown("## AI Pipeline Steps")
    
    for step in steps:
        st.subheader(step["title"])
        for item in step["description"]:
            st.markdown(f"- {item}")
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