import streamlit as st
from config.const import CSV_MATCHED_DATA, VIEWER_FOLDER_NAME
from src.DataViewer import DataViewer
import os
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Delay Causes Analysis",
    page_icon="🔍",
    layout="wide"
)

# Initialize DataViewer instance
viewer = DataViewer()

# Check if data exists
if not viewer.has_data():
    st.stop()

@st.cache_data
def load_cause_metadata():
    """Load all cause metadata files and return mapping dictionaries"""
    metadata_files = {
        'category': 'metadata_train_causes.csv',
        'detailed': 'metadata_train_causes_detailed.csv', 
        'third': 'metadata_third_train_causes.csv'
    }
    
    mappings = {}
    
    for level, filename in metadata_files.items():
        file_path = os.path.join(VIEWER_FOLDER_NAME, "metadata", filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if level == 'category':
                    mappings[level] = dict(zip(df['categoryCode'], df['categoryName_en']))
                elif level == 'detailed':
                    mappings[level] = dict(zip(df['detailedCategoryCode'], df['detailedCategoryName_en']))
                elif level == 'third':
                    mappings[level] = dict(zip(df['thirdCategoryCode'], df['thirdCategoryName_en']))
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")
                mappings[level] = {}
        else:
            mappings[level] = {}
    
    return mappings

@st.cache_data
def extract_causes_from_matched_files(selected_periods):
    """Extract all causes data from matched files for selected periods"""
    all_causes_data = []
    processed_files = 0
    
    # Convert selected_periods to a hashable format for caching
    period_items = []
    for year, months in selected_periods.items():
        for month in months:
            period_items.append((year, month))
    
    for year, month in period_items:
        file_name = f"{CSV_MATCHED_DATA.replace('.csv', '')}_{year}_{str(month).zfill(2)}.csv"
        file_path = os.path.join(VIEWER_FOLDER_NAME, "matched_data", file_name)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                processed_files += 1
                
                # Process each train
                for idx, row in df.iterrows():
                    train_number = row.get('trainNumber', 'Unknown')
                    departure_date = row.get('departureDate', 'Unknown')
                    
                    # Extract timetable data
                    if 'timeTableRows' in row and pd.notna(row['timeTableRows']):
                        timetable_str = str(row['timeTableRows']).replace('nan', 'None')
                        try:
                            timetable_data = literal_eval(timetable_str)
                            timetable_df = pd.DataFrame(timetable_data)
                            
                            # Look for causes in timetable
                            if 'causes' in timetable_df.columns:
                                causes_data = timetable_df[
                                    timetable_df['causes'].notna() & 
                                    timetable_df['causes'].apply(lambda x: 
                                        not (isinstance(x, str) and x in ('[]', '{}')) and
                                        not (isinstance(x, (list, dict)) and len(x) == 0)
                                    )
                                ]
                                
                                # Process each row with causes
                                for _, cause_row in causes_data.iterrows():
                                    station_name = cause_row.get('stationName', 'Unknown')
                                    station_type = cause_row.get('type', 'Unknown')
                                    delay_minutes = cause_row.get('differenceInMinutes', 0)
                                    
                                    # Parse causes
                                    try:
                                        causes = cause_row['causes']
                                        if isinstance(causes, str):
                                            causes = literal_eval(causes)
                                        
                                        if isinstance(causes, list):
                                            for cause in causes:
                                                if isinstance(cause, dict):
                                                    cause_data = {
                                                        'train_number': train_number,
                                                        'departure_date': departure_date,
                                                        'year': year,
                                                        'month': month,
                                                        'station_name': station_name,
                                                        'station_type': station_type,
                                                        'delay_minutes': delay_minutes,
                                                        'categoryCode': cause.get('categoryCode'),
                                                        'detailedCategoryCode': cause.get('detailedCategoryCode'),
                                                        'thirdCategoryCode': cause.get('thirdCategoryCode')
                                                    }
                                                    all_causes_data.append(cause_data)
                                    except Exception as e:
                                        # Skip causes that can't be parsed
                                        continue
                        except Exception as e:
                            # Skip timetables that can't be parsed
                            continue
                            
            except Exception as e:
                st.warning(f"Error processing file {file_name}: {e}")
    
    st.success(f"✅ Successfully processed {processed_files} data files")
    return pd.DataFrame(all_causes_data)

def plot_causes_by_category(causes_df, mappings, top_n=15):
    """Create bar plot of delay causes by main category"""
    if causes_df.empty:
        return None, None
    
    # Count delays by category code
    category_counts = causes_df.groupby('categoryCode').agg({
        'delay_minutes': 'count'
    }).round(1)
    
    category_counts.columns = ['delay_count']
    category_counts = category_counts.reset_index()
    
    # Add category names
    category_counts['category_name'] = category_counts['categoryCode'].map(
        mappings.get('category', {})
    ).fillna('Unknown Category')
    
    # Sort by delay count and take top N
    category_counts = category_counts.sort_values('delay_count', ascending=False).head(top_n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Count of delays
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
    bars = ax.barh(range(len(category_counts)), category_counts['delay_count'], color=colors)
    ax.set_yticks(range(len(category_counts)))
    ax.set_yticklabels([f"{row['category_name']}\n({row['categoryCode']})" 
                        for _, row in category_counts.iterrows()], fontsize=10)
    ax.set_xlabel('Number of Delayed Trains', fontsize=12)
    ax.set_title(f'Top {top_n} Delay Causes by Frequency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig, category_counts

def plot_causes_timeline(causes_df, mappings):
    """Create timeline plot showing delay causes over time"""
    if causes_df.empty:
        return None
    
    # Check if we have enough time periods
    time_periods = causes_df.groupby(['year', 'month']).size()
    if len(time_periods) < 2:
        return None
    
    # Create monthly aggregation
    monthly_causes = causes_df.groupby(['year', 'month', 'categoryCode']).agg({
        'delay_minutes': 'count'
    }).reset_index()
    monthly_causes.columns = ['year', 'month', 'categoryCode', 'delay_count']
    
    # Add category names
    monthly_causes['category_name'] = monthly_causes['categoryCode'].map(
        mappings.get('category', {})
    ).fillna('Unknown')
    
    # Create date column
    monthly_causes['date'] = pd.to_datetime(
        monthly_causes[['year', 'month']].assign(day=1)
    )
    
    # Get top 6 categories for cleaner visualization
    top_categories = causes_df['categoryCode'].value_counts().head(6).index
    monthly_causes = monthly_causes[monthly_causes['categoryCode'].isin(top_categories)]
    
    if monthly_causes.empty:
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Use different colors and markers for each category
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_categories)))
    markers = ['o', 's', '^', 'D', 'v', '*']
    
    # Plot lines for each category
    for i, category_code in enumerate(top_categories):
        category_data = monthly_causes[monthly_causes['categoryCode'] == category_code]
        category_name = mappings.get('category', {}).get(category_code, f'Code {category_code}')
        
        if not category_data.empty:
            ax.plot(category_data['date'], category_data['delay_count'], 
                   marker=markers[i % len(markers)], label=f'{category_name} ({category_code})', 
                   linewidth=2, markersize=8, color=colors[i])
    
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Number of Delayed Trains', fontsize=12)
    ax.set_title('Delay Causes Timeline (Top 6 Categories)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_detailed_causes_for_category(causes_df, mappings, category_code, top_n=10):
    """Create detailed breakdown for a specific category"""
    if causes_df.empty:
        return None, None
    
    # Filter for the selected category
    category_data = causes_df[causes_df['categoryCode'] == category_code]
    
    if category_data.empty:
        return None, None
    
    # Group by detailed category code
    detailed_counts = category_data.groupby('detailedCategoryCode').agg({
        'delay_minutes': 'count'
    }).round(1)
    
    detailed_counts.columns = ['delay_count']
    detailed_counts = detailed_counts.reset_index()
    
    # Add detailed category names
    detailed_counts['detailed_name'] = detailed_counts['detailedCategoryCode'].map(
        mappings.get('detailed', {})
    ).fillna('Unknown Detailed Category')
    
    # Sort and limit
    detailed_counts = detailed_counts.sort_values('delay_count', ascending=False).head(top_n)
    
    if detailed_counts.empty:
        return None, None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(detailed_counts)))
    bars = ax.barh(range(len(detailed_counts)), detailed_counts['delay_count'], color=colors)
    ax.set_yticks(range(len(detailed_counts)))
    ax.set_yticklabels([f"{row['detailed_name']}\n({row['detailedCategoryCode']})" 
                       for _, row in detailed_counts.iterrows()], fontsize=10)
    ax.set_xlabel('Number of Delayed Trains', fontsize=12)
    
    category_name = mappings.get('category', {}).get(category_code, f'Code {category_code}')
    ax.set_title(f'Detailed Breakdown: {category_name} (Code {category_code})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
               f'{int(width):,}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig, detailed_counts

def create_causes_heatmap(causes_df, mappings):
    """Create a heatmap showing delay causes by month"""
    if causes_df.empty:
        return None
    
    # Check if we have multiple months
    if len(causes_df['month'].unique()) < 2:
        return None
    
    # Get top 10 categories for readable heatmap
    top_categories = causes_df['categoryCode'].value_counts().head(10).index
    filtered_df = causes_df[causes_df['categoryCode'].isin(top_categories)]
    
    # Create pivot table
    heatmap_data = filtered_df.groupby(['month', 'categoryCode']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='categoryCode', columns='month', values='count')
    heatmap_pivot = heatmap_pivot.fillna(0)
    
    # Add category names to index
    category_names = []
    for code in heatmap_pivot.index:
        name = mappings.get('category', {}).get(code, 'Unknown')
        category_names.append(f"{name}\n({code})")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Number of Incidents'})
    
    ax.set_yticklabels(category_names, rotation=0)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Delay Cause Category', fontsize=12)
    ax.set_title('Delay Causes by Month (Top 10 Categories)', fontsize=14, fontweight='bold')
    
    # Set month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    current_labels = [int(x) for x in heatmap_pivot.columns]
    new_labels = [month_names[month-1] for month in current_labels]
    ax.set_xticklabels(new_labels)
    
    plt.tight_layout()
    return fig

# Main page content
def main():
    st.title("🔍 Railway Delay Causes Analysis")
    st.markdown("""
    This page analyzes delay causes across all matched train and weather data files, 
    showing patterns and trends in what causes train delays in the Finnish railway system.
    """)
    
    # Get the date dictionary
    viewer.get_date_range()
    date_dict = viewer.get_date_dict()
    
    if not date_dict:
        st.error("No matched data files found.")
        st.stop()
    
    # Automatically load all available data files
    selected_periods = {}
    for year, months in date_dict.items():
        selected_periods[year] = months
    
    # Show summary of all files being loaded
    total_files = sum(len(months) for months in selected_periods.values())
    all_years = sorted(selected_periods.keys())
    
    if len(all_years) == 1:
        year_range = str(all_years[0])
    elif len(all_years) == 2:
        year_range = f"{all_years[0]}-{all_years[1]}"
    else:
        year_range = f"{all_years[0]}-{all_years[-1]}"
    
    st.info(f"📁 Loading all available data: {total_files} files across {len(all_years)} years ({year_range})")
    
    # Load metadata
    with st.spinner("Loading cause metadata..."):
        mappings = load_cause_metadata()
    
    if not any(mappings.values()):
        st.error("No cause metadata found. Please ensure metadata files are available in the metadata folder.")
        st.stop()
    
    with st.spinner("Extracting causes from matched files..."):
        causes_df = extract_causes_from_matched_files(selected_periods)
    
    if causes_df.empty:
        st.warning("⚠️ No delay causes data found in the selected files.")
        st.info("This could happen if:")
        st.markdown("""
        - The selected files don't contain trains with delay causes recorded
        - The 'causes' column is empty in the timetable data  
        - There are parsing issues with the causes data
        """)
        st.stop()
    
    # Calculate summary metrics for sidebar
    total_incidents = len(causes_df)
    unique_trains = causes_df['train_number'].nunique()
    unique_causes = causes_df['categoryCode'].nunique()
    avg_delay_per_incident = causes_df['delay_minutes'].mean()
    
    # SIDEBAR CONFIGURATION
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Context")
    st.sidebar.markdown(f"**Files Processed**: {total_files}")
    st.sidebar.markdown(f"**Total Incidents**: {total_incidents:,}")
    st.sidebar.markdown(f"**Unique Categories**: {unique_causes}")
    st.sidebar.markdown(f"**Period**: {year_range}")
    st.sidebar.markdown("---")
    
    # SIDEBAR PLOT SELECTION
    st.sidebar.subheader("📈 Chart Selection")
    
    # Define plot options
    plot_options = {
        "📊 Main Categories": "main_categories",
        "📈 Timeline": "timeline", 
        "🌡️ Seasonal Patterns": "seasonal_patterns",
        "🔍 Detailed Breakdown": "detailed_breakdown",
        "📋 Data Tables": "data_tables"
    }
    
    selected_plot = st.sidebar.radio(
        "Choose visualization:",
        options=list(plot_options.keys()),
        index=0
    )
    
    # Get the plot key
    plot_key = plot_options[selected_plot]
    
    # Show current selection info
    st.sidebar.markdown(f"**Current View**: {selected_plot}")
    
    # Add plot-specific controls in sidebar
    if plot_key == "main_categories":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Chart Controls:**")
        # Get the actual number of categories for dynamic range
        max_categories = causes_df['categoryCode'].nunique() if not causes_df.empty else 50
        top_n = st.sidebar.slider("Categories to display", min_value=5, max_value=max_categories, value=max_categories, key="main_top_n")
    elif plot_key == "detailed_breakdown":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Chart Controls:**")
        detailed_top_n = st.sidebar.slider("Detailed causes to display", min_value=5, max_value=20, value=10, key="detailed_top_n")
        # Category selection will be in main area since it needs the data
    else:
        # Set default values for other chart types
        max_categories = causes_df['categoryCode'].nunique() if not causes_df.empty else 50
        top_n = max_categories
        detailed_top_n = 10
    
    # MAIN CONTENT AREA - Display selected plot
    st.subheader("📈 Delay Causes Analysis")
    
    if plot_key == "main_categories":
        st.markdown("### Top Delay Causes by Category")
        
        fig_main, category_counts = plot_causes_by_category(causes_df, mappings, top_n)
        if fig_main:
            st.pyplot(fig_main)
            
            # Show insights
            if not category_counts.empty:
                worst_cause = category_counts.iloc[0]
                best_cause = category_counts.iloc[-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🔴 **Most Frequent**: {worst_cause['category_name']} (Code {worst_cause['categoryCode']}) "
                              f"- {worst_cause['delay_count']:,} incidents")
                
                with col2:
                    st.info(f"🟢 **Least Frequent**: {best_cause['category_name']} (Code {best_cause['categoryCode']}) "
                            f"- {best_cause['delay_count']:,} incidents")
        else:
            st.error("Could not generate main categories plot.")
    
    elif plot_key == "timeline":
        st.markdown("### Delay Causes Over Time")
        
        fig_timeline = plot_causes_timeline(causes_df, mappings)
        if fig_timeline:
            st.pyplot(fig_timeline)
            st.info("💡 This chart shows the top 6 most frequent delay causes over time.")
        else:
            st.info("⏳ Timeline analysis requires data spanning multiple time periods.")
    
    elif plot_key == "seasonal_patterns":
        st.markdown("### Seasonal Patterns")
        
        fig_heatmap = create_causes_heatmap(causes_df, mappings)
        if fig_heatmap:
            st.pyplot(fig_heatmap)
            st.info("🌡️ This heatmap shows how different delay causes vary by month, revealing seasonal patterns.")
        else:
            st.info("📅 Seasonal analysis requires data from multiple months.")
    
    elif plot_key == "detailed_breakdown":
        st.markdown("### Detailed Breakdown by Category")
        
        # Category selection for detailed analysis
        available_categories = causes_df['categoryCode'].value_counts()
        
        if not available_categories.empty:
            # Create selectbox with category names
            category_options = []
            for code in available_categories.index:
                name = mappings.get('category', {}).get(code, 'Unknown')
                category_options.append(f"{name} ({code}) - {available_categories[code]} incidents")
            
            selected_option = st.selectbox("Select a category for detailed analysis:", category_options)
            
            if selected_option:
                # Extract category code from selection
                selected_category_code = selected_option.split('(')[1].split(')')[0]
                
                fig_detailed, detailed_counts = plot_detailed_causes_for_category(
                    causes_df, mappings, selected_category_code, detailed_top_n
                )
                
                if fig_detailed:
                    st.pyplot(fig_detailed)
                    
                    if not detailed_counts.empty:
                        top_detailed = detailed_counts.iloc[0]
                        st.success(f"🎯 **Most common detailed cause**: {top_detailed['detailed_name']} "
                                  f"({top_detailed['detailedCategoryCode']}) - {top_detailed['delay_count']:,} incidents")
                else:
                    st.warning(f"No detailed data available for category {selected_category_code}")
        else:
            st.warning("No category data available for detailed analysis.")
    
    elif plot_key == "data_tables":
        st.markdown("### Data Tables")
        
        # Summary by category
        st.markdown("#### Summary by Main Category")
        if not causes_df.empty:
            summary_df = causes_df.groupby('categoryCode').agg({
                'delay_minutes': ['count', 'mean']
            }).round(2)
            summary_df.columns = ['Incident_Count', 'Avg_Delay_Minutes']
            summary_df = summary_df.reset_index()
            
            # Add category names
            summary_df['Category_Name'] = summary_df['categoryCode'].map(
                mappings.get('category', {})
            ).fillna('Unknown')
            
            # Reorder columns
            summary_df = summary_df[['categoryCode', 'Category_Name', 'Incident_Count', 'Avg_Delay_Minutes']]
            summary_df = summary_df.sort_values('Incident_Count', ascending=False)
            
            st.dataframe(
                summary_df.style.format({
                    'Incident_Count': '{:,}',
                    'Avg_Delay_Minutes': '{:.1f}'
                }),
                use_container_width=True
            )
        
        # Raw data sample
        st.markdown("#### Raw Data Sample (First 100 Records)")
        if not causes_df.empty:
            # Add readable category names to raw data
            display_df = causes_df.copy()
            display_df['category_name'] = display_df['categoryCode'].map(
                mappings.get('category', {})
            ).fillna('Unknown')
            
            # Reorder columns for better readability
            display_columns = ['train_number', 'departure_date', 'station_name', 'station_type', 
                             'delay_minutes', 'categoryCode', 'category_name', 'detailedCategoryCode']
            display_df = display_df[[col for col in display_columns if col in display_df.columns]]
            
            st.dataframe(display_df.head(100), use_container_width=True)
        
    # Download section
    st.subheader("💾 Download Data")
    
    if not causes_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare summary data for download
            summary_for_download = causes_df.groupby('categoryCode').agg({
                'delay_minutes': ['count', 'mean']
            }).round(2)
            summary_for_download.columns = ['Incident_Count', 'Avg_Delay_Minutes']
            summary_for_download = summary_for_download.reset_index()
            summary_for_download['Category_Name'] = summary_for_download['categoryCode'].map(
                mappings.get('category', {})
            ).fillna('Unknown')
            
            csv_summary = summary_for_download.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv_summary,
                file_name="delay_causes_summary.csv",
                mime="text/csv"
            )
        
        with col2:
            # Prepare raw data for download (with category names)
            raw_for_download = causes_df.copy()
            raw_for_download['category_name'] = raw_for_download['categoryCode'].map(
                mappings.get('category', {})
            ).fillna('Unknown')
            raw_for_download['detailed_name'] = raw_for_download['detailedCategoryCode'].map(
                mappings.get('detailed', {})
            ).fillna('Unknown')
            
            csv_raw = raw_for_download.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Data CSV",
                data=csv_raw,
                file_name="delay_causes_full_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()