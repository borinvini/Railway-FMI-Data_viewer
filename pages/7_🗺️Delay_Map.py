import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Delay Analysis Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Constants
DELAY_TABLE_PATH = "data/viewers/delay_table.csv"

# Helper functions
def load_delay_data():
    """Load and validate the delay table data"""
    if not os.path.exists(DELAY_TABLE_PATH):
        st.error(f"⚠️ Delay data file not found at: {DELAY_TABLE_PATH}")
        return None
    
    try:
        df = pd.read_csv(DELAY_TABLE_PATH)
        
        # Validate required columns
        required_columns = [
            'year', 'month', 'day_of_month', 'day_of_week',
            'delay_count_by_day', 'total_schedules_by_day', 
            'total_delay_minutes', 'max_delay_minutes',
            'total_trains_on_route', 'avg_delay_minutes'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
            
        # Create date column for easier analysis
        df['date'] = pd.to_datetime(df[['year', 'month', 'day_of_month']].rename(columns={'day_of_month': 'day'}))
        
        # Calculate normalized delays (delay percentage)
        df['delay_percentage'] = (df['delay_count_by_day'] / df['total_schedules_by_day']) * 100
        
        # Create month-year column for grouping
        df['month_year'] = df['date'].dt.to_period('M')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading delay data: {e}")
        return None

def create_monthly_delay_summary(df):
    """Create monthly summary statistics"""
    monthly_summary = df.groupby(['year', 'month']).agg({
        'delay_count_by_day': 'sum',
        'total_schedules_by_day': 'sum',
        'total_delay_minutes': 'sum',
        'max_delay_minutes': 'max',
        'total_trains_on_route': 'sum',
        'avg_delay_minutes': 'mean'
    }).reset_index()
    
    # Calculate monthly normalized delays
    monthly_summary['monthly_delay_percentage'] = (
        monthly_summary['delay_count_by_day'] / monthly_summary['total_schedules_by_day']
    ) * 100
    
    # Create month-year string for display
    monthly_summary['month_year_str'] = (
        monthly_summary['year'].astype(str) + '-' + 
        monthly_summary['month'].astype(str).str.zfill(2)
    )
    
    return monthly_summary

def create_year_range_string(selected_years):
    """Create a formatted year range string from selected years"""
    if not selected_years:
        return ""
    
    sorted_years = sorted(selected_years)
    
    if len(sorted_years) == 1:
        return str(sorted_years[0])
    elif len(sorted_years) == 2:
        return f"{sorted_years[0]}-{sorted_years[1]}"
    else:
        # For multiple non-consecutive years, show range
        return f"{sorted_years[0]}-{sorted_years[-1]}"

def plot_aggregated_normalized_delays(monthly_summary, selected_years):
    """Create visualization for normalized delays aggregated across all selected years"""
    # Aggregate data across all selected years for each month
    aggregated_data = monthly_summary.groupby('month').agg({
        'delay_count_by_day': 'sum',
        'total_schedules_by_day': 'sum'
    }).reset_index()
    
    # Calculate normalized delays for the aggregated data
    aggregated_data['aggregated_delay_percentage'] = (
        aggregated_data['delay_count_by_day'] / aggregated_data['total_schedules_by_day']
    ) * 100
    
    # Create month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    aggregated_data['month_label'] = aggregated_data['month'].map({
        i+1: month_labels[i] for i in range(12)
    })
    
    # Create dynamic year range string
    year_range = create_year_range_string(selected_years)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create single bar chart with gradient colors
    bars = ax.bar(aggregated_data['month_label'], aggregated_data['aggregated_delay_percentage'], 
                  color=plt.cm.viridis(np.linspace(0, 1, len(aggregated_data))), alpha=0.8)
    
    ax.set_title(f"Aggregated Normalized Delays by Month ({year_range})", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Delay Percentage (%)", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, percentage in zip(bars, aggregated_data['aggregated_delay_percentage']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Format y-axis to show percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig, aggregated_data

def plot_delays_per_month_year(monthly_summary):
    """Create visualization for total delays per month per year"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Map month numbers to labels for better display
    monthly_summary['month_label'] = monthly_summary['month'].map({
        i+1: month_labels[i] for i in range(12)
    })
    
    # Create grouped bar chart
    sns.barplot(data=monthly_summary, x='month_label', y='delay_count_by_day', 
                hue='year', ax=ax, palette='Set1')
    
    ax.set_title("Total Delays per Month by Year", fontsize=16, fontweight='bold')
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Number of Delays", fontsize=12)
    ax.legend(title="Year", title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis to show values in thousands
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}K' if x >= 1000 else f'{x:.0f}'))
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_normalized_delays(monthly_summary):
    """Create visualization for normalized delays (percentage)"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Map month numbers to labels for better display
    monthly_summary['month_label'] = monthly_summary['month'].map({
        i+1: month_labels[i] for i in range(12)
    })
    
    # Create grouped bar chart
    sns.barplot(data=monthly_summary, x='month_label', y='monthly_delay_percentage', 
                hue='year', ax=ax, palette='Set1')
    
    ax.set_title("Normalized Delays (Percentage) per Month by Year", fontsize=16, fontweight='bold')
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Delay Percentage (%)", fontsize=12)
    ax.legend(title="Year", title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis to show percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_delay_heatmap(df):
    """Create a heatmap showing delay patterns by day of week and month"""
    # Map day of week numbers to names
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    # Create pivot table for heatmap
    heatmap_data = df.groupby(['month', 'day_of_week'])['delay_percentage'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='month', values='delay_percentage')
    
    # Reorder days of week
    heatmap_pivot.index = heatmap_pivot.index.map(day_names)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex(day_order)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heatmap_pivot, 
        annot=True, 
        fmt='.1f', 
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Average Delay Percentage (%)'}
    )
    ax.set_title('Average Delay Percentage by Day of Week and Month', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_delay_severity_distribution(df):
    """Create visualization showing distribution of delay severity"""
    # Create delay severity categories
    df['delay_severity'] = pd.cut(
        df['avg_delay_minutes'],
        bins=[0, 5, 10, 15, 20, float('inf')],
        labels=['Very Low (0-5 min)', 'Low (5-10 min)', 'Medium (10-15 min)', 
               'High (15-20 min)', 'Very High (20+ min)']
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color palette
    colors = ['green', 'gold', 'orange', 'red', 'darkred']
    
    # Count values and create bar plot
    severity_counts = df['delay_severity'].value_counts().reindex([
        'Very Low (0-5 min)', 'Low (5-10 min)', 'Medium (10-15 min)', 
        'High (15-20 min)', 'Very High (20+ min)'
    ])
    
    bars = ax.bar(severity_counts.index, severity_counts.values, color=colors)
    
    ax.set_title("Distribution of Days by Average Delay Severity", fontsize=16, fontweight='bold')
    ax.set_xlabel("Delay Severity Category", fontsize=12)
    ax.set_ylabel("Number of Days", fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_seasonal_trends(monthly_summary):
    """Create seasonal trends visualization"""
    # Add season information
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    monthly_summary['season'] = monthly_summary['month'].apply(get_season)
    
    seasonal_data = monthly_summary.groupby(['year', 'season']).agg({
        'delay_count_by_day': 'sum',
        'monthly_delay_percentage': 'mean'
    }).reset_index()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Total delays by season
    sns.barplot(data=seasonal_data, x='season', y='delay_count_by_day', 
                hue='year', ax=ax1, palette='Set1')
    ax1.set_title('Total Delays by Season', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Total Number of Delays', fontsize=12)
    ax1.legend(title='Year')
    
    # Format y-axis to show values in thousands
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}K' if x >= 1000 else f'{x:.0f}'))
    
    # Plot 2: Average delay percentage by season
    sns.barplot(data=seasonal_data, x='season', y='monthly_delay_percentage', 
                hue='year', ax=ax2, palette='Set1')
    ax2.set_title('Average Delay Percentage by Season', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Average Delay Percentage (%)', fontsize=12)
    ax2.legend(title='Year')
    
    # Format y-axis to show percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    return fig

def plot_yearly_comparison(monthly_summary):
    """Create yearly comparison visualization"""
    yearly_data = monthly_summary.groupby('year').agg({
        'delay_count_by_day': 'sum',
        'total_schedules_by_day': 'sum',
        'total_delay_minutes': 'sum',
        'avg_delay_minutes': 'mean'
    }).reset_index()
    
    yearly_data['yearly_delay_percentage'] = (
        yearly_data['delay_count_by_day'] / yearly_data['total_schedules_by_day']
    ) * 100
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Total delays per year
    bars1 = ax1.bar(yearly_data['year'], yearly_data['delay_count_by_day'], 
                    color='steelblue', alpha=0.7)
    ax1.set_title('Total Delays per Year', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Total Delays', fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height/1000)}K',
                ha='center', va='bottom', fontweight='bold')
    
    # Delay percentage per year
    bars2 = ax2.bar(yearly_data['year'], yearly_data['yearly_delay_percentage'], 
                    color='coral', alpha=0.7)
    ax2.set_title('Delay Percentage per Year', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Delay Percentage (%)', fontsize=12)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Total schedules per year
    bars3 = ax3.bar(yearly_data['year'], yearly_data['total_schedules_by_day'], 
                    color='lightgreen', alpha=0.7)
    ax3.set_title('Total Schedules per Year', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Total Schedules', fontsize=12)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height/1000)}K',
                ha='center', va='bottom', fontweight='bold')
    
    # Average delay minutes per year
    bars4 = ax4.bar(yearly_data['year'], yearly_data['avg_delay_minutes'], 
                    color='gold', alpha=0.7)
    ax4.set_title('Average Delay Duration per Year', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Year', fontsize=12)
    ax4.set_ylabel('Average Delay (minutes)', fontsize=12)
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main application
def main():
    st.title("🗺️ Railway Delay Analysis Dashboard")
    
    # Important information about delay definition
    st.info("""
    📋 This analysis considers delays for **long distance trains only** and defines a delay as **5 minutes or higher**. 
    Delays under 5 minutes are not included in this dataset. Ref: [Väylä (Finnish Transport Infrastructure Agency)](https://vayla.fi/en/transport-network/data/statistics/railway-statistics).
    """)
    
    # Load data
    df = load_delay_data()
    
    if df is None:
        st.stop()
    
    # Data overview
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_days = len(df)
        st.metric("Total Days Analyzed", f"{total_days:,}")
    
    with col2:
        total_delays = df['delay_count_by_day'].sum()
        st.metric("Total Delays", f"{total_delays:,}")
    
    with col3:
        total_schedules = df['total_schedules_by_day'].sum()
        st.metric("Total Schedules", f"{total_schedules:,}")
    
    with col4:
        overall_delay_rate = (total_delays / total_schedules) * 100
        st.metric("Overall Delay Rate", f"{overall_delay_rate:.2f}%")
    
    # Date range info
    st.info(f"📅 **Data Period**: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Create monthly summary
    monthly_summary = create_monthly_delay_summary(df)
    
    # Year selection for plots
    st.subheader("🎯 Year Selection for Analysis")
    
    available_years = sorted(monthly_summary['year'].unique())
    
    # Set default to 2024 if available, otherwise use the most recent year
    if 'selected_years' not in st.session_state:
        if 2024 in available_years:
            st.session_state.selected_years = [2024]
        else:
            st.session_state.selected_years = [max(available_years)]
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Select All Years"):
            st.session_state.selected_years = available_years
    
    with col1:
        selected_years = st.multiselect(
            "Select years to display in charts:",
            options=available_years,
            default=st.session_state.selected_years,
            help="You can select multiple years to compare them side by side",
            key="year_selector"
        )
    
    # Update session state
    if selected_years:
        st.session_state.selected_years = selected_years
    
    # Filter data based on selected years
    if not selected_years:
        st.warning("⚠️ Please select at least one year to display charts.")
        st.stop()
    
    filtered_monthly_summary = monthly_summary[monthly_summary['year'].isin(selected_years)]
    
    if filtered_monthly_summary.empty:
        st.error("❌ No data available for the selected years.")
        st.stop()
    
    # Show selection info and create dynamic year range
    year_range = create_year_range_string(selected_years)
    
    # Main visualizations
    st.subheader("📈 Monthly Delay Analysis")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        f"Aggregated Delays (%) ({year_range})",
        "Total Delays by Month", 
        "Normalized Delays (%)", 
        "Delay Patterns", 
        "Seasonal Analysis",
        "Yearly Comparison"
    ])
    
    with tab1:
        st.markdown(f"### Aggregated Normalized Delays ({year_range})")
        st.markdown("This chart combines all selected years into a single view, showing the overall delay percentage for each month across the entire selected period.")
        
        fig_agg, aggregated_data = plot_aggregated_normalized_delays(filtered_monthly_summary, selected_years)
        st.pyplot(fig_agg)
        
    with tab2:
        st.markdown("### Total Number of Delays per Month by Year")
        st.markdown("This chart shows the absolute number of delays for each month across different years.")
        
        fig1 = plot_delays_per_month_year(filtered_monthly_summary)
        st.pyplot(fig1)

    with tab3:
        st.markdown("### Normalized Delays (Percentage) per Month by Year")
        st.markdown("This chart shows delay rates as a percentage of total schedules, allowing for fair comparison across months with different traffic volumes.")
        
        fig2 = plot_normalized_delays(filtered_monthly_summary)
        st.pyplot(fig2)

    with tab4:
        st.markdown("### Delay Patterns Analysis")
        
        # Heatmap
        st.markdown("#### Average Delay Percentage by Day of Week and Month")
        fig3 = plot_delay_heatmap(df)
        st.pyplot(fig3)
        
        # Delay severity distribution
        st.markdown("#### Distribution of Days by Average Delay Severity")
        fig4 = plot_delay_severity_distribution(df)
        st.pyplot(fig4)
    
    with tab5:
        st.markdown("### Seasonal Delay Analysis")
        st.markdown("Compare delay patterns across different seasons to identify weather or operational impacts.")
        
        fig5 = plot_seasonal_trends(filtered_monthly_summary)
        st.pyplot(fig5)
    
    with tab6:
        st.markdown("### Yearly Comparison")
        st.markdown("Compare overall performance metrics across different years.")
        
        fig6 = plot_yearly_comparison(filtered_monthly_summary)
        st.pyplot(fig6)
    
    # Explanation of normalized delay
    st.info("""
    📊 **Normalized Delay Explanation**: 
    
    ```
    Normalized Delay (%) = (Total Number of Delays ÷ Total Number of Schedules) × 100
    ```
    
    This metric shows what percentage of all scheduled trains experienced delays, making it easier to compare months with different numbers of total train schedules.
    """)  


    # Detailed statistics
    st.subheader("📋 Detailed Statistics")
    
    with st.expander("View Monthly Summary Data", expanded=False):
        st.dataframe(
            filtered_monthly_summary.style.format({
                'delay_count_by_day': '{:,}',
                'total_schedules_by_day': '{:,}',
                'total_delay_minutes': '{:,}',
                'monthly_delay_percentage': '{:.2f}%',
                'avg_delay_minutes': '{:.2f}'
            })
        )
    
    with st.expander("View Raw Daily Data", expanded=False):
        # Show filtered daily data
        filtered_daily_df = df[df['year'].isin(selected_years)]
        display_df = filtered_daily_df[['date', 'delay_count_by_day', 'total_schedules_by_day', 
                        'delay_percentage', 'total_delay_minutes', 'avg_delay_minutes']].copy()
        
        st.dataframe(
            display_df.style.format({
                'delay_count_by_day': '{:,}',
                'total_schedules_by_day': '{:,}',
                'delay_percentage': '{:.2f}%',
                'total_delay_minutes': '{:,}',
                'avg_delay_minutes': '{:.2f}'
            })
        )

    
    # Download options
    st.subheader("💾 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Monthly Summary CSV"):
            csv = filtered_monthly_summary.to_csv(index=False)
            years_suffix = "_".join(map(str, sorted(selected_years)))
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name=f"monthly_delay_summary_{years_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Download Full Daily Data CSV"):
            # Filter daily data for selected years
            filtered_daily_df = df[df['year'].isin(selected_years)]
            csv = filtered_daily_df.to_csv(index=False)
            years_suffix = "_".join(map(str, sorted(selected_years)))
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name=f"daily_delay_data_{years_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()