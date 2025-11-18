import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import os
import io
import glob
from datetime import datetime
from ast import literal_eval
from wordcloud import WordCloud

# IEEE Standard Settings for publication-quality figures
IEEE_SETTINGS = {
    'figure.figsize': (3.5, 2.8),  # Double-column width × height (inches)
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,  # 8pt for IEEE
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.0,  # 1pt for main plot lines
    'lines.markersize': 3,   # 3pt for markers
    'grid.linewidth': 0.5,   # 0.5pt for grid lines
    'axes.linewidth': 0.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}

# Page configuration
st.set_page_config(
    page_title="Delay Analysis Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Constants - Updated default path
DELAY_TABLE_DEFAULT_PATH = "data/viewers/delay_maps/delay_table_differenceInMinutes.csv"
DELAY_MAPS_DIRECTORY = "data/viewers/delay_maps"

# Day of week mapping (1-based indexing as used in the data)
DAY_OF_WEEK_MAPPING = {
    1: "Monday",
    2: "Tuesday", 
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday"
}

# Helper functions
def find_available_delay_files():
    """Find all available delay table files in the delay_maps directory"""
    if not os.path.exists(DELAY_MAPS_DIRECTORY):
        return []
    
    # Look for CSV files that start with "delay_table"
    pattern = os.path.join(DELAY_MAPS_DIRECTORY, "delay_table*.csv")
    files = glob.glob(pattern)
    
    # Return just the filenames, not full paths
    return [os.path.basename(f) for f in files]

def load_delay_data(file_path=None):
    """Load and validate the delay table data from specified path"""
    if file_path is None:
        file_path = DELAY_TABLE_DEFAULT_PATH
    
    if not os.path.exists(file_path):
        st.error(f"⚠️ Delay data file not found at: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
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
        
        # Check if the new top_10_common_delays column exists
        has_common_delays = 'top_10_common_delays' in df.columns
        if not has_common_delays:
            st.warning("⚠️ The 'top_10_common_delays' column is not available in this dataset. Some advanced analyses will be skipped.")
            
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

def get_season(month):
    """Convert month number to season"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def plot_aggregated_normalized_delays(monthly_summary, selected_years):
    """
    Create IEEE-compliant visualization for aggregated normalized delays by month.
    
    IEEE Standards Applied:
    - Font: Times New Roman, 8pt
    - Figure size: 7.16" × 2.8" (double-column width)
    - Line width: 1.0pt for main lines
    - Marker size: 3pt
    - Grid lines: 0.5pt
    - No title (use caption below figure)
    - Reduced alpha on background zones
    
    Parameters:
    -----------
    monthly_summary : pd.DataFrame
        DataFrame with monthly delay summary data
    selected_years : list
        List of years to include in analysis
    
    Returns:
    --------
    tuple: (fig, aggregated_data) for display and download functionality
    """
    # Aggregate data across all selected years by month
    aggregated_data = monthly_summary.groupby('month').agg({
        'delay_count_by_day': 'sum',
        'total_schedules_by_day': 'sum'
    }).reset_index()
    
    # Calculate normalized delays for the aggregated data
    aggregated_data['aggregated_delay_percentage'] = (
        aggregated_data['delay_count_by_day'] / aggregated_data['total_schedules_by_day']
    ) * 100
    
    # Create month labels
    month_labels = ['1', '2', '3', '4', '5', '6',
                   '7', '8', '9', '10', '11', '12']
    
    aggregated_data['month_label'] = aggregated_data['month'].map({
        i+1: month_labels[i] for i in range(12)
    })
    
    # Create dynamic year range string
    year_range = create_year_range_string(selected_years)
    
    # Apply IEEE settings
    with plt.rc_context(IEEE_SETTINGS):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        
        # Create color mapping based on delay percentage values
        norm = Normalize(
            aggregated_data['aggregated_delay_percentage'].min(), 
            aggregated_data['aggregated_delay_percentage'].max()
        )
        colors = get_cmap('RdYlGn_r')(norm(aggregated_data['aggregated_delay_percentage']))
        
        # Create bar chart with IEEE-compliant styling
        bars = ax.bar(
            aggregated_data['month_label'], 
            aggregated_data['aggregated_delay_percentage'], 
            color=colors, 
            alpha=0.8, 
            edgecolor='black', 
            linewidth=0.5
        )
        
        # NO TITLE - IEEE figures use captions below
        # (title removed for IEEE compliance)
        
        # Axis labels with IEEE font
        ax.set_xlabel("Month", fontsize=8, family='serif')
        ax.set_ylabel("Delay Percentage Normalized (%)", fontsize=8, family='serif')

        # Set y-axis limit to 30%
        ax.set_ylim(0, 31)
        
        # Grid with reduced prominence (0.5pt, reduced alpha)
        ax.grid(True, alpha=0.2, linewidth=0.5, axis='y', linestyle='-')
        
        # Add value labels on bars
        for bar, percentage in zip(bars, aggregated_data['aggregated_delay_percentage']):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.15,
                f'{percentage:.1f}',
                ha='center', 
                va='bottom', 
                fontsize=7,  # Slightly smaller for value labels
                family='serif'
            )
        
        # Format y-axis to show percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}%'))
        
        # Tick parameters with IEEE line widths
        ax.tick_params(width=0.5, labelsize=8)
        
        # No rotation for x-axis labels
        plt.xticks(rotation=0)
        
        # Tight layout to optimize space
        plt.tight_layout()
        
        return fig, aggregated_data
    
def save_figure_as_pdf(fig):
    """
    Save matplotlib figure as PDF (IEEE preferred format).
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    
    Returns:
    --------
    bytes : bytes
        PDF file content as bytes
    """
    bytes_io = io.BytesIO()
    fig.savefig(
        bytes_io, 
        format='pdf', 
        dpi=300,  # High resolution for publication
        bbox_inches='tight',
        pad_inches=0.05  # Minimal padding for IEEE
    )
    bytes_io.seek(0)
    return bytes_io.getvalue()

def plot_aggregated_seasonal_delays(monthly_summary, selected_years):
    """Create visualization for normalized delays aggregated across all selected years by season"""
    # Add season information to monthly summary
    monthly_summary_with_season = monthly_summary.copy()
    monthly_summary_with_season['season'] = monthly_summary_with_season['month'].apply(get_season)
    
    # Aggregate data across all selected years for each season
    aggregated_seasonal_data = monthly_summary_with_season.groupby('season').agg({
        'delay_count_by_day': 'sum',
        'total_schedules_by_day': 'sum'
    }).reset_index()
    
    # Calculate normalized delays for the aggregated seasonal data
    aggregated_seasonal_data['aggregated_delay_percentage'] = (
        aggregated_seasonal_data['delay_count_by_day'] / aggregated_seasonal_data['total_schedules_by_day']
    ) * 100
    
    # Define season order and colors
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    season_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']  # Blue, Green, Orange, Red
    
    # Reorder the data according to season order
    aggregated_seasonal_data['season'] = pd.Categorical(
        aggregated_seasonal_data['season'], 
        categories=season_order, 
        ordered=True
    )
    aggregated_seasonal_data = aggregated_seasonal_data.sort_values('season')
    
    # Create dynamic year range string
    year_range = create_year_range_string(selected_years)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create single bar chart with seasonal colors
    bars = ax.bar(aggregated_seasonal_data['season'], aggregated_seasonal_data['aggregated_delay_percentage'], 
                  color=season_colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_title(f"Aggregated Normalized Delays by Season ({year_range})", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Delay Percentage (%)", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, percentage in zip(bars, aggregated_seasonal_data['aggregated_delay_percentage']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Format y-axis to show percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig, aggregated_seasonal_data

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
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.1f}K' if x >= 1000 else f'{x:.0f}'))
    
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
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_delay_heatmap(df):
    """Create a heatmap showing delay patterns by day of week and month"""
    # Use the predefined day of week mapping
    day_names = DAY_OF_WEEK_MAPPING
    
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
    # Create delay severity categories (starting from 5 min since delays <5 min are not in dataset)
    df['delay_severity'] = pd.cut(
        df['avg_delay_minutes'],
        bins=[5, 10, 15, 20, float('inf')],
        labels=['Low (5-10 min)', 'Medium (10-15 min)', 
               'High (15-20 min)', 'Very High (20+ min)']
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color palette (removed green since we don't have Very Low category)
    colors = ['gold', 'orange', 'red', 'darkred']
    
    # Count values and create bar plot
    severity_counts = df['delay_severity'].value_counts().reindex([
        'Low (5-10 min)', 'Medium (10-15 min)', 
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
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.1f}K' if x >= 1000 else f'{x:.0f}'))
    
    # Plot 2: Average delay percentage by season
    sns.barplot(data=seasonal_data, x='season', y='monthly_delay_percentage', 
                hue='year', ax=ax2, palette='Set1')
    ax2.set_title('Average Delay Percentage by Season', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Average Delay Percentage (%)', fontsize=12)
    ax2.legend(title='Year')
    
    # Format y-axis to show percentage
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
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
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))
    
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
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
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

# NEW DELAY DISTRIBUTION FUNCTIONS
def plot_daily_delay_distribution(df, selected_years):
    """
    Create visualization showing the distribution of daily average delay durations by year.
    This shows what delay ranges were most common on a day-to-day basis.
    """
    # Filter data for selected years
    filtered_df = df[df['year'].isin(selected_years)]
    
    if filtered_df.empty:
        return None, None
    
    # Define delay range bins (in minutes)
    bins = [5, 7, 9, 11, 13, 15, 18, 22, float('inf')]
    labels = ['5-7', '7-9', '9-11', '11-13', '13-15', '15-18', '18-22', '22+']
    
    # Create delay range categories
    filtered_df = filtered_df.copy()
    filtered_df['delay_range'] = pd.cut(
        filtered_df['avg_delay_minutes'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    # Count occurrences by year and delay range
    delay_counts = filtered_df.groupby(['year', 'delay_range']).size().reset_index(name='count')
    
    # Pivot for easier plotting
    delay_pivot = delay_counts.pivot(index='delay_range', columns='year', values='count').fillna(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for years
    colors = get_cmap('Set1')(np.linspace(0, 1, len(selected_years)))
    
    # Create grouped bar chart
    x = np.arange(len(labels))
    width = 0.8 / len(selected_years)
    
    for i, year in enumerate(selected_years):
        if year in delay_pivot.columns:
            values = delay_pivot[year].values
            bars = ax.bar(x + i * width, values, width, 
                         label=str(year), color=colors[i], alpha=0.8)
            
            # Add value labels on bars (only for values > 0)
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title('Distribution of Daily Average Delay Ranges by Year', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Daily Average Delay Range (minutes)', fontsize=12)
    ax.set_ylabel('Number of Days', fontsize=12)
    ax.set_xticks(x + width * (len(selected_years) - 1) / 2)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(title='Year', title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, delay_pivot

def plot_delay_pattern_heatmap(df, selected_years):
    """
    Create a heatmap showing delay range patterns by year and month.
    """
    # Filter data for selected years
    filtered_df = df[df['year'].isin(selected_years)]
    
    if filtered_df.empty:
        return None
    
    # Define delay range bins
    bins = [5, 8, 11, 14, 17, 20, float('inf')]
    labels = ['5-8', '8-11', '11-14', '14-17', '17-20', '20+']
    
    # Create delay range categories
    filtered_df = filtered_df.copy()
    filtered_df['delay_range'] = pd.cut(
        filtered_df['avg_delay_minutes'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    # Count by year-month and delay range
    heatmap_data = filtered_df.groupby(['year', 'month', 'delay_range']).size().reset_index(name='count')
    
    # Create year-month column
    heatmap_data['year_month'] = heatmap_data['year'].astype(str) + '-' + heatmap_data['month'].astype(str).str.zfill(2)
    
    # Pivot for heatmap
    heatmap_pivot = heatmap_data.pivot_table(
        index='delay_range', 
        columns='year_month', 
        values='count', 
        fill_value=0
    )
    
    # Convert to integers to fix formatting issue
    heatmap_pivot = heatmap_pivot.astype(int)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create heatmap with proper formatting
    sns.heatmap(
        heatmap_pivot, 
        annot=True, 
        fmt='d', 
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Number of Days'},
        linewidths=0.5
    )
    
    ax.set_title('Delay Range Patterns by Year-Month', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year-Month', fontsize=12)
    ax.set_ylabel('Daily Average Delay Range (minutes)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_delay_violin_distribution(df, selected_years):
    """
    Create violin plots showing the distribution shape of daily average delays by year.
    """
    # Filter data for selected years
    filtered_df = df[df['year'].isin(selected_years)]
    
    if filtered_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create violin plot
    violin_parts = ax.violinplot([filtered_df[filtered_df['year'] == year]['avg_delay_minutes'].values 
                                 for year in selected_years], 
                                positions=range(len(selected_years)),
                                showmeans=True, showmedians=True)
    
    # Customize violin colors
    colors = get_cmap('Set1')(np.linspace(0, 1, len(selected_years)))
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Set labels
    ax.set_title('Distribution of Daily Average Delays by Year', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Daily Average Delay (minutes)', fontsize=12)
    ax.set_xticks(range(len(selected_years)))
    ax.set_xticklabels([str(year) for year in selected_years])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = []
    for year in selected_years:
        year_data = filtered_df[filtered_df['year'] == year]['avg_delay_minutes']
        median_val = year_data.median()
        mode_val = year_data.mode().iloc[0] if not year_data.mode().empty else "N/A"
        stats_text.append(f"{year}: Median={median_val:.1f}min")
    
    ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    return fig

def parse_common_delays(df):
    """
    Parse the top_10_common_delays column and return a DataFrame with expanded delay values.
    """
    parsed_data = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('top_10_common_delays')):
            try:
                # Parse the string representation of the list
                delay_list_str = str(row['top_10_common_delays']).strip()
                
                # Skip empty or invalid strings
                if not delay_list_str or delay_list_str in ['nan', 'None', '[]', '']:
                    continue
                
                delay_list = literal_eval(delay_list_str)
                
                # Make sure it's a list
                if not isinstance(delay_list, list):
                    continue
                
                # Add each delay value with its position in the top 10
                for position, delay_value in enumerate(delay_list, 1):
                    # Make sure delay_value is numeric
                    try:
                        delay_value = float(delay_value)
                        if pd.isna(delay_value):
                            continue
                    except (ValueError, TypeError):
                        continue
                        
                    parsed_data.append({
                        'date': row['date'] if 'date' in row else f"{row['year']}-{row['month']:02d}-{row['day_of_month']:02d}",
                        'year': row['year'],
                        'month': row['month'],
                        'day_of_month': row['day_of_month'],
                        'day_of_week': row['day_of_week'],
                        'delay_value': delay_value,
                        'position_in_top10': position,
                        'total_delays_that_day': row['delay_count_by_day']
                    })
            except Exception as e:
                # Skip rows where parsing fails
                continue
    
    return pd.DataFrame(parsed_data)

def plot_most_common_delay_values(df, selected_years, top_n=50):
    """
    Create a word cloud showing the most frequently occurring delay values across all selected years.
    """
    # Filter data and parse common delays
    filtered_df = df[df['year'].isin(selected_years)]
    parsed_delays = parse_common_delays(filtered_df)
    
    if parsed_delays.empty:
        return None, None
    
    # Count frequency of each delay value across all days - increased to show more values
    delay_counts = parsed_delays['delay_value'].value_counts().head(top_n)
    
    if delay_counts.empty:
        return None, None
    
    # Create word cloud data - use just the number
    wordcloud_dict = {}
    for delay_val, count in delay_counts.items():
        # Use just the number as text
        delay_text = str(int(delay_val))
        wordcloud_dict[delay_text] = count
    
    # Create word cloud
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Custom colormap - use colors that represent delay severity
    def delay_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # Extract delay value from word (it's already just the number)
        delay_val = int(word)
        
        # Color based on delay severity: green for short delays, red for long delays
        if delay_val <= 6:
            return f"rgb(34, 139, 34)"  # Green for short delays
        elif delay_val <= 10:
            return f"rgb(255, 165, 0)"  # Orange for medium delays  
        elif delay_val <= 15:
            return f"rgb(255, 69, 0)"   # Red-orange for longer delays
        else:
            return f"rgb(139, 0, 0)"    # Dark red for very long delays
    
    # Generate word cloud with denser parameters
    wordcloud = WordCloud(
        width=1000,  # Reduced width for denser packing
        height=600,  # Reduced height for denser packing
        background_color='white',
        max_words=top_n,
        relative_scaling=0.3,  # Reduced to make size differences less dramatic
        min_font_size=25,      # Increased minimum font size
        max_font_size=100,      # Reduced maximum font size
        color_func=delay_color_func,
        prefer_horizontal=0.5, # More balanced horizontal/vertical
        random_state=42,
        collocations=False,    # Prevent word combinations
        margin=5,              # Reduced margin for tighter packing
        scale=2                # Higher resolution for better quality
    ).generate_from_frequencies(wordcloud_dict)
    
    # Display word cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Most Frequently Occurring Delay Values (minutes) ({create_year_range_string(selected_years)})', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, delay_counts

def plot_delay_value_by_position(df, selected_years):
    """
    Show which delay values appear most frequently in each position of the top 10.
    """
    # Filter data and parse common delays
    filtered_df = df[df['year'].isin(selected_years)]
    parsed_delays = parse_common_delays(filtered_df)
    
    if parsed_delays.empty:
        return None
    
    # Get the most common delay values for positions 1-5 (top 5 positions)
    position_analysis = parsed_delays[parsed_delays['position_in_top10'] <= 5].groupby(
        ['position_in_top10', 'delay_value']
    ).size().reset_index(name='frequency')
    
    # Get top 3 delay values for each position
    top_delays_by_position = position_analysis.groupby('position_in_top10').apply(
        lambda x: x.nlargest(3, 'frequency')
    ).reset_index(drop=True)
    
    if top_delays_by_position.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a pivot table for easier plotting
    pivot_data = top_delays_by_position.pivot_table(
        index='delay_value', 
        columns='position_in_top10', 
        values='frequency', 
        fill_value=0
    )
    
    # Convert to integers to fix formatting issue
    pivot_data = pivot_data.astype(int)
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='d', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Frequency'})
    
    ax.set_title(f'Delay Values by Position in Daily Top 10 ({create_year_range_string(selected_years)})', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Position in Daily Top 10', fontsize=12)
    ax.set_ylabel('Delay Value (minutes)', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_common_delays_by_month(df, selected_years):
    """
    Show seasonal patterns in the most common delay values.
    """
    # Filter data and parse common delays
    filtered_df = df[df['year'].isin(selected_years)]
    parsed_delays = parse_common_delays(filtered_df)
    
    if parsed_delays.empty:
        return None
    
    # Focus on the most common delay values overall
    top_delay_values = parsed_delays['delay_value'].value_counts().head(8).index
    
    # Filter for these top delay values and count by month
    monthly_delay_patterns = parsed_delays[
        parsed_delays['delay_value'].isin(top_delay_values)
    ].groupby(['month', 'delay_value']).size().reset_index(name='frequency')
    
    if monthly_delay_patterns.empty:
        return None
    
    # Create pivot table for heatmap
    pivot_monthly = monthly_delay_patterns.pivot_table(
        index='delay_value', 
        columns='month', 
        values='frequency', 
        fill_value=0
    )
    
    # Convert to integers to fix formatting issue
    pivot_monthly = pivot_monthly.astype(int)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(pivot_monthly, annot=True, fmt='d', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Frequency in Top 10'})
    
    ax.set_title(f'Seasonal Patterns of Most Common Delay Values ({create_year_range_string(selected_years)})', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Delay Value (minutes)', fontsize=12)
    
    # Set month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if len(pivot_monthly.columns) > 0:
        ax.set_xticklabels([month_labels[i-1] for i in sorted(pivot_monthly.columns)])
    
    plt.tight_layout()
    return fig

def plot_delay_consistency_analysis(df, selected_years):
    """
    Analyze how consistent the top delay values are across different years.
    """
    # Filter data and parse common delays
    filtered_df = df[df['year'].isin(selected_years)]
    parsed_delays = parse_common_delays(filtered_df)
    
    if parsed_delays.empty or len(selected_years) < 2:
        return None
    
    # Get top delay values for each year
    yearly_top_delays = {}
    for year in selected_years:
        year_delays = parsed_delays[parsed_delays['year'] == year]
        top_delays = year_delays['delay_value'].value_counts().head(10)
        yearly_top_delays[year] = set(top_delays.index)
    
    # Find common delay values across all years
    if len(selected_years) >= 2:
        common_delays = set.intersection(*yearly_top_delays.values())
        
        # Create comparison data
        comparison_data = []
        for delay_val in sorted(common_delays):
            row = {'delay_value': delay_val}
            for year in selected_years:
                year_delays = parsed_delays[
                    (parsed_delays['year'] == year) & 
                    (parsed_delays['delay_value'] == delay_val)
                ]
                row[f'frequency_{year}'] = len(year_delays)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create grouped bar chart
            x = np.arange(len(comparison_df))
            width = 0.8 / len(selected_years)
            
            colors = get_cmap('Set1')(np.linspace(0, 1, len(selected_years)))
            
            for i, year in enumerate(selected_years):
                col_name = f'frequency_{year}'
                if col_name in comparison_df.columns:
                    bars = ax.bar(x + i * width, comparison_df[col_name], 
                                 width, label=str(year), color=colors[i], alpha=0.8)
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{int(height)}',
                                   ha='center', va='bottom', fontsize=9)
            
            ax.set_title('Consistency of Most Common Delay Values Across Years', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Delay Value (minutes)', fontsize=12)
            ax.set_ylabel('Frequency in Top 10', fontsize=12)
            ax.set_xticks(x + width * (len(selected_years) - 1) / 2)
            ax.set_xticklabels([f'{int(val)}' for val in comparison_df['delay_value']])
            ax.legend(title='Year')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            return fig
    
    return None

# Main application
def main():
    st.title("🗺️ Railway Delay Analysis Dashboard")
    
    # FILE SELECTION SECTION
    st.subheader("📁 Data Source Selection")
    
    # Find available delay files
    available_files = find_available_delay_files()
    
    if not available_files:
        st.error(f"⚠️ No delay table files found in `{DELAY_MAPS_DIRECTORY}`. Please ensure the directory exists and contains delay_table*.csv files.")
        st.stop()
    
    # Set default file
    default_file = "delay_table_differenceInMinutes.csv"
    if default_file in available_files:
        default_index = available_files.index(default_file)
    else:
        default_index = 0
    
    # File selection interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_file = st.selectbox(
            "Select delay analysis file:",
            options=available_files,
            index=default_index,
            help="Choose which delay analysis dataset to load. Different files may contain different target variables or time periods."
        )
    
    with col2:
        if st.button("🔄 Refresh File List"):
            st.experimental_rerun()
    
    # Display selected file info
    selected_file_path = os.path.join(DELAY_MAPS_DIRECTORY, selected_file)
    
    # Extract target variable from filename for display
    target_variable = "Unknown"
    if "differenceInMinutes" in selected_file:
        target_variable = "Delay Duration (minutes)"
    elif "trainDelayed" in selected_file:
        target_variable = "Delay Occurrence (binary)"
    elif "cancelled" in selected_file:
        target_variable = "Cancellation"
    
    st.info(f"📊 **Selected Dataset**: `{selected_file}` | **Target Variable**: {target_variable}")
    
    # Important information about delay definition
    st.info("""
    📋 This analysis considers delays for **long distance trains only** and defines a delay as **5 minutes or higher**. 
    Delays under 5 minutes are not included in this dataset. Ref: [Väylä (Finnish Transport Infrastructure Agency)](https://vayla.fi/en/transport-network/data/statistics/railway-statistics).
    """)
    
    # Load data
    df = load_delay_data(selected_file_path)
    
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
    
    # SIDEBAR CONFIGURATION
    st.sidebar.header("🎯 Analysis Configuration")
    
    # Show current file in sidebar
    st.sidebar.markdown(f"**Current File**: `{selected_file}`")
    st.sidebar.markdown(f"**Target Variable**: {target_variable}")
    st.sidebar.markdown("---")
    
    # Year selection for plots
    st.sidebar.subheader("Year Selection")
    
    available_years = sorted(monthly_summary['year'].unique())
    
    # Set default to 2024 if available, otherwise use the most recent year
    if 'selected_years' not in st.session_state:
        if 2024 in available_years:
            st.session_state.selected_years = [2024]
        else:
            st.session_state.selected_years = [max(available_years)]
    
    # Select All Years button
    if st.sidebar.button("Select All Years"):
        st.session_state.selected_years = available_years
    
    # Year multiselect
    selected_years = st.sidebar.multiselect(
        "Select years to display:",
        options=available_years,
        default=st.session_state.selected_years,
        help="You can select multiple years to compare them side by side"
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
    
    # SIDEBAR PLOT SELECTION
    st.sidebar.subheader("📈 Chart Selection")
    
    # Define plot options
    plot_options = {
        f"📊 Aggregated Delays (%) - {year_range}": "aggregated_delays",
        "📈 Total Delays by Month": "total_delays",
        "📊 Normalized Delays (%)": "normalized_delays",
        "🌦️ Seasonal Analysis": "seasonal_analysis",
        "📅 Yearly Comparison": "yearly_comparison",
        "📆 Day of Week Analysis": "day_of_week"
    }
    
    selected_plot = st.sidebar.radio(
        "Choose visualization:",
        options=list(plot_options.keys()),
        index=0
    )
    
    # Get the plot key
    plot_key = plot_options[selected_plot]
    
    # MAIN CONTENT AREA - Display selected plot
    st.subheader("📈 Monthly Delay Analysis")
    
    if plot_key == "aggregated_delays":
        st.subheader("📊 Monthly Delay Analysis - Aggregated Normalized Delays")
        
        st.info("""
        **IEEE Publication Format**: This figure follows IEEE publication standards with:
        - Times New Roman font at 8pt
        - 7.16" × 2.8" figure size (double-column width)
        - Vector format export (PDF) for publication-quality
        """)
        
        # Create IEEE-compliant figure
        fig_agg, aggregated_data = plot_aggregated_normalized_delays(filtered_monthly_summary, selected_years)
        
        # Display the figure
        st.pyplot(fig_agg)
        
        # Add IEEE-style caption below the figure
        year_range = create_year_range_string(selected_years)
        st.caption(
            f"Fig. 1. Monthly delay analysis showing aggregated normalized delays across {year_range}. "
            f"Data represents the percentage of trains delayed ≥5 minutes relative to total scheduled "
            f"trains per month. Color intensity indicates delay severity (green: low delays, red: high delays)."
        )
        
        # Create download buttons for IEEE formats
        st.markdown("---")
        st.markdown("### 📥 Download IEEE-Compliant Figure")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # PDF download button (IEEE preferred)
            pdf_bytes = save_figure_as_pdf(fig_agg)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"delay_analysis_ieee_{year_range}.pdf",
                mime="application/pdf",
                help="Download in PDF format (IEEE preferred vector format)",
                use_container_width=True
            )
        
        with col2:
            # Also offer CSV data download
            csv_data = aggregated_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Data (CSV)",
                data=csv_data,
                file_name=f"delay_analysis_data_{year_range}.csv",
                mime="text/csv",
                help="Download the underlying data",
                use_container_width=True
            )
        
        with col3:
            st.info("💡 **Tip**: Use PDF for publications. The figure follows IEEE standards for journal submissions.")
        
        # Display summary statistics
        st.markdown("---")
        st.markdown("### 📈 Aggregated Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_delay_month = aggregated_data.loc[aggregated_data['aggregated_delay_percentage'].idxmax()]
            st.metric(
                label="Worst Month",
                value=max_delay_month['month_label'],
                delta=f"{max_delay_month['aggregated_delay_percentage']:.1f}%"
            )
        
        with col2:
            min_delay_month = aggregated_data.loc[aggregated_data['aggregated_delay_percentage'].idxmin()]
            st.metric(
                label="Best Month",
                value=min_delay_month['month_label'],
                delta=f"{min_delay_month['aggregated_delay_percentage']:.1f}%"
            )
        
        with col3:
            avg_delay = aggregated_data['aggregated_delay_percentage'].mean()
            st.metric(
                label="Average",
                value=f"{avg_delay:.1f}%"
            )
        
        with col4:
            total_delays = aggregated_data['delay_count_by_day'].sum()
            st.metric(
                label="Total Delays",
                value=f"{total_delays:,}"
            )
        
        # Show data table (expandable)
        with st.expander("🔍 View Detailed Monthly Data"):
            display_df = aggregated_data[['month_label', 'aggregated_delay_percentage', 
                                          'delay_count_by_day', 'total_schedules_by_day']].copy()
            display_df.columns = ['Month', 'Delay %', 'Total Delays', 'Total Schedules']
            st.dataframe(
                display_df.style.format({
                    'Delay %': '{:.2f}%',
                    'Total Delays': '{:,}',
                    'Total Schedules': '{:,}'
                }).background_gradient(subset=['Delay %'], cmap='RdYlGn_r'),
                use_container_width=True
            )
        
        # Close figure to free memory
        plt.close(fig_agg)
        
    elif plot_key == "total_delays":
        st.markdown("### Total Number of Delays per Month by Year")
        st.markdown("This chart shows the absolute number of delays for each month across different years.")
        
        fig1 = plot_delays_per_month_year(filtered_monthly_summary)
        st.pyplot(fig1)

    elif plot_key == "normalized_delays":
        st.markdown("### Normalized Delays (Percentage) per Month by Year")
        st.markdown("This chart shows delay rates as a percentage of total schedules, allowing for fair comparison across months with different traffic volumes.")
        
        fig2 = plot_normalized_delays(filtered_monthly_summary)
        st.pyplot(fig2)

    elif plot_key == "seasonal_analysis":
        st.markdown("### Seasonal Delay Analysis")
        st.markdown("Compare delay patterns across different seasons to identify weather or operational impacts.")
        
        # Add season definition note
        st.info("""
        🗓️ **Season Definitions:**
        - **Winter**: December, January, February
        - **Spring**: March, April, May  
        - **Summer**: June, July, August
        - **Autumn**: September, October, November
        """)
        
        # First plot: Year-by-year seasonal comparison
        st.markdown("#### Seasonal Analysis by Year")
        st.markdown("This view shows how delays vary by season across different years, allowing for year-to-year comparison.")
        
        fig5 = plot_seasonal_trends(filtered_monthly_summary)
        st.pyplot(fig5)
        
        # Second plot: Aggregated seasonal analysis
        st.markdown("#### Aggregated Seasonal Analysis")
        st.markdown(f"This chart combines all selected years ({year_range}) into a single seasonal view, showing the overall delay percentage for each season across the entire selected period.")
        
        fig_seasonal_agg, seasonal_agg_data = plot_aggregated_seasonal_delays(filtered_monthly_summary, selected_years)
        st.pyplot(fig_seasonal_agg)
        
        # Show seasonal insights
        if not seasonal_agg_data.empty:
            worst_season = seasonal_agg_data.loc[seasonal_agg_data['aggregated_delay_percentage'].idxmax()]
            best_season = seasonal_agg_data.loc[seasonal_agg_data['aggregated_delay_percentage'].idxmin()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="🔴 Highest Delay Season", 
                    value=f"{worst_season['season']}", 
                    delta=f"{worst_season['aggregated_delay_percentage']:.1f}%"
                )
            
            with col2:
                st.metric(
                    label="🟢 Lowest Delay Season", 
                    value=f"{best_season['season']}", 
                    delta=f"{best_season['aggregated_delay_percentage']:.1f}%"
                )
    
    elif plot_key == "yearly_comparison":
        st.markdown("### Yearly Comparison")
        st.markdown("Compare overall performance metrics across different years.")
        
        # Original yearly comparison charts
        st.markdown("#### Overall Performance Metrics")
        fig6 = plot_yearly_comparison(filtered_monthly_summary)
        st.pyplot(fig6)
        
        # NEW: Delay Distribution Analysis
        st.markdown("#### Most Common Daily Average Delay Ranges")
        st.markdown("""
        This analysis shows what delay durations were most common on a day-to-day basis for each year. 
        For example, if 2020 has a high bar in the '9-11' range, it means many days that year had average delays between 9-11 minutes.
        """)
        
        # Filter daily data for selected years
        filtered_daily_df = df[df['year'].isin(selected_years)]
        
        if not filtered_daily_df.empty:
            # Distribution bar chart
            fig_dist, delay_pivot = plot_daily_delay_distribution(filtered_daily_df, selected_years)
            if fig_dist:
                st.pyplot(fig_dist)
                
                # Show insights
                st.markdown("##### 📊 Key Insights")
                insights_cols = st.columns(len(selected_years))
                
                for i, year in enumerate(selected_years):
                    with insights_cols[i]:
                        if year in delay_pivot.columns:
                            year_data = delay_pivot[year]
                            most_common_range = year_data.idxmax()
                            most_common_count = year_data.max()
                            
                            st.metric(
                                label=f"🎯 {year} Most Common Range",
                                value=f"{most_common_range} min",
                                delta=f"{int(most_common_count)} days"
                            )
            
            # Violin plot for distribution shape
            st.markdown("#### Distribution Shape Analysis")
            st.markdown("Violin plots show the full distribution shape of daily average delays, revealing patterns beyond just averages.")
            
            fig_violin = plot_delay_violin_distribution(filtered_daily_df, selected_years)
            if fig_violin:
                st.pyplot(fig_violin)
            
            # Heatmap for monthly patterns
            st.markdown("#### Monthly Delay Range Patterns")
            st.markdown("This heatmap reveals seasonal patterns in delay ranges across different months and years.")
            
            fig_heatmap = plot_delay_pattern_heatmap(filtered_daily_df, selected_years)
            if fig_heatmap:
                st.pyplot(fig_heatmap)
            
            # NEW: Common Delay Values Analysis
            if 'top_10_common_delays' in filtered_daily_df.columns:
                st.markdown("#### Most Common Specific Delay Values")
                st.markdown("""
                This analysis examines the specific delay durations that occur most frequently each day. 
                The word cloud shows which exact delay values (e.g., 5, 6, 7 minutes) are most problematic across the railway system,
                with larger text indicating more frequent delays and colors representing delay severity.
                """)
                
                # Most frequently occurring delay values - now as word cloud
                st.markdown("##### 🎯 Most Frequently Occurring Delay Values")
                fig_common, delay_counts = plot_most_common_delay_values(filtered_daily_df, selected_years)
                if fig_common:
                    st.pyplot(fig_common)
                    
                    # Add explanation of the word cloud
                    st.info("""
                    💡 **Word Cloud Guide:**
                    - **Size**: Larger text = more frequent delays
                    - **Colors**: 🟢 Green (≤6min) → 🟠 Orange (7-10min) → 🔴 Red (11-15min) → 🔵 Dark Red (>15min)
                    - **Density**: More delay values packed together for comprehensive view
                    """)
                else:
                    st.warning("No common delay data available for visualization.")
                
                # Position analysis
                st.markdown("##### 📊 Delay Values by Position in Daily Rankings")
                st.markdown("This shows which delay values most commonly appear in each position of the daily top 10 rankings.")
                
                fig_position = plot_delay_value_by_position(filtered_daily_df, selected_years)
                if fig_position:
                    st.pyplot(fig_position)
                else:
                    st.warning("No position ranking data available for visualization.")
                
                # Seasonal patterns in common delays
                st.markdown("##### 🌦️ Seasonal Patterns of Common Delay Values")
                st.markdown("This heatmap shows how different delay values vary by season, revealing weather-related patterns.")
                
                fig_seasonal_delays = plot_common_delays_by_month(filtered_daily_df, selected_years)
                if fig_seasonal_delays:
                    st.pyplot(fig_seasonal_delays)
                else:
                    st.warning("No seasonal delay pattern data available for visualization.")
                
                # Year-over-year consistency (only if multiple years selected)
                if len(selected_years) > 1:
                    st.markdown("##### 🔄 Consistency Across Years")
                    st.markdown("This analysis shows how consistent the most problematic delay values are across different years.")
                    
                    fig_consistency = plot_delay_consistency_analysis(filtered_daily_df, selected_years)
                    if fig_consistency:
                        st.pyplot(fig_consistency)
                    else:
                        st.info("No common delay values found across all selected years for comparison.")
                
                # Summary statistics for common delays
                st.markdown("##### 📈 Common Delay Statistics")
                
                # Parse all common delays for statistics
                parsed_delays = parse_common_delays(filtered_daily_df)
                if not parsed_delays.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        most_common_delay = parsed_delays['delay_value'].mode().iloc[0]
                        st.metric("Most Common Delay Value", f"{int(most_common_delay)} min")
                    
                    with col2:
                        unique_delay_values = parsed_delays['delay_value'].nunique()
                        st.metric("Unique Delay Values", f"{unique_delay_values}")
                    
                    with col3:
                        avg_delay_in_top10 = parsed_delays['delay_value'].mean()
                        st.metric("Average Delay in Top 10", f"{avg_delay_in_top10:.1f} min")
                    
                    with col4:
                        max_delay_in_top10 = parsed_delays['delay_value'].max()
                        st.metric("Max Delay in Top 10", f"{int(max_delay_in_top10)} min")
                else:
                    st.warning("No valid common delay data found for statistics calculation.")
            else:
                st.info("⚠️ Common delay values analysis not available - missing 'top_10_common_delays' column in dataset.")
        else:
            st.warning("⚠️ No daily data available for the selected years.")
    
    elif plot_key == "day_of_week":
        st.markdown("### Day of the Week Delay Analysis")
        
        # Heatmap
        st.markdown("#### Average Delay Percentage by Day of Week and Month")
        fig3 = plot_delay_heatmap(df)
        st.pyplot(fig3)
        
        # Delay severity distribution
        st.markdown("#### Distribution of Days by Average Delay Severity")
        fig4 = plot_delay_severity_distribution(df)
        st.pyplot(fig4)
    
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