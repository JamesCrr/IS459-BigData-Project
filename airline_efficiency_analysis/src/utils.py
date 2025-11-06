"""
Utility Functions
Common helper functions for the airline efficiency analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os


def setup_plotting_style():
    """Set consistent plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def save_dataframe(df: pd.DataFrame, filename: str, output_dir: str = "../outputs/"):
    """
    Save dataframe to CSV
    
    Args:
        df: DataFrame to save
        filename: Output filename
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"   ✓ Saved: {filepath}")
    return filepath


def save_figure(fig, filename: str, output_dir: str = "../outputs/"):
    """
    Save matplotlib figure
    
    Args:
        fig: Figure object
        filename: Output filename
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {filepath}")
    return filepath


def print_section_header(title: str, char: str = "=", width: int = 60):
    """Print formatted section header"""
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def calculate_percentiles(series: pd.Series, percentiles: List[int] = [25, 50, 75, 90, 95]) -> Dict:
    """
    Calculate percentiles for a series
    
    Args:
        series: Pandas series
        percentiles: List of percentile values
        
    Returns:
        Dictionary of percentile values
    """
    result = {}
    for p in percentiles:
        result[f'p{p}'] = np.percentile(series.dropna(), p)
    return result


def format_minutes(minutes: float) -> str:
    """
    Format minutes as hours:minutes
    
    Args:
        minutes: Number of minutes
        
    Returns:
        Formatted string
    """
    if pd.isna(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def format_delay(delay_minutes: float) -> str:
    """
    Format delay with sign
    
    Args:
        delay_minutes: Delay in minutes (positive = late, negative = early)
        
    Returns:
        Formatted string
    """
    if pd.isna(delay_minutes):
        return "N/A"
    sign = "+" if delay_minutes > 0 else ""
    return f"{sign}{delay_minutes:.0f} min"


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get human-readable memory usage of dataframe
    
    Args:
        df: DataFrame
        
    Returns:
        Formatted string with memory usage
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    if memory_bytes < 1024**2:
        return f"{memory_bytes / 1024:.2f} KB"
    elif memory_bytes < 1024**3:
        return f"{memory_bytes / (1024**2):.2f} MB"
    else:
        return f"{memory_bytes / (1024**3):.2f} GB"


def create_bins_labels(values: List, labels: List[str]) -> Tuple:
    """
    Create bins and labels for pd.cut
    
    Args:
        values: List of bin edges
        labels: List of bin labels
        
    Returns:
        Tuple of (bins, labels)
    """
    return values, labels


def highlight_top_n(s: pd.Series, n: int = 5, color: str = 'yellow') -> List[str]:
    """
    Highlight top N values in a pandas series for styling
    
    Args:
        s: Pandas series
        n: Number of top values to highlight
        color: Background color
        
    Returns:
        List of CSS styles
    """
    top_n_values = s.nlargest(n)
    return [f'background-color: {color}' if v in top_n_values.values else '' for v in s]


def safe_division(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0) -> pd.Series:
    """
    Safe division handling zero denominators
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result series
    """
    return np.where(denominator != 0, numerator / denominator, fill_value)


def convert_hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """
    Convert HHMM time format to minutes since midnight
    
    Args:
        hhmm: Series with HHMM format (e.g., 1530 = 3:30 PM)
        
    Returns:
        Series with minutes since midnight
    """
    hours = (hhmm // 100).astype(int)
    minutes = (hhmm % 100).astype(int)
    return hours * 60 + minutes


def get_season(month: int) -> str:
    """
    Get season from month
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test format functions
    print(format_minutes(125))  # Should print "2h 5m"
    print(format_delay(15))     # Should print "+15 min"
    print(format_delay(-5))     # Should print "-5 min"
    
    # Test percentiles
    data = pd.Series([1, 2, 3, 4, 5, 10, 20, 30, 40, 50])
    print(calculate_percentiles(data))
    
    print("✓ Utilities working correctly")
