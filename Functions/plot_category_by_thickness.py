import itertools
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
from pandas.api.types import is_numeric_dtype, is_object_dtype
from matplotlib.container import BarContainer
from typing import List, Optional, Union, Sequence, Dict, Tuple, Any
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import re

# Soil consistency libraries for intelligent sorting
SOIL_CONSISTENCY_LIBRARY = ['St', 'VSt', 'H', 'F', 'D', 'VD', 'L', 'MD', 'VL', 'S', 'VS', 'Fr']
CONSISTENCY_STRENGTH_ORDER = ['Fr', 'VS', 'S', 'F', 'St', 'VSt', 'H', 'VL', 'L', 'MD', 'D', 'VD']

def plot_category_by_thickness(
    # === Essential Data Parameters ===
    df: pd.DataFrame,
    value_col: str = 'Value',
    category_col: str = 'Category',
    
    # === Plot Appearance ===
    title: Optional[str] = None,
    title_suffix: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 6),
    title_fontsize: int = 14,
    title_fontweight: str = 'bold',
    
    # === Category Options ===
    category_order: Optional[List[str]] = None,
    x_axis_sort: str = 'smart_consistency',
    legend_order: Optional[List[str]] = None,
    legend_sort: str = 'same_as_x',
    
    # === Axis Configuration ===
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    label_fontweight: str = 'bold',
    tick_fontsize: int = 11,
    show_percentage_labels: bool = True,
    percentage_decimal_places: int = 1,
    
    # === Display Options ===
    show_plot: bool = True,
    show_legend: bool = True,
    show_grid: bool = True,
    grid_axis: str = 'y',
    legend_fontsize: int = 10,
    legend_loc: str = 'best',
    
    # === Output Control ===
    output_filepath: Optional[str] = None,
    save_plot: Optional[bool] = None,
    dpi: int = 300,
    
    # === Visual Customization ===
    colors: Optional[Union[List[str], Dict[str, str]]] = None,
    bar_width: float = 0.8,
    bar_alpha: float = 0.8,
    bar_edgecolor: str = 'black',
    bar_linewidth: float = 0.6,
    bar_hatch: Optional[str] = None,
    rotation: float = 0,
    value_label_fontsize: int = 10,
    
    # === Advanced Styling Options ===
    plot_style: Optional[str] = 'seaborn-v0_8-whitegrid',
    grid_style: Optional[Dict[str, Any]] = None,
    axis_style: Optional[Dict[str, Any]] = None,
    legend_style: Optional[Dict[str, Any]] = None,
    label_style: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a professional bar chart with categories on x-axis and values on y-axis.
    
    This general-purpose function plots pre-calculated values without any internal calculations.
    Provides comprehensive styling and ordering controls for any type of categorical data.
    
    Parameters
    ----------
    === Essential Data Parameters ===
    df : pd.DataFrame
        DataFrame containing pre-calculated values and categories.
        Must contain columns specified by value_col and category_col.
        
    value_col : str, default 'Value'
        Column name containing numerical values to plot.
        Values should be positive numbers (percentages, counts, measurements, etc.).
        
    category_col : str, default 'Category'
        Column name containing categorical data for x-axis.
        Example: 'Category', 'Type', 'Group'
        
    === Plot Appearance ===
    title : str, optional
        Custom plot title. If None, uses intelligent default.
        Example: "Category Distribution"
        
    title_suffix : str, optional
        Text to append to default title.
        Example: ": Project XYZ" → "Default Title: Project XYZ"
        
    figsize : tuple, default (9, 6)
        Figure size in inches (width, height).
        Example: (10, 8) for larger plots
        
    title_fontsize : int, default 14
        Font size for the main plot title.
        
    title_fontweight : str, default 'bold'
        Font weight for the main plot title.
        Options: 'normal', 'bold', 'light', 'heavy'
        
    === Category Options ===
    category_order : list of str, optional
        Custom order for categories on x-axis. Overrides x_axis_sort if provided.
        If None, uses x_axis_sort parameter for ordering.
        Example: ['TypeA', 'TypeB', 'TypeC']
        
    x_axis_sort : str, default 'smart_consistency'
        Sorting method for x-axis categories. Options:
        - 'smart_consistency': Intelligent sorting based on data type:
          * Numerical-only: Alphabetical ascending
          * Consistency-only: Follow strength order (Fr,VS,S,F,St,VSt,H,VL,L,MD,D,VD)
          * Mixed: Consistency first (strength order), then numerical (alphabetical)
        - 'descending': Sort by values (highest first)
        - 'ascending': Sort by values (lowest first)  
        - 'alphabetical': Sort alphabetically by category name
        - 'reverse_alphabetical': Sort reverse alphabetically
        
    legend_order : list of str, optional
        Custom order for legend entries. Overrides legend_sort if provided.
        If None, uses legend_sort parameter for ordering.
        Example: ['TypeC', 'TypeA', 'TypeB']
        
    legend_sort : str, default 'same_as_x'
        Sorting method for legend entries. Options:
        - 'same_as_x': Use same order as x-axis
        - 'descending': Sort by values (highest first)
        - 'ascending': Sort by values (lowest first)
        - 'alphabetical': Sort alphabetically by category name
        - 'reverse_alphabetical': Sort reverse alphabetically
        
    === Axis Configuration ===
    xlim : tuple of (float, float), optional
        X-axis limits as (min, max). If None, uses automatic scaling.
        Example: (0, 10) to limit x-axis range
        
    ylim : tuple of (float, float), optional
        Y-axis limits as (min, max). If None, uses 0 to 100% range.
        Example: (0, 50) to limit to 50% maximum
        
    xlabel : str, optional
        Custom x-axis label. If None, uses category_col name.
        Example: "Material Type", "Formation"
        
    ylabel : str, optional
        Custom y-axis label. If None, uses "Value".
        Example: "Percentage (%)", "Count", "Amount"
        
    xlabel_fontsize : int, default 12
        Font size for x-axis label.
        
    ylabel_fontsize : int, default 12
        Font size for y-axis label.
        
    label_fontweight : str, default 'bold'
        Font weight for axis labels.
        Options: 'normal', 'bold', 'light', 'heavy'
        
    tick_fontsize : int, default 11
        Font size for tick labels on both axes.
        
    show_percentage_labels : bool, default True
        Whether to show value labels on top of bars.
        
    percentage_decimal_places : int, default 1
        Number of decimal places for value labels.
        Example: 1 → "25.3", 0 → "25"
        
    === Display Options ===
    show_plot : bool, default True
        Whether to display the plot. Set False for batch processing.
        
    show_legend : bool, default True
        Whether to show the legend. Set False to hide legend.
        
    show_grid : bool, default True
        Whether to show grid lines for easier reading.
        
    grid_axis : str, default 'y'
        Which axis to show grid lines. Options: 'x', 'y', 'both'
        
    legend_fontsize : int, default 10
        Font size for legend text.
        
    legend_loc : str, default 'best'
        Legend location. Options: 'best', 'upper right', 'upper left', 'lower left', 
        'lower right', 'right', 'center left', 'center right', 'lower center', 
        'upper center', 'center'
        
    === Output Control ===
    output_filepath : str, optional
        Full path to save the plot. If None, plot is not saved.
        Example: "/path/to/output/category_distribution.png"
        
    save_plot : bool, optional
        Whether to save the plot. If None, automatically saves when output_filepath is provided.
        Set to False to prevent saving even when output_filepath is given.
        
    dpi : int, default 300
        Resolution for saved plots in dots per inch.
        Higher values create higher quality images.
        
    === Visual Customization ===
    colors : list or dict, optional
        Colors for categories. Can be list of colors or dict mapping categories to colors.
        If None, uses professional colorblind-friendly palette.
        Example: ['orange', 'green', 'blue'] or {'TypeA': 'orange', 'TypeB': 'green'}
        
    bar_width : float, default 0.8
        Width of bars (0.1 to 1.0). Smaller values create thinner bars with more spacing.
        
    bar_hatch : str, optional
        Hatch pattern for bars. Useful for accessibility and black/white printing.
        Options: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
        Example: '///' for diagonal lines, '|||' for vertical lines
        
    bar_alpha : float, default 0.8
        Transparency of bars (0.0 to 1.0).
        Lower values create more transparent bars.
        
    bar_edgecolor : str, default 'black'
        Color of bar edges. Use 'none' for no edges.
        
    bar_linewidth : float, default 0.6
        Width of bar edge lines.
        
    rotation : float, default 0
        Rotation angle for x-axis labels in degrees.
        Example: 45 for diagonal labels
        
    value_label_fontsize : int, default 9
        Font size for value labels displayed on top of bars.
        
    === Advanced Styling Options ===
    plot_style : str, default 'seaborn-v0_8-whitegrid'
        Matplotlib style to use for the plot.
        Example: 'seaborn-v0_8-colorblind', 'classic', 'bmh'
        
    grid_style : dict, optional
        Grid styling parameters. If None, uses default grid style.
        Example: {'linestyle': ':', 'color': 'blue', 'alpha': 0.3}
        
    axis_style : dict, optional
        Axis styling parameters. If None, uses default axis style.
        Example: {'xlabel_fontsize': 14, 'ylabel_fontsize': 14}
        
    legend_style : dict, optional
        Legend styling parameters. If None, uses default legend style.
        Example: {'frameon': True, 'shadow': True}
        
    label_style : dict, optional
        Label styling parameters for value labels.
        Example: {'fontsize': 10, 'fontweight': 'bold'}
    
    Returns
    -------
    None
        Function creates and optionally saves the plot.
    
    Examples
    --------
    **Basic usage:**
    >>> plot_category_by_thickness(data)
    
    **With customization:**
    >>> plot_category_by_thickness(data, 
    ...                           title="Value Distribution", 
    ...                           category_order=['TypeA', 'TypeB'],
    ...                           bar_width=0.6)
    
    **Advanced styling:**
    >>> plot_category_by_thickness(data,
    ...                           colors={'TypeA': 'orange', 'TypeB': 'green'},
    ...                           title_fontsize=16,
    ...                           bar_hatch='///',
    ...                           grid_axis='both')
    """
    
    
    # === Input Validation ===
    # 1. Type validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")
    
    # 2. Column existence validation
    required_cols = [value_col, category_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    # 3. Data type validation
    if not is_numeric_dtype(df[value_col]):
        raise TypeError(f"Column '{value_col}' must contain numeric data.")
    
    if not is_object_dtype(df[category_col]) and not is_numeric_dtype(df[category_col]):
        warnings.warn(f"Column '{category_col}' should contain categorical data.")
    
    # 4. Parameter validation
    if percentage_decimal_places < 0:
        warnings.warn("'percentage_decimal_places' cannot be negative. Using 0.")
        percentage_decimal_places = 0
    
    if bar_alpha < 0 or bar_alpha > 1:
        warnings.warn("'bar_alpha' must be between 0 and 1. Using 0.8.")
        bar_alpha = 0.8
    
    # Validate sorting parameters
    valid_sorts = ['smart_consistency', 'descending', 'ascending', 'alphabetical', 'reverse_alphabetical']
    if x_axis_sort not in valid_sorts:
        warnings.warn(f"Invalid 'x_axis_sort': {x_axis_sort}. Using 'smart_consistency'.")
        x_axis_sort = 'smart_consistency'
    
    valid_legend_sorts = valid_sorts + ['same_as_x']
    if legend_sort not in valid_legend_sorts:
        warnings.warn(f"Invalid 'legend_sort': {legend_sort}. Using 'same_as_x'.")
        legend_sort = 'same_as_x'
    
    # === Data Processing ===
    # Create working copy and remove missing values
    data = df.copy()
    data = data.dropna(subset=[value_col, category_col])
    
    if data.empty:
        raise ValueError("No valid data remaining after removing missing values.")
    
    # Apply category normalization (general-purpose)
    data[category_col] = data[category_col].apply(_normalize_category)
    
    # Create series with category as index and values - no calculations, just organize data
    values_series = data.set_index(category_col)[value_col]
    
    # If multiple rows per category, sum them (but data should ideally be pre-aggregated)
    if values_series.index.duplicated().any():
        values_series = values_series.groupby(level=0).sum()
        warnings.warn("Multiple rows found per category. Values have been summed.")
    
    # Order categories for x-axis
    if category_order:
        # Use custom order, but only include categories that exist in data
        existing_categories = set(values_series.index)
        category_order_filtered = [cat for cat in category_order if cat in existing_categories]
        # Add any remaining categories not in custom order
        remaining_categories = existing_categories - set(category_order_filtered)
        final_order = category_order_filtered + sorted(remaining_categories)
        x_axis_values = values_series.reindex(final_order)
    else:
        # Apply sorting based on x_axis_sort parameter
        x_axis_values = _sort_categories(values_series, x_axis_sort)
    
    # Order categories for legend (independent of x-axis)
    if legend_order:
        # Use custom legend order
        existing_categories = set(values_series.index)
        legend_order_filtered = [cat for cat in legend_order if cat in existing_categories]
        remaining_categories = existing_categories - set(legend_order_filtered)
        legend_final_order = legend_order_filtered + sorted(remaining_categories)
        legend_values = values_series.reindex(legend_final_order)
    else:
        # Apply sorting based on legend_sort parameter
        if legend_sort == 'same_as_x':
            legend_values = x_axis_values
        else:
            legend_values = _sort_categories(values_series, legend_sort)
    
    print(f"Processing {len(values_series)} categories")
    
    # === Style Application ===
    # Apply matplotlib style
    if plot_style:
        try:
            plt.style.use(plot_style)
        except Exception as e:
            warnings.warn(f"Could not apply style '{plot_style}': {e}. Using default.")
    
    # Default style dictionaries
    default_grid_style = {'linestyle': '--', 'color': 'grey', 'alpha': 0.35}
    default_axis_style = {
        'xlabel_fontsize': xlabel_fontsize, 'xlabel_fontweight': label_fontweight,
        'ylabel_fontsize': ylabel_fontsize, 'ylabel_fontweight': label_fontweight,
        'title_fontsize': title_fontsize, 'title_fontweight': title_fontweight
    }
    default_legend_style = {'fontsize': legend_fontsize, 'loc': legend_loc}
    default_label_style = {'fontsize': value_label_fontsize, 'fontweight': 'normal', 'ha': 'center', 'va': 'bottom'}
    
    # Apply custom styling with defaults as fallback
    grid_params = {**default_grid_style, **(grid_style or {})}
    axis_params = {**default_axis_style, **(axis_style or {})}
    legend_params = {**default_legend_style, **(legend_style or {})}
    label_params = {**default_label_style, **(label_style or {})}
    
    # === Color Assignment ===
    # Default color palette (professional and colorblind-friendly)
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Determine final colors
    if colors:
        if isinstance(colors, dict):
            final_colors = colors
        else:
            # List of colors
            final_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(x_axis_values.index)}
    else:
        # Use default color palette
        final_colors = {}
        for i, category in enumerate(x_axis_values.index):
            final_colors[category] = default_colors[i % len(default_colors)]
    
    # === Figure Creation ===
    fig, ax = plt.subplots(figsize=figsize)
    
    # === Main Plotting ===
    # Create bar chart using x-axis ordering
    categories = x_axis_values.index
    values = x_axis_values.values
    bar_colors = [final_colors.get(cat, default_colors[i % len(default_colors)]) 
                  for i, cat in enumerate(categories)]
    
    bars = ax.bar(categories, values, 
                  width=bar_width,
                  color=bar_colors, 
                  alpha=bar_alpha,
                  edgecolor=bar_edgecolor, 
                  linewidth=bar_linewidth,
                  hatch=bar_hatch)
    
    # Add value labels on bars
    if show_percentage_labels:  # Keep parameter name for backward compatibility
        for bar, value in zip(bars, values):
            height = bar.get_height()
            label_text = f"{value:.{percentage_decimal_places}f}"
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_text, **label_params)
    
    # === Axis Customization ===
    # Set labels
    x_label = xlabel if xlabel is not None else category_col
    y_label = ylabel if ylabel is not None else 'Value'
    
    ax.set_xlabel(x_label, 
                  fontsize=axis_params['xlabel_fontsize'], 
                  fontweight=axis_params['xlabel_fontweight'])
    ax.set_ylabel(y_label, 
                  fontsize=axis_params['ylabel_fontsize'], 
                  fontweight=axis_params['ylabel_fontweight'])
    
    # Set title
    if title:
        plot_title = title
    else:
        plot_title = f"Distribution by {category_col}"
    
    if title_suffix:
        plot_title += title_suffix
    
    ax.set_title(plot_title, 
                 fontsize=axis_params['title_fontsize'], 
                 fontweight=axis_params['title_fontweight'])
    
    # Set axis limits
    if xlim:
        ax.set_xlim(xlim)
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        # Auto-scale based on data with some padding
        max_value = max(values)
        min_value = min(0, min(values))  # Start at 0 or below if negative values
        ax.set_ylim(min_value, max_value * 1.1)
    
    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Rotate x-axis labels if specified
    if rotation != 0:
        ax.tick_params(axis='x', rotation=rotation)
    
    # === Grid ===
    if show_grid:
        ax.grid(True, axis=grid_axis, **grid_params)
    
    # === Legend ===
    # Note: For bar charts, legend is typically not needed as categories are on x-axis
    # But we maintain consistency with project conventions
    if show_legend and len(categories) <= 10:  # Only show legend if not too many categories
        # Use legend ordering (may be different from x-axis ordering)
        legend_categories = legend_values.index
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=final_colors.get(cat, default_colors[i % len(default_colors)]), 
                                       alpha=bar_alpha, edgecolor=bar_edgecolor) 
                          for i, cat in enumerate(legend_categories)]
        ax.legend(legend_elements, legend_categories, **legend_params)
    
    # === Layout Optimization ===
    plt.tight_layout()
    
    # === Save and Display ===
    # Determine if we should save the plot
    should_save = save_plot
    if should_save is None:
        # Auto-save if output_filepath is provided
        should_save = output_filepath is not None
    
    if should_save and output_filepath:
        try:
            plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {output_filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    elif should_save and not output_filepath:
        warnings.warn("save_plot=True but no output_filepath provided. Plot not saved.")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    print("--- Plotting finished ---")


def _sort_categories(values_series: pd.Series, sort_method: str) -> pd.Series:
    """
    Sort categories based on specified method with intelligent consistency handling.
    
    Parameters
    ----------
    values_series : pd.Series
        Series with category names as index and numerical values
        
    sort_method : str
        Sorting method: 'smart_consistency', 'descending', 'ascending', 'alphabetical', 'reverse_alphabetical'
        
    Returns
    -------
    pd.Series
        Sorted series
    """
    if sort_method == 'smart_consistency':
        return _sort_categories_intelligently(values_series)
    elif sort_method == 'descending':
        return values_series.sort_values(ascending=False)
    elif sort_method == 'ascending':
        return values_series.sort_values(ascending=True)
    elif sort_method == 'alphabetical':
        return values_series.sort_index(ascending=True)
    elif sort_method == 'reverse_alphabetical':
        return values_series.sort_index(ascending=False)
    else:
        # Default fallback
        return _sort_categories_intelligently(values_series)


def _sort_categories_intelligently(values_series: pd.Series) -> pd.Series:
    """
    Intelligently sort categories based on data type composition.
    
    Logic:
    - Numerical-only: Alphabetical ascending 
    - Consistency-only: Follow CONSISTENCY_STRENGTH_ORDER
    - Mixed: Consistency first (strength order), then numerical/text (alphabetical)
    
    Parameters
    ----------
    values_series : pd.Series
        Series with category names as index and numerical values
        
    Returns
    -------
    pd.Series
        Intelligently sorted series
    """
    categories = list(values_series.index)
    
    # Separate categories into consistency and non-consistency groups
    consistency_categories = []
    non_consistency_categories = []
    
    for cat in categories:
        cat_str = str(cat).strip()
        if cat_str in SOIL_CONSISTENCY_LIBRARY:
            consistency_categories.append(cat_str)
        else:
            non_consistency_categories.append(cat_str)
    
    # Determine data type composition
    has_consistency = len(consistency_categories) > 0
    has_non_consistency = len(non_consistency_categories) > 0
    
    if has_consistency and not has_non_consistency:
        # Pure consistency data - use strength order
        ordered_categories = _order_by_consistency_strength(consistency_categories)
    elif has_non_consistency and not has_consistency:
        # Pure numerical/text data - use alphabetical
        ordered_categories = _order_alphabetically(non_consistency_categories)
    else:
        # Mixed data - consistency first (strength order), then others (alphabetical) 
        consistency_ordered = _order_by_consistency_strength(consistency_categories)
        non_consistency_ordered = _order_alphabetically(non_consistency_categories)
        ordered_categories = consistency_ordered + non_consistency_ordered
    
    # Reindex the series according to the intelligent ordering
    return values_series.reindex(ordered_categories)


def _order_by_consistency_strength(consistency_categories: List[str]) -> List[str]:
    """
    Order consistency categories according to CONSISTENCY_STRENGTH_ORDER.
    
    Parameters
    ----------
    consistency_categories : List[str]
        List of consistency category names
        
    Returns
    -------
    List[str]
        Ordered list following strength order
    """
    # Create mapping from consistency to order index
    strength_order_map = {cons: idx for idx, cons in enumerate(CONSISTENCY_STRENGTH_ORDER)}
    
    # Sort by strength order, with fallback for unknown consistency types
    def get_order_key(category):
        return strength_order_map.get(category, 999)  # Unknown types go to end
    
    return sorted(consistency_categories, key=get_order_key)


def _order_alphabetically(categories: List[str]) -> List[str]:
    """
    Order categories alphabetically with smart numerical handling.
    
    For mixed alphanumeric like '1a', '1b', '3', '5a' -> ['1a', '1b', '3', '5a']
    
    Parameters
    ----------
    categories : List[str]
        List of category names
        
    Returns
    -------
    List[str]
        Alphabetically ordered list
    """
    def smart_sort_key(category):
        """Create sort key that handles mixed alphanumeric properly."""
        cat_str = str(category)
        
        # Try to extract numeric component for smarter sorting
        import re
        match = re.match(r'^(\d+)', cat_str)
        if match:
            # Has leading number - sort by number first, then by rest of string
            numeric_part = int(match.group(1))
            text_part = cat_str[len(match.group(1)):]
            return (0, numeric_part, text_part)  # 0 = numerical category priority
        else:
            # Pure text - sort alphabetically
            return (1, 0, cat_str)  # 1 = text category priority
    
    return sorted(categories, key=smart_sort_key)


def _normalize_category(category_value):
    """
    Normalize category values using basic standardization.
    
    Applies basic string normalization to category values for consistency.
    This is a general-purpose function that can be customized for specific domains.
    
    Parameters
    ----------
    category_value : str or any
        Category value to normalize. Can be string or other types.
        
    Returns
    -------
    str or original type
        Normalized category name, or original value if no normalization needed.
    """
    if pd.isna(category_value):
        return category_value
    
    # Convert to string and apply basic normalization
    if isinstance(category_value, str):
        # Remove extra whitespace and convert to consistent case
        normalized = str(category_value).strip()
        # You can add domain-specific normalization patterns here
        return normalized
    
    # Return original if not a string
    return category_value