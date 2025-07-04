import itertools
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline  # Keep import for optional use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns  # For KDE plots (optional)
import os
import warnings
from pandas.api.types import is_numeric_dtype, is_object_dtype
from matplotlib.container import BarContainer  # NEW IMPORT
from typing import List, Optional, Union, Sequence, Dict, Tuple, Any
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import re


def plot_SPT_vs_depth_cohesive(
    # === Essential Data Parameters ===
    df: pd.DataFrame,
    depth_col: str = 'From_mbgl',
    spt_col: str = 'SPT N',
    
    # === Plot Appearance ===
    title: Optional[str] = None,
    title_suffix: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    
    # === Category Options ===
    category_col: Optional[str] = None,
    
    # === Axis Configuration ===
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    xtick_interval: Optional[float] = 10.0,
    ytick_interval: Optional[float] = None,
    invert_yaxis: bool = True,
    
    # === Display Options ===
    show_plot: bool = True,
    show_legend: bool = True,
    show_strength_indicators: bool = True,
    show_strength_boundaries: bool = True,
    show_strength_boundary_ticks: bool = True,
    strength_indicator_position: float = 0.15,
    
    # === Output Control ===
    output_filepath: Optional[str] = None,
    dpi: int = 300,
    
    # === Visual Customization ===
    marker_size: int = 40,
    marker_alpha: float = 0.7,
    marker_edge_lw: float = 0.5,
    
    # === Font Styling ===
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_fontsize: int = 11,
    strength_indicator_fontsize: int = 10,
    
    # === Advanced Styling Options ===
    plot_style: Optional[str] = 'seaborn-v0_8-whitegrid',
    grid_style: Optional[Dict[str, Any]] = None,
    axis_style: Optional[Dict[str, Any]] = None,
    legend_loc: str = 'best',
    legend_style: Optional[Dict[str, Any]] = None,
    scatter_style: Optional[Dict[str, Any]] = None,
    boundary_line_style: Optional[Dict[str, Any]] = None,
    strength_indicator_style: Optional[Dict[str, Any]] = None,
    palette: Optional[Union[List[str], Dict[Any, str]]] = None,
    
    # === Plot Cleanup Control ===
    close_plot: bool = True
    ):
    """
    Create professional SPT vs Depth scatter plot for cohesive soils with consistency classification indicators.
    
    This function creates scatter plots showing Standard Penetration Test (SPT) N values versus 
    depth with cohesive soil consistency classification boundaries and indicators. 
    Designed for geotechnical analysis following standard cohesive soil consistency classifications.
    
    Parameters
    ----------
    === Essential Data Parameters ===
    df : pd.DataFrame
        DataFrame containing SPT and depth data with optional category classifications.
        
    depth_col : str, default 'From_mbgl'
        Column name containing depth values in meters below ground level.
        Common alternatives: 'Depth_m', 'From_m', 'Depth (m)', 'From (mbgl)'
        
    spt_col : str, default 'SPT N'
        Column name containing SPT N values (blows per 30cm).
        Common alternatives: 'SPT', 'N_value', 'SPT_N', 'N60'
        
    === Plot Appearance ===
    title : str, optional
        Custom plot title. If None, uses default "SPT vs Depth (Cohesive Soils)".
        Example: "Borehole SPT Analysis - Cohesive Layers"
        
    title_suffix : str, optional
        Text to append to default title.
        Example: ": Phase 2 Results" → "SPT vs Depth (Cohesive Soils): Phase 2 Results"
        
    figsize : tuple of (float, float), default (9, 6)
        Figure size in inches (width, height). Standard landscape format.
        Examples: (12, 8) for detailed analysis, (8, 5) for reports
        
    === Category Options ===
    category_col : str, optional
        Column name for categorizing data points (colors/markers by category).
        Examples: 'Material', 'Geology_Origin', 'Formation', 'Soil_Type'
        Enables geological intelligence for color assignment.
        
    === Axis Configuration ===
    xlim : tuple of (float, float), optional
        X-axis (SPT N) limits as (min_value, max_value).
        If None, automatically determined from data with padding.
        Example: xlim=(0, 60) sets SPT axis from 0 to 60 blows
        
    ylim : tuple of (float, float), optional
        Y-axis (depth) limits as (min_value, max_value).
        If None, automatically starts from 0 at top and extends to max depth + padding.
        Example: ylim=(0, 50) shows depths from surface to 50m
        
    xtick_interval : float, default 10.0
        X-axis (SPT N) tick spacing. Creates evenly spaced ticks.
        Example: xtick_interval=5.0 creates ticks every 5 SPT N values (0, 5, 10, 15, ...)
        If None, uses matplotlib's automatic tick spacing.
        
    ytick_interval : float, optional
        Y-axis (depth) tick spacing. Creates evenly spaced ticks.
        Example: ytick_interval=5.0 creates ticks every 5 meters (0, 5, 10, 15, ...)
        If None, uses matplotlib's automatic tick spacing.
        
    invert_yaxis : bool, default True
        Whether to invert y-axis (depth increases downward from 0 at top).
        Standard for geological plots where surface is at top, depth increases downward.
        
    === Display Options ===
    show_plot : bool, default True
        Whether to display plot on screen. Set False for batch processing.
        
    show_legend : bool, default True
        Whether to show legend for categories (if category_col provided).
        
    show_strength_indicators : bool, default True
        Whether to show SPT consistency classification labels between boundary lines.
        Displays: VS, S, F, St, VSt, H labels positioned between respective boundaries.
        
    show_strength_boundaries : bool, default True
        Whether to show vertical lines at SPT classification boundaries.
        Lines at: 2, 4, 8, 15, 30 SPT N (standard cohesive soil boundaries)
        
    show_strength_boundary_ticks : bool, default True
        Whether to show strength boundary values as x-axis tick labels.
        When True, displays ticks at boundary values (0, 2, 4, 8, 15, 30).
        Takes precedence over xtick_interval when enabled.
        
    strength_indicator_position : float, default 0.15
        Vertical position of strength indicator labels as fraction from top of plot.
        0.0 = very top, 0.5 = middle, 1.0 = bottom
        Example: 0.1 = 10% down from top, 0.9 = 90% down (near bottom)
        
    === Output Control ===
    output_filepath : str, optional
        Full path to save plot including filename and extension.
        Example: 'results/spt_depth_cohesive.png'
        If provided, plot will be automatically saved to this location.
        
    dpi : int, default 300
        Resolution for saved figure in dots per inch.
        Standard values: 150 (draft), 300 (publication), 600 (high-res)
        
    === Visual Customization ===
    marker_size : int, default 40
        Size of scatter points in points^2.
        Examples: 60-80 for presentations, 30 for dense data
        
    marker_alpha : float, default 0.7
        Point transparency (0.0=invisible, 1.0=solid).
        Use lower values (0.5) for overlapping points.
        
    marker_edge_lw : float, default 0.5
        Width of point borders in points. Set to 0 for no borders.
        
    === Font Styling ===
    title_fontsize : int, default 14
        Font size for main plot title.
        
    label_fontsize : int, default 12
        Font size for axis labels.
        
    tick_fontsize : int, default 10
        Font size for axis tick labels.
        
    legend_fontsize : int, default 11
        Font size for legend text.
        
    strength_indicator_fontsize : int, default 10
        Font size for strength classification text labels.
        
    === Advanced Styling Options ===
    plot_style : str, optional, default 'seaborn-v0_8-whitegrid'
        Matplotlib style to apply to the plot.
        Common options: 'default', 'seaborn-v0_8-whitegrid', 'ggplot', 'bmh'
        Set to None to use current matplotlib style settings.
        
    grid_style : dict, optional
        Dictionary of grid styling parameters.
        Default: {'linestyle': '--', 'color': 'grey', 'alpha': 0.3}
        Example: {'linestyle': ':', 'color': 'blue', 'alpha': 0.5}
        
    axis_style : dict, optional
        Dictionary of axis label styling parameters.
        Available keys: 'xlabel_fontsize', 'ylabel_fontsize', 'title_fontsize',
        'xlabel_fontweight', 'ylabel_fontweight', 'title_fontweight'
        
    legend_loc : str, default 'best'
        Legend location. Standard matplotlib positions:
        'best', 'upper right', 'upper left', 'lower left', 'lower right'
        
    legend_style : dict, optional
        Dictionary of legend styling parameters.
        Example: {'frameon': True, 'shadow': True, 'framealpha': 0.9}
        
    scatter_style : dict, optional
        Dictionary of scatter plot styling parameters.
        Example: {'edgecolors': 'black', 'linewidths': 0.8}
        
    boundary_line_style : dict, optional
        Dictionary of strength boundary line styling.
        Default: {'linestyle': '--', 'alpha': 0.6, 'linewidth': 1.0}
        Example: {'linestyle': ':', 'color': 'red', 'alpha': 0.8}
        
    strength_indicator_style : dict, optional
        Dictionary of strength indicator text styling.
        Example: {'fontweight': 'bold', 'fontsize': 12}
        
    palette : list or dict, optional
        Custom color palette for categories.
        List: ['blue', 'red', 'green'] (cycles through categories)
        Dict: {'Alluvial': 'orange', 'Residual': 'green'} (specific mapping)
        If None, uses intelligent geological defaults.
        
    Returns
    -------
    None
    
    Examples
    --------
    **Basic usage:**
    >>> plot_SPT_vs_depth_cohesive(df)
    
    **With custom columns:**
    >>> plot_SPT_vs_depth_cohesive(df, depth_col='Depth_m', spt_col='N_value')
    
    **With geological categories:**
    >>> plot_SPT_vs_depth_cohesive(df, category_col='Material', 
    ...                            title="Site Investigation - Cohesive Layers")
    
    **Custom axis limits and styling:**
    >>> plot_SPT_vs_depth_cohesive(df, xlim=(0, 40), ylim=(0, 30),
    ...                            marker_size=60, marker_alpha=0.8)
    
    **Advanced styling:**
    >>> plot_SPT_vs_depth_cohesive(df,
    ...                            plot_style='ggplot',
    ...                            grid_style={'linestyle': ':', 'alpha': 0.4},
    ...                            legend_style={'frameon': True, 'shadow': True},
    ...                            boundary_line_style={'color': 'red', 'alpha': 0.7})
    
    **Custom strength indicator positioning:**
    >>> plot_SPT_vs_depth_cohesive(df, strength_indicator_position=0.1)  # Near top
    >>> plot_SPT_vs_depth_cohesive(df, strength_indicator_position=0.8)  # Near bottom
    
    **Save without displaying:**
    >>> plot_SPT_vs_depth_cohesive(df, show_plot=False,
    ...                            output_filepath='spt_cohesive_analysis.png')
    """
    # === Input Validation ===
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    # Validate required columns exist
    required_cols = [depth_col, spt_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    # Validate category column if provided
    if category_col and category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in DataFrame.")
    
    # Validate axis limits
    if xlim is not None and not (isinstance(xlim, tuple) and len(xlim) == 2):
        warnings.warn("xlim must be a tuple (min, max). Ignoring.")
        xlim = None
    if ylim is not None and not (isinstance(ylim, tuple) and len(ylim) == 2):
        warnings.warn("ylim must be a tuple (min, max). Ignoring.")
        ylim = None
    
    # No validation needed - output_filepath presence determines saving
    
    # === Data Preparation ===
    data = df.copy()
    
    # Convert numeric columns and handle errors
    data[depth_col] = pd.to_numeric(data[depth_col], errors='coerce')
    data[spt_col] = pd.to_numeric(data[spt_col], errors='coerce')
    
    # Remove rows with missing essential data
    initial_rows = len(data)
    data = data.dropna(subset=[depth_col, spt_col])
    if len(data) < initial_rows:
        print(f"INFO: Removed {initial_rows - len(data)} rows with missing depth/SPT data.")
    
    if data.empty:
        print("WARNING: No valid data points after cleaning. Skipping plot.")
        return
    
    # === SPT Consistency Classification System (Cohesive Soils) ===
    strength_boundaries = [2, 4, 8, 15, 30]
    strength_labels = ['VS', 'S', 'F', 'St', 'VSt', 'H']
    strength_ranges = [
        (0, 2, 'VS'),      # Very Soft: 0-2
        (2, 4, 'S'),       # Soft: 2-4
        (4, 8, 'F'),       # Firm: 4-8
        (8, 15, 'St'),     # Stiff: 8-15
        (15, 30, 'VSt'),   # Very Stiff: 15-30
        (30, float('inf'), 'H')  # Hard: >30
    ]
    
    # === Geological Color Intelligence ===
    def _normalize_geological_category(category_value):
        """Normalize geological categories using flexible pattern matching."""
        if pd.isna(category_value):
            return category_value
        
        cat_str = str(category_value).upper().strip()
        
        # ALLUVIAL/ALLUVIUM patterns
        alluvial_patterns = ['ALLUVIAL', 'ALLUVIUM', 'QA', 'QUAT', 'QUATERNARY']
        if any(pattern in cat_str for pattern in alluvial_patterns):
            return 'ALLUVIAL'
        
        # RESIDUAL/WEATHERED patterns  
        residual_patterns = ['RESIDUAL', 'RS', 'XW', 'WEATHERED', 'EXTREMELY WEATHERED']
        if any(pattern in cat_str for pattern in residual_patterns):
            return 'RESIDUAL'
        
        # FILL patterns
        fill_patterns = ['FILL', 'FILLING', 'ENGINEERED FILL', 'CONTROLLED FILL']
        if any(pattern in cat_str for pattern in fill_patterns):
            return 'FILL'
        
        # Standard geological units
        geological_units = {'DCF': 'DCF', 'RIN': 'RIN', 'RJBW': 'RJBW', 'TOS': 'TOS'}
        if cat_str in geological_units:
            return geological_units[cat_str]
        
        return category_value
    
    # Define geological color scheme
    geological_colors = {
        'ALLUVIAL': 'darkorange',
        'RESIDUAL': 'green', 
        'FILL': 'lightblue',
        'DCF': 'brown',
        'RIN': 'purple',
        'RJBW': 'red',
        'TOS': 'blue'
    }
    
    # === Style Application ===
    if plot_style:
        try: 
            plt.style.use(plot_style)
        except: 
            print(f"Warning: matplotlib style '{plot_style}' not found.")
    
    # === Create Plot ===
    fig, ax = plt.subplots(figsize=figsize)
    
    # === Default Styling ===
    default_grid_style = {'linestyle': '--', 'color': 'grey', 'alpha': 0.3}
    default_axis_style = {
        'xlabel_fontsize': label_fontsize, 'xlabel_fontweight': 'bold',
        'ylabel_fontsize': label_fontsize, 'ylabel_fontweight': 'bold', 
        'title_fontsize': title_fontsize, 'title_fontweight': 'bold'
    }
    default_scatter_style = {'edgecolors': 'black', 'zorder': 5}
    default_boundary_style = {'linestyle': '--', 'color': 'gray', 'alpha': 0.6, 'linewidth': 1.0}
    default_strength_indicator_style = {'fontsize': strength_indicator_fontsize}
    
    # Apply styling with user overrides
    grid_params = {**default_grid_style, **(grid_style or {})}
    axis_params = {**default_axis_style, **(axis_style or {})}
    scatter_params = {**default_scatter_style, **(scatter_style or {})}
    boundary_params = {**default_boundary_style, **(boundary_line_style or {})}
    strength_params = {**default_strength_indicator_style, **(strength_indicator_style or {})}
    
    # === Main Plotting Logic ===
    if category_col:
        # Plot by categories with color coding
        categories = data[category_col].dropna().unique()
        
        # Set up color scheme
        if isinstance(palette, dict):
            color_map = palette
        elif isinstance(palette, list):
            color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
        else:
            # Use geological intelligence
            color_map = {}
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            color_idx = 0
            
            for cat in categories:
                normalized_cat = _normalize_geological_category(cat)
                if normalized_cat in geological_colors:
                    color_map[cat] = geological_colors[normalized_cat]
                else:
                    color_map[cat] = default_colors[color_idx % len(default_colors)]
                    color_idx += 1
        
        # Plot each category
        for category in categories:
            cat_data = data[data[category_col] == category]
            
            ax.scatter(cat_data[spt_col], cat_data[depth_col],
                      label=str(category),
                      color=color_map.get(category, '#1f77b4'),
                      s=marker_size,
                      alpha=marker_alpha,
                      linewidths=marker_edge_lw,
                      **scatter_params)
    else:
        # Plot all data with single color
        ax.scatter(data[spt_col], data[depth_col],
                  color='#1f77b4',
                  s=marker_size,
                  alpha=marker_alpha,
                  linewidths=marker_edge_lw,
                  **scatter_params)
    
    # === Add Strength Classification Boundaries ===
    if show_strength_boundaries:
        y_min_plot, y_max_plot = ax.get_ylim()
        for boundary in strength_boundaries:
            ax.axvline(x=boundary, ymin=0, ymax=1, **boundary_params)
    
    # === Axis Configuration ===
    # Set axis labels
    ax.set_xlabel(f'{spt_col}', 
                  fontsize=axis_params['xlabel_fontsize'], 
                  fontweight=axis_params['xlabel_fontweight'])
    ax.set_ylabel(f'{depth_col}', 
                  fontsize=axis_params['ylabel_fontsize'], 
                  fontweight=axis_params['ylabel_fontweight'])
    
    # Handle title
    final_title = None
    if title is not None:
        final_title = title
    else:
        final_title = 'SPT vs Depth '
        if title_suffix and isinstance(title_suffix, str) and title_suffix.strip():
            final_title += f": {title_suffix.strip()}"
    
    if final_title:
        ax.set_title(final_title, 
                     fontsize=axis_params['title_fontsize'], 
                     fontweight=axis_params['title_fontweight'])
    
    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        # Default SPT range for cohesive soils
        ax.set_xlim(0, 50)
    
    # Set custom x-axis ticks
    current_xlim = ax.get_xlim()
    x_start = current_xlim[0]
    x_end = current_xlim[1]
    
    all_ticks = set()  # Use set to automatically handle duplicates
    
    # Add strength boundary ticks if enabled
    if show_strength_boundary_ticks:
        all_ticks.update([0] + strength_boundaries)
    
    # Add regular interval ticks if specified
    if xtick_interval is not None:
        interval_ticks = np.arange(x_start, x_end + xtick_interval, xtick_interval)
        all_ticks.update(interval_ticks)
    
    # Convert to sorted list and apply
    if all_ticks:
        final_ticks = sorted(list(all_ticks))
        ax.set_xticks(final_ticks)
        ax.set_xticklabels([str(int(x)) if x.is_integer() else str(x) for x in final_ticks])
    
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Auto-scale with padding, ensure starts from 0 at top
        depth_min, depth_max = data[depth_col].min(), data[depth_col].max()
        depth_range = depth_max - depth_min
        padding = max(0.05 * depth_range, 0.5)  # At least 0.5m padding
        
        # Start from 0 (or slightly above surface if data starts below surface)
        y_min = 0 if depth_min <= 1.0 else max(0, depth_min - padding)
        y_max = depth_max + padding
        ax.set_ylim(y_min, y_max)
    
    # Set custom y-axis ticks if interval specified
    if ytick_interval is not None:
        current_ylim = ax.get_ylim()
        y_start = current_ylim[0]
        y_end = current_ylim[1]
        ytick_values = np.arange(y_start, y_end + ytick_interval, ytick_interval)
        ax.set_yticks(ytick_values)
    
    # Invert y-axis if requested (standard for depth plots)
    if invert_yaxis:
        ax.invert_yaxis()
    
    # Apply grid
    ax.grid(True, **grid_params)
    
    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # === Add Legend ===
    if show_legend and category_col:
        # Check if there are any legend entries to display
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            # Default legend styling with outline
            default_legend_params = {
                'loc': legend_loc, 
                'fontsize': legend_fontsize,
                'frameon': True,
                'facecolor': 'white',
                'framealpha': 0.9,
                'edgecolor': 'black'
            }
            
            if legend_style:
                default_legend_params.update(legend_style)
            ax.legend(**default_legend_params)
    
    # === Add Strength Indicator Labels ===
    if show_strength_indicators:
        ylim_current = ax.get_ylim()
        
        # Position at specified fraction down from top of plot
        if invert_yaxis:
            # When inverted, top of plot is at minimum y value
            text_y = ylim_current[0] + strength_indicator_position * (ylim_current[1] - ylim_current[0])
        else:
            # Normal orientation, top is at maximum y value  
            text_y = ylim_current[1] - strength_indicator_position * (ylim_current[1] - ylim_current[0])
        
        # Place text labels between boundary lines
        for min_val, max_val, label in strength_ranges:
            if max_val == float('inf'):
                # For the last category (H), position within plot bounds
                xlim_current = ax.get_xlim()
                xlim_max = xlim_current[1]
                # Position at 90% of plot width or just beyond last boundary, whichever fits better
                ideal_position = strength_boundaries[-1] + (strength_boundaries[-1] - strength_boundaries[-2]) / 2
                text_x = min(xlim_max * 0.9, ideal_position)
            else:
                # Calculate arithmetic mean for linear positioning
                text_x = (min_val + max_val) / 2
            
            ax.text(text_x, text_y, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='bold',
                    **strength_params)
    
    # === Layout Optimization ===
    plt.tight_layout()
    
    # === Save and Display ===
    if output_filepath:
        try:
            dir_name = os.path.dirname(output_filepath)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"Created directory: {dir_name}")
            plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {output_filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    if show_plot:
        plt.show()
    
    if close_plot:
        plt.close(fig)
    
    print("--- SPT vs Depth (Cohesive) plotting finished ---")