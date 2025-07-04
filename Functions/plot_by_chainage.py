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


def plot_by_chainage(
    # === Essential Data Parameters ===
    df: pd.DataFrame,
    chainage_col: str,
    property_col: str,
    
    # === Plot Appearance ===
    title: Optional[str] = None,
    title_suffix: Optional[str] = None,
    property_name: Optional[str] = None,
    chainage_name: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    
    # === Category Options ===
    category_by_col: Optional[str] = None,
    color_by_col: Optional[str] = None,
    colormap: str = 'viridis',
    
    # === Axis Configuration ===
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    xtick_interval: Optional[float] = None,
    ytick_interval: Optional[float] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    invert_yaxis: bool = False,
    use_log_scale_x: bool = False,
    use_log_scale_y: bool = False,
    
    # === Classification Zones ===
    classification_zones: Optional[Dict[str, Tuple[float, float]]] = None,
    zone_colors: Optional[Dict[str, str]] = None,
    show_zone_boundaries: bool = False,
    show_zone_labels: bool = False,
    zone_label_position: float = 0.85,
    zone_orientation: str = 'horizontal',  # 'horizontal' or 'vertical'
    
    # === Connection Lines ===
    connect_points: bool = False,
    line_style: str = '-',
    line_width: float = 1.0,
    line_alpha: float = 0.7,
    
    # === Display Options ===
    show_plot: bool = True,
    show_legend: bool = True,
    show_grid: bool = True,
    show_colorbar: bool = True,
    legend_outside: bool = False,
    legend_position: str = 'right',
    
    # === Output Control ===
    output_filepath: Optional[str] = None,
    dpi: int = 300,
    
    # === Visual Customization ===
    marker_size: int = 50,
    marker_alpha: float = 0.8,
    marker_edge_lw: float = 0.5,
    marker_style: str = 'o',
    
    # === Font Styling ===
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_fontsize: int = 11,
    zone_label_fontsize: int = 10,
    
    # === Advanced Styling Options ===
    plot_style: Optional[str] = 'seaborn-v0_8-whitegrid',
    grid_style: Optional[Dict[str, Any]] = None,
    axis_style: Optional[Dict[str, Any]] = None,
    legend_loc: str = 'best',
    legend_style: Optional[Dict[str, Any]] = None,
    scatter_style: Optional[Dict[str, Any]] = None,
    line_style_dict: Optional[Dict[str, Any]] = None,
    zone_boundary_style: Optional[Dict[str, Any]] = None,
    zone_label_style: Optional[Dict[str, Any]] = None,
    palette: Optional[Union[List[str], Dict[Any, str]]] = None,
    colorbar_label: Optional[str] = None,
    colorbar_position: str = 'right',
    color_limits: Optional[Tuple[float, float]] = None,
    
    # === Backward Compatibility ===
    category_col: Optional[str] = None,  # DEPRECATED: Use category_by_col instead
    
    # === Plot Cleanup Control ===
    close_plot: bool = True
    ):
    """
    Create professional property vs chainage scatter/line plot with optional classification zones.
    
    This function creates plots showing any engineering property versus chainage with optional 
    classification zones, connection lines, and boundaries. Designed for geotechnical analysis 
    along linear infrastructure projects (roads, railways, pipelines, etc.).
    
    Parameters
    ----------
    === Essential Data Parameters ===
    df : pd.DataFrame
        DataFrame containing chainage and property data with optional category classifications.
        
    chainage_col : str
        Column name containing chainage values (distance along alignment).
        Examples: 'Chainage', 'Chainage_m', 'CH', 'Distance_m', 'Stationing'
        
    property_col : str
        Column name containing engineering property values.
        Examples: 'CBR_%', 'Su_kPa', 'Moisture_Content_%', 'PI', 'Density_kg_m3', 'Elevation_m'
        
    === Plot Appearance ===
    title : str, optional
        Custom plot title. If None, uses default based on property and chainage names.
        Example: "CBR Analysis Along Project Alignment"
        
    title_suffix : str, optional
        Text to append to default title.
        Example: ": Phase 2 Investigation" → "Property vs Chainage: Phase 2 Investigation"
        
    property_name : str, optional
        Display name for the property. If None, uses property_col name.
        Example: 'CBR (%)' instead of 'CBR_%'
        
    chainage_name : str, optional
        Display name for chainage. If None, uses chainage_col name.
        Example: 'Chainage (m)' instead of 'Chainage'
        
    figsize : tuple of (float, float), default (12, 6)
        Figure size in inches (width, height). Wide format for chainage plots.
        Examples: (15, 8) for detailed analysis, (10, 5) for reports
        
    === Category Options ===
    category_by_col : str, optional
        Column name for categorizing data points (different markers/shapes by category).
        Examples: 'Material', 'Geology_Origin', 'Formation', 'Test_Type', 'Hole_ID'
        Enables geological intelligence for color assignment and marker styles.
        
    color_by_col : str, optional
        Column name for continuous color mapping (third dimension).
        Creates color gradient based on values in this column.
        Examples: 'Depth_m', 'Moisture_Content', 'Sample_Quality', 'Age_days'
        
    colormap : str, default 'viridis'
        Matplotlib colormap for continuous color mapping.
        Common options: 'viridis', 'plasma', 'coolwarm', 'terrain', 'RdYlGn'
        Only used when color_by_col is specified.
        
    === Axis Configuration ===
    xlim : tuple of (float, float), optional
        X-axis (chainage) limits as (min_value, max_value).
        If None, automatically determined from data with padding.
        Example: xlim=(0, 5000) sets chainage from 0 to 5000m
        
    ylim : tuple of (float, float), optional
        Y-axis (property) limits as (min_value, max_value).
        If None, automatically determined from data with padding.
        Example: ylim=(0, 100) sets property from 0 to 100
        
    xtick_interval : float, optional
        X-axis (chainage) tick spacing. Creates evenly spaced ticks.
        Example: xtick_interval=500.0 creates ticks every 500m (0, 500, 1000, ...)
        If None, uses matplotlib's automatic tick spacing.
        
    ytick_interval : float, optional
        Y-axis (property) tick spacing. Creates evenly spaced ticks.
        Example: ytick_interval=10.0 creates ticks every 10 units
        If None, uses matplotlib's automatic tick spacing.
        
    xlabel : str, optional
        Custom label for x-axis (chainage axis). If None, uses chainage display name.
        Example: 'Distance Along Alignment (m)' instead of default column name.
        
    ylabel : str, optional
        Custom label for y-axis (property axis). If None, uses property display name.
        Example: 'CBR (%)' instead of default column name.
        
    invert_yaxis : bool, default False
        Whether to invert y-axis (higher values at bottom).
        Useful for properties like elevation where lower values might be plotted at top.
        
    use_log_scale_x : bool, default False
        Whether to use logarithmic scale for x-axis (chainage).
        Rarely used but available for special cases.
        
    use_log_scale_y : bool, default False
        Whether to use logarithmic scale for y-axis (property values).
        Useful for properties spanning multiple orders of magnitude.
        
    === Classification Zones ===
    classification_zones : dict, optional
        Dictionary defining classification zones with boundaries.
        For horizontal zones (property-based): {'Zone_Name': (min_value, max_value), ...}
        For vertical zones (chainage-based): {'Zone_Name': (min_chainage, max_chainage), ...}
        Example: {'Low': (0, 25), 'Medium': (25, 50), 'High': (50, float('inf'))}
        
    zone_colors : dict, optional
        Colors for classification zones backgrounds.
        Format: {'Zone_Name': 'color', ...}
        If None, uses default color scheme.
        
    show_zone_boundaries : bool, default False
        Whether to show lines at zone classification boundaries.
        
    show_zone_labels : bool, default False
        Whether to show zone classification labels on plot.
        
    zone_label_position : float, default 0.85
        Position of zone labels as fraction along the plot.
        For horizontal zones: 0.0 = left, 0.5 = center, 1.0 = right
        For vertical zones: 0.0 = bottom, 0.5 = middle, 1.0 = top
        
    zone_orientation : str, default 'horizontal'
        Orientation of classification zones:
        'horizontal': zones based on property values (horizontal bands)
        'vertical': zones based on chainage values (vertical bands)
        
    === Connection Lines ===
    connect_points : bool, default False
        Whether to connect data points with lines.
        Useful for showing trends along chainage.
        
    line_style : str, default '-'
        Line style for connecting points.
        Options: '-', '--', '-.', ':', 'None'
        
    line_width : float, default 1.0
        Width of connection lines in points.
        
    line_alpha : float, default 0.7
        Transparency of connection lines (0.0=invisible, 1.0=solid).
        
    === Display Options ===
    show_plot : bool, default True
        Whether to display plot on screen. Set False for batch processing.
        
    show_legend : bool, default True
        Whether to show legend for categories (if category_by_col provided).
        
    show_grid : bool, default True
        Whether to show grid lines on plot.
        
    show_colorbar : bool, default True
        Whether to show colorbar for continuous color mapping.
        Only applies when color_by_col is specified.
        
    legend_outside : bool, default False
        Whether to place legend outside the plot area.
        Prevents legend from obscuring data points.
        
    legend_position : str, default 'right'
        Position for outside legend placement.
        Options: 'right', 'left', 'top', 'bottom'
        Only used when legend_outside=True.
        
    === Output Control ===
    output_filepath : str, optional
        Full path to save plot including filename and extension.
        Example: 'results/cbr_chainage_analysis.png'
        If provided, plot will be automatically saved to this location.
        
    dpi : int, default 300
        Resolution for saved figure in dots per inch.
        Standard values: 150 (draft), 300 (publication), 600 (high-res)
        
    === Visual Customization ===
    marker_size : int, default 50
        Size of scatter points in points^2.
        Examples: 80-100 for presentations, 30-40 for dense data
        
    marker_alpha : float, default 0.8
        Point transparency (0.0=invisible, 1.0=solid).
        Use lower values (0.6) for overlapping points.
        
    marker_edge_lw : float, default 0.5
        Width of point borders in points. Set to 0 for no borders.
        
    marker_style : str, default 'o'
        Marker style for scatter points.
        Options: 'o', 's', '^', 'v', 'D', '*', '+', 'x'
        
    === Font Styling ===
    title_fontsize : int, default 14
        Font size for main plot title.
        
    label_fontsize : int, default 12
        Font size for axis labels.
        
    tick_fontsize : int, default 10
        Font size for axis tick labels.
        
    legend_fontsize : int, default 11
        Font size for legend text.
        
    zone_label_fontsize : int, default 10
        Font size for zone classification text labels.
        
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
        
    line_style_dict : dict, optional
        Dictionary of line styling parameters for connections.
        Example: {'color': 'blue', 'linestyle': '--', 'alpha': 0.8}
        
    zone_boundary_style : dict, optional
        Dictionary of zone boundary line styling.
        Default: {'linestyle': '--', 'alpha': 0.6, 'linewidth': 1.0}
        Example: {'linestyle': ':', 'color': 'red', 'alpha': 0.8}
        
    zone_label_style : dict, optional
        Dictionary of zone label text styling.
        Example: {'fontweight': 'bold', 'fontsize': 12}
        
    palette : list or dict, optional
        Custom color palette for categories.
        List: ['blue', 'red', 'green'] (cycles through categories)
        Dict: {'BH001': 'blue', 'BH002': 'red'} (specific mapping)
        If None, uses intelligent geological defaults.
        
    colorbar_label : str, optional
        Custom label for colorbar. If None, uses color_by_col name.
        Example: 'Depth (mbgl)' instead of 'Depth_m'
        
    colorbar_position : str, default 'right'
        Position for colorbar placement.
        Options: 'right', 'left', 'top', 'bottom'
        
    color_limits : tuple of (float, float), optional
        Manual limits for color scale as (min_value, max_value).
        If None, uses full range of color_by_col data.
        Example: color_limits=(0, 30) for depth range 0-30m
        
    Returns
    -------
    None
    
    Examples
    --------
    **Basic usage:**
    >>> plot_by_chainage(df, 'Chainage', 'CBR_%')
    
    **Categorical grouping:**
    >>> plot_by_chainage(df, 'Chainage', 'CBR_%',
    ...                  category_by_col='Material',
    ...                  xlabel='Distance Along Alignment (m)',
    ...                  ylabel='CBR (%)',
    ...                  title='CBR Analysis by Material Type')
    
    **Third dimension - Color by depth:**
    >>> plot_by_chainage(df, 'Chainage', 'CBR_%',
    ...                  color_by_col='Depth_m',
    ...                  colormap='viridis',
    ...                  colorbar_label='Depth (mbgl)',
    ...                  xlabel='Distance Along Alignment (m)',
    ...                  ylabel='CBR (%)',
    ...                  title='CBR Analysis with Depth Gradient')
    
    **Multi-dimensional analysis:**
    >>> plot_by_chainage(df, 'Chainage', 'Emerson_Class',
    ...                  category_by_col='Formation',      # Different markers
    ...                  color_by_col='Depth_m',           # Color gradient
    ...                  marker_size=60,
    ...                  colormap='plasma',
    ...                  legend_outside=True,
    ...                  legend_position='top',            # Legend outside top
    ...                  colorbar_position='right',        # Colorbar on right
    ...                  colorbar_label='Depth (mbgl)',
    ...                  xlabel='Chainage (m)',
    ...                  ylabel='Emerson Class Number',
    ...                  title='Emerson Classification - Multi-Dimensional View')
    
    **Classification zones with gradients:**
    >>> zones = {'Low': (0, 15), 'Medium': (15, 30), 'High': (30, float('inf'))}
    >>> plot_by_chainage(df, 'Chainage', 'CBR_%',
    ...                  color_by_col='Depth_m',
    ...                  classification_zones=zones,
    ...                  show_zone_boundaries=True,
    ...                  show_zone_labels=True,
    ...                  zone_orientation='horizontal',
    ...                  colormap='coolwarm',
    ...                  color_limits=(0, 30),
    ...                  figsize=(15, 8),
    ...                  legend_outside=True,
    ...                  xlabel='Distance Along Alignment (m)',
    ...                  ylabel='CBR (%)',
    ...                  colorbar_label='Sample Depth (m)',
    ...                  title='CBR Classification Zones with Depth Analysis')
    
    **Ground surface profile with connected points:**
    >>> plot_by_chainage(df, 'Chainage', 'Elevation_m', 
    ...                  category_by_col='Hole_ID',
    ...                  color_by_col='Depth_to_Bedrock',
    ...                  connect_points=True,
    ...                  line_style='-',
    ...                  colormap='terrain',
    ...                  classification_zones={'Section_A': (0, 2000), 
    ...                                       'Section_B': (2000, 4500)},
    ...                  zone_orientation='vertical',
    ...                  show_zone_boundaries=True,
    ...                  figsize=(16, 6),
    ...                  legend_outside=True,
    ...                  legend_position='right',
    ...                  colorbar_position='bottom',
    ...                  title='Geotechnical Profile - Ground Surface & Bedrock')
    
    **Quality control dashboard:**
    >>> plot_by_chainage(df, 'Chainage', 'CBR_%',
    ...                  category_by_col='Hole_ID',
    ...                  color_by_col='Sample_Quality',
    ...                  classification_zones={'Poor': (0, 2), 'Fair': (2, 7), 
    ...                                       'Good': (7, 10)},
    ...                  zone_colors={'Poor': 'red', 'Fair': 'yellow', 'Good': 'green'},
    ...                  show_zone_boundaries=True,
    ...                  colormap='RdYlGn',
    ...                  legend_outside=True,
    ...                  legend_position='right',
    ...                  colorbar_position='bottom',
    ...                  output_filepath='cbr_quality_analysis.png')
    """
    # === Input Validation ===
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    # === Backward Compatibility Handling ===
    if category_col is not None and category_by_col is None:
        warnings.warn("Parameter 'category_col' is deprecated. Use 'category_by_col' instead.", 
                     DeprecationWarning, stacklevel=2)
        category_by_col = category_col
    elif category_col is not None and category_by_col is not None:
        warnings.warn("Both 'category_col' and 'category_by_col' provided. Using 'category_by_col'.", 
                     UserWarning, stacklevel=2)
    
    # Validate required columns exist
    required_cols = [chainage_col, property_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    # Validate category column if provided
    if category_by_col and category_by_col not in df.columns:
        raise ValueError(f"Category column '{category_by_col}' not found in DataFrame.")
    
    # Validate color column if provided
    if color_by_col and color_by_col not in df.columns:
        raise ValueError(f"Color column '{color_by_col}' not found in DataFrame.")
    
    # Validate axis limits
    if xlim is not None and not (isinstance(xlim, tuple) and len(xlim) == 2):
        warnings.warn("xlim must be a tuple (min, max). Ignoring.")
        xlim = None
    if ylim is not None and not (isinstance(ylim, tuple) and len(ylim) == 2):
        warnings.warn("ylim must be a tuple (min, max). Ignoring.")
        ylim = None
    
    # Validate zone orientation
    if zone_orientation not in ['horizontal', 'vertical']:
        warnings.warn("zone_orientation must be 'horizontal' or 'vertical'. Using 'horizontal'.")
        zone_orientation = 'horizontal'
    
    # Validate color limits
    if color_limits is not None and not (isinstance(color_limits, tuple) and len(color_limits) == 2):
        warnings.warn("color_limits must be a tuple (min, max). Ignoring.")
        color_limits = None
    
    # Validate legend position
    valid_legend_positions = ['right', 'left', 'top', 'bottom']
    if legend_position not in valid_legend_positions:
        warnings.warn(f"legend_position must be one of {valid_legend_positions}. Using 'right'.")
        legend_position = 'right'
    
    # Validate colorbar position
    valid_colorbar_positions = ['right', 'left', 'top', 'bottom']
    if colorbar_position not in valid_colorbar_positions:
        warnings.warn(f"colorbar_position must be one of {valid_colorbar_positions}. Using 'right'.")
        colorbar_position = 'right'
    
    # === Data Preparation ===
    data = df.copy()
    
    # Convert numeric columns and handle errors
    data[chainage_col] = pd.to_numeric(data[chainage_col], errors='coerce')
    data[property_col] = pd.to_numeric(data[property_col], errors='coerce')
    
    # Convert color column if provided
    if color_by_col:
        data[color_by_col] = pd.to_numeric(data[color_by_col], errors='coerce')
    
    # Remove rows with missing essential data
    initial_rows = len(data)
    essential_cols = [chainage_col, property_col]
    if color_by_col:
        essential_cols.append(color_by_col)
    
    data = data.dropna(subset=essential_cols)
    if len(data) < initial_rows:
        removed_count = initial_rows - len(data)
        print(f"INFO: Removed {removed_count} rows with missing essential data.")
    
    if data.empty:
        print("WARNING: No valid data points after cleaning. Skipping plot.")
        return
    
    # Sort by chainage for proper line connections
    data = data.sort_values(by=chainage_col)
    
    # === Display Names ===
    display_property_name = property_name if property_name is not None else property_col
    display_chainage_name = chainage_name if chainage_name is not None else chainage_col
    
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
        
        # Borehole patterns (BH001, BH-01, etc.)
        borehole_patterns = [r'BH\d+', r'BH-\d+', r'BH_\d+', r'BORE\d+', r'HOLE\d+']
        for pattern in borehole_patterns:
            if re.search(pattern, cat_str):
                return cat_str  # Keep original borehole ID
        
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
    default_line_style = {'alpha': line_alpha, 'linewidth': line_width, 'zorder': 3}
    default_zone_boundary_style = {'linestyle': '--', 'color': 'gray', 'alpha': 0.4, 'linewidth': 1.0}
    default_zone_label_style = {'fontsize': zone_label_fontsize}
    
    # Apply styling with user overrides
    grid_params = {**default_grid_style, **(grid_style or {})}
    axis_params = {**default_axis_style, **(axis_style or {})}
    scatter_params = {**default_scatter_style, **(scatter_style or {})}
    line_params = {**default_line_style, **(line_style_dict or {})}
    zone_boundary_params = {**default_zone_boundary_style, **(zone_boundary_style or {})}
    zone_label_params = {**default_zone_label_style, **(zone_label_style or {})}
    
    # === Main Plotting Logic ===
    # Prepare color data for continuous coloring
    color_data = None
    if color_by_col:
        color_data = data[color_by_col].values
        # Apply color limits if specified
        if color_limits:
            color_data = np.clip(color_data, color_limits[0], color_limits[1])
    
    # Initialize colorbar object for later use
    colorbar_obj = None
    
    if category_by_col and not color_by_col:
        # === Categorical coloring only ===
        categories = data[category_by_col].dropna().unique()
        
        # Set up color scheme
        if isinstance(palette, dict):
            color_map = palette
        elif isinstance(palette, list):
            color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
        else:
            # Use geological intelligence
            color_map = {}
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
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
            cat_data = data[data[category_by_col] == category]
            
            # Scatter plot
            ax.scatter(cat_data[chainage_col], cat_data[property_col],
                      label=str(category),
                      color=color_map.get(category, '#1f77b4'),
                      s=marker_size,
                      alpha=marker_alpha,
                      linewidths=marker_edge_lw,
                      marker=marker_style,
                      **scatter_params)
            
            # Connection lines if requested
            if connect_points and len(cat_data) > 1:
                ax.plot(cat_data[chainage_col], cat_data[property_col],
                       color=color_map.get(category, '#1f77b4'),
                       linestyle=line_style,
                       **line_params)
    
    elif color_by_col and not category_by_col:
        # === Continuous coloring only ===
        scatter = ax.scatter(data[chainage_col], data[property_col],
                           c=color_data,
                           cmap=colormap,
                           s=marker_size,
                           alpha=marker_alpha,
                           linewidths=marker_edge_lw,
                           marker=marker_style,
                           **scatter_params)
        
        # Store scatter object for colorbar
        colorbar_obj = scatter
        
        # Connection lines if requested (single color)
        if connect_points and len(data) > 1:
            ax.plot(data[chainage_col], data[property_col],
                   color='gray',
                   linestyle=line_style,
                   alpha=line_alpha * 0.7,  # Make lines more transparent
                   **line_params)
    
    elif category_by_col and color_by_col:
        # === Hybrid: Categories with continuous coloring ===
        categories = data[category_by_col].dropna().unique()
        
        # Set up marker styles for categories
        marker_styles = ['o', 's', '^', 'v', 'D', '*', 'P', 'X', 'h', '+']
        category_markers = {cat: marker_styles[i % len(marker_styles)] 
                          for i, cat in enumerate(categories)}
        
        # Plot each category with continuous coloring
        for category in categories:
            cat_data = data[data[category_by_col] == category]
            cat_color_data = cat_data[color_by_col].values
            
            if color_limits:
                cat_color_data = np.clip(cat_color_data, color_limits[0], color_limits[1])
            
            # Scatter plot with category marker and continuous color
            scatter = ax.scatter(cat_data[chainage_col], cat_data[property_col],
                               c=cat_color_data,
                               cmap=colormap,
                               label=str(category),
                               s=marker_size,
                               alpha=marker_alpha,
                               linewidths=marker_edge_lw,
                               marker=category_markers.get(category, 'o'),
                               **scatter_params)
            
            # Store the last scatter object for colorbar (all use same colormap)
            colorbar_obj = scatter
            
            # Connection lines if requested
            if connect_points and len(cat_data) > 1:
                ax.plot(cat_data[chainage_col], cat_data[property_col],
                       linestyle=line_style,
                       alpha=line_alpha * 0.7,
                       **line_params)
    
    else:
        # === No categorical or continuous coloring ===
        ax.scatter(data[chainage_col], data[property_col],
                  color='#1f77b4',
                  s=marker_size,
                  alpha=marker_alpha,
                  linewidths=marker_edge_lw,
                  marker=marker_style,
                  **scatter_params)
        
        # Connection lines if requested
        if connect_points and len(data) > 1:
            ax.plot(data[chainage_col], data[property_col],
                   color='#1f77b4',
                   linestyle=line_style,
                   **line_params)
    
    # === Set Scales ===
    if use_log_scale_x:
        ax.set_xscale('log')
    if use_log_scale_y:
        ax.set_yscale('log')
    
    # Placeholder for zone boundaries - will be added after axis limits are set
    
    # === Axis Configuration ===
    # Set axis labels with custom titles or defaults
    x_label = xlabel if xlabel is not None else display_chainage_name
    y_label = ylabel if ylabel is not None else display_property_name
    
    ax.set_xlabel(x_label, 
                  fontsize=axis_params['xlabel_fontsize'], 
                  fontweight=axis_params['xlabel_fontweight'])
    ax.set_ylabel(y_label, 
                  fontsize=axis_params['ylabel_fontsize'], 
                  fontweight=axis_params['ylabel_fontweight'])
    
    # Handle title
    final_title = None
    if title is not None:
        final_title = title
    else:
        final_title = f'{display_property_name} vs {display_chainage_name}'
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
        chainage_min, chainage_max = data[chainage_col].min(), data[chainage_col].max()
        if use_log_scale_x:
            # Auto-scale for log scale, ensuring positive values
            x_min = min(chainage_min * 0.1, chainage_min * 0.5)
            x_max = max(chainage_max * 10, chainage_max * 2)
            ax.set_xlim(x_min, x_max)
        else:
            # Auto-scale with padding
            chainage_range = chainage_max - chainage_min
            padding = max(0.05 * chainage_range, 10)  # At least 10m padding
            x_min = max(0, chainage_min - padding) if chainage_min >= 0 else chainage_min - padding
            x_max = chainage_max + padding
            ax.set_xlim(x_min, x_max)
    
    # Set custom x-axis ticks if interval specified
    if xtick_interval is not None:
        current_xlim = ax.get_xlim()
        x_start = current_xlim[0]
        x_end = current_xlim[1]
        xtick_values = np.arange(x_start, x_end + xtick_interval, xtick_interval)
        ax.set_xticks(xtick_values)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Auto-scale with padding
        prop_min, prop_max = data[property_col].min(), data[property_col].max()
        if use_log_scale_y:
            # Auto-scale for log scale, ensuring positive values
            y_min = min(prop_min * 0.1, prop_min * 0.5)
            y_max = max(prop_max * 10, prop_max * 2)
            ax.set_ylim(y_min, y_max)
        else:
            prop_range = prop_max - prop_min
            padding = max(0.1 * prop_range, 0.01 * prop_max)
            y_min = max(0, prop_min - padding) if prop_min >= 0 else prop_min - padding
            y_max = prop_max + padding
            ax.set_ylim(y_min, y_max)
    
    # Set custom y-axis ticks if interval specified
    if ytick_interval is not None:
        current_ylim = ax.get_ylim()
        y_start = current_ylim[0]
        y_end = current_ylim[1]
        ytick_values = np.arange(y_start, y_end + ytick_interval, ytick_interval)
        ax.set_yticks(ytick_values)
    
    # Invert y-axis if requested
    if invert_yaxis:
        ax.invert_yaxis()
    
    # Apply grid
    if show_grid:
        ax.grid(True, **grid_params)
    
    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # === Add Classification Zones (after axis limits are set) ===
    if classification_zones:
        # Get current axis limits for proper zone rendering
        xlim_current = ax.get_xlim()
        ylim_current = ax.get_ylim()
        
        # Default zone colors if not provided
        default_zone_colors = {
            'Low': 'lightcoral', 'Medium': 'lightyellow', 'High': 'lightgreen',
            'Poor': 'lightcoral', 'Fair': 'lightyellow', 'Good': 'lightgreen',
            'Weak': 'lightcoral', 'Moderate': 'lightyellow', 'Strong': 'lightgreen'
        }
        zone_color_map = {**default_zone_colors, **(zone_colors or {})}
        
        if zone_orientation == 'horizontal':
            # Horizontal zones based on property values
            for zone_name, (min_val, max_val) in classification_zones.items():
                # Handle infinite boundaries
                y_min = max(min_val, ylim_current[0]) if min_val != float('-inf') else ylim_current[0]
                y_max = min(max_val, ylim_current[1]) if max_val != float('inf') else ylim_current[1]
                
                # Only proceed if zone is visible in current view
                if y_max > y_min:
                    # Always add zone background with appropriate color
                    zone_color = zone_color_map.get(zone_name, 'lightgray')
                    ax.axhspan(y_min, y_max, alpha=0.2, color=zone_color, zorder=1)
                
                # Add zone boundaries
                if show_zone_boundaries:
                    if min_val != float('-inf') and min_val >= ylim_current[0] and min_val <= ylim_current[1]:
                        ax.axhline(y=min_val, **zone_boundary_params)
                    if max_val != float('inf') and max_val >= ylim_current[0] and max_val <= ylim_current[1]:
                        ax.axhline(y=max_val, **zone_boundary_params)
        
        else:  # vertical zones
            # Vertical zones based on chainage values
            for zone_name, (min_val, max_val) in classification_zones.items():
                # Handle infinite boundaries
                x_min = max(min_val, xlim_current[0]) if min_val != float('-inf') else xlim_current[0]
                x_max = min(max_val, xlim_current[1]) if max_val != float('inf') else xlim_current[1]
                
                # Only proceed if zone is visible in current view
                if x_max > x_min:
                    # Always add zone background with appropriate color
                    zone_color = zone_color_map.get(zone_name, 'lightgray')
                    ax.axvspan(x_min, x_max, alpha=0.2, color=zone_color, zorder=1)
                
                # Add zone boundaries
                if show_zone_boundaries:
                    if min_val != float('-inf') and min_val >= xlim_current[0] and min_val <= xlim_current[1]:
                        ax.axvline(x=min_val, **zone_boundary_params)
                    if max_val != float('inf') and max_val >= xlim_current[0] and max_val <= xlim_current[1]:
                        ax.axvline(x=max_val, **zone_boundary_params)
    
    # === Add Legend ===
    if show_legend and category_by_col:
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
            
            # Handle outside legend placement
            if legend_outside:
                bbox_positions = {
                    'right': (1.05, 0.5),
                    'left': (-0.05, 0.5),
                    'top': (0.5, 1.05),
                    'bottom': (0.5, -0.05)
                }
                anchor_positions = {
                    'right': 'center left',
                    'left': 'center right', 
                    'top': 'lower center',
                    'bottom': 'upper center'
                }
                
                default_legend_params.update({
                    'bbox_to_anchor': bbox_positions.get(legend_position, (1.05, 0.5)),
                    'loc': anchor_positions.get(legend_position, 'center left')
                })
            
            if legend_style:
                default_legend_params.update(legend_style)
            ax.legend(**default_legend_params)
    
    # === Add Colorbar ===
    if show_colorbar and colorbar_obj is not None:
        # Determine colorbar label
        cbar_label = colorbar_label if colorbar_label is not None else (
            color_by_col if color_by_col else 'Value'
        )
        
        # Smart positioning: avoid conflicts with outside legend
        final_colorbar_position = colorbar_position
        if legend_outside and legend_position == colorbar_position:
            # Conflict detected - move colorbar to different position
            if colorbar_position == 'right':
                final_colorbar_position = 'bottom'
            elif colorbar_position == 'left':
                final_colorbar_position = 'bottom'
            elif colorbar_position == 'top':
                final_colorbar_position = 'right'
            elif colorbar_position == 'bottom':
                final_colorbar_position = 'right'
        
        # Create colorbar with smart positioning and appropriate sizing
        if final_colorbar_position == 'right':
            cbar = plt.colorbar(colorbar_obj, ax=ax, location='right', shrink=0.8, fraction=0.02, pad=0.05)
        elif final_colorbar_position == 'left':
            cbar = plt.colorbar(colorbar_obj, ax=ax, location='left', shrink=0.8, fraction=0.02, pad=0.05)
        elif final_colorbar_position == 'top':
            cbar = plt.colorbar(colorbar_obj, ax=ax, location='top', shrink=0.6, fraction=0.03, pad=0.08)
        elif final_colorbar_position == 'bottom':
            cbar = plt.colorbar(colorbar_obj, ax=ax, location='bottom', shrink=0.6, fraction=0.03, pad=0.08)
        else:
            cbar = plt.colorbar(colorbar_obj, ax=ax, shrink=0.8, fraction=0.02, pad=0.05)
        
        # Style the colorbar
        cbar.set_label(cbar_label, fontsize=label_fontsize, fontweight='bold')
        cbar.ax.tick_params(labelsize=tick_fontsize)
    
    # === Add Zone Labels ===
    if show_zone_labels and classification_zones:
        # Get current axis limits for zone labels (may have changed after colorbar)
        xlim_current = ax.get_xlim()
        ylim_current = ax.get_ylim()
        
        if zone_orientation == 'horizontal':
            # Horizontal zones - labels positioned along x-axis
            text_x = xlim_current[0] + zone_label_position * (xlim_current[1] - xlim_current[0])
            
            for zone_name, (min_val, max_val) in classification_zones.items():
                # Calculate text position within visible area and zone bounds
                zone_y_min = max(min_val, ylim_current[0]) if min_val != float('-inf') else ylim_current[0]
                zone_y_max = min(max_val, ylim_current[1]) if max_val != float('inf') else ylim_current[1]
                
                # Only add label if zone is visible
                if zone_y_max > zone_y_min:
                    if use_log_scale_y and zone_y_min > 0 and zone_y_max > 0:
                        text_y = np.sqrt(zone_y_min * zone_y_max)
                    else:
                        text_y = (zone_y_min + zone_y_max) / 2
                    
                    ax.text(text_x, text_y, zone_name,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                            **zone_label_params)
        
        else:  # vertical zones
            # Vertical zones - labels positioned along y-axis
            text_y = ylim_current[0] + zone_label_position * (ylim_current[1] - ylim_current[0])
            
            for zone_name, (min_val, max_val) in classification_zones.items():
                # Calculate text position within visible area and zone bounds
                zone_x_min = max(min_val, xlim_current[0]) if min_val != float('-inf') else xlim_current[0]
                zone_x_max = min(max_val, xlim_current[1]) if max_val != float('inf') else xlim_current[1]
                
                # Only add label if zone is visible
                if zone_x_max > zone_x_min:
                    if use_log_scale_x and zone_x_min > 0 and zone_x_max > 0:
                        text_x = np.sqrt(zone_x_min * zone_x_max)
                    else:
                        text_x = (zone_x_min + zone_x_max) / 2
                    
                    ax.text(text_x, text_y, zone_name,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                            **zone_label_params)
    
    # === Layout Optimization ===
    # Adjust layout based on outside elements
    if legend_outside or (show_colorbar and colorbar_obj is not None) or (show_zone_labels and classification_zones):
        try:
            # More padding for outside elements
            pad = 2.0 if legend_outside else 1.5
            plt.tight_layout(pad=pad)
        except:
            # Fallback: manual subplot adjustment
            try:
                if legend_outside and legend_position == 'right':
                    plt.subplots_adjust(right=0.85)
                elif legend_outside and legend_position == 'left':
                    plt.subplots_adjust(left=0.15)
                elif legend_outside and legend_position == 'top':
                    plt.subplots_adjust(top=0.90)
                elif legend_outside and legend_position == 'bottom':
                    plt.subplots_adjust(bottom=0.15)
                else:
                    plt.tight_layout()
            except:
                pass  # Skip layout adjustment if it causes issues
    else:
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
    
    print("--- Property vs Chainage plotting finished ---")