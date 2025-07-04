import itertools
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline # Keep import for optional use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns # For KDE plots (optional)
import os
import warnings
from pandas.api.types import is_numeric_dtype, is_object_dtype
from matplotlib.container import BarContainer # NEW IMPORT
from typing import List, Optional, Union, Sequence, Dict, Tuple, Any
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import re


def plot_UCS_Is50(
    # === Required Data Parameters ===
    datasets: List[Dict[str, Any]],
    
    # === Plot Appearance ===
    title: Optional[str] = None,
    title_suffix: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 7),
    
    # === Category Options ===
    category_by: Optional[str] = None,
    
    # === Axis Limits ===
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    
    # === Axis Ticks ===
    xticks: Optional[List[float]] = None,
    yticks: Optional[List[float]] = None,
    xtick_interval: Optional[float] = None,
    ytick_interval: Optional[float] = None,
    
    # === Trendline Options ===
    show_trendlines: bool = True,
    show_equations: bool = True,
    show_r_squared: bool = True,
    
    # === Display Options ===
    show_plot: bool = True,
    show_legend: bool = True,
    
    # === Save Options ===
    output_filepath: Optional[str] = None,
    save_dpi: int = 300,
    
    # === Advanced Styling Options ===
    plot_style: Optional[str] = 'seaborn-v0_8-whitegrid',
    grid_style: Optional[Dict[str, Any]] = None,
    axis_style: Optional[Dict[str, Any]] = None,
    legend_loc: Optional[str] = 'best',
    legend_fontsize: Optional[Union[int, float, str]] = 11,
    legend_style: Optional[Dict[str, Any]] = None,
    scatter_alpha: Optional[float] = 0.65,
    scatter_size: Optional[Union[int, float]] = 35,
    scatter_style: Optional[Dict[str, Any]] = None,
    trendline_style: Optional[Dict[str, Any]] = None,
    equation_position: Optional[Tuple[float, float]] = None,
    equation_style: Optional[Dict[str, Any]] = None,
    
    # === Plot Cleanup Control ===
    close_plot: bool = True
    ):
    """
    Create scatter plots of UCS vs Is50 data with flexible dataset support and optional trendlines.
    
    This function creates scatter plots comparing UCS (Unconfined Compressive Strength) against 
    one or more Is50 (Point Load Index) datasets, with optional trendline analysis and equations.
    
    Parameters
    ----------
    === Required Data Parameters ===
    datasets : list of dict
        **REQUIRED** List of dataset configurations. Each dataset specifies its own DataFrame.
        Each dictionary defines one dataset with these keys:
        
        REQUIRED keys:
        - 'data_df': pd.DataFrame - DataFrame containing the data for this dataset
        - 'x_col': str - Column name in data_df containing Is50 values
        - 'y_col': str - Column name in data_df containing UCS values
        
        OPTIONAL keys (auto-generated if not provided):
        - 'label': str - Legend label (default: "{x_col_name} Data")
        - 'color': str - Scatter point color (default: auto-assigned from palette)
        - 'marker': str - Point marker style (default: auto-assigned: 'o', 's', '^', etc.)
        - 'trend_color': str - Trendline color (default: same as point color)
        - 'show_trendline': bool - Show trendline for this dataset (default: uses global show_trendlines)
        
        Example single dataset from one DataFrame:
        >>> datasets = [{'data_df': df, 'x_col': 'Is50a (MPa)', 'y_col': 'UCS (MPa)', 'label': 'Is50a'}]
        
        Example multiple datasets from same DataFrame:
        >>> datasets = [
        ...     {'data_df': df, 'x_col': 'Is50a (MPa)', 'y_col': 'UCS (MPa)', 'label': 'Axial'},
        ...     {'data_df': df, 'x_col': 'Is50d (MPa)', 'y_col': 'UCS (MPa)', 'label': 'Diametral'}
        ... ]
        
        Example multiple datasets from different DataFrames:
        >>> datasets = [
        ...     {'data_df': lab_results, 'x_col': 'Is50', 'y_col': 'UCS', 'label': 'Lab Results'},
        ...     {'data_df': field_data, 'x_col': 'Is50_field', 'y_col': 'UCS_field', 'label': 'Field Data'}
        ... ]
        
    === Plot Appearance ===
    title : str, optional
        Custom plot title. If None, uses default "UCS vs Is50 Comparison".
        If provided, completely replaces the default title.
        
    title_suffix : str, optional
        Text to append to default title. Only used if title=None.
        Creates title: "UCS vs Is50 Comparison: {title_suffix}"
        
    figsize : tuple of (float, float), default (9, 7)
        Figure size in inches as (width, height).
        Larger values create bigger plots. Standard sizes: (8,6), (10,8), (12,9)
        
    === Category Options ===
    category_by : str, optional
        Column name to split datasets into categories. If provided, each dataset will be
        automatically split by unique values in this column, with each category getting
        its own color/marker and legend entry.
        Example: category_by='Material' splits data by material types
        Format in legend: "{dataset_label} - {category_value}"
        
    === Axis Limits ===
    xlim : tuple of (float, float), optional
        X-axis (Is50) limits as (min_value, max_value).
        If None, automatically determined from data with 5% padding.
        Example: xlim=(0, 10) sets x-axis from 0 to 10 MPa
        
    ylim : tuple of (float, float), optional
        Y-axis (UCS) limits as (min_value, max_value).
        If None, automatically determined from data with 5% padding.
        Example: ylim=(0, 200) sets y-axis from 0 to 200 MPa
        
    === Axis Ticks ===
    xticks : list of float, optional
        Custom x-axis tick positions. Overrides automatic tick placement.
        Example: xticks=[0, 2, 4, 6, 8, 10] creates ticks at these exact positions
        
    yticks : list of float, optional
        Custom y-axis tick positions. Overrides automatic tick placement.
        Example: yticks=[0, 50, 100, 150, 200] creates ticks at these exact positions
        
    xtick_interval : float, optional
        Automatic x-axis tick spacing. Creates evenly spaced ticks.
        Example: xtick_interval=1.0 creates ticks every 1 unit (0, 1, 2, 3, ...)
        Ignored if xticks is provided.
        
    ytick_interval : float, optional
        Automatic y-axis tick spacing. Creates evenly spaced ticks.
        Example: ytick_interval=25.0 creates ticks every 25 units (0, 25, 50, 75, ...)
        Ignored if yticks is provided.
        
    === Trendline Options ===
    show_trendlines : bool, default True
        Global control for showing trendlines on all datasets.
        Can be overridden per dataset using 'show_trendline' in datasets config.
        Trendlines are forced through origin (0,0) for geotechnical correlation.
        
    show_equations : bool, default True
        Whether to show trendline equations in bottom-right text box.
        Format: "Is50a: UCS = 15.23*Is50a, R²=0.85"
        Only appears if show_trendlines=True and equations exist.
        
    show_r_squared : bool, default True
        Whether to include R² correlation coefficients in equation text.
        Only relevant if show_equations=True.
        
    === Display Options ===
    show_plot : bool, default True
        Whether to display the plot on screen.
        Set to False for batch processing or when only saving plots.
        
    show_legend : bool, default True
        Whether to show legend with dataset labels and trendline info.
        Legend auto-positioned to avoid overlapping with data.
        
    === Save Options ===
    output_filepath : str, optional
        Full file path to save the plot (including extension).
        If None, plot is not saved to file.
        Creates directories if they don't exist.
        Example: "/path/to/output/ucs_is50_comparison.png"
        
    save_dpi : int, default 300
        Resolution for saved figure in dots per inch.
        Higher values create larger, sharper files.
        Standard values: 150 (draft), 300 (publication), 600 (high-res)
        
    === Advanced Styling Options ===
    plot_style : str, optional, default 'seaborn-v0_8-whitegrid'
        Matplotlib style to apply to the plot.
        Common options: 'default', 'seaborn-v0_8-whitegrid', 'ggplot', 'bmh', 'classic'
        Set to None to use current matplotlib style settings.
        
    grid_style : dict, optional
        Dictionary of grid styling parameters to pass to ax.grid().
        Default: {'linestyle': '--', 'color': 'grey', 'alpha': 0.35}
        Example: {'linestyle': ':', 'color': 'blue', 'alpha': 0.5, 'linewidth': 0.8}
        
    axis_style : dict, optional
        Dictionary of axis label and title styling parameters.
        Available keys: 'xlabel_fontsize', 'xlabel_fontweight', 'ylabel_fontsize', 
        'ylabel_fontweight', 'title_fontsize', 'title_fontweight'
        Default: all font sizes 12 (labels) or 14 (title), all fontweight 'bold'
        Example: {'xlabel_fontsize': 14, 'title_fontweight': 'normal'}
        
    legend_loc : str, optional, default 'best'
        Legend location. Standard matplotlib legend locations:
        'best', 'upper right', 'upper left', 'lower left', 'lower right',
        'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        
    legend_fontsize : int, float, or str, optional, default 11
        Legend font size. Can be numeric or string ('xx-small', 'x-small', 'small',
        'medium', 'large', 'x-large', 'xx-large').
        
    legend_style : dict, optional
        Dictionary of additional legend styling parameters to pass to ax.legend().
        Example: {'frameon': True, 'fancybox': True, 'shadow': True, 'framealpha': 0.9,
        'facecolor': 'lightgray', 'edgecolor': 'black'}
        
    scatter_alpha : float, optional, default 0.65
        Transparency level for scatter points (0.0 = fully transparent, 1.0 = opaque).
        
    scatter_size : int or float, optional, default 35
        Size of scatter points in points^2.
        
    scatter_style : dict, optional
        Dictionary of additional scatter plot styling parameters to pass to ax.scatter().
        Example: {'edgecolors': 'black', 'linewidths': 0.5, 'zorder': 10}
        
    trendline_style : dict, optional
        Dictionary of trendline styling parameters to pass to ax.plot().
        Default: {'linestyle': '--', 'zorder': 4}
        Example: {'linestyle': '-', 'linewidth': 2, 'alpha': 0.8}
        
    equation_position : tuple of (float, float), optional
        Position for equation text box in axes coordinates (0-1).
        Default: (0.98, 0.02) (bottom-right corner)
        Example: (0.02, 0.98) for top-left corner
        
    equation_style : dict, optional
        Dictionary of equation text box styling parameters to pass to ax.text().
        Default: {'fontsize': 12, 'verticalalignment': 'bottom', 
        'horizontalalignment': 'right', 'bbox': dict(boxstyle='round,pad=0.4', 
        facecolor='white', alpha=0.8)}
        Example: {'fontsize': 10, 'bbox': dict(boxstyle='square', facecolor='yellow')}
        
    Returns
    -------
    None
    
    Examples
    --------
    **Minimal usage (single dataset):**
    >>> datasets = [{'data_df': df, 'x_col': 'Is50a (MPa)', 'y_col': 'UCS (MPa)'}]
    >>> plot_UCS_Is50(datasets)
    
    **Single dataset with custom styling:**
    >>> datasets = [{'data_df': df, 'x_col': 'Is50a (MPa)', 'y_col': 'UCS (MPa)', 
    ...              'label': 'Is50a Data', 'color': 'blue', 'marker': 'o'}]
    >>> plot_UCS_Is50(datasets, title="My UCS vs Is50 Analysis")
    
    **Multiple datasets from same DataFrame (auto-styled):**
    >>> datasets = [
    ...     {'data_df': df, 'x_col': 'Is50a (MPa)', 'y_col': 'UCS (MPa)'},  # Auto-assigned orange, circle
    ...     {'data_df': df, 'x_col': 'Is50d (MPa)', 'y_col': 'UCS (MPa)'},  # Auto-assigned blue, square
    ...     {'data_df': df, 'x_col': 'Is50_calc (MPa)', 'y_col': 'UCS (MPa)'}  # Auto-assigned green, triangle
    ... ]
    >>> plot_UCS_Is50(datasets)
    
    **Multiple datasets from different DataFrames:**
    >>> datasets = [
    ...     {'data_df': lab_results, 'x_col': 'Is50', 'y_col': 'UCS', 'label': 'Lab Results'},
    ...     {'data_df': field_data, 'x_col': 'Is50_field', 'y_col': 'UCS_field', 'label': 'Field Data'},
    ...     {'data_df': calculated, 'x_col': 'Is50_calc', 'y_col': 'UCS_calc', 'label': 'Calculated'}
    ... ]
    >>> plot_UCS_Is50(datasets, title_suffix="Multi-Source Comparison")
    
    **Custom axis limits and ticks:**
    >>> plot_UCS_Is50(datasets, 
    ...               xlim=(0, 12), ylim=(0, 250),
    ...               xticks=[0, 2, 4, 6, 8, 10, 12], 
    ...               yticks=[0, 50, 100, 150, 200, 250])
    
    **Regular tick intervals:**
    >>> plot_UCS_Is50(datasets, 
    ...               xtick_interval=1.5, ytick_interval=25,
    ...               figsize=(10, 8))
    
    **Scatter plots only (no trendlines):**
    >>> plot_UCS_Is50(datasets, show_trendlines=False)
    
    **Show trendlines but hide equations:**
    >>> plot_UCS_Is50(datasets, show_equations=False)
    
    **Save without displaying:**
    >>> plot_UCS_Is50(datasets, 
    ...               show_plot=False, 
    ...               output_filepath="/path/to/ucs_is50_plot.png",
    ...               save_dpi=600)
    
    **Per-dataset trendline control:**
    >>> datasets = [
    ...     {'data_df': df, 'x_col': 'Is50a (MPa)', 'y_col': 'UCS (MPa)', 'show_trendline': True},
    ...     {'data_df': df, 'x_col': 'Is50d (MPa)', 'y_col': 'UCS (MPa)', 'show_trendline': False}
    ... ]
    >>> plot_UCS_Is50(datasets, show_trendlines=True)  # Global setting overridden per dataset
    
    **Different column names per dataset:**
    >>> datasets = [
    ...     {'data_df': df1, 'x_col': 'Is50', 'y_col': 'UCS'},
    ...     {'data_df': df2, 'x_col': 'Is50_value', 'y_col': 'UCS_corrected'}
    ... ]
    >>> plot_UCS_Is50(datasets)
    
    **Advanced styling examples:**
    
    **Custom matplotlib style and grid:**
    >>> plot_UCS_Is50(datasets, 
    ...               plot_style='ggplot',
    ...               grid_style={'linestyle': ':', 'color': 'blue', 'alpha': 0.3})
    
    **Custom legend positioning and styling:**
    >>> plot_UCS_Is50(datasets,
    ...               legend_loc='upper left',
    ...               legend_fontsize=13,
    ...               legend_style={'frameon': True, 'shadow': True, 'fancybox': True})
    
    **Custom scatter and trendline styling:**
    >>> plot_UCS_Is50(datasets,
    ...               scatter_alpha=0.8,
    ...               scatter_size=50,
    ...               scatter_style={'edgecolors': 'black', 'linewidths': 0.8},
    ...               trendline_style={'linestyle': '-', 'linewidth': 3, 'alpha': 0.9})
    
    **Custom axis styling:**
    >>> plot_UCS_Is50(datasets,
    ...               axis_style={'xlabel_fontsize': 16, 'ylabel_fontsize': 16,
    ...                          'title_fontsize': 18, 'title_fontweight': 'normal'})
    
    **Custom equation positioning and styling:**
    >>> plot_UCS_Is50(datasets,
    ...               equation_position=(0.02, 0.98),  # Top-left corner
    ...               equation_style={'fontsize': 10, 
    ...                              'bbox': dict(boxstyle='square', facecolor='lightblue', alpha=0.7)})
    
    **Comprehensive styling example:**
    >>> plot_UCS_Is50(datasets,
    ...               plot_style='bmh',
    ...               grid_style={'linestyle': '-', 'alpha': 0.2},
    ...               axis_style={'xlabel_fontsize': 14, 'title_fontweight': 'normal'},
    ...               legend_loc='lower right',
    ...               legend_style={'framealpha': 0.9, 'edgecolor': 'black'},
    ...               scatter_alpha=0.7,
    ...               scatter_size=40,
    ...               trendline_style={'linewidth': 2.5},
    ...               equation_position=(0.05, 0.95))
    """
    # === Input Validation ===
    if not datasets or len(datasets) == 0:
        raise ValueError("At least one dataset must be provided.")
    
    # Validate and prepare datasets
    processed_datasets = []
    default_colors = ['orange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    default_markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
    
    for i, dataset in enumerate(datasets):
        # Validate required fields
        required_fields = ['data_df', 'x_col', 'y_col']
        for field in required_fields:
            if field not in dataset:
                raise ValueError(f"Dataset {i} missing required '{field}' field.")
        
        data_df = dataset['data_df']
        x_col = dataset['x_col']
        y_col = dataset['y_col']
        
        # Validate DataFrame
        if data_df is None or data_df.empty:
            raise ValueError(f"Dataset {i}: DataFrame is empty.")
        
        # Validate columns exist
        if x_col not in data_df.columns:
            raise ValueError(f"Dataset {i}: X column '{x_col}' not found in DataFrame.")
        if y_col not in data_df.columns:
            raise ValueError(f"Dataset {i}: Y column '{y_col}' not found in DataFrame.")
        
        # Check if category_by column exists (if specified)
        if category_by and category_by not in data_df.columns:
            raise ValueError(f"Dataset {i}: Category column '{category_by}' not found in DataFrame.")
        
        # Handle category splitting if category_by is specified
        if category_by:
            # Get unique categories, excluding NaN values
            categories = data_df[category_by].dropna().unique()
            categories = sorted(categories)  # Sort for consistent ordering
            
            for cat_idx, category in enumerate(categories):
                # Filter data for this category
                cat_data = data_df[data_df[category_by] == category].copy()
                
                if not cat_data.empty:
                    # Create new dataset for this category
                    cat_dataset = dataset.copy()
                    cat_dataset['data_df'] = cat_data
                    
                    # Generate category-specific label
                    original_label = dataset.get('label', f"{x_col.split(' ')[0]} Data")
                    cat_dataset['label'] = f"{original_label} - {category}"
                    
                    # Auto-generate missing fields for category
                    total_idx = len(processed_datasets) + cat_idx
                    if 'color' not in cat_dataset:
                        cat_dataset['color'] = default_colors[total_idx % len(default_colors)]
                    if 'marker' not in cat_dataset:
                        cat_dataset['marker'] = default_markers[total_idx % len(default_markers)]
                    if 'trend_color' not in cat_dataset:
                        cat_dataset['trend_color'] = cat_dataset['color']
                    if 'show_trendline' not in cat_dataset:
                        cat_dataset['show_trendline'] = show_trendlines
                        
                    processed_datasets.append(cat_dataset)
        else:
            # No category splitting - process dataset as normal
            processed_dataset = dataset.copy()
            if 'label' not in processed_dataset:
                processed_dataset['label'] = f"{x_col.split(' ')[0]} Data"
            if 'color' not in processed_dataset:
                processed_dataset['color'] = default_colors[i % len(default_colors)]
            if 'marker' not in processed_dataset:
                processed_dataset['marker'] = default_markers[i % len(default_markers)]
            if 'trend_color' not in processed_dataset:
                processed_dataset['trend_color'] = processed_dataset['color']
            if 'show_trendline' not in processed_dataset:
                processed_dataset['show_trendline'] = show_trendlines
                
            processed_datasets.append(processed_dataset)

    # === Prepare Data ===
    # Prepare data for each dataset and calculate trends
    dataset_data = []
    for dataset in processed_datasets:
        data_df = dataset['data_df']
        x_col = dataset['x_col']
        y_col = dataset['y_col']
        
        # Extract and convert data from this dataset's DataFrame
        x_data = pd.to_numeric(data_df[x_col], errors='coerce').copy()
        y_data = pd.to_numeric(data_df[y_col], errors='coerce').copy()
        
        # Calculate trend if needed
        slope, r2, x_trend, y_trend = None, None, None, None
        if dataset['show_trendline']:
            slope, r2, x_trend, y_trend = _calculate_forced_trend(x_data, y_data)
        
        dataset_data.append({
            'config': dataset,
            'x_data': x_data,
            'y_data': y_data,
            'slope': slope,
            'r2': r2,
            'x_trend': x_trend,
            'y_trend': y_trend
        })

    # === Create Plot ===
    # Apply matplotlib style
    if plot_style:
        try: 
            plt.style.use(plot_style)
        except: 
            print(f"Warning: matplotlib style '{plot_style}' not found.")
            pass
        
    fig, ax = plt.subplots(figsize=figsize)
    all_valid_x, all_valid_y = [], []

    # === Plot All Datasets ===
    for i, data in enumerate(dataset_data):
        config = data['config']
        x_data = data['x_data']
        y_data = data['y_data']
        
        # Plot scatter points
        valid_idx = x_data.notna() & y_data.notna()
        if valid_idx.any():
            x_plot = x_data[valid_idx]
            y_plot = y_data[valid_idx]
            all_valid_x.append(x_plot)
            all_valid_y.append(y_plot)
            
            # Default scatter styling
            default_scatter_style = {'zorder': 5}
            
            # Build scatter parameters
            scatter_params = {
                'x': x_plot, 
                'y': y_plot,
                'label': config['label'],
                'color': config['color'],
                'marker': config['marker'],
                'alpha': scatter_alpha,
                's': scatter_size,
                **default_scatter_style
            }
            
            # Apply custom scatter styling if provided
            if scatter_style is not None:
                scatter_params.update(scatter_style)
            
            ax.scatter(**scatter_params)
            
            # Plot trendline if enabled
            if (data['slope'] is not None and data['x_trend'] is not None and 
                len(data['x_trend']) > 1 and config['show_trendline']):
                
                trend_label = f"Trend ({config['x_col'].split(' ')[0]})" if show_equations else None
                
                # Default trendline styling
                default_trendline_style = {'linestyle': '--', 'zorder': 4}
                
                # Build trendline parameters
                trendline_params = {
                    'color': config['trend_color'],
                    'label': trend_label,
                    **default_trendline_style
                }
                
                # Apply custom trendline styling if provided
                if trendline_style is not None:
                    trendline_params.update(trendline_style)
                
                ax.plot(data['x_trend'], data['y_trend'], **trendline_params)

    # === Axis Customizations ===
    # Default axis styling
    default_axis_style = {
        'xlabel_fontsize': 12,
        'xlabel_fontweight': 'bold',
        'ylabel_fontsize': 12,
        'ylabel_fontweight': 'bold',
        'title_fontsize': 14,
        'title_fontweight': 'bold'
    }
    
    # Apply custom axis styling if provided
    if axis_style is not None:
        axis_params = {**default_axis_style, **axis_style}
    else:
        axis_params = default_axis_style
    
    # Set axis labels with styling
    ax.set_xlabel('Is50 (MPa)', 
                  fontsize=axis_params['xlabel_fontsize'], 
                  fontweight=axis_params['xlabel_fontweight'])
    
    # Use first dataset's y_col for y-axis label, or generic if datasets have different y-columns
    y_cols = [dataset['y_col'] for dataset in processed_datasets]
    if len(set(y_cols)) == 1:
        # All datasets use the same y-column
        ylabel = f'{y_cols[0]}'
    else:
        # Multiple different y-columns, use generic label
        ylabel = 'UCS (MPa)'
    
    ax.set_ylabel(ylabel, 
                  fontsize=axis_params['ylabel_fontsize'], 
                  fontweight=axis_params['ylabel_fontweight'])
    
    # Handle title
    final_title = None
    if title is not None:
        final_title = title
    else:
        final_title = 'UCS vs Is50 Comparison'
        if title_suffix and isinstance(title_suffix, str) and title_suffix.strip():
            final_title += f": {title_suffix.strip()}"
    
    if final_title:
        ax.set_title(final_title, 
                     fontsize=axis_params['title_fontsize'], 
                     fontweight=axis_params['title_fontweight'])
    
    # === Enhanced Axis Limits and Ticks ===
    # Handle axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    elif all_valid_x:
        max_x = pd.concat(all_valid_x).max()
        ax.set_xlim(0, max_x * 1.05)
    else:
        ax.set_xlim(left=0)
        
    if ylim is not None:
        ax.set_ylim(ylim)
    elif all_valid_y:
        max_y = pd.concat(all_valid_y).max()
        ax.set_ylim(0, max_y * 1.05)
    else:
        ax.set_ylim(bottom=0)
    
    # Handle custom ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    elif xtick_interval is not None:
        current_xlim = ax.get_xlim()
        xtick_values = np.arange(0, current_xlim[1] + xtick_interval, xtick_interval)
        ax.set_xticks(xtick_values)
    
    if yticks is not None:
        ax.set_yticks(yticks)
    elif ytick_interval is not None:
        current_ylim = ax.get_ylim()
        ytick_values = np.arange(0, current_ylim[1] + ytick_interval, ytick_interval)
        ax.set_yticks(ytick_values)
    
    # Apply grid styling
    default_grid_style = {'linestyle': '--', 'color': 'grey', 'alpha': 0.35}
    if grid_style is not None:
        grid_params = {**default_grid_style, **grid_style}
    else:
        grid_params = default_grid_style
    ax.grid(True, **grid_params)

    # --- Hide X-axis 0 Tick Label ---
    fig.canvas.draw()
    xticks = ax.get_xticks(); xticklabels_objs = ax.get_xticklabels()
    new_xticklabels = []; tick_found_zero = False
    current_labels = [lbl.get_text() for lbl in xticklabels_objs]
    for i, tick in enumerate(xticks):
        if np.isclose(tick, 0.0): new_xticklabels.append(''); tick_found_zero = True
        elif i < len(current_labels): new_xticklabels.append(current_labels[i])
        else: new_xticklabels.append(f'{tick:.1f}')
    if tick_found_zero: ax.set_xticks(xticks); ax.set_xticklabels(new_xticklabels)

    # === Add Legend ===
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            # Default legend styling
            default_legend_style = {}
            
            # Build legend parameters
            legend_params = {'loc': legend_loc, 'fontsize': legend_fontsize}
            
            # Apply custom legend styling if provided
            if legend_style is not None:
                legend_params.update(legend_style)
            
            ax.legend(**legend_params)

    # === Add Combined Equation Text Box ===
    if show_equations:
        combined_lines = []
        
        for data in dataset_data:
            config = data['config']
            slope = data['slope']
            r2 = data['r2']
            
            if slope is not None and config['show_trendline']:
                # Build equation text using dataset label
                eq_parts = [f"{config['label']}: UCS = {slope:.2f}*Is50"]
                
                if show_r_squared and r2 is not None:
                    eq_parts.append(f"$R^2$={r2:.2f}")
                
                eq_text = ", ".join(eq_parts)
                combined_lines.append(eq_text)

        combined_eq_text = "\n".join(combined_lines)

        if combined_eq_text:  # Only add text box if equations exist
            # Default equation positioning and styling
            default_eq_position = (0.98, 0.02)
            default_eq_style = {
                'fontsize': 12,
                'verticalalignment': 'bottom',
                'horizontalalignment': 'right',
                'bbox': dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
            }
            
            # Use custom position if provided
            eq_position = equation_position if equation_position is not None else default_eq_position
            
            # Build equation text parameters
            eq_text_params = {
                'transform': ax.transAxes,
                **default_eq_style
            }
            
            # Apply custom equation styling if provided
            if equation_style is not None:
                eq_text_params.update(equation_style)
            
            ax.text(eq_position[0], eq_position[1], combined_eq_text, **eq_text_params)

    plt.tight_layout()

    # === Save or Show ===
    if output_filepath:
        try:
            dir_name = os.path.dirname(output_filepath)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"Created directory: {dir_name}")
            plt.savefig(output_filepath, dpi=save_dpi, bbox_inches='tight')
            print(f"Plot successfully saved to: {output_filepath}")
        except Exception as e:
            print(f"Error saving plot to {output_filepath}: {e}")
    
    if show_plot:
        plt.show()
    
    if close_plot:
        plt.close(fig)

def _calculate_forced_trend(x_data, y_data):
    """Calculates slope, R², and trendline points for line forced through zero."""
    if not isinstance(x_data, pd.Series): x_data = pd.Series(x_data)
    if not isinstance(y_data, pd.Series): y_data = pd.Series(y_data)
    valid_idx = x_data.notna() & y_data.notna()
    x = x_data[valid_idx]; y = y_data[valid_idx]
    if len(x) < 2: return None, None, None, None
    sum_xy = np.sum(x * y); sum_x_sq = np.sum(x * x)
    slope = sum_xy / sum_x_sq if sum_x_sq != 0 else 0
    y_pred = slope * x; ss_res = np.sum((y - y_pred) ** 2); ss_tot_uncentered = np.sum(y ** 2)
    r_squared = 1 - (ss_res / ss_tot_uncentered) if ss_tot_uncentered != 0 else 0
    r_squared = max(0, r_squared) if slope != 0 else 0
    if x.empty: x_trend = np.array([0])
    elif x.min() == x.max() and len(x) > 0: x_trend = np.array([x.min(), x.max()])
    elif len(x) > 0: x_trend = np.linspace(x.min(), x.max(), 100)
    else: x_trend = np.array([0])
    y_trend = slope * x_trend
    return slope, r_squared, x_trend, y_trend

