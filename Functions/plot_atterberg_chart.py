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


def plot_atterberg_chart(
        # Essential data parameters (required, most common)
        df: pd.DataFrame,
        ll_col: str = 'LL (%)',
        pi_col: str = 'PI (%)',
        category_col: str = 'Geology_Origin',
        
        # Common customization (frequently used)
        title: Optional[str] = None,
        title_suffix: Optional[str] = None,
        xlim: Optional[Tuple[Optional[float], Optional[float]]] = (0, 100),
        ylim: Optional[Tuple[Optional[float], Optional[float]]] = (0, 80),
        
        # Output control (important for saving)
        output_filepath: Optional[str] = None,
        show_plot: bool = True,
        save_plot: bool = True,
        
        # Visual customization (moderate importance)
        figsize: tuple = (7, 5),
        marker_color: Optional[Union[str, List[str]]] = None,
        marker_shape: Optional[Union[str, List[str]]] = None,
        marker_size: int = 40,
        show_legend: bool = True,
        include_background: bool = True,
        include_zone_labels: bool = True,
        
        # Fine-tuning (advanced users)
        marker_alpha: float = 0.8,
        marker_edge_lw: float = 0.5,
        dpi: int = 300,
        
        # Font sizes (least frequently changed)
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        tick_fontsize: int = 10,
        legend_fontsize: int = 9,
        legend_title_fontsize: int = 10,
        zone_label_fontsize: int = 9,
        
        # Legend positioning (advanced)
        legend_loc: str = 'best',
        legend_bbox_to_anchor: Optional[tuple] = None,
        
        # Plot cleanup control
        close_plot: bool = True,
):
    """
    Create professional Atterberg Limits chart for geotechnical soil classification.
    
    The Atterberg chart plots Plasticity Index (PI) vs Liquid Limit (LL) with standard
    geotechnical reference lines (A-line and U-line) to classify soil types according
    to the Unified Soil Classification System (USCS).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing soil test data with LL, PI, and classification columns.
        
    ll_col : str, default 'LL (%)'
        Column name for Liquid Limit values (typically 0-100%).
        
    pi_col : str, default 'PI (%)'  
        Column name for Plasticity Index values (typically 0-80%).
        
    category_col : str, default 'Geology_Origin'
        Column name for soil categories/classifications to color-code points.
        Examples: 'Geology_Origin', 'Soil_Type', 'Formation'
        
    title : str, optional
        Custom plot title. If None, uses default "Atterberg Limits Chart".
        Example: "Project XYZ - Soil Classification"
        
    title_suffix : str, optional
        Text to append to default title. 
        Example: ": Alluvium Samples" → "Atterberg Limits Chart: Alluvium Samples"
        
    xlim, ylim : tuple of (float, float), default (0,100), (0,80)
        Axis limits as (min, max). Standard ranges cover most soil types.
        Examples: xlim=(0, 120), ylim=(0, 100) for high-plasticity soils.
        
    output_filepath : str, optional
        Full path to save plot including filename and extension.
        Example: 'results/atterberg_chart.png'
        If None, plot is not saved to file.
        
    show_plot : bool, default True
        Whether to display plot on screen. Set False for batch processing.
        
    save_plot : bool, default True
        Whether to save plot to file. Requires output_filepath when True.
        
    figsize : tuple of (float, float), default (8, 6)
        Figure size in inches (width, height). 
        Examples: (12, 9) for presentations, (6, 4.5) for reports.
        
    marker_color : str or list, optional
        Color(s) for data points. Examples:
        - Single color: 'red', 'blue', '#FF5733'
        - Multiple colors: ['blue', 'red', 'green'] (cycles through categories)
        - None: Uses intelligent defaults (orange for alluvium, green for residual, etc.)
        
    marker_shape : str or list, optional
        Shape(s) for data points. Examples:
        - Single shape: 'o' (circles), 's' (squares), '^' (triangles), 'D' (diamonds)
        - Multiple shapes: ['o', 's', '^'] (cycles through categories)
        - None: Uses default variety of shapes
        
    marker_size : int, default 40
        Size of data points. Examples: 60-80 for presentations, 30 for dense data.
        
    show_legend : bool, default True
        Whether to show legend explaining point colors/shapes and reference lines.
        
    include_background : bool, default True
        Whether to show A-line, U-line, and soil classification zones.
        Set False for clean plots with just data points.
        
    include_zone_labels : bool, default True
        Whether to show soil type labels (e.g., "CH or OH", "CL or OL").
        Only applies when include_background=True.
        
    marker_alpha : float, default 0.8
        Point transparency (0.0=invisible, 1.0=solid). 
        Use lower values (0.6) for overlapping points.
        
    marker_edge_lw : float, default 0.5
        Width of point borders in points. Set to 0 for no borders.
        
    dpi : int, default 300
        Image resolution for saved plots. Examples: 150 (draft), 300 (standard), 600 (publication).
        
    title_fontsize : int, default 14
        Font size for main plot title.
        
    label_fontsize : int, default 12  
        Font size for axis labels (x and y axis titles).
        
    tick_fontsize : int, default 10
        Font size for axis tick labels (numbers on axes).
        
    legend_fontsize : int, default 9
        Font size for legend text.
        
    legend_title_fontsize : int, default 10
        Font size for legend title.
        
    zone_label_fontsize : int, default 9
        Font size for soil classification zone labels (e.g., "CH or OH").
        
    legend_loc : str, default 'best'
        Legend position. Options:
        - 'best': Matplotlib automatically chooses optimal position
        - 'upper right', 'upper left', 'lower left', 'lower right'
        - 'center left', 'center right', 'upper center', 'lower center'
        
    legend_bbox_to_anchor : tuple of (float, float), optional
        Fine-tune legend position relative to plot area. Examples:
        - (1.05, 1): Outside plot area, top-right
        - (0.5, -0.15): Below plot, centered  
        - (0, 1): Inside plot, top-left corner
        - None: Uses legend_loc setting only
        
        Coordinate system: (0,0) = bottom-left of plot, (1,1) = top-right of plot.
        Values > 1 place legend outside plot area.
    
    Examples
    --------
    Basic usage with default column names:
    >>> plot_atterberg_chart(df)
    
    Specify custom columns:
    >>> plot_atterberg_chart(df, ll_col='Liquid_Limit', pi_col='Plasticity_Index', 
    ...                      category_col='Soil_Type')
    
    Custom styling for presentation:
    >>> plot_atterberg_chart(df, title="Site ABC Soil Classification", 
    ...                      figsize=(12, 9), marker_size=60,
    ...                      title_fontsize=18, label_fontsize=14)
    
    Save without displaying (batch processing):
    >>> plot_atterberg_chart(df, output_filepath='charts/atterberg.png',
    ...                      show_plot=False)
    
    Custom colors and shapes:
    >>> plot_atterberg_chart(df, marker_color=['red', 'blue', 'green'],
    ...                      marker_shape=['o', 's', '^'])
    
    Position legend outside plot:
    >>> plot_atterberg_chart(df, legend_bbox_to_anchor=(1.05, 1),
    ...                      legend_loc='upper left')
    
    Notes
    -----
    Geotechnical Reference Lines:
    - **A-line**: Separates clays (above line) from silts (below line)
    - **U-line**: Upper limit line for natural soils; points above may indicate unusual soils
    - **Vertical lines**: At LL=50 separate high plasticity (H) from low plasticity (L)
    
    Soil Classification Zones (when include_zone_labels=True):
    - CH/OH: High plasticity clays/organic soils
    - CL/OL: Low plasticity clays/organic soils  
    - MH/OH: High plasticity silts/organic soils
    - ML/OL: Low plasticity silts/organic soils
    - CI/OI: Intermediate plasticity clays/organic soils
    - CL-ML: Clay-silt mixtures
    
    The chart follows ASTM D2487 and USCS classification standards.
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
    required_cols = [ll_col, pi_col, category_col]
    for col in required_cols:
        if col not in df.columns: raise ValueError(f"DataFrame must contain column: '{col}'.")
    if marker_color is not None and not isinstance(marker_color, (str, list)):
        raise TypeError("marker_color must be None, a string, or a list of strings.")
    if marker_shape is not None and not isinstance(marker_shape, (str, list)):
        raise TypeError("marker_shape must be None, a string, or a list of strings.")
    if isinstance(marker_color, list) and not marker_color:
        raise ValueError("If marker_color is a list, it cannot be empty.")
    if isinstance(marker_shape, list) and not marker_shape:
        raise ValueError("If marker_shape is a list, it cannot be empty.")

    # --- Filepath Handling ---
    if save_plot:
        if output_filepath is None: raise ValueError("If 'save_plot' is True, 'output_filepath' must be provided.")
        if not isinstance(output_filepath, str) or not output_filepath: raise ValueError(
            "'output_filepath' must be a non-empty string.")
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.isdir(output_dir):
            try:
                print(f"Output directory '{output_dir}' does not exist. Creating it.")
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"Error creating output directory '{output_dir}'. Plot will not be saved. Error: {e}")
                save_plot = False

    # --- Data Preprocessing ---
    df_plot = df.copy()
    initial_rows = len(df_plot)
    for col in [ll_col, pi_col]: df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    # Also convert category column to string to handle potential numeric categories and ease comparison
    # This line matches the reference script structure
    df_plot[category_col] = df_plot[category_col].astype(str)
    # Drop rows if LL, PI, or Category (now string) is NaN/NaT/None
    # Note: dropna on string column usually only drops None/NaN before conversion,
    # but keeping it here for consistency with reference script's intent.
    # Empty strings '' or the string 'nan' will NOT be dropped here.
    df_plot.dropna(subset=[ll_col, pi_col, category_col], inplace=True)
    rows_after_drop = len(df_plot)
    if rows_after_drop < initial_rows:
        print(
            f"INFO: Removed {initial_rows - rows_after_drop} rows with NaN/non-numeric/missing values in LL, PI, or Category.")
    if df_plot.empty: print(f"Warning: DataFrame empty after cleaning. Skipping plot."); return

    # --- Calculate data ranges for dynamic axis limits ---
    ll_min, ll_max = df_plot[ll_col].min(), df_plot[ll_col].max()
    pi_min, pi_max = df_plot[pi_col].min(), df_plot[pi_col].max()

    # Add some padding to the ranges
    ll_padding = max(5, (ll_max - ll_min) * 0.1) if ll_max > ll_min else 5
    pi_padding = max(3, (pi_max - pi_min) * 0.1) if pi_max > pi_min else 3

    # Extract axis limits from parameters
    x_min, x_max = xlim if xlim else (0, 100)
    y_min, y_max = ylim if ylim else (0, 80)

    print(f"INFO: Data ranges - LL: {ll_min:.1f} to {ll_max:.1f}, PI: {pi_min:.1f} to {pi_max:.1f}")
    print(f"INFO: Plot ranges - X: {x_min} to {x_max:.1f}, Y: {y_min} to {y_max:.1f}")

    # --- Nested Helper Function for Background ---
    def plot_casagrande_background(ax, draw_labels=True):
        ll_a_stop = 25.48;
        ll_u_stop = 16.33
        ll_a_slope = np.linspace(ll_a_stop, min(x_max, 200), 200);
        pi_a_slope = 0.73 * (ll_a_slope - 20)
        ll_u_slope = np.linspace(ll_u_stop, min(x_max, 200), 200);
        pi_u_slope = 0.9 * (ll_u_slope - 8)
        line_lw = 1.0
        
        # Calculate where A-line intersects with PI=7
        # A-line equation: PI = 0.73*(LL-20) for LL > 25.48
        # For PI = 7: 7 = 0.73*(LL-20) => LL = 7/0.73 + 20 ≈ 29.59
        ll_a_intersect_7 = 7 / 0.73 + 20
        
        ax.hlines(y=4, xmin=0, xmax=ll_a_stop, linestyles='solid', colors='black', lw=line_lw)
        ax.plot(ll_a_slope, pi_a_slope, 'k-', label='A-line', lw=line_lw)
        ax.hlines(y=7, xmin=0, xmax=ll_a_intersect_7, linestyles='dashed', colors='dimgray', lw=line_lw)
        ax.plot(ll_u_slope, pi_u_slope, color='dimgray', linestyle='--', label='U-line', lw=line_lw)

        def get_a_line_pi(ll_val):
            return 4 if ll_val <= ll_a_stop else 0.73 * (ll_val - 20)

        def get_u_line_pi(ll_val):
            return 7 if ll_val <= ll_u_stop else 0.9 * (ll_val - 8)

        for x_vert in [35, 50]:
            if x_vert <= x_max:  # Only draw vertical lines if they're within the plot range
                y_bottom = get_a_line_pi(x_vert) if x_vert != 50 else 0
                y_top = min(get_u_line_pi(x_vert), y_max)  # Don't extend lines beyond plot area
                y_bottom = min(y_bottom, y_top)
                ax.vlines(x=x_vert, ymin=y_bottom, ymax=y_top, linestyles='solid', colors='black', lw=line_lw)
        if draw_labels:
            labels_positions = [(67, 17.5, "MH or OH"), (57.5, 37, "CH o OH"), (42, 23, "CI or OI"),
                                (28, 12, "CL or OL"), (20, 5.5, "CL-ML"), (37, 3, "ML or OL")]
            for (xx, yy, txt) in labels_positions:
                # Only draw labels that are within the plot boundaries
                if xx <= x_max and yy <= y_max:
                    ax.text(xx, yy, txt, ha='center', va='center', fontsize=zone_label_fontsize,
                            bbox=dict(facecolor='white', alpha=0.75, edgecolor='lightgray', lw=0.5, pad=1.5))

        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Liquid Limit (LL) %', fontsize=label_fontsize)
        ax.set_ylabel('Plasticity Index (PI) %', fontsize=label_fontsize)

        # Set ticks dynamically based on the axis ranges
        x_major_step = 10 if x_max <= 150 else 20
        x_minor_step = 5 if x_max <= 150 else 10
        y_major_step = 10 if y_max <= 100 else 20
        y_minor_step = 5 if y_max <= 100 else 10

        ax.set_xticks(np.arange(0, x_max + x_major_step, x_major_step))
        ax.set_xticks(np.arange(0, x_max + x_minor_step, x_minor_step), minor=True)
        ax.set_yticks(np.arange(0, y_max + y_major_step, y_major_step))
        ax.set_yticks(np.arange(0, y_max + y_minor_step, y_minor_step), minor=True)

        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True, which='major', linestyle='--', color='lightgrey', alpha=0.6, lw=0.5)
        ax.grid(True, which='minor', linestyle=':', color='lightgrey', alpha=0.4, lw=0.5)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    if include_background:
        plot_casagrande_background(ax, draw_labels=include_zone_labels)
    else:
        ax.set_xlabel(f'{ll_col} (%)', fontsize=label_fontsize)
        ax.set_ylabel(f'{pi_col} (%)', fontsize=label_fontsize)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        # Apply axis limits for non-background plots too
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # --- Set up Color and Shape Cycling/Assignment ---
    unique_categories = sorted(df_plot[category_col].unique())

    # Define base default styles
    internal_default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                               '#bcbd22', '#17becf']
    internal_default_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    # Define specific default styles for certain categories (case-insensitive keys)
    specific_style_defaults = {
        'alluvial': {'color': 'darkorange', 'shape': 'o'},  # Covers ALLUVIAL, Alluvial, alluvial
        'alluvium': {'color': 'darkorange', 'shape': 'o'},  # Covers ALLUVIUM, Alluvium, alluvium
        'residual': {'color': 'green', 'shape': '^'},
        'fill': {'color': 'lightblue', 'shape': 's'}
    }

    # Determine the source for colors and shapes based on user input
    # If user provides a color/shape, that overrides everything
    use_specific_defaults = marker_color is None and marker_shape is None

    if isinstance(marker_color, str):
        color_source = itertools.repeat(marker_color)
    elif isinstance(marker_color, list):
        color_source = itertools.cycle(marker_color)
    else:
        color_source = itertools.cycle(internal_default_colors)  # Base default cycle

    if isinstance(marker_shape, str):
        shape_source = itertools.repeat(marker_shape)
    elif isinstance(marker_shape, list):
        shape_source = itertools.cycle(marker_shape)
    else:
        shape_source = itertools.cycle(internal_default_markers)  # Base default cycle

    # Build the style map
    style_map = {}
    assigned_specific_default_count = 0
    for cat in unique_categories:
        # Skip assigning style if category is effectively empty after conversion (e.g., 'nan', '')
        # This prevents plotting points with missing categories
        if not cat or cat.lower() == 'nan':
            print(f"INFO: Skipping category '{cat}' as it represents missing data.")
            continue

        cat_lower = cat.lower()  # For case-insensitive matching
        specific_style = None
        if use_specific_defaults:
            specific_style = specific_style_defaults.get(cat_lower)

        if specific_style:
            # Use the specific default style
            style_map[cat] = specific_style
            assigned_specific_default_count += 1
        else:
            # Assign next available color/shape from the general sources
            style_map[cat] = {'color': next(color_source), 'shape': next(shape_source)}

    if use_specific_defaults and assigned_specific_default_count > 0:
        print("INFO: Applied specific default styles for ALLUVIAL/ALLUVIUM, RESIDUAL, FILL where applicable.")

    # --- Scatter Plotting Loop ---
    skipped_categories = []
    for category, group_df in df_plot.groupby(category_col):
        # Get the style; only plot if a style was assigned (i.e., not skipped above)
        style = style_map.get(category)
        if style:
            ax.scatter(group_df[ll_col], group_df[pi_col],
                       marker=style['shape'], s=marker_size, label=category,
                       color=style['color'], alpha=marker_alpha,
                       edgecolors='black', lw=marker_edge_lw)
        elif category not in style_map and (not category or category.lower() == 'nan'):
            # Track categories that were skipped because they were missing/nan
            if category not in skipped_categories:
                skipped_categories.append(category)

    if skipped_categories:
        print(f"INFO: Did not plot points for categories representing missing data: {skipped_categories}")

    # --- Final Touches ---
    # Construct Title with Custom or Default Options
    final_title = None
    if title is not None:
        final_title = title
    else:
        # Use default title
        final_title = 'Atterberg Limits Chart'
        if title_suffix and isinstance(title_suffix, str) and title_suffix.strip():
            final_title += f": {title_suffix.strip()}"

    if final_title:
        ax.set_title(final_title, fontsize=title_fontsize, fontweight='bold', pad=15)

    # --- Legend ---
    handles, labels = ax.get_legend_handles_labels()
    
    if show_legend:
        # Separate line handles (A-line, U-line) from scatter handles
        line_handles = [h for h, l in zip(handles, labels) if l in ('A-line', 'U-line')]
        line_labels = [l for l in labels if l in ('A-line', 'U-line')]
        
        scatter_handles = [h for h, l in zip(handles, labels) if l not in ('A-line', 'U-line')]
        scatter_labels = [l for l in labels if l not in ('A-line', 'U-line')]
        
        # Create legend with both lines and scatter points
        legend_handles = []
        legend_labels = []
        
        # Add line handles first (A-line, U-line)
        if include_background and line_handles:
            legend_handles.extend(line_handles)
            legend_labels.extend(line_labels)
            
        # Add scatter handles for categories that were actually plotted
        if scatter_handles:
            unique_handles_labels = {l: h for l, h in zip(scatter_labels, scatter_handles) if l in style_map}
            if unique_handles_labels:
                legend_handles.extend(unique_handles_labels.values())
                legend_labels.extend(unique_handles_labels.keys())
        
        # Create the combined legend
        if legend_handles:
            ax.legend(handles=legend_handles, labels=legend_labels,
                      title=category_col.replace('_', ' ').title(),
                      fontsize=legend_fontsize, title_fontsize=legend_title_fontsize,
                      loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor,
                      frameon=True, facecolor='white', framealpha=0.75, edgecolor='darkgrey',
                      markerscale=0.9)

    plt.tight_layout()

    # --- Saving ---
    if save_plot:
        try:
            plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved successfully to: {output_filepath}")
        except Exception as e:
            print(f"Error saving plot to '{output_filepath}': {e}")

    # --- Showing ---
    if show_plot: plt.show()

    # --- Cleanup ---
    if close_plot:
        plt.close(fig)
    print(f"--- Plotting finished for Atterberg Limits ({category_col}) ---\n")