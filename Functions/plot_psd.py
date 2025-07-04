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

def plot_psd(
    # === Tier 1: Essential Data Parameters ===
    df: pd.DataFrame,
    hole_id_col: str = 'Hole_ID',
    depth_col: str = 'From_mbgl',
    size_col: str = 'Sieve_Size_mm',
    percent_col: str = 'Percentage passing (%)',
    max_plots: int = None,
    
    # === Tier 2: Plot Appearance ===
    title: Optional[str] = None,
    title_suffix: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 9),
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    
    # === Tier 3: Category & Color Options ===
    color_by: Optional[str] = None,
    palette: Optional[dict] = None,
    show_color_mappings: bool = False,
    
    # === Tier 4: Axis Configuration ===
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    xlabel_fontweight: str = 'bold',
    ylabel_fontweight: str = 'bold',
    tick_fontsize: int = 14,
    tick_pad: int = 12,
    rotation: float = 0,
    
    # === Tier 5: Display Options ===
    show_plot: bool = True,
    show_legend: bool = True,
    show_grid: bool = True,
    grid_axis: str = 'both',  # 'x', 'y', or 'both'
    grid_style: str = '--',
    
    # === Tier 6: Output Controls ===
    output_filepath: Optional[str] = None,
    save_dpi: int = 300,
    save_bbox_inches: str = "tight",
    close_plot: bool = True,
    
    # === Tier 7: Visual Customization ===
    line_width: float = 1.5,
    marker_style: str = 'o',
    marker_size: float = 4,
    show_markers: bool = True,
    alpha: float = 1.0,
    smooth_curves: bool = False,
    
    # === Tier 8: Advanced Formatting ===
    formatting_options: Optional[dict] = None,
    
    # === Deprecated Parameters (backward compatibility) ===
    xmin: Optional[float] = None,  # DEPRECATED - use xlim instead
    xmax: Optional[float] = None,  # DEPRECATED - use xlim instead
):
    """
    Plots Particle Size Distribution (PSD) curves from a DataFrame with comprehensive style control.
    
    Follows 8-tier parameter organization for clarity and consistency:
    
    TIER 1 - ESSENTIAL DATA PARAMETERS:
        df (pd.DataFrame): DataFrame containing the PSD data.
                          Must include a numeric column with particle sizes in mm.
        hole_id_col (str): Column name for the sample/hole identifier. Default: 'Hole_ID'.
        depth_col (str): Column name for the sample depth. Default: 'From_mbgl'.
        size_col (str): Column name for particle size in mm (numeric). Default: 'Sieve_Size_mm'.
        percent_col (str): Column name for percentage passing (numeric). Default: 'Percentage passing (%)'.
        max_plots (int, optional): Maximum number of curves to plot. Default: None (plot all).
    
    TIER 2 - PLOT APPEARANCE:
        title (str, optional): Custom title for the plot. Default: None (auto-generated).
        title_suffix (str, optional): String to append to the default plot title. Default: None.
        figsize (tuple): Figure size (width, height) in inches. Default: (14, 9).
        xlim (tuple, optional): Tuple of (min, max) for x-axis limits. Default: None (0.001, 1000).
        ylim (tuple, optional): Tuple of (min, max) for y-axis limits. Default: None (0, 100).
        xlabel (str, optional): Custom x-axis label. Default: None ('Particle Size (mm)').
        ylabel (str, optional): Custom y-axis label. Default: None ('Percent Passing (%)').
    
    TIER 3 - CATEGORY & COLOR OPTIONS:
        color_by (str, optional): Column name to use for coloring lines. Default: None.
        palette (dict, optional): Dictionary mapping color_by values to colors. Default: None.
        show_color_mappings (bool): Display available geological color mappings. Default: False.
    
    TIER 4 - AXIS CONFIGURATION:
        xlabel_fontsize (int): Font size for x-axis label. Default: 14.
        ylabel_fontsize (int): Font size for y-axis label. Default: 14.
        xlabel_fontweight (str): Font weight for x-axis label. Default: 'bold'.
        ylabel_fontweight (str): Font weight for y-axis label. Default: 'bold'.
        tick_fontsize (int): Font size for tick labels. Default: 14.
        tick_pad (int): Padding for tick labels. Default: 12.
        rotation (float): Rotation angle for x-axis labels. Default: 0.
    
    TIER 5 - DISPLAY OPTIONS:
        show_plot (bool): Whether to display the plot. Default: True.
        show_legend (bool): Whether to show the legend. Default: True.
        show_grid (bool): Whether to show grid lines. Default: True.
        grid_axis (str): Which axes to show grid ('x', 'y', 'both'). Default: 'both'.
        grid_style (str): Grid line style. Default: '--'.
    
    TIER 6 - OUTPUT CONTROLS:
        output_filepath (str, optional): File path to save the figure. Default: None.
        save_dpi (int): DPI for saved figure. Default: 300.
        save_bbox_inches (str): Bbox parameter for saving. Default: "tight".
        close_plot (bool): Whether to close plot after creation. Default: True.
    
    TIER 7 - VISUAL CUSTOMIZATION:
        line_width (float): Width of plotted lines. Default: 1.5.
        marker_style (str): Style of markers ('o', '.', 'x', etc.). Default: 'o'.
        marker_size (float): Size of markers. Default: 4.
        show_markers (bool): Whether to show data point markers. Default: True.
        alpha (float): Line transparency (0-1). Default: 1.0.
        smooth_curves (bool): Whether to use spline interpolation. Default: False.
    
    TIER 8 - ADVANCED FORMATTING:
        formatting_options (dict, optional): Dictionary to override advanced formatting settings.
                                           See function code for available options. Default: None.
    
    DEPRECATED PARAMETERS:
        xmin (float): DEPRECATED - use xlim instead.
        xmax (float): DEPRECATED - use xlim instead.
    
    Enhanced Features:
    - Intelligent geological category recognition with flexible pattern matching
    - Automatic color assignment for common geological units (ALLUVIAL, RESIDUAL, FILL, etc.)
    - Special legend ordering for consistency data (VS → VD progression)
    - Comprehensive style control through direct parameters
    - Advanced formatting through formatting_options dictionary
    
    Example:
        # Basic usage
        plot_psd(df, xlim=(0.001, 100), show_markers=True)
        
        # Advanced styling
        plot_psd(df, 
                color_by='Geology', 
                title='Project PSD Analysis',
                xlabel_fontsize=16,
                line_width=2.0,
                alpha=0.8)
    """
    
    # --- Handle deprecated parameters ---
    if xmin is not None or xmax is not None:
        warnings.warn(
            "Parameters 'xmin' and 'xmax' are deprecated and will be removed in a future version. "
            "Please use 'xlim=(min, max)' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if xlim is None and xmin is not None and xmax is not None:
            xlim = (xmin, xmax)
        elif xlim is None:
            # Use defaults if only one is provided
            xlim = (xmin if xmin is not None else 0.001, 
                   xmax if xmax is not None else 1000)
    
    # Set default xlim/ylim if not provided
    if xlim is None:
        xlim = (0.001, 1000)
    if ylim is None:
        ylim = (0, 100)
    
    # --- Default Formatting Settings ---
    default_formatting = {
        # --- Smoothing & Advanced Markers ---
        "smooth_points": 100,             # Number of points if interpolated curve is used
        "smooth_spline_k": 3,             # Spline degree if smoothing is used
        
        # --- Main Plot Area (ax1) ---
        "ax1_pos": [0.1, 0.28, 0.85, 0.63],
        "axis_label_pad": 20,
        "title_fontsize": 16,
        "title_fontweight": 'bold',
        "title_pad": 20,
        "major_grid_linestyle": '--',
        "major_grid_linewidth": 0.8,
        "major_grid_color": 'gray',
        "minor_grid_linestyle": ':',
        "minor_grid_linewidth": 0.4,
        "minor_grid_color": 'lightgray',
        "boundary_line_color": '#555555',
        "boundary_line_style": '-.',
        "boundary_line_width": 1.0,
        "spine_linewidth": 1.5,

        # --- Legend (ax1) ---
        "legend_fontsize": 10,
        "legend_title_fontsize": 11,
        "legend_bbox_to_anchor": (1.02, 1),
        "legend_loc": 'upper left',

        # --- Classification Bar (ax2) ---
        "ax2_pos": [0.1, 0.06, 0.85, 0.108],
        "ax2_border_color": 'black',
        "ax2_border_linewidth": 1.5,
        "ax2_main_sep_color": 'black',
        "ax2_main_sep_style": '-',
        "ax2_main_sep_linewidth": 1.5,
        "ax2_sub_sep_color": 'black',
        "ax2_sub_sep_style": '--',
        "ax2_sub_sep_linewidth": 1,
        "ax2_horz_div_color": 'black',
        "ax2_horz_div_style": '--',
        "ax2_horz_div_linewidth": 1,
        "ax2_main_label_fontsize": 10,
        "ax2_main_label_fontweight": 'bold',
        "ax2_sub_label_fontsize": 10,
        "ax2_num_label_fontsize": 10,
    }

    # Merge user options with defaults
    fmt = default_formatting.copy()
    if formatting_options:
        if isinstance(formatting_options, dict):
             fmt.update(formatting_options)
        else:
             print("Warning: 'formatting_options' provided but is not a dictionary. Using default formatting.")


    # --- Default Color Settings ---
    base_default_colors = plt.cm.tab10.colors
    
    # Enhanced flexible category mapping for geological units
    def normalize_geological_category(category):
        """
        Normalize geological category names to standard forms for consistent color mapping.
        Handles case variations, abbreviations, and common geological terminology.
        
        Enhanced to support both exact matches and substring patterns with comprehensive
        geological category recognition including common typos and variations.
        """
        if pd.isna(category) or str(category).strip() == "":
            return None
            
        original_cat = str(category).strip()
        cat_str = original_cat.upper()
        
        # ALLUVIAL/ALLUVIUM variations - comprehensive pattern matching
        alluvial_exact = ['ALLUVIAL', 'ALLUVIUM', 'QA', 'QUAT']
        alluvial_patterns = [
            'ALLUVIUM SOILS', 'ALLUVIAL DEPOSITS', 'ALLUVIAL SOILS', 
            'QUATERNARY ALLUVIUM', 'QUATERNARY ALLUVIAL', 'ALLUVIAL SEDIMENT'
        ]
        
        # Check exact matches first (faster)
        if cat_str in alluvial_exact:
            return 'ALLUVIAL'
        # Then check substring patterns
        if any(pattern in cat_str for pattern in alluvial_patterns):
            return 'ALLUVIAL'
            
        # RESIDUAL/WEATHERED variations - comprehensive pattern matching
        residual_exact = ['RESIDUAL', 'RS', 'XW', 'RS_XW', 'RS-XW']
        residual_patterns = [
            'RS TO XW', 'EXTREMELY WEATHERED', 'EXTREME WEATHERED', 'EXTREMED WEATHERED',
            'EXTREMELY WEATHERED ROCK', 'WEATHERED ROCK', 'RESIDUAL SOIL', 
            'RESIDUAL SOILS', 'WEATHERED', 'WEATHERED MATERIAL'
        ]
        
        # Check exact matches first
        if cat_str in residual_exact:
            return 'RESIDUAL'
        # Handle specific user-mentioned variations
        if 'RS_XW' in cat_str or 'RS-XW' in cat_str:
            return 'RESIDUAL'
        # Check substring patterns
        if any(pattern in cat_str for pattern in residual_patterns):
            return 'RESIDUAL'
            
        # FILL variations - comprehensive pattern matching
        fill_exact = ['FILL']
        fill_patterns = ['FILLING', 'ENGINEERED FILL', 'CONTROLLED FILL', 'STRUCTURAL FILL']
        
        if cat_str in fill_exact:
            return 'FILL'
        if any(pattern in cat_str for pattern in fill_patterns):
            return 'FILL'
            
        # Standard geological unit abbreviations (exact matches only)
        geological_units = {
            'DCF': 'DCF',
            'RIN': 'RIN', 
            'RJBW': 'RJBW',
            'TOS': 'TOS'
        }
        
        if cat_str in geological_units:
            return geological_units[cat_str]
            
        # Return original if no match found
        return original_cat
    
    # Define colors for normalized categories
    specific_defaults = {
        'ALLUVIAL': 'darkorange',
        'RESIDUAL': 'green', 
        'FILL': 'lightblue',
        'DCF': 'brown',
        'RIN': 'purple',
        'RJBW': 'red',
        'TOS': 'blue'
    }

    # --- Display Available Color Mappings (if requested) ---
    if show_color_mappings:
        print("\n" + "="*60)
        print("AVAILABLE GEOLOGICAL COLOR MAPPINGS")
        print("="*60)
        print("The following patterns are recognized and assigned specific colors:")
        print(f"\n🟠 ALLUVIAL/ALLUVIUM → {specific_defaults['ALLUVIAL']}")
        print("   Exact matches: 'ALLUVIAL', 'Alluvial', 'ALLUVIUM', 'Alluvium', 'QA', 'QUAT'")
        print("   Patterns: 'Alluvium soils', 'Quaternary alluvium', 'Alluvial deposits'")
        
        print(f"\n🟢 RESIDUAL/WEATHERED → {specific_defaults['RESIDUAL']}")
        print("   Exact matches: 'RESIDUAL', 'Residual', 'RS', 'XW', 'RS_XW', 'RS-XW'")
        print("   Patterns: 'Extremely weathered', 'Extremed weathered', 'Weathered rock',")
        print("             'Residual soil', 'RS TO XW', 'Weathered material'")
        
        print(f"\n🔵 FILL → {specific_defaults['FILL']}")
        print("   Exact matches: 'FILL', 'Fill'")
        print("   Patterns: 'Filling', 'Engineered fill', 'Controlled fill'")
        
        print("\n📋 OTHER GEOLOGICAL UNITS:")
        for unit, color in specific_defaults.items():
            if unit not in ['ALLUVIAL', 'RESIDUAL', 'FILL']:
                print(f"   {unit} → {color}")
        
        print("\n💡 TIPS:")
        print("   • Pattern matching is case-insensitive")
        print("   • Both exact matches and substring patterns are checked")
        print("   • User-provided palette overrides these defaults")
        print("   • Unrecognized categories get assigned from default color palette")
        print("="*60 + "\n")

    # --- Data Handling and Validation ---
    data = df.copy()
    data.columns = data.columns.str.strip()

    required_cols = [hole_id_col, depth_col, size_col, percent_col]
    if color_by and color_by not in data.columns:
        print(f"Warning: color_by column '{color_by}' not found in DataFrame. Ignoring coloring.")
        color_by = None
        palette = None
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not pd.api.types.is_numeric_dtype(data[size_col]):
         raise TypeError(f"The size column '{size_col}' must contain numeric values (in mm). Please convert units beforehand.")
    if not pd.api.types.is_numeric_dtype(data[percent_col]):
         data[percent_col] = pd.to_numeric(data[percent_col], errors='coerce')
         initial_rows = len(data)
         data.dropna(subset=[percent_col], inplace=True)
         if len(data) < initial_rows:
             print(f"Warning: {initial_rows - len(data)} rows dropped due to non-numeric '{percent_col}' values.")

    if color_by and palette and not isinstance(palette, dict):
         raise TypeError("If provided, 'palette' must be a dictionary.")

    # --- Identify Groups (Samples) ---
    try:
        all_groups = list(data.groupby([hole_id_col, depth_col]).groups.keys())
    except KeyError:
        raise KeyError(f"Could not group data. Check if '{hole_id_col}' and '{depth_col}' exist in the DataFrame.")

    if not all_groups:
        print("Warning: No data groups found after cleaning. No plot generated.")
        return

    groups = all_groups[:max_plots] if max_plots is not None else all_groups
    if not groups: return

    # --- Subset Data ---
    try:
        index_cols = [hole_id_col, depth_col]
        temp_data = data.set_index(index_cols)
        groups_index = pd.MultiIndex.from_tuples(groups, names=index_cols)
        subset = temp_data[temp_data.index.isin(groups_index)].reset_index()
    except KeyError as e:
         print(f"Warning: Could not create subset. Check column names '{hole_id_col}', '{depth_col}'. Error: {e}. No plot generated.")
         return
    except Exception as e:
        print(f"An unexpected error occurred during subsetting: {e}. No plot generated.")
        return

    if subset.empty:
        print("Warning: No valid data remaining after subsetting. Cannot plot.")
        return

    # --- Setup Palette if using color_by ---
    current_palette = None
    if color_by:
        if color_by not in subset.columns:
             print(f"Error: color_by column '{color_by}' not found in the subset. Cannot apply colors.")
             color_by = None
        else:
            unique_categories = subset[color_by].dropna().unique()
            
            if palette is None:
                # Create mapping using normalized categories  
                generated_palette = {}
                category_mapping = {}  # Track original -> normalized mappings
                
                # First, normalize all categories and create reverse mapping
                for cat in unique_categories:
                    normalized = normalize_geological_category(cat)
                    category_mapping[cat] = normalized
                
                # Assign colors based on normalized categories
                color_idx = 0
                used_colors = set(specific_defaults.values())
                
                for cat in unique_categories:
                    normalized_cat = category_mapping[cat]
                    
                    # Check if normalized category has a specific default color
                    if normalized_cat in specific_defaults:
                        generated_palette[cat] = specific_defaults[normalized_cat]
                    else:
                        # Assign from base color palette
                        assigned_color = base_default_colors[color_idx % len(base_default_colors)]
                        while assigned_color in used_colors and color_idx < len(base_default_colors)*2:
                             color_idx += 1
                             assigned_color = base_default_colors[color_idx % len(base_default_colors)]
                        generated_palette[cat] = assigned_color
                        used_colors.add(assigned_color)
                        color_idx += 1
                        
                current_palette = generated_palette
            else:
                current_palette = palette
                missing_cats = [cat for cat in unique_categories if cat not in current_palette]
                if missing_cats:
                    print(f"Warning: User-provided palette missing colors for categories: {missing_cats}. These will use default line colors.")


    # --- Classification Boundaries ---
    main_bounds = {
        "CLAY": (0.001, 0.002), "SILT": (0.002, 0.060),
        "SAND": (0.060, 2.0),   "GRAVEL": (2.0, 60.0),
        "COBBLE": (60.0, 200.0), "BOULDERS": (200.0, 1000.0)
    }
    sub_bounds = {
        "SILT": {"Fine": (0.002, 0.006), "Medium": (0.006, 0.020), "Coarse": (0.020, 0.060)},
        "SAND": {"Fine": (0.060, 0.200), "Medium": (0.200, 0.600), "Coarse": (0.600, 2.0)},
        "GRAVEL": {"Fine": (2.0, 6.0), "Medium": (6.0, 20.0), "Coarse": (20.0, 60.0)}
    }
    boulder_upper = min(main_bounds["BOULDERS"][1], xlim[1])
    boundaries = sorted([b[0] for b in main_bounds.values()][1:] + [boulder_upper])
    boundaries = [b for b in boundaries if xlim[0] < b < xlim[1]]
    major_ticks = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    major_ticks = [t for t in major_ticks if xlim[0] <= t <= xlim[1]]

    # --- Plotting ---
    fig = plt.figure(figsize=figsize)

    # --- Main PSD Plot Area (ax1) ---
    ax1 = fig.add_axes(fmt['ax1_pos'])

    lines_for_legend = {}
    skipped_groups = []

    for group_key, grp in subset.groupby([hole_id_col, depth_col], sort=False):
        if grp.empty: continue

        # Ensure enough points for spline interpolation if used
        grp = grp.sort_values(size_col)
        grp = grp.drop_duplicates(subset=[size_col], keep='first')
        min_points_needed = fmt['smooth_spline_k'] + 1
        can_smooth = len(grp) >= min_points_needed

        label = f"{group_key[0]}:{group_key[1]}"
        color = None
        category_label = None
        cat_value = None

        if color_by:
            if color_by not in grp.columns:
                 skipped_groups.append(group_key); continue
            cat_value = grp[color_by].iloc[0]
            if pd.isna(cat_value) or str(cat_value).strip() == "":
                skipped_groups.append(group_key); continue
            category_label = str(cat_value)
            if current_palette:
                color = current_palette.get(cat_value, None)

        # --- Plotting Logic ---
        x_orig = grp[size_col]
        y_orig = grp[percent_col]

        temp_line, = ax1.plot([],[], color=color)
        line_color = temp_line.get_color()
        temp_line.remove()

        if smooth_curves and can_smooth:
            try:
                x_smooth = np.logspace(np.log10(x_orig.min()), np.log10(x_orig.max()), fmt['smooth_points'])
                spl = make_interp_spline(x_orig, y_orig, k=fmt['smooth_spline_k'])
                y_smooth = spl(x_smooth)
                y_smooth = np.clip(y_smooth, 0, 100)
                line, = ax1.plot(x_smooth, y_smooth, label=label, color=line_color,
                                 linewidth=line_width, alpha=alpha)
            except Exception as e:
                 print(f"Warning: Could not create smooth curve for {label}. Plotting straight lines. Error: {e}")
                 line, = ax1.plot(x_orig, y_orig, label=label, color=line_color,
                                  linewidth=line_width, alpha=alpha)
        else:
            # Plot with straight lines
            line, = ax1.plot(x_orig, y_orig, label=label, color=line_color,
                             linewidth=line_width,
                             marker=marker_style if show_markers else None,
                             markersize=marker_size if show_markers else None,
                             alpha=alpha,
                             linestyle='-'
                            )

        if color_by and category_label is not None and category_label not in lines_for_legend:
             if line_color is not None:
                 lines_for_legend[category_label] = line

    if skipped_groups:
        skipped_labels = [f"{key[0]}:{key[1]}" for key in skipped_groups]
        print(f"Info: Skipped plotting {len(skipped_groups)} group(s) due to missing/empty values in '{color_by}' column or insufficient points for smoothing: {', '.join(skipped_labels)}")

    # --- Formatting ax1 ---
    ax1.set_xscale('log')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xticks(major_ticks)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{int(y)}'))
    
    # Set axis labels with custom options
    ax1.set_xlabel(xlabel if xlabel else 'Particle Size (mm)', 
                   fontsize=xlabel_fontsize, 
                   fontweight=xlabel_fontweight, 
                   labelpad=fmt['axis_label_pad'])
    ax1.set_ylabel(ylabel if ylabel else 'Percent Passing (%)', 
                   fontsize=ylabel_fontsize, 
                   fontweight=ylabel_fontweight, 
                   labelpad=fmt['axis_label_pad'])

    # Construct Title
    if title is not None:
        plot_title = title
    else:
        plot_title = 'Particle Size Distribution Curve'
        if title_suffix and isinstance(title_suffix, str) and title_suffix.strip():
            plot_title += f": {title_suffix.strip()}"
    ax1.set_title(plot_title, fontsize=fmt['title_fontsize'], fontweight=fmt['title_fontweight'], pad=fmt['title_pad'])

    # Tick parameters
    ax1.tick_params(axis='x', labelsize=tick_fontsize, pad=tick_pad, rotation=rotation)
    ax1.tick_params(axis='y', labelsize=tick_fontsize, pad=tick_pad)
    
    # Grid settings
    if show_grid:
        if grid_axis in ['x', 'both']:
            ax1.grid(which='major', axis='x', linestyle=grid_style if grid_style else fmt['major_grid_linestyle'], 
                    linewidth=fmt['major_grid_linewidth'], color=fmt['major_grid_color'])
            ax1.grid(which='minor', axis='x', linestyle=fmt['minor_grid_linestyle'], 
                    linewidth=fmt['minor_grid_linewidth'], color=fmt['minor_grid_color'])
        if grid_axis in ['y', 'both']:
            ax1.grid(which='major', axis='y', linestyle=grid_style if grid_style else fmt['major_grid_linestyle'], 
                    linewidth=fmt['major_grid_linewidth'], color=fmt['major_grid_color'])
            ax1.grid(which='minor', axis='y', linestyle=fmt['minor_grid_linestyle'], 
                    linewidth=fmt['minor_grid_linewidth'], color=fmt['minor_grid_color'])
    
    ax1.minorticks_on()
    
    # Boundary lines
    for b in boundaries:
        ax1.axvline(b, color=fmt['boundary_line_color'], linestyle=fmt['boundary_line_style'], linewidth=fmt['boundary_line_width'])
    
    # Spine formatting
    for spine in ax1.spines.values():
        spine.set_linewidth(fmt['spine_linewidth'])

    # --- Legend Handling ---
    if show_legend:
        if color_by and lines_for_legend:
            # Check if we're dealing with consistency data and apply geological ordering
            consistency_columns = ['consistency', 'consist', 'cons']  # Common consistency column variations
            is_consistency = any(col.lower() in color_by.lower() for col in consistency_columns)
            
            if is_consistency:
                # Geological consistency order from Very Soft to Very Dense
                consistency_order = ['VS', 'S', 'F', 'St', 'VSt', 'H', 'VL', 'L', 'MD', 'D', 'VD']
                
                # Create ordered legend based on geological consistency progression
                ordered_lines = {}
                
                # Build a mapping of available labels (case-sensitive exact matching)
                available_labels = {}
                for label in lines_for_legend.keys():
                    label_str = str(label).strip()
                    # Direct case-sensitive matching
                    if label_str in consistency_order:
                        available_labels[label_str] = label
                
                # Add items in geological order if they exist in the data
                for consistency in consistency_order:
                    if consistency in available_labels:
                        original_label = available_labels[consistency]
                        ordered_lines[original_label] = lines_for_legend[original_label]
                
                # Add any remaining items that weren't in the predefined order (fallback)
                for label, line in lines_for_legend.items():
                    if label not in ordered_lines:
                        ordered_lines[label] = line
                        
                # Apply the ordered legend
                ax1.legend(ordered_lines.values(), ordered_lines.keys(),
                          title=color_by,
                          fontsize=fmt['legend_fontsize'], title_fontsize=fmt['legend_title_fontsize'],
                          bbox_to_anchor=fmt['legend_bbox_to_anchor'], loc=fmt['legend_loc'])
            else:
                # Default behavior for non-consistency columns
                ax1.legend(lines_for_legend.values(), lines_for_legend.keys(),
                          title=color_by,
                          fontsize=fmt['legend_fontsize'], title_fontsize=fmt['legend_title_fontsize'],
                          bbox_to_anchor=fmt['legend_bbox_to_anchor'], loc=fmt['legend_loc'])
        elif not color_by and len(ax1.get_lines()) > 0:
             handles, labels = ax1.get_legend_handles_labels()
             ax1.legend(handles=handles, labels=labels,
                        title=f"{hole_id_col} : {depth_col}",
                        fontsize=fmt['legend_fontsize'], title_fontsize=fmt['legend_title_fontsize'],
                        bbox_to_anchor=fmt['legend_bbox_to_anchor'], loc=fmt['legend_loc'])


    # --- Classification Bar Area (ax2) ---
    ax2 = fig.add_axes(fmt['ax2_pos'])
    ax2.set_xscale('log')
    ax2.set_xlim(xlim)
    ax2.set_ylim(-0.1, 1)
    ax2.axis('off')
    ax2.plot([xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]], [0, 0, 1, 1, 0],
             color=fmt['ax2_border_color'], linewidth=fmt['ax2_border_linewidth'], clip_on=False)
    for b in boundaries:
        ax2.vlines(b, ymin=0, ymax=1, color=fmt['ax2_main_sep_color'], linestyle=fmt['ax2_main_sep_style'], linewidth=fmt['ax2_main_sep_linewidth'],
                   transform=ax2.get_xaxis_transform())
    for cat, subs in sub_bounds.items():
        sub_boundaries = sorted([high for _, high in subs.values()])[:-1]
        for high in sub_boundaries:
             if xlim[0] < high < xlim[1]:
                ax2.vlines(high, ymin=0, ymax=0.5, color=fmt['ax2_sub_sep_color'], linestyle=fmt['ax2_sub_sep_style'],
                           linewidth=fmt['ax2_sub_sep_linewidth'], transform=ax2.get_xaxis_transform())
    for cat in ['SILT', 'SAND', 'GRAVEL']:
        if cat in main_bounds:
            low, high = main_bounds[cat]
            draw_low = max(low, xlim[0])
            draw_high = min(high, xlim[1])
            if draw_low < draw_high:
                 ax2.hlines(0.5, draw_low, draw_high, color=fmt['ax2_horz_div_color'], linestyle=fmt['ax2_horz_div_style'],
                            linewidth=fmt['ax2_horz_div_linewidth'], transform=ax2.get_xaxis_transform())
    center_labels = ['CLAY', 'COBBLE', 'BOULDERS']
    for label, (low_b, high_b) in main_bounds.items():
        low = max(xlim[0], low_b); high = min(xlim[1], high_b)
        if low >= high: continue
        # Fixed: For CLAY, when plot extends below geological boundary, 
        # CLAY should span from plot start to geological upper bound
        calc_low = xlim[0] if label == 'CLAY' and xlim[0] < low_b else low
        if calc_low <= 0: calc_low = 1e-9
        mid = np.sqrt(calc_low * high)
        y_pos = 0.5 if label in center_labels else 0.75
        ax2.text(mid, y_pos, label, ha='center', va='center',
                 fontsize=fmt['ax2_main_label_fontsize'], fontweight=fmt['ax2_main_label_fontweight'])
    for cat, subs in sub_bounds.items():
        for sub_label, (low_s, high_s) in subs.items():
            low = max(xlim[0], low_s); high = min(xlim[1], high_s)
            if low >= high: continue
            if low <= 0: low = 1e-9
            mid = np.sqrt(low * high)
            ax2.text(mid, 0.25, sub_label, ha='center', va='center', fontsize=fmt['ax2_sub_label_fontsize'])
    all_vals = set(boundaries)
    for subs in sub_bounds.values():
        for _, high in subs.values():
             if xlim[0] < high < xlim[1]: all_vals.add(high)
    sorted_vals = sorted(list(all_vals))
    original_boundaries = sorted([b[0] for b in main_bounds.values()][1:] + [min(main_bounds["BOULDERS"][1], xlim[1])])
    for val in sorted_vals:
        weight = 'bold' if val in original_boundaries else 'normal'
        ax2.text(val, -0.05, f'{val:g}', ha='center', va='top',
                 transform=ax2.get_xaxis_transform(), fontsize=fmt['ax2_num_label_fontsize'], fontweight=weight)

    # --- Save Figure ---
    if output_filepath:
        try:
            # Adjust layout slightly before saving if legend is outside
            if fmt.get("legend_bbox_to_anchor", (0,0))[0] >= 1.0: # Check if legend x-pos is outside main axes
                 plt.subplots_adjust(right=0.8) # Example adjustment, might need tuning
            plt.savefig(output_filepath, dpi=save_dpi, bbox_inches=save_bbox_inches)
            print(f"Figure saved to: {output_filepath}")
        except Exception as e:
            print(f"Error saving figure to {output_filepath}: {e}")

    # --- Display Plot ---
    if show_plot:
        plt.show()
    
    # --- Close Plot (optional) ---
    if close_plot:
        plt.close(fig)