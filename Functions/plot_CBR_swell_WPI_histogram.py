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


def _find_stacking_column(df: pd.DataFrame, stack_col: Optional[str] = None) -> Optional[str]:
    """
    Find the best matching column for stacking based on flexible name patterns.
    
    Args:
        df: Input DataFrame
        stack_col: User-specified column name (if provided, will try to find flexible matches)
        
    Returns:
        Column name to use for stacking, or None if not found or disabled
    """
    if stack_col is None:
        return None
        
    # If exact column exists, use it
    if stack_col in df.columns:
        return stack_col
        
    # Common patterns for geological/stacking columns
    common_patterns = [
        'map_symbol', 'map symbol', 'mapsymbol',
        'geology', 'geological', 'geo',
        'material', 'mat', 'rock_type', 'rocktype',
        'formation', 'unit', 'lithology', 'lith',
        'symbol', 'code', 'group', 'class'
    ]
    
    # Try case-insensitive exact matches first
    for col in df.columns:
        if col.lower() == stack_col.lower():
            return col
            
    # Try pattern matching
    stack_col_lower = stack_col.lower().replace('_', '').replace(' ', '')
    for col in df.columns:
        col_clean = col.lower().replace('_', '').replace(' ', '')
        if stack_col_lower in col_clean or col_clean in stack_col_lower:
            return col
            
    # Try common patterns if user didn't specify exact match
    for pattern in common_patterns:
        for col in df.columns:
            col_clean = col.lower().replace('_', '').replace(' ', '')
            pattern_clean = pattern.replace('_', '').replace(' ', '')
            if pattern_clean in col_clean:
                return col
                
    return None


def plot_CBR_swell_WPI_histogram(
    # === Data Parameters ===
    data_df: pd.DataFrame,
    facet_col: str = 'Name',
    category_col: str = 'category', 
    category_order: Optional[List[str]] = None,
    facet_order: Optional[List[str]] = None,
    
    # === Stacking Control ===
    enable_stacking: bool = True,
    stack_col: Optional[str] = 'map_symbol',
    
    # === Display Control ===
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    yticks: Optional[List[float]] = None,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    title_suffix: Optional[str] = None,
    
    # === Figure Settings ===
    figsize: Tuple[float, float] = (10, 6),
    figure_dpi: int = 100,
    save_dpi: int = 300,
    style: Optional[str] = 'seaborn-v0_8-colorblind',
    
    # === Styling Parameters ===
    cmap_name: Optional[str] = None,
    bar_alpha: float = 0.9,
    bar_edgecolor: str = 'black',
    bar_linewidth: float = 0.6,
    subplot_title_fontsize: int = 14,
    subplot_title_fontweight: str = 'bold',
    axis_label_fontsize: int = 13,
    axis_label_fontweight: str = 'bold',
    tick_fontsize: int = 12,
    tick_direction: str = 'out',
    legend_fontsize: int = 11,
    legend_title_fontsize: int = 12,
    show_grid: bool = True,
    grid_axis: str = 'y',
    grid_linestyle: str = '--',
    grid_linewidth: float = 0.6,
    grid_alpha: float = 0.4,
    bottom_margin: float = 0.12,
    legend_loc: str = 'center left',
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = (0.87, 0.5),
    
    # === Output Parameters ===
    show_plot: bool = True,
    show_legend: bool = True,
    output_filepath: Optional[str] = None,
    
    # === Plot Cleanup Control ===
    close_plot: bool = True
    ) -> None:
    """
    Create stacked histogram plots for CBR, Swell, and WPI data with flexible stacking options.
    
    This function creates histogram plots with optional stacking by any categorical column 
    (geology, material, etc.), faceted by a specified column, with extensive customization options.
    
    Parameters
    ----------
    === Data Parameters ===
    data_df : pd.DataFrame
        Input DataFrame containing the data to plot
    facet_col : str, default 'Name'
        Column name to create separate subplots (facets) for
    category_col : str, default 'category'
        Column name containing the categories for histogram bars (x-axis)
    category_order : list of str, optional
        Specific order for categories. If None, uses sorted order
    facet_order : list of str, optional
        Specific order for facets. If None, uses sorted order
        
    === Stacking Control ===
    enable_stacking : bool, default True
        Whether to enable stacking by a categorical column
    stack_col : str or None, default 'map_symbol'
        Column name for stacking. Uses flexible matching (case-insensitive, handles 
        variations like 'Map_symbol', 'map symbol', 'geology', etc.). Set to None to disable stacking.
        
    === Display Control ===
    xlim : tuple of (float, float), optional
        X-axis limits as (min, max)
    ylim : tuple of (float, float), optional  
        Y-axis limits as (min, max)
    yticks : list of float, optional
        Custom y-axis tick positions (e.g., [0, 5, 10, 15, 20])
    xlabel : str, optional
        X-axis title. If None, no x-axis title is shown
    title : str, optional
        Overall plot title
    title_suffix : str, optional
        Suffix to add to default title 'CBR, Swell and WPI Histogram'
        
    === Figure Settings ===
    figsize : tuple of (float, float), default (8, 6)
        Figure size in inches as (width, height)
    figure_dpi : int, default 100
        DPI for figure display
    save_dpi : int, default 300
        DPI for saved figure
    style : str, default 'seaborn-v0_8-colorblind'
        Matplotlib style to use
        
    === Styling Parameters ===
    cmap_name : str, optional
        Colormap name for stacking colors. Auto-selected if None
    bar_alpha : float, default 0.9
        Transparency of bars (0=transparent, 1=opaque)
    bar_edgecolor : str, default 'black'
        Color of bar edges
    bar_linewidth : float, default 0.6
        Width of bar edges
    subplot_title_fontsize : int, default 14
        Font size for subplot titles
    subplot_title_fontweight : str, default 'bold'
        Font weight for subplot titles
    axis_label_fontsize : int, default 13
        Font size for axis labels
    axis_label_fontweight : str, default 'bold'
        Font weight for axis labels
    tick_fontsize : int, default 12
        Font size for tick labels
    tick_direction : str, default 'out'
        Direction of ticks ('in', 'out', 'inout')
    legend_fontsize : int, default 11
        Font size for legend text
    legend_title_fontsize : int, default 12
        Font size for legend title
    show_grid : bool, default True
        Whether to show grid lines
    grid_axis : str, default 'y'
        Which axes to show grid on ('x', 'y', 'both')
    grid_linestyle : str, default '--'
        Style of grid lines
    grid_linewidth : float, default 0.6
        Width of grid lines
    grid_alpha : float, default 0.4
        Transparency of grid lines
    bottom_margin : float, default 0.12
        Bottom margin for layout
    legend_loc : str, default 'center left'
        Legend location
    legend_bbox_to_anchor : tuple of (float, float), default (0.87, 0.5)
        Legend anchor position
        
    === Output Parameters ===
    show_plot : bool, default True
        Whether to display the plot
    show_legend : bool, default True
        Whether to show the legend (only relevant when stacking is enabled)
    output_filepath : str, optional
        File path to save the plot. If None, plot is not saved
        
    Returns
    -------
    None
    
    Examples
    --------
    Basic usage with default stacking:
    >>> plot_CBR_swell_WPI_histogram(df, facet_col='Test_Type', category_col='CBR_Range')
    
    Disable stacking for simple frequency histogram:
    >>> plot_CBR_swell_WPI_histogram(df, enable_stacking=False)
    
    Custom y-axis ticks and x-axis label:
    >>> plot_CBR_swell_WPI_histogram(df, yticks=[0, 5, 10, 15, 20], xlabel='CBR Categories')
    
    Stack by material instead of geology:
    >>> plot_CBR_swell_WPI_histogram(df, stack_col='Material')
    """
    # === Input Validation ===
    if data_df is None or data_df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    # Validate required columns
    required_cols = [facet_col, category_col]
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")
    
    # Handle stacking column detection
    actual_stack_col = None
    if enable_stacking and stack_col is not None:
        actual_stack_col = _find_stacking_column(data_df, stack_col)
        if actual_stack_col is None:
            print(f"WARNING: Stack column '{stack_col}' not found. Disabling stacking.")
            enable_stacking = False
        else:
            print(f"INFO: Using '{actual_stack_col}' for stacking (matched from '{stack_col}')")
    elif enable_stacking and stack_col is None:
        enable_stacking = False
        print("INFO: Stacking disabled (stack_col is None)")
    
    # Determine if we should show legend
    show_legend = show_legend and enable_stacking

    # --- Apply Base Style ---
    context_manager = plt.style.context(style) if style else warnings.catch_warnings()
    with context_manager:

        # === 1. Prepare Data ===
        df_processed = data_df.copy()
        
        # Ensure category column is proper type
        if pd.api.types.is_numeric_dtype(df_processed[category_col]):
            if df_processed[category_col].isnull().any():
                df_processed[category_col] = df_processed[category_col].fillna('NaN').astype(str)
            else:
                df_processed[category_col] = df_processed[category_col].astype(str)
        elif df_processed[category_col].isnull().any():
            df_processed[category_col] = df_processed[category_col].fillna('NaN')
        
        # Handle stacking column preparation
        all_symbols = []
        ordered_symbols_by_count = []
        
        if enable_stacking and actual_stack_col:
            # Ensure stacking column is proper type
            if pd.api.types.is_numeric_dtype(df_processed[actual_stack_col]):
                if df_processed[actual_stack_col].isnull().any():
                    df_processed[actual_stack_col] = df_processed[actual_stack_col].fillna('NaN').astype(str)
                else:
                    df_processed[actual_stack_col] = df_processed[actual_stack_col].astype(str)
            elif df_processed[actual_stack_col].isnull().any():
                df_processed[actual_stack_col] = df_processed[actual_stack_col].fillna('NaN')
            
            # Get stacking values and ordering
            stack_values = df_processed[actual_stack_col].dropna().unique()
            if len(stack_values) > 0:
                symbol_total_counts = df_processed[actual_stack_col].value_counts()
                ordered_symbols_by_count = symbol_total_counts.index.tolist()
                all_symbols = sorted(stack_values)
            else:
                warnings.warn(f"No valid data in '{actual_stack_col}'.")
                enable_stacking = False
        available_categories = df_processed[category_col].unique() # Category order
        if category_order is None: category_order_actual = sorted(available_categories); print(f"INFO: Using sorted categories: {category_order_actual}")
        else:
            category_order_actual = [cat for cat in category_order if cat in available_categories]; missing_cats = [cat for cat in category_order if cat not in available_categories]
            if missing_cats: warnings.warn(f"Categories specified but not found: {missing_cats}")
            if not category_order_actual: warnings.warn("No specified categories found. Defaulting."); category_order_actual = sorted(available_categories)
        category_dtype = pd.CategoricalDtype(categories=category_order_actual, ordered=True); df_processed[category_col] = df_processed[category_col].astype(category_dtype)
        actual_facets_in_data = df_processed[facet_col].unique() # Facet order
        if facet_order:
            facet_names_to_plot = [name for name in facet_order if name in actual_facets_in_data]; missing_facets = [name for name in facet_order if name not in actual_facets_in_data]
            if missing_facets: warnings.warn(f"Facets specified but not found: {missing_facets}")
            remaining_facets = sorted([name for name in actual_facets_in_data if name not in facet_names_to_plot]); facet_names_to_plot.extend(remaining_facets)
        else: facet_names_to_plot = sorted(actual_facets_in_data)
        if not facet_names_to_plot: raise ValueError(f"No values found in '{facet_col}'.")
        # Prepare plot data
        plot_data: Dict[str, pd.DataFrame] = {}
        for name in facet_names_to_plot:
            df_facet = df_processed[df_processed[facet_col] == name]
            
            if enable_stacking and actual_stack_col:
                # Create stacked counts
                counts = df_facet.groupby([category_col, actual_stack_col], observed=False).size().unstack(fill_value=0)
                counts = counts.reindex(columns=all_symbols, fill_value=0)
            else:
                # Create simple frequency counts without stacking
                counts = df_facet.groupby(category_col, observed=False).size()
                counts = pd.DataFrame({'count': counts})
                
            plot_data[name] = counts

        # --- 2. Create Figure & Axes ---
        n_facets = len(facet_names_to_plot)
        if n_facets == 0: print("No data facets to plot."); return
        figsize_inches = figsize
        # Make sure sharey=True is set
        fig, axes = plt.subplots(1, n_facets, figsize=figsize_inches, sharey=True, sharex=True, squeeze=False)
        axes = axes.flatten()

        # --- Define Colors ---
        # [SAME AS PREVIOUS VERSION - uses linspace(0.1, 0.9)]
        color_map: Dict[Any, Any] = {}
        if all_symbols:
            effective_cmap_name = cmap_name;
            if not effective_cmap_name:
                if len(all_symbols) <= 10: effective_cmap_name = 'tab10'
                elif len(all_symbols) <= 20: effective_cmap_name = 'tab20'
                else: effective_cmap_name = 'viridis'
            try: cmap = plt.get_cmap(effective_cmap_name); palette = cmap(np.linspace(0.1, 0.9, len(all_symbols))); color_map = {symbol: color for symbol, color in zip(all_symbols, palette)}
            except ValueError: warnings.warn(f"Invalid cmap '{effective_cmap_name}'. Using fallback."); cmap = plt.get_cmap('viridis'); palette = cmap(np.linspace(0.1, 0.9, len(all_symbols))); color_map = {symbol: color for symbol, color in zip(all_symbols, palette)}
        else: warnings.warn("No symbols found for stacking/coloring.")


        # === 3. Plotting Loop ===
        max_y_val = 0
        plot_handles_labels: Dict[str, plt.Artist] = {}
        
        for i, name in enumerate(facet_names_to_plot):
            ax = axes[i]
            data = plot_data[name]
            categories_to_plot = data.index.astype(str).tolist()
            
            if enable_stacking and actual_stack_col:
                # Stacked plotting
                bottoms = np.zeros(len(categories_to_plot))
                for symbol in ordered_symbols_by_count:
                    if symbol in data.columns:
                        counts = data.loc[categories_to_plot, symbol].values
                        if np.any(counts > 0):
                            bar = ax.bar(categories_to_plot, counts, label=symbol, 
                                       bottom=bottoms, color=color_map.get(symbol, 'grey'), 
                                       alpha=bar_alpha, edgecolor=bar_edgecolor, 
                                       linewidth=bar_linewidth)
                            bottoms += counts
                            plot_handles_labels.setdefault(symbol, bar)
                max_y_val = max(max_y_val, bottoms.max())
            else:
                # Simple frequency plotting (no stacking)
                counts = data['count'].values
                bar = ax.bar(categories_to_plot, counts, 
                           color='steelblue', alpha=bar_alpha, 
                           edgecolor=bar_edgecolor, linewidth=bar_linewidth)
                max_y_val = max(max_y_val, counts.max())

            # --- 4. Subplot Styling ---
            ax.set_title(name, fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight, pad=10)
            ax.tick_params(axis='x', rotation=0, labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)
            ax.tick_params(axis='both', which='major', direction=tick_direction)
            if show_grid:
                if grid_axis in ['x', 'both']: ax.xaxis.grid(True, linestyle=grid_linestyle, linewidth=grid_linewidth, alpha=grid_alpha)
                if grid_axis in ['y', 'both']: ax.yaxis.grid(True, linestyle=grid_linestyle, linewidth=grid_linewidth, alpha=grid_alpha)
                ax.set_axisbelow(True)
            else: ax.grid(False)
            ax.spines[['top', 'right']].set_visible(False); ax.spines[['left', 'bottom']].set_linewidth(0.8)
            ax.set_xticks(range(len(categories_to_plot)))
            ax.set_xticklabels(categories_to_plot)
            
            # Set custom y-axis ticks if provided
            if yticks is not None:
                ax.set_yticks(yticks)
            else:
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))


        # === 5. Global Styling, Limits, & Legend ===
        axes[0].set_ylabel('Frequency count', fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
        
        # Set x-axis label if provided
        if xlabel is not None:
            for ax in axes:
                ax.set_xlabel(xlabel, fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)

        # *** Handle axis limits with standardized parameters ***
        # Y-axis limits
        if ylim is not None:
            axes[0].set_ylim(ylim)
        elif max_y_val > 0:
            final_y_limit = max_y_val * 1.05 # Add 5% padding
            final_y_limit = max(final_y_limit, 1) # Ensure limit is at least 1
            print(f"INFO: Setting shared Y-axis limit to: {final_y_limit:.2f}")
            # Apply limit to the first axis (sharey=True propagates it)
            axes[0].set_ylim(bottom=0, top=final_y_limit)
        elif len(axes)>0 : # If no data plotted, set a default small limit
             axes[0].set_ylim(bottom=0, top=1)
             
        # X-axis limits (if specified)
        if xlim is not None:
            for ax in axes:
                ax.set_xlim(xlim)
                
        # Re-apply locator to get nice ticks within the range (unless custom yticks provided)
        if yticks is None:
            for ax_loop in axes:
                ax_loop.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Legend creation logic with show_legend control
        legend_created = False
        if show_legend and enable_stacking and all_symbols and plot_handles_labels:
            legend_handles = []
            legend_labels = []
            for symbol in reversed(ordered_symbols_by_count):
                if symbol in plot_handles_labels:
                    legend_handles.append(plot_handles_labels[symbol])
                    legend_labels.append(symbol)
            if legend_handles:
                legend_title = actual_stack_col if actual_stack_col else 'Category'
                fig.legend(legend_handles, legend_labels, title=legend_title, 
                         bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc, 
                         fontsize=legend_fontsize, title_fontsize=legend_title_fontsize, 
                         frameon=False)
                legend_created = True

        # --- 6. Layout & Save/Show ---
        # [SAME AS PREVIOUS VERSION - uses bottom_margin, adjusts right_margin]
        left_margin = 0.05; top_margin = 0.92
        right_margin_default = 0.95; right_margin_shrunk = 0.86
        right_margin = right_margin_shrunk if legend_created and legend_bbox_to_anchor is not None else right_margin_default
        layout_rect = [left_margin, bottom_margin, right_margin, top_margin]
        try: fig.tight_layout(rect=layout_rect)
        except ValueError:
              warnings.warn("Tight layout failed. Using subplots_adjust fallback.")
              plt.subplots_adjust(left=layout_rect[0], bottom=layout_rect[1], right=layout_rect[2], top=layout_rect[3], wspace=0.15)

        # Add global title if specified
        if title is not None:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        elif title_suffix is not None:
            default_title = f'CBR, Swell and WPI Histogram'
            fig.suptitle(f'{default_title}: {title_suffix}', fontsize=16, fontweight='bold')
        
        # Saving / Showing logic with standardized parameters
        if output_filepath:
            try:
                dir_name = os.path.dirname(output_filepath);
                if dir_name and not os.path.exists(dir_name): os.makedirs(dir_name); print(f"Created dir: {dir_name}")
                plt.savefig(output_filepath, dpi=save_dpi, bbox_inches='tight'); print(f"Plot saved to: {output_filepath}")
            except Exception as e: print(f"Error saving plot: {e}")
        
        if show_plot:
            plt.show()

        # Cleanup
        if close_plot:
            plt.close(fig)
        print(f"--- Plotting finished for {facet_col} ---")

