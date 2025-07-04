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


def plot_emerson_by_origin(
        # --- Core Data Args ---
        df: pd.DataFrame,
        origin_col: str,
        emerson_col: str = 'Emerson class',  # Column name containing Emerson class data
        # --- Output Control Args ---
        output_filepath: str = None,
        # FULL path including filename (e.g., 'output/plots/my_chart.png'). Required if save_plot=True.
        save_plot: bool = True,  # Whether to save the plot to a file
        show_plot: bool = True,  # Whether to display the plot
        dpi: int = 300,  # Resolution (dots per inch) for saving
        # --- Plotting Style Args ---
        xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        title: Optional[str] = None,
        title_suffix: Optional[str] = None,
        show_legend: bool = True,
        figsize: tuple = (12, 6),  # Figure dimensions (width, height)
        cmap_name: str = 'viridis',  # Matplotlib colormap name for bars
        bar_width: float = 1,  # Width of bars
        bar_alpha: float = 0.8,  # Transparency of bars
        edge_color: str = 'black',  # Color of bar edges ('none' for no edges)
        # --- Font Size Args ---
        title_fontsize: int = 15,  # Font size for the main title
        label_fontsize: int = 13,  # Font size for x and y axis labels
        tick_fontsize: int = 12,  # Font size for x and y axis tick labels
        legend_fontsize: int = 11,  # Font size for legend text
        legend_title_fontsize: int = 12,  # Font size for legend title
        # --- Legend Position Args ---
        legend_loc: str = 'best',  # Anchor point for legend positioning relative to bbox
        legend_bbox_to_anchor: tuple = (1.01, 1),  # Position relative to axes (set to None for default placement)
        
        # === Plot Cleanup Control ===
        close_plot: bool = True
):
    """
    Generates and saves a bar chart showing the distribution of Emerson classes
    based on a specified origin column, with adjustable styling parameters.
    Accepts a full filepath for saving.

    Args:
        df (pd.DataFrame): Input DataFrame with Emerson class data and origin_col.
        origin_col (str): Name of the column representing the origin category.
        emerson_col (str): Name of the column containing Emerson class data. 
                          Default 'Emerson class'. Supports flexible naming.

        output_filepath (str | None): Full path including desired filename
                                      (e.g., 'output/plots/my_chart.png').
                                      Required if save_plot is True. Default None.
        save_plot (bool): If True, saves the plot to output_filepath. Default True.
        show_plot (bool): If True, displays the plot. Default True.
        dpi (int): Resolution for saved plot. Default 300.

        figsize (tuple): Figure dimensions (width, height). Default (12, 6).
        cmap_name (str): Matplotlib colormap name for bars. Default 'viridis'.
        bar_width (float): Width of bars. Default 0.7.
        bar_alpha (float): Transparency (0-1) of bars. Default 0.9.
        edge_color (str): Color of bar edges ('none' for no edge). Default 'black'.
        title_fontsize (int): Font size for the plot title. Default 15.
        label_fontsize (int): Font size for axis labels. Default 12.
        tick_fontsize (int): Font size for axis tick labels. Default 11.
        legend_fontsize (int): Font size for legend text. Default 11.
        legend_title_fontsize (int): Font size for legend title. Default 12.
        legend_loc (str): Anchor point for legend positioning. Default 'upper left'.
        legend_bbox_to_anchor (tuple | None): Legend position relative to axes.
                                             Default (1.02, 1) (outside right). Set to None for automatic placement.
    """

    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if emerson_col not in df.columns:
        raise ValueError(f"DataFrame must contain the specified Emerson column: '{emerson_col}'.")
    if origin_col not in df.columns:
        raise ValueError(f"DataFrame must contain the specified origin column: '{origin_col}'.")

    # --- Filepath Handling ---
    if save_plot:
        if output_filepath is None:
            # Raise error if saving is requested but no path is given
            raise ValueError("If 'save_plot' is True, 'output_filepath' must be provided.")
        if not isinstance(output_filepath, str) or not output_filepath:
            raise ValueError("'output_filepath' must be a non-empty string.")

        # Extract directory and create if it doesn't exist
        output_dir = os.path.dirname(output_filepath)
        # Handle case where path is just a filename (dirname is '')
        if output_dir and not os.path.isdir(output_dir):
            try:
                print(f"Output directory '{output_dir}' does not exist. Creating it.")
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"Error: Could not create output directory '{output_dir}'. Plot will not be saved. Error: {e}")
                save_plot = False  # Disable saving if folder creation fails

    # --- Data Preprocessing ---
    # (Keeping previous NaN handling)
    df_processed = df.copy()
    initial_rows = len(df_processed)
    df_processed.dropna(subset=[emerson_col, origin_col], inplace=True)
    rows_after_drop = len(df_processed)
    if rows_after_drop < initial_rows:
        print(f"INFO: Removed {initial_rows - rows_after_drop} rows with NaN values "
              f"in '{emerson_col}' or '{origin_col}'.")

    if df_processed.empty:
        print(f"Warning: DataFrame is empty after handling NaNs for '{origin_col}'. Skipping plot.")
        return

    try:
        df_processed[emerson_col] = pd.to_numeric(df_processed[emerson_col], errors='coerce')
        conversion_fails = df_processed[emerson_col].isna().sum()
        if conversion_fails > 0:
            new_na = conversion_fails - (initial_rows - rows_after_drop)
            if new_na > 0:
                warnings.warn(
                    f"INFO: Found {int(new_na)} non-numeric value(s) in '{emerson_col}'. "  # Ensure int for warning
                    f"These rows will also be dropped.", UserWarning)
            df_processed.dropna(subset=[emerson_col], inplace=True)
            if df_processed.empty:
                print(
                    f"Warning: DataFrame became empty after removing non-numeric '{emerson_col}' values. Skipping plot.")
                return
    except Exception as e:
        warnings.warn(
            f"Could not reliably convert '{emerson_col}' to numeric. Proceeding with original data types for grouping. Error: {e}",
            UserWarning)

    # --- Data Grouping and Pivoting ---
    grouped = (
        df_processed
        .groupby([emerson_col, origin_col], observed=False)
        .size()
        .reset_index(name="count")
    )

    if grouped.empty:
        print(f"Warning: No data after grouping for '{origin_col}'. Skipping plot.")
        return

    try:
        pivot_df = grouped.pivot(
            index=emerson_col,
            columns=origin_col,
            values="count"
        ).fillna(0)
    except Exception as e:
        print(f"Error during pivoting for '{origin_col}': {e}")
        return

    # Sort index numerically if possible
    try:
        pivot_df = pivot_df.sort_index()
    except TypeError:
        warnings.warn("Index could not be sorted numerically. Using default order.", UserWarning)

    print(f"\n--- Generating Plot for: {origin_col} ---")

    # --- Plotting Setup ---
    num_categories = len(pivot_df.columns)
    if num_categories == 0:
        print(f"Warning: Pivot table for '{origin_col}' has no columns to plot. Skipping.")
        return

    try:
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.1, 0.9, num_categories))
    except ValueError:
        warnings.warn(f"Invalid cmap_name '{cmap_name}'. Using default 'viridis'.", UserWarning)
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0.1, 0.9, num_categories))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    pivot_df.plot(
        kind="bar", ax=ax, width=bar_width, color=colors, alpha=bar_alpha,
        edgecolor=edge_color, linewidth=0.8 if edge_color != 'none' else 0,
        legend=False  # Disable automatic legend - will be controlled by show_legend parameter
    )

    # --- Axis and Label Styling ---
    ax.set_xlabel(emerson_col, fontsize=label_fontsize, labelpad=12)
    ax.set_ylabel("Frequency Count", fontsize=label_fontsize, labelpad=12)
    
    # Handle title with custom or default options
    if title is not None:
        title_str = title
    else:
        title_str = f"Emerson Class Distribution"
        if title_suffix and isinstance(title_suffix, str) and title_suffix.strip():
            title_str += f": {title_suffix.strip()}"
    
    ax.set_title(title_str, fontsize=title_fontsize, fontweight='bold', pad=18)

    # --- Tick Styling ---
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.index, rotation=0, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    
    # Handle axis limits with backward compatibility
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Spine and Grid Styling ---
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8);
    ax.spines['bottom'].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, linewidth=0.5)
    ax.xaxis.grid(False)

    # --- Legend Styling ---
    if show_legend:
        legend_title = origin_col.replace('_', ' ').title()
        ax.legend(
            title=legend_title, fontsize=legend_fontsize, title_fontsize=legend_title_fontsize,
            frameon=False, bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc
        )

    # --- Layout Adjustment ---
    if legend_bbox_to_anchor:
        try:
            if legend_bbox_to_anchor[0] > 1.0:
                # This simple adjustment might need fine-tuning depending on figure/font size
                plt.subplots_adjust(right=0.85)
        except (TypeError, IndexError):
            pass
    plt.tight_layout()

    # --- Saving ---
    if save_plot:
        try:
            # Use the full output_filepath directly
            plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved successfully to: {output_filepath}")
        except Exception as e:
            print(f"Error saving plot to '{output_filepath}': {e}")

    # --- Showing ---
    if show_plot:
        plt.show()

    # --- Cleanup ---
    if close_plot:
        plt.close(fig)
    print(f"--- Plotting finished for: {origin_col} ---\n")


