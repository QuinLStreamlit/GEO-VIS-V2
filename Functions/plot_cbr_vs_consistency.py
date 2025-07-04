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

def plot_cbr_vs_consistency(
    data_df,
    x_column='Consistency',
    y_column='CBR (%)',
    category_order=None,
    figsize=(10, 6),
    show_xlabel=False,       # Controls X-axis label visibility
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    title=None,              # Optional custom plot title
    title_suffix: Optional[str] = None,
    show_plot: bool = True,
    show_legend: bool = True,
    output_filepath: Optional[str] = None,
    save_filename=None,      # Kept for backward compatibility
    save_dpi=300
    ):
    """
    Generates a Scatter Plot (stripplot) using black points for y_column vs. x_column.
    X-axis title is optional (default off). Tick labels are bold.
    Optionally saves the plot to a file, otherwise displays it.
    (Boxplot functionality has been removed).

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        x_column (str, optional): Categorical column for x-axis. Defaults to 'Consistency'.
        y_column (str, optional): Numerical column for y-axis. Defaults to 'CBR (%)'.
        category_order (list, optional): Order for x-axis categories. Defaults to None (auto-order).
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (10, 6).
        show_xlabel (bool, optional): If True, displays the x-axis title (label). Defaults to False.
        title (str, optional): Custom plot title. If None, defaults to '<y_column> vs. <x_column>'.
        save_filename (str, optional): Path and filename to save the plot. If None, displays the plot.
        save_dpi (int): DPI resolution for saving file. Defaults to 300.
    """
    # Handle backward compatibility for output file
    if output_filepath is None and save_filename is not None:
        output_filepath = save_filename
        
    # --- Input Validation ---
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("Input 'data_df' must be a pandas DataFrame.")
    required_cols = [x_column, y_column]
    if not all(col in data_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data_df.columns]
        raise ValueError(f"Input DataFrame missing required columns: {missing} (using x='{x_column}', y='{y_column}')")
    if category_order is not None and not isinstance(category_order, list):
        raise TypeError("Input 'category_order' must be a list or None.")

    # --- Data Preparation ---
    df_plot = data_df[required_cols].copy()
    df_plot[y_column] = pd.to_numeric(df_plot[y_column], errors='coerce')
    original_rows = len(df_plot)
    df_plot.dropna(subset=[x_column, y_column], inplace=True)
    removed_rows = original_rows - len(df_plot)
    if removed_rows > 0:
        print(f"Note: Removed {removed_rows} rows with missing/invalid values in '{x_column}' or '{y_column}'.")
    if df_plot.empty:
        print(f"Error: No valid data remains for plotting ('{y_column}' vs '{x_column}').")
        return

    # --- Determine Full Category Order for Axis ---
    present_categories_all = df_plot[x_column].unique()
    if category_order is not None:
        temp_order = [cat for cat in category_order if cat in present_categories_all]
        temp_order.extend(sorted([cat for cat in present_categories_all if cat not in temp_order]))
        full_category_order_axis = temp_order or sorted(present_categories_all)
        if not temp_order:
            print(f"Warning: Categories provided in 'category_order' not found. Using alphabetical.")
    else:
        if x_column == 'Consistency':
            standard = ['Very soft', 'Soft', 'Firm', 'Stiff', 'Very stiff', 'Hard']
            temp_order = [cat for cat in standard if cat in present_categories_all]
            temp_order.extend(sorted([cat for cat in present_categories_all if cat not in temp_order]))
            full_category_order_axis = temp_order or sorted(present_categories_all)
            if not any(cat in present_categories_all for cat in standard):
                print(f"Warning: Standard 'Consistency' categories not found. Using alphabetical order.")
        else:
            full_category_order_axis = sorted(present_categories_all)

    # --- Create Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    sns.stripplot(
        data=df_plot,
        x=x_column,
        y=y_column,
        order=full_category_order_axis,
        color='black',
        jitter=True, alpha=0.6, size=5,
        ax=ax
    )

    # --- Title Handling with custom or default options ---
    if title is not None:
        title_str = title
    else:
        x_label = x_column.replace('_', ' ').title()
        y_label = y_column
        title_str = f'{y_label} vs. {x_label}'
        if title_suffix and isinstance(title_suffix, str) and title_suffix.strip():
            title_str += f": {title_suffix.strip()}"
    
    ax.set_title(title_str, fontsize=18, fontweight='bold', pad=15)

    # --- Axis Labels ---
    if show_xlabel:
        ax.set_xlabel(x_column.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('')
    ax.set_ylabel(y_column, fontsize=14, fontweight='bold')

        # increase padding between x-label and plot
    ax.xaxis.labelpad = 10  # try 8–12 and tweak to taste

    # --- Tick Settings ---
    ax.set_xticks(range(len(full_category_order_axis)))
    ax.set_xticklabels(full_category_order_axis)
    ax.tick_params(axis='both', which='both', direction='out')
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')
    
    # --- Handle axis limits with standardized parameters ---
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    # --- Save or Show with standardized parameters ---
    if output_filepath:
        try:
            directory = os.path.dirname(output_filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(output_filepath, dpi=save_dpi, bbox_inches='tight')
            print(f"Plot successfully saved to: {output_filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

