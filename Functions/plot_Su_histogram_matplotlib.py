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

def plot_Su_histogram_matplotlib(df, consistency,
                                 column='Su_kPa',
                                 bin_width=25,
                                 min_for_box=6,
                                 xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
                                 ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
                                 title: Optional[str] = None,
                                 title_suffix: Optional[str] = None,
                                 show_plot: bool = True,
                                 show_legend: bool = True,
                                 output_filepath: Optional[str] = None,
                                 output_dir='Output/figure/Triaxial',  # Kept for backward compatibility
                                 save=True):  # Kept for backward compatibility
    # Handle backward compatibility for output file
    if output_filepath is None and save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe = re.sub(r'[^0-9A-Za-z_-]', '_', consistency)
        fname = f"Triaxial_test_Su_{safe}.png"
        output_filepath = os.path.join(output_dir, fname)
        
    data = df[column].dropna()
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'axes.labelsize': 16,
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'grid.color': 'lightgrey',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
    })

    # Optional top boxplot
    if len(data) >= min_for_box:
        ax2 = make_axes_locatable(ax).append_axes("top", "20%", pad=0.1, sharex=ax)
        ax2.boxplot([data], vert=False, patch_artist=True,
                    boxprops=dict(facecolor='white', edgecolor='black'))
        ax2.axis('off')

    # Histogram with fixed range and bin_width, capture counts & patches
    bins = np.arange(0, 401, bin_width)
    counts, _, patches = ax.hist(data, bins=bins,
                                 edgecolor='black',
                                 linewidth=1.2, zorder=2)
    ax.set_axisbelow(True)
    ax.grid(True)

    if len(data) >= min_for_box:
        q1, med, mu = data.quantile(.25), data.median(), data.mean()
        ax.text(.98, .95,
                f"25th pct: {q1:.1f}\nMedian: {med:.1f}\nMean: {mu:.1f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='black', alpha=0.6),
                zorder=3)

    # Labels, ticks & limits
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency Count')
    
    # Handle axis limits with backward compatibility
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, 400)
        
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(bin_width))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Handle title with custom or default options
    if title is not None:
        title_str = title
    else:
        title_str = f'Triaxial Test: {column} for consistency = {consistency}'
        if title_suffix and isinstance(title_suffix, str) and title_suffix.strip():
            title_str += f": {title_suffix.strip()}"
    
    fig.subplots_adjust(top=0.85)
    fig.suptitle(title_str, y=0.95)

    # Save figure with standardized parameters
    if output_filepath:
        fig.savefig(output_filepath, dpi=100, bbox_inches='tight')

    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

