import itertools
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


def plot_effective_c_phi_histograms(df, consistency,
                                    bin_width_c=50,
                                    bin_width_phi=5,
                                    min_for_box=10,
                                    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
                                    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
                                    title: Optional[str] = None,
                                    title_suffix: Optional[str] = None,
                                    show_plot: bool = True,
                                    show_legend: bool = True,
                                    output_filepath: Optional[str] = None,
                                    output_dir='Output/figure/Triaxial',  # Kept for backward compatibility
                                    save=True):  # Kept for backward compatibility
    """
    1×2 subplot:
      - Left:  Effective c (kPa), x in 0–400, bins = bin_width_c
      - Right: Effective phi (deg), x in 0–50,  bins = bin_width_phi

    Optional top boxplot if ≥ min_for_box points, bar‐count labels,
    y‐limit 0–5, and auto‐save PNG.
    """
    # Handle backward compatibility for output file
    if output_filepath is None and save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe = re.sub(r'[^0-9A-Za-z_-]', '_', consistency)
        fname = f"Effective_c_phi_histograms_{safe}.png"
        output_filepath = os.path.join(output_dir, fname)
        
    cols = ['Effective c (kPa)', 'Effective phi (deg)']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        show_box = len(data) >= min_for_box

        # optional boxplot
        if show_box:
            ax_box = make_axes_locatable(ax)\
                .append_axes("top", "20%", pad=0.1, sharex=ax)
            ax_box.boxplot([data], vert=False, patch_artist=True,
                           boxprops=dict(facecolor='white', edgecolor='black'))
            ax_box.axis('off')

        # set ranges and bin width per column
        if col == 'Effective phi (deg)':
            x_min, x_max, bw = 0, 50, bin_width_phi
        else:
            x_min, x_max, bw = 0, 400, bin_width_c

        bins = np.arange(x_min, x_max + bw, bw)
        counts, _, patches = ax.hist(data, bins=bins,
                                     edgecolor='black', linewidth=1.2, zorder=2)
        ax.set_axisbelow(True)
        ax.grid(True)

        # stats box if boxplot shown
        if show_box:
            q1, med, mu = data.quantile(.25), data.median(), data.mean()
            ax.text(.98, .95,
                    f"25th pct: {q1:.1f}\nMedian: {med:.1f}\nMean: {mu:.1f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='white',
                              edgecolor='black', alpha=0.6),
                    zorder=3)

        # labels, ticks & limits
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency Count')
        
        # Handle axis limits with backward compatibility
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(x_min, x_max)
            
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(0, 5)
            
        ax.xaxis.set_major_locator(mticker.MultipleLocator(bw))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Handle title with custom or default options
    if title is not None:
        title_str = title
    else:
        title_str = f'Triaxial Test: Effective c & φ for consistency = {consistency}'
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


