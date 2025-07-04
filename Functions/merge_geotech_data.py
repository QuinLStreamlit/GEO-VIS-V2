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


def merge_geotech_data(main_df, secondary_df, depth_difference_threshold,
                          main_suffix='_main', secondary_suffix='_secondary'):
    """
    As-of–merge main_df and secondary_df by nearest From_mbgl within
    depth_difference_threshold, matching on Hole_ID and Material.

    Returns a DataFrame with these first columns:
      • Hole_ID
      • Material
      • Material{secondary_suffix}
      • From_mbgl
      • From_mbgl{secondary_suffix}

    Then:
      – any other main_df columns
      – any overlapping secondary columns (suffixed)
      – any secondary-only columns (like your Is_50a_MPa)
    """
    # 1) Validate inputs
    for df, name in [(main_df,'main_df'), (secondary_df,'secondary_df')]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
    required = ['Hole_ID','Material','From_mbgl']
    for name, df in [('main_df',main_df), ('secondary_df',secondary_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
    if depth_difference_threshold < 0:
        raise ValueError("depth_difference_threshold cannot be negative.")

    # 2) Preserve original index to re-join later
    main = main_df.reset_index().rename(columns={'index':'_orig_idx'})
    sec  = secondary_df.copy()

    # 3) Duplicate the secondary Material/From_mbgl so they come through
    sec[f"Material{secondary_suffix}"]  = sec['Material']
    sec[f"From_mbgl{secondary_suffix}"] = sec['From_mbgl']

    # 4) Drop rows missing keys
    m = main.dropna(subset=['Hole_ID','Material']).copy()
    s = sec.dropna(subset=['Hole_ID','Material']).copy()
    if m.empty or s.empty:
        # nothing to merge → just append empty secondary cols
        out = main_df.copy()
        for c in secondary_df.columns:
            if c not in main_df.columns and c not in required:
                out[f"{c}{secondary_suffix}"] = np.nan
        return out

    # 5) Force types & drop bad From_mbgl
    for df in (m, s):
        df['Hole_ID']   = df['Hole_ID'].astype(str)
        df['Material']  = df['Material'].astype(str)
        df['From_mbgl'] = pd.to_numeric(df['From_mbgl'], errors='coerce')
    m = m.dropna(subset=['From_mbgl'])
    s = s.dropna(subset=['From_mbgl'])
    if m.empty or s.empty:
        out = main_df.copy()
        for c in secondary_df.columns:
            if c not in main_df.columns and c not in required:
                out[f"{c}{secondary_suffix}"] = np.nan
        return out

    # 6) Sort so merge_asof’s “on” key is globally ascending
    sort_keys = ['From_mbgl','Hole_ID','Material']
    m_sorted = m.sort_values(by=sort_keys).reset_index(drop=True)
    s_sorted = s.sort_values(by=sort_keys).reset_index(drop=True)

    # 7) Perform the as-of merge on From_mbgl
    merged = pd.merge_asof(
        left = m_sorted,
        right= s_sorted,
        on   ='From_mbgl',
        by   =['Hole_ID','Material'],
        tolerance          = depth_difference_threshold,
        direction          ='nearest',
        suffixes           =(main_suffix, secondary_suffix),
        allow_exact_matches=True
    )

    # 8) Build list of new columns to bring back
    key_cols = ['Hole_ID','Material','From_mbgl']
    # secondary-only columns (e.g. Is_50a_MPa)
    sec_only = [c for c in secondary_df.columns
                if c not in main_df.columns and c not in key_cols]
    # any overlapping columns that got suffixed
    suffixed = [c for c in merged.columns if c.endswith(secondary_suffix)]
    new_cols = sec_only + suffixed

    # 9) Merge back onto original main via the preserved index
    back = pd.merge(
        main,
        merged[['_orig_idx'] + new_cols],
        on   ='_orig_idx',
        how  ='left'
    ).drop(columns=['_orig_idx'])

    # 10) Reorder so that Material & From_mbgl from both DF come first
    first_cols = [
        'Hole_ID',
        'Material',
        f"Material{secondary_suffix}",
        'From_mbgl',
        f"From_mbgl{secondary_suffix}"
    ]
    main_rest = [c for c in main_df.columns if c not in ['Hole_ID','Material','From_mbgl']]
    # any new_cols not already in first_cols
    extra_new = [c for c in new_cols if c not in first_cols]

    final_order = first_cols + main_rest + extra_new
    # include any leftover columns at the end
    final_cols = [c for c in final_order if c in back.columns] + \
                 [c for c in back.columns if c not in final_order]

    return back[final_cols]
