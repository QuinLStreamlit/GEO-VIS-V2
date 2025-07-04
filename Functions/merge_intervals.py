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


def merge_intervals(main_df, geology_df):
    """
    Merges geology data into main_df based on dominant interval overlap
    or point-in-interval matching (if main_df['To_mbgl'] is missing/invalid).
    Automatically detects and maps column names for hole ID and depth columns.

    Args:
        main_df (pd.DataFrame): Left DataFrame. Must contain hole ID and from depth columns.
        geology_df (pd.DataFrame): Right DataFrame. Must contain hole ID and from depth columns.

    Returns:
        pd.DataFrame: main_df merged with relevant data from geology_df (suffix '_geo').
    """
    if not isinstance(main_df, pd.DataFrame) or not isinstance(geology_df, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")

    return _match_logic_with_fallback(main_df, geology_df)

def _detect_column_mapping(df, column_type):
    """
    Detects and maps column names based on common naming patterns.

    Args:
        df (pd.DataFrame): DataFrame to analyze
        column_type (str): Type of column to detect ('hole_id', 'from_depth', 'to_depth')

    Returns:
        str or None: The detected column name, or None if not found
    """
    columns = df.columns.tolist()

    if column_type == 'hole_id':
        # Common patterns for hole ID columns
        patterns = [
            r'^hole[_\s]*id$',
            r'^bh[_\s]*id$',
            r'^borehole[_\s]*id$',
            r'^drill[_\s]*hole[_\s]*id$',
            r'^hole[_\s]*name$',
            r'^bh[_\s]*name$',
            r'^id$'
        ]
    elif column_type == 'from_depth':
        # Common patterns for "from" depth columns
        patterns = [
            r'^from[_\s]*m(bgl)?$',
            r'^from[_\s]*\([^)]*m[^)]*\)$',
            r'^from[_\s]*depth$',
            r'^from$',
            r'^start[_\s]*depth$',
            r'^top[_\s]*depth$'
        ]
    elif column_type == 'to_depth':
        # Common patterns for "to" depth columns
        patterns = [
            r'^to[_\s]*m(bgl)?$',
            r'^to[_\s]*\([^)]*m[^)]*\)$',
            r'^to[_\s]*depth$',
            r'^to$',
            r'^end[_\s]*depth$',
            r'^bottom[_\s]*depth$'
        ]
    else:
        return None

    # Try to match patterns (case insensitive)
    for pattern in patterns:
        for col in columns:
            if re.match(pattern, col.lower().strip()):
                return col

    return None

def _standardize_dataframe_columns(df, df_name):
    """
    Standardizes column names in a DataFrame by detecting and mapping them.

    Args:
        df (pd.DataFrame): DataFrame to standardize
        df_name (str): Name of the DataFrame for error messages

    Returns:
        tuple: (standardized_df, column_mapping)
    """
    df_copy = df.copy()
    column_mapping = {}

    # Detect hole ID column
    hole_id_col = _detect_column_mapping(df, 'hole_id')
    if hole_id_col is None:
        raise ValueError(
            f"{df_name} missing hole ID column. Expected patterns: 'Hole_ID', 'BH_ID', 'Borehole_ID', etc.")

    # Detect from depth column
    from_depth_col = _detect_column_mapping(df, 'from_depth')
    if from_depth_col is None:
        raise ValueError(
            f"{df_name} missing 'from' depth column. Expected patterns: 'From_mbgl', 'From_m', 'From (m)', 'From', etc.")

    # Detect to depth column (optional)
    to_depth_col = _detect_column_mapping(df, 'to_depth')

    # Create column mapping
    column_mapping['hole_id'] = hole_id_col
    column_mapping['from_depth'] = from_depth_col
    if to_depth_col:
        column_mapping['to_depth'] = to_depth_col

    # Rename columns to standard names
    rename_dict = {
        hole_id_col: 'Hole_ID',
        from_depth_col: 'From_mbgl'
    }
    if to_depth_col:
        rename_dict[to_depth_col] = 'To_mbgl'

    df_copy = df_copy.rename(columns=rename_dict)

    return df_copy, column_mapping

def _match_logic_with_fallback(main_df, geology_df):
    """
    Core logic for merging geology data into main_df with automatic column detection.
    Tolerant to missing 'To_mbgl' column and various column naming conventions.
    """

    # Standardize column names for both DataFrames
    try:
        main_df_std, main_mapping = _standardize_dataframe_columns(main_df, "main_df")
        geology_df_std, geo_mapping = _standardize_dataframe_columns(geology_df, "geology_df")
    except ValueError as e:
        raise ValueError(f"Column detection failed: {str(e)}")

    matched_rows = []

    # Convert required 'From_mbgl' column and coerce errors
    main_df_std['From_mbgl'] = pd.to_numeric(main_df_std['From_mbgl'], errors='coerce')
    geology_df_std['From_mbgl'] = pd.to_numeric(geology_df_std['From_mbgl'], errors='coerce')

    # Convert optional 'To_mbgl' column IF IT EXISTS, coercing errors
    if 'To_mbgl' in main_df_std.columns:
        main_df_std['To_mbgl'] = pd.to_numeric(main_df_std['To_mbgl'], errors='coerce')
    if 'To_mbgl' in geology_df_std.columns:
        geology_df_std['To_mbgl'] = pd.to_numeric(geology_df_std['To_mbgl'], errors='coerce')

    # --- Pre-group geology data ---
    # Determine if geology can define intervals (requires To_mbgl column to exist)
    can_do_interval_geo = 'To_mbgl' in geology_df_std.columns

    # Essential cols for filtering geology now include Hole_ID and From_mbgl
    geo_essential_cols = ['Hole_ID', 'From_mbgl']
    if can_do_interval_geo:
        geo_essential_cols.append('To_mbgl')  # Add To_mbgl if it exists

    # Drop rows where any essential available columns are NA
    geology_df_valid = geology_df_std.dropna(subset=geo_essential_cols)

    # Ensure geology intervals are valid IF To_mbgl exists
    if can_do_interval_geo:
        geology_df_valid = geology_df_valid[geology_df_valid['To_mbgl'] > geology_df_valid['From_mbgl']]

    # Group valid geology data if any exists
    if not geology_df_valid.empty:
        try:
            geo_grouped = geology_df_valid.groupby('Hole_ID')
            valid_hole_ids = set(geo_grouped.groups.keys())
        except KeyError:
            geo_grouped = None
            valid_hole_ids = set()
    else:
        geo_grouped = None
        valid_hole_ids = set()

    # --- Main Loop ---
    for i, row in main_df_std.iterrows():
        # Get original row data to preserve original column names in output
        original_row = main_df.iloc[i].to_dict()

        hole_id = row.get('Hole_ID', None)
        main_from = row.get('From_mbgl', np.nan)
        main_to = row.get('To_mbgl', np.nan)

        # Check minimum requirements
        if pd.isna(hole_id) or hole_id not in valid_hole_ids or pd.isna(main_from) or geo_grouped is None:
            matched_rows.append(original_row)
            continue

        # Get relevant geology subset for this Hole_ID
        sub_geo_indices = geo_grouped.groups.get(hole_id)
        if sub_geo_indices is None:
            matched_rows.append(original_row)
            continue
        sub_geo_df = geology_df_valid.loc[sub_geo_indices]

        best_geo_data_to_merge = None

        # --- Case 1: Attempt Overlap Calc ---
        can_do_overlap = (not pd.isna(main_to) and main_to > main_from and can_do_interval_geo)

        if can_do_overlap:
            overlaps = []
            for idx, geo_row in sub_geo_df.iterrows():
                g_from = geo_row['From_mbgl']
                g_to = geo_row['To_mbgl']
                overlap_len = max(0, min(main_to, g_to) - max(main_from, g_from))
                overlaps.append((idx, overlap_len))

            if overlaps:
                best_idx, best_overlap = max(overlaps, key=lambda x: x[1])
                if best_overlap > 1e-9:
                    best_geo_data_to_merge = sub_geo_df.loc[best_idx]

        # --- Case 2: Attempt Point-in-Interval ---
        elif best_geo_data_to_merge is None and can_do_interval_geo:
            for idx, geo_row in sub_geo_df.iterrows():
                g_from = geo_row['From_mbgl']
                g_to = geo_row['To_mbgl']
                if g_from <= main_from < g_to:
                    best_geo_data_to_merge = geo_row
                    break

        # --- Merge data if a match was found ---
        if best_geo_data_to_merge is not None:
            # Get original geology row to preserve original column names
            original_geo_idx = best_geo_data_to_merge.name
            original_geo_row = geology_df.loc[original_geo_idx]

            for col in geology_df.columns:
                if col != geo_mapping['hole_id']:  # Skip the hole ID column
                    original_row[col + '_geo'] = original_geo_row.get(col, pd.NA)

        matched_rows.append(original_row)

    return pd.DataFrame(matched_rows)

def test_column_detection():
    """
    Test function to demonstrate the robust column detection capabilities.
    """
    # Test data with different column naming conventions
    main_data = {
        'BH_ID': ['DH001', 'DH002', 'DH003'],
        'From (m)': [0, 5, 10],
        'To (m)': [5, 10, 15],
        'Sample_Type': ['Core', 'Cutting', 'Core']
    }

    geo_data = {
        'Borehole_ID': ['DH001', 'DH001', 'DH002', 'DH002', 'DH003'],
        'from': [0, 3, 5, 8, 10],
        'to': [3, 6, 8, 12, 13],
        'Rock_Type': ['Granite', 'Schist', 'Granite', 'Shale', 'Limestone']
    }

    main_df = pd.DataFrame(main_data)
    geology_df = pd.DataFrame(geo_data)

    print("Original DataFrames:")
    print("Main DF:")
    print(main_df)
    print("\nGeology DF:")
    print(geology_df)

    # Test the merge function
    try:
        result = merge_intervals(main_df, geology_df)
        print("\nMerged Result:")
        print(result)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

