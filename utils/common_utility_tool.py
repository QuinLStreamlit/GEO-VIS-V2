"""
Common Utility Tools

This module contains shared utility functions used across multiple analysis modules.
It provides common functionality to avoid code duplication and maintain consistency.
Includes centralized default parameters for all analysis types.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Optional, Tuple, Dict, Any

# Import required functions from data_processing
try:
    from .data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_numerical_properties_smart
except ImportError:
    try:
        from data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_numerical_properties_smart
    except ImportError:
        # Fallback if data_processing is not available
        extract_test_columns = None
        create_test_dataframe = None
        get_standard_id_columns = None
        get_numerical_properties_smart = None


def calculate_map_zoom_and_center(lat_data, lon_data):
    """
    Calculate appropriate zoom level and center point for map based on data bounds.
    
    Args:
        lat_data: Array-like of latitude values
        lon_data: Array-like of longitude values
        
    Returns:
        tuple: (zoom_level, center_dict)
    """
    import numpy as np
    
    # Calculate center
    lat_center = float(np.mean(lat_data))
    lon_center = float(np.mean(lon_data))
    
    # Calculate the span of data
    lat_span = float(np.max(lat_data) - np.min(lat_data))
    lon_span = float(np.max(lon_data) - np.min(lon_data))
    
    # Calculate appropriate zoom level based on span
    # This formula provides a good approximation for most cases
    max_span = max(lat_span, lon_span)
    if max_span > 0:
        # Adjust zoom based on span - larger spans need lower zoom
        if max_span > 10:
            zoom_level = 4
        elif max_span > 5:
            zoom_level = 5
        elif max_span > 2:
            zoom_level = 6
        elif max_span > 1:
            zoom_level = 7
        elif max_span > 0.5:
            zoom_level = 8
        elif max_span > 0.2:
            zoom_level = 9
        elif max_span > 0.1:
            zoom_level = 10
        elif max_span > 0.05:
            zoom_level = 11
        else:
            zoom_level = 12
    else:
        zoom_level = 12
    
    return zoom_level, {'lat': lat_center, 'lon': lon_center}


def get_numerical_properties(df: pd.DataFrame, include_spatial: bool = False) -> List[str]:
    """
    Get list of numerical properties suitable for spatial analysis using smart detection.
    
    Args:
        df: DataFrame to analyze
        include_spatial: Whether to include spatial columns (Chainage, coordinates)
        
    Returns:
        List[str]: List of numerical column names organized by property type
    """
    if get_numerical_properties_smart is not None:
        return get_numerical_properties_smart(df, include_spatial=include_spatial)
    else:
        # Fallback implementation if data_processing is not available
        numeric_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                if include_spatial or col not in ['Chainage', 'Easting', 'Northing', 'X', 'Y', 'Longitude', 'Latitude']:
                    numeric_cols.append(col)
        return numeric_cols


def filter_valid_chainage_data(df: pd.DataFrame, property_col: str) -> pd.DataFrame:
    """
    Filter data to include only records with valid chainage and property values.
    
    Args:
        df: DataFrame to filter
        property_col: Property column name
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if 'Chainage' not in df.columns or property_col not in df.columns:
        return pd.DataFrame()
    
    return df.dropna(subset=['Chainage', property_col])


def parse_tuple(value_str: str, default: tuple) -> tuple:
    """
    Parse tuple string safely with fallback to default.
    
    Args:
        value_str: String representation of tuple like "(10, 6)"
        default: Default tuple to return if parsing fails
        
    Returns:
        tuple: Parsed tuple or default
    """
    try:
        # Remove spaces and evaluate the string as a tuple
        clean_str = value_str.strip()
        if clean_str.startswith('(') and clean_str.strip().endswith(')'):
            return eval(clean_str)
        else:
            return default
    except:
        return default


def get_categorical_properties(df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical properties suitable for grouping/faceting.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List[str]: List of categorical column names
    """
    categorical_cols = []
    
    # Common categorical properties from Jupyter notebook
    target_categories = [
        'Geology_Orgin', 'Consistency', 'Hole_ID', 'Type', 
        'Material', 'Formation', 'map_symbol', 'Map_symbol'
    ]
    
    for col in df.columns:
        if col in target_categories:
            # Check if column has reasonable number of unique values for grouping
            unique_count = df[col].nunique()
            if 1 < unique_count <= 20:  # Reasonable for grouping
                categorical_cols.append(col)
    
    return categorical_cols


def find_map_symbol_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the map symbol column using flexible pattern matching.
    
    Args:
        df: DataFrame to search in
        
    Returns:
        str: Column name if found, None otherwise
    """
    # Common patterns for map symbol columns
    map_symbol_patterns = [
        'map_symbol', 'map symbol', 'mapsymbol',
        'geology', 'geological', 'geo',
        'material', 'mat', 'rock_type', 'rocktype',
        'formation', 'unit', 'lithology', 'lith',
        'symbol', 'code', 'group', 'class'
    ]
    
    # Try exact matches first (case-insensitive)
    for col in df.columns:
        if col.lower() in ['map_symbol', 'map symbol', 'mapsymbol']:
            return col
    
    # Try pattern matching
    for pattern in map_symbol_patterns:
        for col in df.columns:
            col_clean = col.lower().replace('_', '').replace(' ', '')
            pattern_clean = pattern.replace('_', '').replace(' ', '')
            if pattern_clean in col_clean:
                return col
                
    return None


def detect_cbr_swell_column(df: pd.DataFrame) -> Optional[str]:
    """
    Smart detection of CBR Swell (%) column with flexible naming patterns.
    
    Args:
        df: DataFrame to search in
        
    Returns:
        str: Column name if found, None otherwise
    """
    if df is None or df.empty:
        return None
    
    # Define patterns for CBR Swell columns (most specific first)
    cbr_swell_patterns = [
        # Exact matches (case-insensitive)
        r'^CBR\s*Swell\s*\(%\)$',
        r'^CBR\s*Swell\s*%$',
        r'^CBR_Swell_\(%\)$',
        r'^CBR_Swell_%$',
        
        # Common variations with parentheses
        r'^CBR\s*Swell\s*\(.*%.*\)$',
        r'^CBR_Swell_\(.*%.*\)$',
        
        # Without parentheses but with % indicator
        r'^CBR\s*Swell.*%.*$',
        r'^CBR_Swell.*%.*$',
        
        # Just CBR Swell (assuming % if no other CBR columns)
        r'^CBR\s*Swell$',
        r'^CBR_Swell$',
        
        # More flexible patterns
        r'.*CBR.*Swell.*%.*',
        r'.*CBR.*SWELL.*%.*',
        r'.*cbr.*swell.*%.*',
    ]
    
    # Try each pattern
    for pattern in cbr_swell_patterns:
        for col in df.columns:
            if re.match(pattern, str(col).strip(), re.IGNORECASE):
                # Additional validation - check if column contains numeric data
                if _is_numeric_column(df, col):
                    return col
    
    return None


def detect_wpi_column(df: pd.DataFrame) -> Optional[str]:
    """
    Smart detection of WPI (Weighted Plasticity Index) column with flexible naming patterns.
    
    Args:
        df: DataFrame to search in
        
    Returns:
        str: Column name if found, None otherwise
    """
    if df is None or df.empty:
        return None
    
    # Define patterns for WPI columns (most specific first)
    wpi_patterns = [
        # Exact matches (case-insensitive)
        r'^WPI$',
        r'^W\.P\.I\.?$',
        r'^W_P_I$',
        
        # Full name variations
        r'^Weighted\s*Plasticity\s*Index$',
        r'^Weighted_Plasticity_Index$',
        r'^WeightedPlasticityIndex$',
        
        # Abbreviation variations
        r'^WPI\s*\(.*\)$',  # WPI with any description in parentheses
        r'^W\.P\.I\s*\(.*\)$',
        
        # Flexible patterns
        r'.*WPI.*',
        r'.*W\.P\.I.*',
        r'.*Weighted.*Plasticity.*Index.*',
        r'.*weighted.*plasticity.*index.*',
    ]
    
    # Try each pattern
    for pattern in wpi_patterns:
        for col in df.columns:
            if re.match(pattern, str(col).strip(), re.IGNORECASE):
                # Additional validation - check if column contains numeric data
                if _is_numeric_column(df, col):
                    return col
    
    return None


def get_cbr_swell_column_candidates(df: pd.DataFrame) -> List[str]:
    """
    Get all potential CBR Swell column candidates for user selection.
    
    Args:
        df: DataFrame to search in
        
    Returns:
        List[str]: List of potential column names
    """
    if df is None or df.empty:
        return []
    
    candidates = []
    
    # Look for columns containing 'CBR' and 'Swell' keywords
    for col in df.columns:
        col_lower = str(col).lower()
        if ('cbr' in col_lower and 'swell' in col_lower) or ('cbr' in col_lower and '%' in col_lower):
            if _is_numeric_column(df, col):
                candidates.append(col)
    
    # Also include any column with 'CBR' if no specific swell columns found
    if not candidates:
        for col in df.columns:
            col_lower = str(col).lower()
            if 'cbr' in col_lower and _is_numeric_column(df, col):
                candidates.append(col)
    
    return candidates


def get_wpi_column_candidates(df: pd.DataFrame) -> List[str]:
    """
    Get all potential WPI column candidates for user selection.
    
    Args:
        df: DataFrame to search in
        
    Returns:
        List[str]: List of potential column names
    """
    if df is None or df.empty:
        return []
    
    candidates = []
    
    # Look for columns containing WPI keywords
    for col in df.columns:
        col_lower = str(col).lower()
        if ('wpi' in col_lower or 
            'w.p.i' in col_lower or 
            ('weighted' in col_lower and 'plasticity' in col_lower) or
            ('w_p_i' in col_lower)):
            if _is_numeric_column(df, col):
                candidates.append(col)
    
    return candidates


def _is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """
    Check if a column contains numeric data (helper function).
    
    Args:
        df: DataFrame containing the column
        col: Column name to check
        
    Returns:
        bool: True if column contains numeric data
    """
    try:
        if col not in df.columns:
            return False
        
        # Check if column dtype is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            return True
        
        # Try to convert a sample to numeric
        sample = df[col].dropna().head(10)
        if sample.empty:
            return False
        
        # Try to convert to numeric
        pd.to_numeric(sample, errors='raise')
        return True
    except:
        return False


def extract_emerson_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Emerson test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: Emerson-specific dataframe
    """
    if (get_standard_id_columns is None or 
        extract_test_columns is None or 
        create_test_dataframe is None):
        # Fallback if data_processing functions are not available
        emerson_cols = [col for col in df.columns if 'emerson' in col.lower()]
        if not emerson_cols:
            return pd.DataFrame()
        
        # Get basic ID columns
        id_cols = [col for col in ['Hole_ID', 'From_mbgl', 'To_mbgl', 'Type', 'Chainage'] 
                  if col in df.columns]
        return df[id_cols + emerson_cols].dropna(subset=emerson_cols, how='all')
    
    # Use the proper data_processing functions
    id_columns = get_standard_id_columns(df)
    emerson_columns = extract_test_columns(df, 'Emerson')
    
    if not emerson_columns:
        return pd.DataFrame()
    
    return create_test_dataframe(df, 'Emerson', id_columns, emerson_columns)


def get_default_parameters(analysis_type: str):
    """
    Get default parameters for specific analysis types.
    Uses Functions file names as keys for perfect 1:1 mapping.
    
    Args:
        analysis_type: Type of analysis (e.g., 'plot_psd', 'plot_UCS_vs_depth')
        
    Returns:
        Dict[str, Any]: Default parameters for the analysis
    """
    defaults = {
        'plot_psd': {
            'figsize': '(10, 6)',
            'color_scheme': 'Set2',
            'line_width': 1.5,
            'marker_size': 4,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'Particle Size Distribution',
            'x_label': 'Particle Size (mm)',
            'y_label': 'Percentage Passing (%)',
            'show_classification': True,
            'show_statistics': True
        },
        'plot_atterberg_chart': {
            'figsize': '(8, 6)',
            'color_scheme': 'Set2',
            'marker_size': 40,
            'alpha': 0.8,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'Plasticity Chart (BS 1377)',
            'x_label': 'Liquid Limit (%)',
            'y_label': 'Plasticity Index (%)',
            'xlim': '(0, 100)',
            'ylim': '(0, 80)',
            'show_a_line': True,
            'show_u_line': True,
            'show_classification_zones': True
        },
        'plot_SPT_vs_depth': {
            'figsize': '(8, 10)',
            'color_scheme': 'Set2',
            'marker_size': 50,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'SPT N-Value Profile',
            'x_label': 'SPT N-Value',
            'y_label': 'Depth (m bgl)',
            'invert_y': True,
            'show_statistics': True
        },
        'plot_UCS_vs_depth': {
            'figsize': '(12, 8)',
            'color_scheme': 'Set2',
            'point_size': 50,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'UCS vs Depth',
            'x_label': 'UCS (MPa)',
            'y_label': 'Depth (m bgl)',
            'show_trend': True,
            'show_stats': True,
            'log_scale': False
        },
        'plot_UCS_Is50': {
            'figsize': '(12, 8)',
            'color_scheme': 'Set2',
            'point_size': 50,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'UCS vs Is50',
            'x_label': 'Is50 (MPa)',
            'y_label': 'UCS (MPa)',
            'show_trend': True,
            'show_stats': True,
            'log_scale': False,
            'correlation_line': True
        },
        'plot_emerson_by_origin': {
            'figsize': '(12, 6)',
            'color_scheme': 'Set2',
            'marker_size': 50,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'Emerson Test Results by Origin'
        },
        'plot_engineering_property_vs_depth': {
            'figsize': '(10, 8)',
            'color_scheme': 'Set2',
            'marker_size': 50,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'Engineering Property vs Depth'
        },
        'plot_by_chainage': {
            'figsize': '(12, 6)',
            'color_scheme': 'Set2',
            'marker_size': 50,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'Property vs Chainage'
        },
        'plot_category_by_thickness': {
            'figsize': '(15, 7)',
            'color_scheme': 'Set2',
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'Distribution by Thickness'
        },
        'plot_histogram': {
            'figsize': '(10, 6)',
            'color_scheme': 'Set2',
            'bins': 20,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'Histogram Analysis'
        },
        'plot_CBR_swell_WPI_histogram': {
            'figsize': '(12, 8)',
            'color_scheme': 'Set2',
            'point_size': 50,
            'alpha': 0.7,
            'grid': True,
            'legend': True,
            'dpi': 300,
            'title': 'CBR Swell vs WPI Analysis'
        }
    }
    
    return defaults.get(analysis_type, {})


def get_color_schemes():
    """
    Get available color schemes for plots.
    
    Returns:
        List[str]: Available color scheme names
    """
    return ['Set2', 'tab10', 'viridis', 'plasma', 'Dark2', 'Paired', 'Set1', 'Set3']


def extract_ucs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract UCS test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: UCS-specific dataframe
    """
    if (get_standard_id_columns is None or 
        extract_test_columns is None or 
        create_test_dataframe is None):
        # Fallback if data_processing functions are not available
        ucs_cols = [col for col in df.columns if 'ucs' in col.lower()]
        if not ucs_cols:
            raise ValueError("No UCS data columns found")
        
        # Get basic ID columns
        id_cols = [col for col in ['Hole_ID', 'From_mbgl', 'To_mbgl', 'Type', 'Chainage'] 
                  if col in df.columns]
        return df[id_cols + ucs_cols].dropna(subset=ucs_cols, how='all')
    
    # Use the proper data_processing functions
    from .data_processing import get_id_columns_from_data
    id_columns = get_id_columns_from_data(df)
    ucs_columns = extract_test_columns(df, 'UCS')
    
    if not ucs_columns:
        raise ValueError("No UCS data columns found")
    
    return create_test_dataframe(df, 'UCS', id_columns, ucs_columns)


def extract_is50_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Is50 test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: Is50-specific dataframe
    """
    if (get_standard_id_columns is None or 
        extract_test_columns is None or 
        create_test_dataframe is None):
        # Fallback if data_processing functions are not available
        is50_cols = [col for col in df.columns if 'is50' in col.lower()]
        if not is50_cols:
            return pd.DataFrame()
        
        # Get basic ID columns
        id_cols = [col for col in ['Hole_ID', 'From_mbgl', 'To_mbgl', 'Type', 'Chainage'] 
                  if col in df.columns]
        return df[id_cols + is50_cols].dropna(subset=is50_cols, how='all')
    
    # Use the proper data_processing functions
    from .data_processing import get_id_columns_from_data
    id_columns = get_id_columns_from_data(df)
    is50_columns = extract_test_columns(df, 'Is_50')
    
    if not is50_columns:
        return pd.DataFrame()
    
    return create_test_dataframe(df, 'Is_50', id_columns, is50_columns)


def get_available_geologies(ucs_data: pd.DataFrame) -> List[str]:
    """
    Get available geological units from UCS data.
    
    Args:
        ucs_data: UCS dataframe
        
    Returns:
        List[str]: List of available geological units
    """
    if 'Geology_Orgin' in ucs_data.columns:
        return ucs_data['Geology_Orgin'].dropna().unique().tolist()
    return []


def filter_ucs_by_geology(ucs_data: pd.DataFrame, geology: str) -> pd.DataFrame:
    """
    Filter UCS data by geological unit.
    
    Args:
        ucs_data: UCS dataframe
        geology: Geological unit to filter by
        
    Returns:
        pd.DataFrame: Filtered UCS dataframe
    """
    if geology == "All" or 'Geology_Orgin' not in ucs_data.columns:
        return ucs_data.copy()
    
    return ucs_data[ucs_data['Geology_Orgin'] == geology].copy()


# Test the functions
if __name__ == "__main__":
    print("Testing default parameters...")
    ucs_defaults = get_default_parameters('plot_UCS_vs_depth')
    print(f"UCS defaults: {list(ucs_defaults.keys())}")
    
    color_schemes = get_color_schemes()
    print(f"Color schemes: {color_schemes}")