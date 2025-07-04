"""
Data Processing Utilities for Geotechnical Analysis Tool

This module handles data loading, validation, filtering, and processing operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Optional streamlit import
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    # Create mock streamlit functions for testing
    class MockStreamlit:
        @staticmethod
        def cache_data(func):
            return func
        @staticmethod
        def warning(msg):
            print(f"WARNING: {msg}")
    st = MockStreamlit()


def get_numerical_properties_smart(df: pd.DataFrame, include_spatial: bool = False) -> List[str]:
    """
    Intelligently detect numerical properties suitable for analysis using flexible regex patterns.
    
    Args:
        df: DataFrame to analyze
        include_spatial: Whether to include spatial columns (Chainage, coordinates)
        
    Returns:
        List[str]: List of numerical column names organized by property type
    """
    import re
    from typing import Dict, Set
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # System columns to exclude (unless specifically requested)
    exclude_patterns = [
        r'^HOLE[_\s]*ID',
        r'^SAMPLE[_\s]*ID', 
        r'^FROM[_\s]*MBGL',
        r'^TO[_\s]*MBGL',
    ]
    
    # Additional exclusion criteria
    def should_exclude_column(col_name: str) -> bool:
        """Check if column should be excluded based on name patterns"""
        # Exclude columns that are purely numerical (like "1", "2", "3")
        if col_name.strip().isdigit():
            return True
        
        # Exclude columns containing "?" (test identifier columns)
        if '?' in col_name:
            return True
            
        return False
    
    if not include_spatial:
        exclude_patterns.extend([
            r'^CHAINAGE',
            r'^EASTING',
            r'^NORTHING',
            r'^LATITUDE',
            r'^LONGITUDE',
        ])
    
    # Engineering property patterns (tiered by confidence)
    property_patterns = {
        'strength': [
            # Specific patterns (with units) - higher priority
            r'UCS.*MPA',              # UCS (MPa), UCS_MPa, etc.
            r'IS50[AD]?.*MPA',        # Is50a (MPa), Is50d (MPa), etc.
            r'CBR.*%',                # CBR (%), CBR_percent, etc.
            r'SPT.*N.*VALUE',         # SPT N Value, SPT_N_Value, etc.
            
            # General patterns (base names) - catch-all
            r'^UCS$',                 # Just "UCS"
            r'UCS[^A-Z]*$',          # UCS followed by non-letters
            r'^IS50[AD]?$',          # Just "Is50a", "Is50d", "Is50"
            r'IS50[AD]?[^A-Z]*$',    # Is50 with modifiers
            r'^CBR$',                 # Just "CBR"
            r'CBR[^A-Z]*$',          # CBR with modifiers
            r'^SPT$',                 # Just "SPT"
            r'SPT.*N$',              # SPT_N, SPT-N, etc.
        ],
        'index': [
            # Specific patterns
            r'LL.*%',                 # LL (%), LL_percent, etc.
            r'PL.*%',                 # PL (%), PL_percent, etc.
            r'PI.*%',                 # PI (%), PI_percent, etc.
            r'LS.*%',                 # LS (%), LS_percent, etc.
            
            # General patterns
            r'^LL$',                  # Just "LL"
            r'^PL$',                  # Just "PL"  
            r'^PI$',                  # Just "PI"
            r'^LS$',                  # Just "LS"
            r'LIQUID.*LIMIT',         # Liquid Limit variations
            r'PLASTIC.*LIMIT',        # Plastic Limit variations
            r'PLASTICITY.*INDEX',     # Plasticity Index variations
        ],
        'moisture': [
            # Specific patterns
            r'MC.*%',                 # MC_%, MC (%), etc.
            r'MOISTURE.*%',           # Moisture Content (%), etc.
            r'WATER.*CONTENT.*%',     # Water Content (%), etc.
            
            # General patterns
            r'^MC$',                  # Just "MC"
            r'MOISTURE',              # Any moisture-related
            r'WATER.*CONTENT',        # Water Content (no %)
        ],
        'density': [
            r'DENSITY.*MG',           # Density (Mg/m3), etc.
            r'UNIT.*WEIGHT',          # Unit Weight, etc.
            r'BULK.*DENSITY',         # Bulk Density, etc.
            r'^DENSITY$',             # Just "Density"
        ],
        'emerson': [
            r'EMERSON.*CLASS',        # Emerson Class, etc.
            r'^EMERSON$',             # Just "Emerson"
        ],
        'wpi': [
            r'^WPI$',                 # Just "WPI"
            r'WEIGHTED.*PLASTICITY',  # Full name variations
        ]
    }
    
    # Categorize columns by patterns
    categorized = {category: [] for category in property_patterns.keys()}
    categorized['other'] = []
    matched_cols: Set[str] = set()
    
    # First, exclude system columns and unwanted patterns
    filtered_cols = []
    for col in numeric_cols:
        col_upper = col.upper()
        exclude = False
        
        # Check system patterns
        for pattern in exclude_patterns:
            if re.search(pattern, col_upper):
                exclude = True
                break
        
        # Check additional exclusion criteria
        if not exclude and should_exclude_column(col):
            exclude = True
        
        if not exclude:
            filtered_cols.append(col)
    
    # Apply pattern matching
    for category, patterns in property_patterns.items():
        for col in filtered_cols:
            if col in matched_cols:
                continue
                
            col_upper = col.upper()
            for pattern in patterns:
                if re.search(pattern, col_upper):
                    categorized[category].append(col)
                    matched_cols.add(col)
                    break
    
    # Add remaining numeric columns to 'other'
    for col in filtered_cols:
        if col not in matched_cols:
            categorized['other'].append(col)
    
    # Flatten in priority order
    priority_order = ['strength', 'index', 'moisture', 'density', 'emerson', 'wpi', 'other']
    result = []
    for category in priority_order:
        result.extend(categorized[category])
    
    return result


@st.cache_data
def load_and_validate_data(file) -> pd.DataFrame:
    """
    Load and validate laboratory data file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        pd.DataFrame: Validated laboratory data
        
    Raises:
        ValueError: If data validation fails
    """
    try:
        # Load Excel file
        df = pd.read_excel(file)
        
        # Basic validation
        if df.empty:
            raise ValueError("File is empty")
        
        # Check for required columns
        required_cols = ['Hole_ID', 'Type', 'From_mbgl', 'Geology_Orgin']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Optional columns missing: {missing_cols}")
        
        # Validate test identifier columns
        test_columns = [col for col in df.columns if col.endswith('?')]
        if not test_columns:
            st.warning("No test identifier columns found (columns ending with '?')")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")


def extract_test_columns(df: pd.DataFrame, test_name: str) -> List[str]:
    """
    Extract data columns for a specific test type following the standard pattern.
    
    Args:
        df: Laboratory data DataFrame
        test_name: Name of the test (without '?')
        
    Returns:
        List[str]: List of data column names for the test
    """
    identifier_col = f"{test_name}?"
    if identifier_col not in df.columns:
        return []
    
    # Find the position of the identifier column
    all_cols = df.columns.tolist()
    start_idx = all_cols.index(identifier_col) + 1
    
    # Find the next identifier column to determine where this test's columns end
    end_idx = len(all_cols)
    for i in range(start_idx, len(all_cols)):
        if all_cols[i].endswith('?'):
            end_idx = i
            break
    
    return all_cols[start_idx:end_idx]


def get_test_availability(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get availability count for each test type by counting unique tests.
    
    Args:
        df: Laboratory data DataFrame
        
    Returns:
        Dict[str, int]: Test type names and their unique test counts
    """
    test_availability = {}
    
    # Find all test identifier columns
    test_columns = [col for col in df.columns if col.endswith('?')]
    
    for test_col in test_columns:
        test_name = test_col.replace('?', '')
        # Filter for records where test is available (value is 'Y')
        test_data = df[df[test_col] == 'Y']
        # Count unique tests by Hole_ID and From_mbgl
        count = len(test_data.drop_duplicates(subset=['Hole_ID', 'From_mbgl']))
        test_availability[test_name] = count
    
    return test_availability


def apply_global_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply global filters to the dataset.
    
    Args:
        df: Laboratory data DataFrame
        filters: Dictionary of filter parameters
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Apply chainage filter
    if filters.get('chainage_range') and 'Chainage' in filtered_df.columns:
        min_chainage, max_chainage = filters['chainage_range']
        
        # Auto-clamp the filter values to actual data boundaries
        data_min = filtered_df['Chainage'].min()
        data_max = filtered_df['Chainage'].max()
        
        # Use the intersection of user range and data range
        effective_min = max(min_chainage, data_min)
        effective_max = min(max_chainage, data_max)
        
        filtered_df = filtered_df[
            (filtered_df['Chainage'] >= effective_min) & 
            (filtered_df['Chainage'] <= effective_max)
        ]
    
    # Apply depth filter
    if filters.get('depth_range') and 'From_mbgl' in filtered_df.columns:
        min_depth, max_depth = filters['depth_range']
        filtered_df = filtered_df[
            (filtered_df['From_mbgl'] >= min_depth) & 
            (filtered_df['From_mbgl'] <= max_depth)
        ]
    
    # Apply geology filter
    if filters.get('selected_geology') and 'Geology_Orgin' in filtered_df.columns:
        selected_geology = filters['selected_geology']
        if selected_geology:  # Only apply if something is selected
            filtered_df = filtered_df[filtered_df['Geology_Orgin'].isin(selected_geology)]
    
    # Apply consistency filter
    if filters.get('selected_consistency') and 'Consistency' in filtered_df.columns:
        selected_consistency = filters['selected_consistency']
        if selected_consistency:  # Only apply if something is selected
            filtered_df = filtered_df[filtered_df['Consistency'].isin(selected_consistency)]
    
    return filtered_df


def create_test_dataframe(df: pd.DataFrame, test_name: str, id_columns: List[str], data_columns: List[str]) -> pd.DataFrame:
    """
    Create a test-specific dataframe including the test identifier column.
    
    Args:
        df: Laboratory data DataFrame
        test_name: Name of the test
        id_columns: List of ID column names
        data_columns: List of test-specific data column names
        
    Returns:
        pd.DataFrame: Test-specific dataframe
    """
    condition = df[f'{test_name}?'] == 'Y'
    
    # Only include columns that actually exist in the dataframe
    available_id_columns = [col for col in id_columns if col in df.columns]
    available_data_columns = [col for col in data_columns if col in df.columns]
    
    # Include the test identifier column along with available ID and data columns
    columns_to_include = available_id_columns + [f'{test_name}?'] + available_data_columns
    
    return df[condition][columns_to_include]


def get_standard_id_columns(df: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Get the standard ID columns used across all test types.
    If a dataframe is provided, uses dynamic detection. Otherwise returns hardcoded list.
    
    Args:
        df: Optional dataframe to detect ID columns from
    
    Returns:
        List[str]: List of ID column names
    """
    if df is not None:
        # Use dynamic detection when dataframe is available
        return get_id_columns_from_data(df)
    else:
        # Fallback to original hardcoded list from Jupyter notebook
        return [
            'Hole_ID', 'Type', 'From_mbgl', 'To_mbgl', 'Chainage', 
            'Surface RL (m AHD)', 'BH Depth (m)', 'Geology_Orgin', 
            'Map_symbol', 'Consistency', 'Report'
        ]


def get_id_columns_from_data(df: pd.DataFrame) -> List[str]:
    """
    Dynamically get ID columns by finding all columns before the first test identifier.
    ID columns are all columns that appear before the first column ending with '?'.
    
    Args:
        df: Laboratory data DataFrame
        
    Returns:
        List[str]: List of ID column names found before first test column
    """
    all_columns = df.columns.tolist()
    
    # Find the index of the first test identifier column (ends with '?')
    first_test_idx = None
    for idx, col in enumerate(all_columns):
        if col.endswith('?'):
            first_test_idx = idx
            break
    
    # If no test columns found, return all columns (shouldn't happen with valid data)
    if first_test_idx is None:
        return all_columns
    
    # Return all columns before the first test column
    return all_columns[:first_test_idx]


def get_dynamic_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Dynamically identify ID columns from the actual data.
    ID columns are typically at the beginning of the dataset before test data starts.
    
    Args:
        df: Laboratory data DataFrame
        
    Returns:
        List[str]: List of actual ID column names found in the data
    """
    all_columns = df.columns.tolist()
    
    # Standard ID patterns to look for
    id_patterns = [
        'hole_id', 'hole', 'borehole', 'id',
        'type', 'sample_type', 
        'from_mbgl', 'to_mbgl', 'depth', 'from', 'to',
        'chainage', 'station', 'distance',
        'northing', 'easting', 'north', 'east', 'x', 'y',
        'latitude', 'longitude', 'lat', 'lon',
        'surface', 'rl', 'elevation', 'level',
        'geology', 'description', 'material',
        'consistency', 'condition', 'state',
        'report', 'reference', 'project'
    ]
    
    # Find columns that match ID patterns
    id_columns = []
    for col in all_columns:
        col_clean = col.lower().replace('(', '').replace(')', '').replace('_', '').replace(' ', '').replace('-', '')
        if any(pattern in col_clean for pattern in id_patterns):
            id_columns.append(col)
        
        # Stop looking once we hit the first test identifier column (ends with '?')
        if col.endswith('?'):
            break
    
    # Include standard hardcoded columns that exist in the data (avoid recursive call)
    hardcoded_standard_columns = [
        'Hole_ID', 'Type', 'From_mbgl', 'To_mbgl', 'Chainage', 
        'Surface RL (m AHD)', 'BH Depth (m)', 'Geology_Orgin', 
        'Map_symbol', 'Consistency', 'Report'
    ]
    for col in hardcoded_standard_columns:
        if col in all_columns and col not in id_columns:
            id_columns.append(col)
    
    return id_columns


@st.cache_data
def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive data summary statistics.
    
    Args:
        df: Laboratory data DataFrame
        
    Returns:
        Dict[str, Any]: Summary statistics
    """
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'test_types': [],
        'geology_distribution': {},
        'depth_range': {},
        'chainage_range': {}
    }
    
    # Test types
    test_columns = [col for col in df.columns if col.endswith('?')]
    for test_col in test_columns:
        test_name = test_col.replace('?', '')
        count = (df[test_col] == 'Y').sum()
        summary['test_types'].append({'name': test_name, 'count': count})
    
    # Geology distribution
    if 'Geology_Orgin' in df.columns:
        summary['geology_distribution'] = df['Geology_Orgin'].value_counts().to_dict()
    
    # Depth range
    if 'From_mbgl' in df.columns:
        summary['depth_range'] = {
            'min': df['From_mbgl'].min(),
            'max': df['From_mbgl'].max(),
            'mean': df['From_mbgl'].mean()
        }
    
    # Chainage range
    if 'Chainage' in df.columns:
        summary['chainage_range'] = {
            'min': df['Chainage'].min(),
            'max': df['Chainage'].max(),
            'mean': df['Chainage'].mean()
        }
    
    return summary


def validate_test_data(df: pd.DataFrame, test_name: str) -> Tuple[bool, str]:
    """
    Validate if sufficient data exists for a specific test analysis.
    
    Args:
        df: Laboratory data DataFrame
        test_name: Name of the test
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    identifier_col = f"{test_name}?"
    
    if identifier_col not in df.columns:
        return False, f"Test type '{test_name}' not found in data"
    
    available_records = (df[identifier_col] == 'Y').sum()
    
    if available_records == 0:
        return False, f"No records available for {test_name} analysis"
    
    if available_records < 5:
        return False, f"Insufficient data for {test_name} analysis ({available_records} records, minimum 5 required)"
    
    return True, f"{available_records} records available for {test_name} analysis"