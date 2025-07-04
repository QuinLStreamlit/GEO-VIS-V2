"""
Particle Size Distribution (PSD) Analysis Module

This module handles PSD data processing, analysis, and visualization for geotechnical applications.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import pyproj
    HAS_PYPROJ = True
    PYPROJ_VERSION = pyproj.__version__
except ImportError as e:
    HAS_PYPROJ = False
    PYPROJ_VERSION = f"Import failed: {e}"
except Exception as e:
    HAS_PYPROJ = False
    PYPROJ_VERSION = f"Other error: {e}"

try:
    from .data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns
    from .plot_defaults import get_default_parameters, get_color_schemes
    from .dashboard_materials import store_material_plot
    HAS_PLOTTING_UTILS = True
except ImportError:
    # For standalone testing
    from data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns
    from plot_defaults import get_default_parameters, get_color_schemes
    try:
        from dashboard_materials import store_material_plot
        HAS_PLOTTING_UTILS = True
    except ImportError:
        HAS_PLOTTING_UTILS = False

# Import Functions from Functions folder
try:
    import sys
    import os
    functions_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Functions')
    if functions_path not in sys.path:
        sys.path.insert(0, functions_path)
    
    from plot_psd import plot_psd
    from parse_to_mm import parse_to_mm
    HAS_FUNCTIONS = True
except ImportError as e:
    HAS_FUNCTIONS = False
    print(f"Warning: Could not import Functions: {e}")



def extract_hydrometer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Hydrometer test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: Hydrometer-specific dataframe
    """
    # Use dynamic ID columns detection by passing dataframe to get_standard_id_columns
    id_columns = get_standard_id_columns(df)
    hydrometer_columns = extract_test_columns(df, 'Hydrometer')
    
    if not hydrometer_columns:
        return pd.DataFrame()
    
    return create_test_dataframe(df, 'Hydrometer', id_columns, hydrometer_columns)


def convert_psd_to_long_format(psd_hydrometer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PSD data to long format exactly like Jupyter notebook.
    
    Args:
        psd_hydrometer_data: Merged PSD and Hydrometer data
        
    Returns:
        pd.DataFrame: Long format PSD data
    """
    try:
        # Import parse_to_mm function from Functions folder
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        functions_path = os.path.join(current_dir, 'Functions')
        if functions_path not in sys.path:
            sys.path.insert(0, functions_path)
        
        from Functions.parse_to_mm import parse_to_mm
        
        # Get ID columns and additional columns for melting using dynamic detection
        id_columns = get_standard_id_columns(psd_hydrometer_data)
        additional_cols = []
        
        # Add test identifier columns if they exist
        for col in ['PSD?', 'Hydrometer?']:
            if col in psd_hydrometer_data.columns:
                additional_cols.append(col)
                
        # Add Soil Particle Density if exists
        if 'Soil Particle Density (t/m3)' in psd_hydrometer_data.columns:
            additional_cols.append('Soil Particle Density (t/m3)')
        
        # Melt to long format exactly like Jupyter notebook
        id_vars = id_columns + additional_cols
        psd_long_format = psd_hydrometer_data.melt(
            id_vars=id_vars,
            var_name='Sieve Size',
            value_name='Percentage passing (%)'
        )
        
        # Remove rows with NaN values
        psd_long_format = psd_long_format.dropna(subset=['Percentage passing (%)'])
        
        # Apply parse_to_mm function exactly like Jupyter notebook
        psd_long_format['Sieve_Size_mm'] = psd_long_format['Sieve Size'].apply(parse_to_mm)
        
        return psd_long_format
        
    except Exception as e:
        if HAS_STREAMLIT:
            st.error(f"Error converting PSD to long format: {e}")
        else:
            print(f"Error converting PSD to long format: {e}")
        return pd.DataFrame()


def extract_psd_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract PSD test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: PSD-specific dataframe
    """
    # Use dynamic ID columns detection by passing dataframe to get_standard_id_columns
    id_columns = get_standard_id_columns(df)
    psd_columns = extract_test_columns(df, 'PSD')
    
    if not psd_columns:
        raise ValueError("No PSD data columns found")
    
    return create_test_dataframe(df, 'PSD', id_columns, psd_columns)


def get_sieve_columns(psd_df: pd.DataFrame) -> List[str]:
    """
    Identify sieve size columns from PSD dataframe.
    
    Args:
        psd_df: PSD dataframe
        
    Returns:
        List[str]: List of sieve column names
    """
    # Standard sieve patterns - look for numeric columns that could be sieve sizes
    potential_sieves = []
    
    # Get list of columns that actually exist in the dataframe using dynamic detection
    standard_id_columns = get_standard_id_columns(psd_df)
    existing_id_columns = [col for col in standard_id_columns if col in psd_df.columns]
    
    for col in psd_df.columns:
        # Skip ID columns and test identifier
        if col in existing_id_columns or col == 'PSD?':
            continue
            
        # Look for columns that might represent sieve sizes
        # Common patterns: numbers, "mm", "Sieve_", etc.
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['sieve', 'mm', 'size', 'pass']):
            potential_sieves.append(col)
        elif col.replace('.', '').replace('_', '').isdigit():
            potential_sieves.append(col)
    
    return potential_sieves


def convert_to_long_format(psd_df: pd.DataFrame, sieve_columns: List[str]) -> pd.DataFrame:
    """
    Convert PSD data to long format for plotting.
    
    Args:
        psd_df: PSD dataframe in wide format
        sieve_columns: List of sieve column names
        
    Returns:
        pd.DataFrame: Long format dataframe
    """
    # Dynamically detect ID columns from the dataframe structure
    all_columns = psd_df.columns.tolist()
    id_columns = []
    
    # Find all columns before the first sieve column
    for col in all_columns:
        if col in sieve_columns:
            break
        id_columns.append(col)
    
    # Ensure we have PSD? and Hydrometer? columns if they exist
    for test_col in ['PSD?', 'Hydrometer?']:
        if test_col in psd_df.columns and test_col not in id_columns:
            id_columns.append(test_col)
    
    # Melt the dataframe
    psd_long = pd.melt(
        psd_df,
        id_vars=id_columns,
        value_vars=sieve_columns,
        var_name='Sieve_Size',
        value_name='Percent_Passing'
    )
    
    # Extract numeric sieve sizes for sorting
    psd_long['Sieve_Size_Numeric'] = psd_long['Sieve_Size'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Remove rows with missing values
    psd_long = psd_long.dropna(subset=['Percent_Passing'])
    
    return psd_long.sort_values(['Hole_ID', 'From_mbgl', 'Sieve_Size_Numeric'])


def calculate_psd_statistics(psd_long: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate PSD statistics (D10, D30, D50, etc.).
    
    Args:
        psd_long: Long format PSD dataframe
        
    Returns:
        pd.DataFrame: Statistics dataframe
    """
    stats_list = []
    
    # Group by sample (Hole_ID + depth)
    grouped = psd_long.groupby(['Hole_ID', 'From_mbgl'])
    
    for (hole_id, depth), group in grouped:
        if len(group) < 3:  # Need minimum points for interpolation
            continue
            
        # Sort by sieve size
        group_sorted = group.sort_values('Sieve_Size_Numeric')
        
        # Calculate percentiles
        sieve_sizes = group_sorted['Sieve_Size_Numeric'].values
        percent_passing = group_sorted['Percent_Passing'].values
        
        # Interpolate to find D10, D30, D50, D60
        try:
            d10 = np.interp(10, percent_passing, sieve_sizes)
            d30 = np.interp(30, percent_passing, sieve_sizes)
            d50 = np.interp(50, percent_passing, sieve_sizes)
            d60 = np.interp(60, percent_passing, sieve_sizes)
            
            # Calculate uniformity coefficient and coefficient of curvature
            cu = d60 / d10 if d10 > 0 else np.nan
            cc = (d30 ** 2) / (d60 * d10) if d60 > 0 and d10 > 0 else np.nan
            
            # Get other sample info
            sample_info = group.iloc[0]
            
            stats_list.append({
                'Hole_ID': hole_id,
                'From_mbgl': depth,
                'To_mbgl': sample_info.get('To_mbgl', np.nan),
                'Geology_Orgin': sample_info.get('Geology_Orgin', ''),
                'Consistency': sample_info.get('Consistency', ''),
                'D10': d10,
                'D30': d30,
                'D50': d50,
                'D60': d60,
                'Cu': cu,
                'Cc': cc,
                'Gravel_Percent': 100 - np.interp(4.75, sieve_sizes, percent_passing),
                'Sand_Percent': np.interp(4.75, sieve_sizes, percent_passing) - np.interp(0.075, sieve_sizes, percent_passing),
                'Fines_Percent': np.interp(0.075, sieve_sizes, percent_passing)
            })
        except Exception:
            continue
    
    return pd.DataFrame(stats_list)


def create_psd_distribution_plot(psd_long: pd.DataFrame, parameters: Dict[str, Any]):
    """
    Create PSD distribution curve plot.
    
    Args:
        psd_long: Long format PSD dataframe
        parameters: Plot parameters
        
    Returns:
        go.Figure: Plotly figure (or None if plotly not available)
    """
    if not HAS_PLOTLY:
        print("Warning: Plotly not available. Cannot create plot.")
        return None
        
    fig = go.Figure()
    
    # Get color scheme
    color_schemes = get_color_schemes()
    colors = color_schemes.get(parameters.get('color_scheme', 'geology_based'), color_schemes['geology_based'])
    
    # Group by the specified grouping variable
    group_by = parameters.get('group_by', 'geology')
    if group_by == 'geology' and 'Geology_Orgin' in psd_long.columns:
        group_col = 'Geology_Orgin'
    elif group_by == 'borehole':
        group_col = 'Hole_ID'
    else:
        group_col = 'Hole_ID'
    
    # Plot by groups
    grouped = psd_long.groupby([group_col, 'Hole_ID', 'From_mbgl'])
    color_idx = 0
    
    for (group_val, hole_id, depth), group_data in grouped:
        group_data_sorted = group_data.sort_values('Sieve_Size_Numeric')
        
        color = colors[color_idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=group_data_sorted['Sieve_Size_Numeric'],
            y=group_data_sorted['Percent_Passing'],
            mode='lines+markers',
            name=f"{hole_id} @ {depth}m ({group_val})",
            line=dict(width=parameters.get('line_width', 1.5), color=color),
            marker=dict(size=parameters.get('marker_size', 4), color=color),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Sieve Size: %{x:.3f} mm<br>' +
                         'Passing: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        color_idx += 1
    
    # Update layout
    fig.update_layout(
        title=parameters.get('title', 'Particle Size Distribution'),
        xaxis_title=parameters.get('x_label', 'Particle Size (mm)'),
        yaxis_title=parameters.get('y_label', 'Percentage Passing (%)'),
        width=parameters.get('figure_size', (10, 6))[0] * 100,
        height=parameters.get('figure_size', (10, 6))[1] * 100,
        showlegend=parameters.get('legend', True),
        hovermode='closest'
    )
    
    # Set axes
    if parameters.get('x_axis') == 'log_scale':
        fig.update_xaxes(type="log", range=[-2, 2])  # 0.01 to 100 mm
    
    fig.update_yaxes(range=[0, 100])
    
    # Add grid
    if parameters.get('grid', True):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def create_gradation_envelope(psd_long: pd.DataFrame, parameters: Dict[str, Any]):
    """
    Create gradation envelope plot.
    
    Args:
        psd_long: Long format PSD dataframe
        parameters: Plot parameters
        
    Returns:
        go.Figure: Plotly figure (or None if plotly not available)
    """
    if not HAS_PLOTLY:
        print("Warning: Plotly not available. Cannot create plot.")
        return None
        
    fig = go.Figure()
    
    # Calculate envelope for each sieve size
    sieve_stats = psd_long.groupby('Sieve_Size_Numeric')['Percent_Passing'].agg(['min', 'max', 'mean']).reset_index()
    
    # Add envelope bands
    fig.add_trace(go.Scatter(
        x=sieve_stats['Sieve_Size_Numeric'],
        y=sieve_stats['max'],
        mode='lines',
        name='Maximum',
        line=dict(color='rgba(0,100,80,0)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=sieve_stats['Sieve_Size_Numeric'],
        y=sieve_stats['min'],
        mode='lines',
        name='Envelope',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(0,100,80,0.5)', width=1)
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=sieve_stats['Sieve_Size_Numeric'],
        y=sieve_stats['mean'],
        mode='lines',
        name='Mean',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='PSD Gradation Envelope',
        xaxis_title='Particle Size (mm)',
        yaxis_title='Percentage Passing (%)',
        width=800,
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(type="log", range=[-2, 2])
    fig.update_yaxes(range=[0, 100])
    
    return fig


def parse_tuple_input(input_str: str, expected_length: int = 2, data_type=float, default_value=None):
    """
    Parse tuple input from string format.
    
    Args:
        input_str: String input like "(14, 9)" or "14, 9"
        expected_length: Expected number of elements
        data_type: Type to convert elements to
        default_value: Default value if parsing fails
        
    Returns:
        tuple: Parsed tuple or default value
    """
    try:
        # Clean the input string
        cleaned = input_str.strip().replace('(', '').replace(')', '')
        # Split by comma and convert to specified type
        values = [data_type(x.strip()) for x in cleaned.split(',')]
        
        if len(values) != expected_length:
            return default_value
            
        return tuple(values)
    except:
        return default_value


def apply_multi_column_filters(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    """
    Apply multiple column filters to PSD data.
    Empty lists mean "show all" for that category (following Jupyter notebook logic).
    
    Args:
        df: PSD long format dataframe
        filter_dict: Dictionary of filter parameters
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    filtered_df = df.copy()
    
    # Apply geology filter (empty list means show all)
    if filter_dict.get('geology') and len(filter_dict['geology']) > 0:
        filtered_df = filtered_df[filtered_df['Geology_Orgin'].isin(filter_dict['geology'])]
    
    # Apply consistency filter (empty list means show all)
    if filter_dict.get('consistency') and len(filter_dict['consistency']) > 0:
        filtered_df = filtered_df[filtered_df['Consistency'].isin(filter_dict['consistency'])]
    
    # Apply hole ID filter (empty list means show all)
    if filter_dict.get('hole_ids') and len(filter_dict['hole_ids']) > 0:
        filtered_df = filtered_df[filtered_df['Hole_ID'].isin(filter_dict['hole_ids'])]
    
    # Apply map symbol filter (empty list means show all)
    if filter_dict.get('map_symbols') and len(filter_dict['map_symbols']) > 0:
        filtered_df = filtered_df[filtered_df['Map_symbol'].isin(filter_dict['map_symbols'])]
    
    # Apply depth range filter
    if filter_dict.get('depth_range'):
        min_depth, max_depth = filter_dict['depth_range']
        filtered_df = filtered_df[
            (filtered_df['From_mbgl'] >= min_depth) & 
            (filtered_df['From_mbgl'] <= max_depth)
        ]
    
    return filtered_df


def get_filter_summary(filtered_data: pd.DataFrame, original_data: pd.DataFrame) -> str:
    """Generate summary of applied filters."""
    filtered_count = len(filtered_data)
    original_count = len(original_data)
    
    if filtered_count == original_count:
        return f"Showing all {original_count:,} data points"
    else:
        percentage = (filtered_count / original_count) * 100
        return f"Showing {filtered_count:,} of {original_count:,} data points ({percentage:.1f}%)"


def render_psd_analysis_tab(filtered_data: pd.DataFrame):
    """
    Render the PSD analysis tab in Streamlit.
    Uses original plotting functions from Functions folder exactly as in Jupyter notebook.
    
    Args:
        filtered_data: Filtered laboratory data
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render tab.")
        return
    
    try:
        # Helper function for parsing tuple inputs
        def parse_tuple(input_str, default):
            try:
                # Remove parentheses and split by comma
                cleaned = input_str.strip().replace('(', '').replace(')', '')
                values = [float(x.strip()) for x in cleaned.split(',')]
                return tuple(values) if len(values) == 2 else default
            except:
                return default
        
        # Extract PSD and Hydrometer data exactly like Jupyter notebook
        psd_data = extract_psd_data(filtered_data)
        
        if psd_data.empty:
            st.warning("No PSD data available with current filters.")
            return
        
        # Try to extract Hydrometer data for merging (following Jupyter notebook workflow)
        try:
            hydrometer_data = extract_hydrometer_data(filtered_data)
        except:
            hydrometer_data = pd.DataFrame()
        
        # Merge PSD and Hydrometer data exactly like Jupyter notebook
        if not hydrometer_data.empty:
            # Use dynamic ID columns from the original data
            id_column_list = get_standard_id_columns(filtered_data)
            
            # Filter to only columns that exist in both dataframes
            psd_columns = set(psd_data.columns)
            hydrometer_columns = set(hydrometer_data.columns)
            merge_columns = [col for col in id_column_list if col in psd_columns and col in hydrometer_columns]
            
            # Perform merge
            if merge_columns:
                psd_hydrometer = psd_data.merge(right=hydrometer_data, how='outer', on=merge_columns)
            else:
                # If no common columns, just concatenate
                st.warning("No common ID columns found between PSD and Hydrometer data. Combining data without merge.")
                psd_hydrometer = pd.concat([psd_data, hydrometer_data], ignore_index=True)
        else:
            psd_hydrometer = psd_data.copy()
        
        # Convert to long format exactly like Jupyter notebook
        psd_long_format = convert_psd_to_long_format(psd_hydrometer)
        
        if psd_long_format.empty:
            st.warning("No valid PSD data after processing.")
            return
        
        # Add custom CSS for white checkbox background
        st.markdown("""
        <style>
        /* Checkbox styling - multiple selectors to ensure compatibility */
        .stCheckbox > label > div[data-testid="stCheckbox"] > div {
            background-color: white !important;
        }
        .stCheckbox > label > div[data-testid="stCheckbox"] > div:checked {
            background-color: white !important;
        }
        .stCheckbox input[type="checkbox"]:checked {
            background-color: white !important;
        }
        div[data-testid="stCheckbox"] > label > div {
            background-color: white !important;
        }
        div[data-testid="stCheckbox"] > label > div[data-checked="true"] {
            background-color: white !important;
        }
        /* Alternative selectors for different Streamlit versions */
        .st-checkbox {
            background-color: white !important;
        }
        .st-checkbox:checked {
            background-color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Get proper column groups following Jupyter notebook pattern
        # Standard ID columns for major identifiers (hole ID, depth, etc.) using dynamic detection
        standard_id_columns = [col for col in get_standard_id_columns(psd_data) if col in psd_data.columns]
        
        # Processed columns from long format conversion (these are what the plotting function expects)
        processed_columns = ['Sieve_Size_mm', 'Percentage passing (%)']
        available_processed_cols = [col for col in processed_columns if col in psd_long_format.columns]
        
        # Smart column detection functions
        def find_column(patterns, columns, default=""):
            for pattern in patterns:
                for col in columns:
                    if pattern.lower() in col.lower():
                        return col
            return default
        
        # Find smart defaults within appropriate column groups
        hole_id_default = find_column(["hole_id", "hole", "borehole", "BH ID", "BH_ID", "Test ID"], standard_id_columns, "Hole_ID")
        depth_default = find_column(["from_mbgl", "depth", "from", "from (m)", "From_mbgl", "From (m)", "From(m)", "From (mbgl)", "From (mgbl)"], standard_id_columns, "From_mbgl")
        size_default = find_column(["sieve_size_mm", "size_mm", "sieve"], available_processed_cols, "Sieve_Size_mm")
        percent_default = find_column(["percentage passing", "percent_passing", "passing"], available_processed_cols, "Percentage passing (%)")
        
        # Helper function to get available values for filter types
        def get_filter_options(filter_type):
            if filter_type == "Geology Origin":
                return sorted(psd_long_format['Geology_Orgin'].dropna().unique())
            elif filter_type == "Consistency":
                return sorted(psd_long_format['Consistency'].dropna().unique())
            elif filter_type == "Map Symbol":
                if 'Map_symbol' in psd_long_format.columns:
                    return sorted(psd_long_format['Map_symbol'].dropna().unique())
                else:
                    return []
            elif filter_type == "Hole ID":
                return sorted(psd_long_format['Hole_ID'].dropna().unique())
            elif filter_type == "Report":
                if 'Report' in psd_long_format.columns:
                    return sorted(psd_long_format['Report'].dropna().unique())
                else:
                    return []
            else:
                return []

        # Main Parameters Box
        with st.expander("Plot Parameters", expanded=True):
            # Row 1: Essential Data Configuration
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                hole_id_col = st.selectbox(
                    "hole_id_col", 
                    standard_id_columns, 
                    index=standard_id_columns.index(hole_id_default) if hole_id_default in standard_id_columns else 0,
                    help="Column containing hole/borehole identifiers"
                )
            with col2:
                depth_col = st.selectbox(
                    "depth_col", 
                    standard_id_columns, 
                    index=standard_id_columns.index(depth_default) if depth_default in standard_id_columns else 0,
                    help="Column containing depth values (from surface)"
                )
            with col3:
                size_col = st.selectbox(
                    "size_col", 
                    available_processed_cols, 
                    index=available_processed_cols.index(size_default) if size_default in available_processed_cols else 0,
                    help="Column containing sieve size in mm"
                )
            with col4:
                percent_col = st.selectbox(
                    "percent_col", 
                    available_processed_cols, 
                    index=available_processed_cols.index(percent_default) if percent_default in available_processed_cols else 0,
                    help="Column containing percentage passing values"
                )
            with col5:
                color_by = st.selectbox("color_by", ["None", "Consistency", "Geology_Orgin", "Hole_ID", "Report"], index=1, help="Group data by color")
            
            # Row 2: Data Filtering
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                filter1_by = st.selectbox(
                    "Filter 1 By",
                    ["None", "Geology Origin", "Consistency", "Map Symbol", "Hole ID", "Report"],
                    index=1,  # Default to "Geology Origin"
                    help="Select first filter type"
                )
            
            with col2:
                # Dynamic Filter 1 Value dropdown based on selection
                if filter1_by == "None":
                    filter1_values = []
                    st.selectbox("Filter 1 Value", ["All"], index=0, disabled=True, help="Select filter type first")
                else:
                    filter1_options = get_filter_options(filter1_by)
                    # Create dropdown with "All" as default option
                    filter1_dropdown_options = ["All"] + filter1_options
                    filter1_selection = st.selectbox(
                        f"{filter1_by}",
                        filter1_dropdown_options,
                        index=0,
                        help=f"Select {filter1_by.lower()} value"
                    )
                    # Convert to list format for filtering logic
                    if filter1_selection == "All":
                        filter1_values = filter1_options  # Include all options
                    else:
                        filter1_values = [filter1_selection]  # Single selection
            
            with col3:
                filter2_by = st.selectbox(
                    "Filter 2 By",
                    ["None", "Geology Origin", "Consistency", "Map Symbol", "Hole ID", "Report"],
                    index=0,
                    help="Select second filter type"
                )
            
            with col4:
                # Dynamic Filter 2 Value dropdown based on selection
                if filter2_by == "None":
                    filter2_values = []
                    st.selectbox("Filter 2 Value", ["All"], index=0, disabled=True, help="Select filter type first")
                else:
                    filter2_options = get_filter_options(filter2_by)
                    # Create dropdown with "All" as default option
                    filter2_dropdown_options = ["All"] + filter2_options
                    filter2_selection = st.selectbox(
                        f"{filter2_by}",
                        filter2_dropdown_options,
                        index=0,
                        help=f"Select {filter2_by.lower()} value"
                    )
                    # Convert to list format for filtering logic
                    if filter2_selection == "All":
                        filter2_values = filter2_options  # Include all options
                    else:
                        filter2_values = [filter2_selection]  # Single selection
            
            with col5:
                show_legend = st.selectbox("Show legend", ["Yes", "No"], index=0, help="Display legend")
            
            # Row 3: Plot Appearance
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                xlim_str = st.text_input("xlim (min, max)", value="(0.001, 1000)", help="X-axis limits as tuple")
            with col2:
                ylim_str = st.text_input("ylim (min, max)", value="(0, 100)", help="Y-axis limits as tuple")
            with col3:
                figsize_str = st.text_input("figsize (w, h)", value="(14, 9)", help="Figure size as tuple")
            with col4:
                title = st.text_input("title", value="", help="Custom plot title")
            with col5:
                show_markers = st.selectbox("Show markers", ["Yes", "No"], index=0, help="Show data markers")
            
            # Row 4: Visual Styling
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                line_width = st.number_input("Line width", min_value=0.5, max_value=5.0, value=1.5, step=0.1, help="Data line width")
            with col2:
                marker_style = st.selectbox("Marker style", ["o", "s", "^", "v", "D", "x", "+"], index=0, help="Marker shape")
            with col3:
                marker_size = st.number_input("Marker size", min_value=1, max_value=15, value=4, help="Marker size")
            with col4:
                alpha = st.number_input("Alpha", min_value=0.1, max_value=1.0, value=0.8, step=0.1, help="Line transparency")
            with col5:
                show_grid = st.selectbox("Show grid", ["Yes", "No"], index=0, help="Show grid lines")

        # Advanced Parameters Box  
        with st.expander("Advanced Parameters", expanded=False):
            # Row 1: Font Controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                title_fontsize = st.number_input("Title font size", min_value=8, max_value=24, value=16, help="Title font size")
            with col2:
                xlabel_fontsize = st.number_input("X-label font size", min_value=8, max_value=20, value=14, help="X-axis label font size")
            with col3:
                ylabel_fontsize = st.number_input("Y-label font size", min_value=8, max_value=20, value=14, help="Y-axis label font size")
            with col4:
                tick_fontsize = st.number_input("Tick font size", min_value=6, max_value=16, value=14, help="Tick labels font size")
            
            # Row 2: Font Weights & Grid Controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                title_fontweight = st.selectbox("Title font weight", ["normal", "bold"], index=1, help="Title font weight")
            with col2:
                xlabel_fontweight = st.selectbox("X-label font weight", ["normal", "bold"], index=1, help="X-axis label font weight")
            with col3:
                ylabel_fontweight = st.selectbox("Y-label font weight", ["normal", "bold"], index=1, help="Y-axis label font weight")
            with col4:
                grid_style = st.selectbox("Grid style", ["--", "-", ":", "-."], index=0, help="Grid line style")
            
            # Row 3: Custom Labels & Positioning
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                xlabel = st.text_input("Custom X-label", value="", help="Custom X-axis label (blank for default)")
            with col2:
                ylabel = st.text_input("Custom Y-label", value="", help="Custom Y-axis label (blank for default)")
            with col3:
                rotation = st.number_input("X-tick rotation", min_value=0, max_value=90, value=0, help="X-axis tick label rotation")
            with col4:
                legend_fontsize = st.number_input("Legend font size", min_value=6, max_value=16, value=10, help="Legend text size")
            
            # Row 4: Advanced Visual Controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                smooth_curves = st.selectbox("Smooth curves", ["No", "Yes"], index=0, help="Use spline interpolation")
            with col2:
                save_dpi = st.number_input("Save DPI", min_value=100, max_value=600, value=300, help="DPI for saved figures")
            with col3:
                tick_pad = st.number_input("Tick padding", min_value=6, max_value=20, value=12, help="Padding around tick labels")
            with col4:
                st.empty()  # Placeholder for future parameters
            
                
        # Apply dynamic filters to data
        filtered_psd = psd_long_format.copy()
        
        # Apply Filter 1
        if filter1_by != "None" and filter1_values:
            if filter1_by == "Geology Origin":
                filtered_psd = filtered_psd[filtered_psd['Geology_Orgin'].isin(filter1_values)]
            elif filter1_by == "Consistency":
                filtered_psd = filtered_psd[filtered_psd['Consistency'].isin(filter1_values)]
            elif filter1_by == "Map Symbol" and 'Map_symbol' in filtered_psd.columns:
                filtered_psd = filtered_psd[filtered_psd['Map_symbol'].isin(filter1_values)]
            elif filter1_by == "Hole ID":
                filtered_psd = filtered_psd[filtered_psd['Hole_ID'].isin(filter1_values)]
            elif filter1_by == "Report" and 'Report' in filtered_psd.columns:
                filtered_psd = filtered_psd[filtered_psd['Report'].isin(filter1_values)]
        
        # Apply Filter 2
        if filter2_by != "None" and filter2_values:
            if filter2_by == "Geology Origin":
                filtered_psd = filtered_psd[filtered_psd['Geology_Orgin'].isin(filter2_values)]
            elif filter2_by == "Consistency":
                filtered_psd = filtered_psd[filtered_psd['Consistency'].isin(filter2_values)]
            elif filter2_by == "Map Symbol" and 'Map_symbol' in filtered_psd.columns:
                filtered_psd = filtered_psd[filtered_psd['Map_symbol'].isin(filter2_values)]
            elif filter2_by == "Hole ID":
                filtered_psd = filtered_psd[filtered_psd['Hole_ID'].isin(filter2_values)]
            elif filter2_by == "Report" and 'Report' in filtered_psd.columns:
                filtered_psd = filtered_psd[filtered_psd['Report'].isin(filter2_values)]
        
        # Generate dynamic title suffix based on applied filters
        def generate_title_suffix():
            suffix_parts = []
            
            # Add Filter 1 to suffix if applied (and not "All")
            if filter1_by != "None" and filter1_values:
                # Get all available options for this filter type to check if "All" is selected
                all_options = get_filter_options(filter1_by)
                # Only add to suffix if it's not "All" (i.e., not all options selected)
                if filter1_values != all_options:
                    if len(filter1_values) == 1:
                        suffix_parts.append(f"{filter1_values[0]}")
                    elif len(filter1_values) <= 3:
                        suffix_parts.append(f"{', '.join(filter1_values)}")
                    else:
                        suffix_parts.append(f"{filter1_by} (Multiple)")
            
            # Add Filter 2 to suffix if applied (and not "All")
            if filter2_by != "None" and filter2_values:
                # Get all available options for this filter type to check if "All" is selected
                all_options = get_filter_options(filter2_by)
                # Only add to suffix if it's not "All" (i.e., not all options selected)
                if filter2_values != all_options:
                    if len(filter2_values) == 1:
                        suffix_parts.append(f"{filter2_values[0]}")
                    elif len(filter2_values) <= 3:
                        suffix_parts.append(f"{', '.join(filter2_values)}")
                    else:
                        suffix_parts.append(f"{filter2_by} (Multiple)")
            
            return f" - {' + '.join(suffix_parts)}" if suffix_parts else ""
        
        # Generate dynamic title suffix
        dynamic_title_suffix = generate_title_suffix()
        
        # Check if filtered data is empty
        if filtered_psd.empty:
            st.warning("No data remains after applying filters. Please adjust your filter criteria.")
            return
        # Parse the tuple inputs
        xlim = parse_tuple(xlim_str, (0.001, 1000))
        ylim = parse_tuple(ylim_str, (0, 100))
        figsize = parse_tuple(figsize_str, (14, 9))
        
        # Create main plot using filtered data
        if not filtered_psd.empty:
            # Prepare data for plotting (ensure column names match Functions expectations)
            plot_data = filtered_psd.copy()
            
            # Map our long format columns to what plot_psd expects
            if 'Sieve_Size_mm' not in plot_data.columns and 'Sieve_Size_Numeric' in plot_data.columns:
                plot_data['Sieve_Size_mm'] = plot_data['Sieve_Size_Numeric']
            
            if 'Percentage passing (%)' not in plot_data.columns and 'Percent_Passing' in plot_data.columns:
                plot_data['Percentage passing (%)'] = plot_data['Percent_Passing']
            
            if HAS_FUNCTIONS:
                try:
                    # Clear any existing figures first
                    plt.close('all')
                    
                    # Call function directly from Functions folder with ALL parameters
                    # Ensure None is passed as None, not string "None"
                    color_by_param = None if color_by == "None" or color_by is None else color_by
                    
                    # Create palette for default blue color when no color_by is specified
                    palette_param = None
                    if color_by_param is None:
                        # Set default blue color for all lines with black outline
                        palette_param = {'default': '#1f77b4'}  # Default matplotlib blue
                    
                    plot_psd(
                        # === Tier 1: Essential Data Parameters ===
                        df=plot_data,
                        hole_id_col=hole_id_col,
                        depth_col=depth_col,
                        size_col=size_col,
                        percent_col=percent_col,
                        max_plots=None,  # Plot all available data without restriction
                        
                        # === Tier 2: Plot Appearance ===
                        title=title if title else None,
                        title_suffix=dynamic_title_suffix,
                        figsize=figsize,
                        xlim=xlim,
                        ylim=ylim,
                        xlabel=xlabel if xlabel else None,
                        ylabel=ylabel if ylabel else None,
                        
                        # === Tier 3: Category & Color Options ===
                        color_by=color_by_param,
                        palette=palette_param,
                        show_color_mappings=False,
                        
                        # === Tier 4: Axis Configuration ===
                        xlabel_fontsize=xlabel_fontsize,
                        ylabel_fontsize=ylabel_fontsize,
                        xlabel_fontweight=xlabel_fontweight,
                        ylabel_fontweight=ylabel_fontweight,
                        tick_fontsize=tick_fontsize,
                        tick_pad=tick_pad,
                        rotation=rotation,
                        
                        # === Tier 5: Display Options ===
                        show_plot=False,
                        show_legend=(show_legend == "Yes") if color_by_param is not None else False,
                        show_grid=(show_grid == "Yes"),
                        grid_axis="both",
                        grid_style=grid_style,
                        
                        # === Tier 6: Output Controls ===
                        output_filepath=None,
                        save_dpi=save_dpi,
                        save_bbox_inches="tight",
                        close_plot=False,
                        
                        # === Tier 7: Visual Customization ===
                        line_width=line_width,
                        marker_style=marker_style,
                        marker_size=marker_size,
                        show_markers=(show_markers == "Yes"),
                        alpha=alpha,
                        smooth_curves=(smooth_curves == "Yes"),
                        
                        # === Tier 8: Advanced Formatting ===
                        formatting_options={
                            # Title formatting
                            "title_fontsize": title_fontsize,
                            "title_fontweight": title_fontweight,
                            "title_pad": 20,
                            
                            # Legend formatting
                            "legend_fontsize": legend_fontsize,
                            "legend_title_fontsize": legend_fontsize,
                            "legend_loc": "upper left",
                            "legend_bbox_to_anchor": (1.02, 1),
                            
                            # Add black outline to plots
                            "spine_linewidth": 2.0,  # Thicker plot outline
                            "ax2_border_linewidth": 2.0,  # Thicker classification bar outline
                        }
                    )
                    
                    # Capture and display the figure in Streamlit using plot size control
                    current_fig = plt.gcf()
                    if current_fig and current_fig.get_axes():
                        # Import the display function with size control
                        from .plotting_utils import display_plot_with_size_control
                        success = display_plot_with_size_control(current_fig)
                    else:
                        success = False
                    
                    # Store the plot for Materials Dashboard and provide download (regardless of success status)
                    try:
                        # Get the current figure for storage and download
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            # Save figure to buffer for dashboard
                            import io
                            buf = io.BytesIO()
                            current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            if success:  # Only store in dashboard if plot was successful
                                store_material_plot('psd_analysis', buf)
                    except Exception as e:
                        pass  # Don't break the main functionality if storage fails
                    
                    # Simple download button with figure reference - always available if there's a current figure
                    from .plot_download_simple import create_simple_download_button
                    create_simple_download_button("psd_analysis", "main", fig=current_fig)
                    
                    # Map visualization (moved before distribution plots)
                    st.markdown("### Test Locations Map")
                    
                    # Create a filtered version of the original data matching the PSD filters
                    # This ensures the map shows only the filtered locations
                    map_filtered_data = psd_hydrometer.copy()
                    
                    # Apply the same filters to the original format data
                    if filter1_by != "None" and filter1_values:
                        if filter1_by == "Geology Origin":
                            map_filtered_data = map_filtered_data[map_filtered_data['Geology_Orgin'].isin(filter1_values)]
                        elif filter1_by == "Consistency":
                            map_filtered_data = map_filtered_data[map_filtered_data['Consistency'].isin(filter1_values)]
                        elif filter1_by == "Map Symbol" and 'Map_symbol' in map_filtered_data.columns:
                            map_filtered_data = map_filtered_data[map_filtered_data['Map_symbol'].isin(filter1_values)]
                        elif filter1_by == "Hole ID":
                            map_filtered_data = map_filtered_data[map_filtered_data['Hole_ID'].isin(filter1_values)]
                        elif filter1_by == "Report" and 'Report' in map_filtered_data.columns:
                            map_filtered_data = map_filtered_data[map_filtered_data['Report'].isin(filter1_values)]
                    
                    # Apply Filter 2
                    if filter2_by != "None" and filter2_values:
                        if filter2_by == "Geology Origin":
                            map_filtered_data = map_filtered_data[map_filtered_data['Geology_Orgin'].isin(filter2_values)]
                        elif filter2_by == "Consistency":
                            map_filtered_data = map_filtered_data[map_filtered_data['Consistency'].isin(filter2_values)]
                        elif filter2_by == "Map Symbol" and 'Map_symbol' in map_filtered_data.columns:
                            map_filtered_data = map_filtered_data[map_filtered_data['Map_symbol'].isin(filter2_values)]
                        elif filter2_by == "Hole ID":
                            map_filtered_data = map_filtered_data[map_filtered_data['Hole_ID'].isin(filter2_values)]
                        elif filter2_by == "Report" and 'Report' in map_filtered_data.columns:
                            map_filtered_data = map_filtered_data[map_filtered_data['Report'].isin(filter2_values)]
                    
                    # Check for coordinate columns for map visualization  
                    # Use dynamic ID columns from original data (before long format conversion)
                    id_columns = get_standard_id_columns(map_filtered_data)  # Use filtered data instead
                    
                    # Precise coordinate matching - avoids false positives
                    def is_coordinate_column(column_name, keywords):
                        col_clean = column_name.lower().replace('(', '').replace(')', '').replace('_', '').replace(' ', '').replace('-', '')
                        # Must be exact match or start with the keyword to avoid false positives
                        return any(col_clean == keyword or col_clean.startswith(keyword) for keyword in keywords)
                    
                    # Northing variations: exact matches and starts-with patterns
                    northing_keywords = ['northing', 'north', 'latitude', 'lat', 'y']
                    potential_lat_cols = [col for col in id_columns if is_coordinate_column(col, northing_keywords)]
                    
                    # Easting variations: exact matches and starts-with patterns  
                    easting_keywords = ['easting', 'east', 'longitude', 'lon', 'x'] 
                    potential_lon_cols = [col for col in id_columns if is_coordinate_column(col, easting_keywords)]
                    
                    if potential_lat_cols and potential_lon_cols:
                        lat_col = potential_lat_cols[0]  # Use first match
                        lon_col = potential_lon_cols[0]  # Use first match
                        
                        # Get coordinate data from original data, not the long-format converted data
                        # Extract unique combinations of Hole_ID and depth to get coordinate data
                        try:
                            # Get unique sample locations from the filtered data
                            sample_locations = map_filtered_data[['Hole_ID', 'From_mbgl']].drop_duplicates()
                            
                            # Merge with coordinate columns and chainage if available
                            merge_cols = ['Hole_ID', 'From_mbgl', lat_col, lon_col]
                            if 'Chainage' in map_filtered_data.columns:
                                merge_cols.append('Chainage')
                            
                            coord_data = sample_locations.merge(
                                map_filtered_data[merge_cols], 
                                on=['Hole_ID', 'From_mbgl'], 
                                how='left'
                            ).dropna(subset=[lat_col, lon_col])
                            
                            if not coord_data.empty and len(coord_data) > 0:
                                # Prepare map data
                                map_data = coord_data.copy()
                                
                                # Convert UTM to WGS84 for mapping if needed
                                if coord_data[lat_col].max() > 1000:  # UTM coordinates detected
                                    if HAS_PYPROJ:
                                        try:
                                            # Determine UTM zone based on easting values
                                            avg_easting = coord_data[lon_col].mean()
                                            if avg_easting < 500000:
                                                utm_zone = 'EPSG:32755'  # Zone 55S
                                            else:
                                                utm_zone = 'EPSG:32756'  # Zone 56S
                                            
                                            utm_crs = pyproj.CRS(utm_zone)
                                            wgs84_crs = pyproj.CRS('EPSG:4326')
                                            transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
                                            
                                            lons, lats = transformer.transform(
                                                coord_data[lon_col].values,
                                                coord_data[lat_col].values
                                            )
                                            
                                            # Create map data with converted coordinates
                                            map_data = coord_data.copy()
                                            map_data['lat'] = lats
                                            map_data['lon'] = lons
                                            
                                            # Display enhanced map with test locations
                                            if HAS_PLOTLY:
                                                import plotly.express as px
                                                import plotly.graph_objects as go
                                                
                                                fig = go.Figure()
                                                
                                                # Add test location points
                                                if 'Chainage' in coord_data.columns:
                                                    map_data_with_chainage = map_data.copy()
                                                    map_data_with_chainage['chainage'] = coord_data['Chainage'].values
                                                    
                                                    fig.add_trace(go.Scattermapbox(
                                                        lat=map_data_with_chainage['lat'],
                                                        lon=map_data_with_chainage['lon'],
                                                        mode='markers',
                                                        marker=dict(
                                                            size=8,
                                                            color='blue',
                                                            opacity=0.8
                                                        ),
                                                        name='Test Locations',
                                                        customdata=map_data_with_chainage['chainage'],
                                                        hovertemplate='<b>Test Location</b><br>' +
                                                                     'Chainage: %{customdata:.0f} m<br>' +
                                                                     'Lat: %{lat:.6f}<br>' +
                                                                     'Lon: %{lon:.6f}<extra></extra>'
                                                    ))
                                                else:
                                                    fig.add_trace(go.Scattermapbox(
                                                        lat=map_data['lat'],
                                                        lon=map_data['lon'],
                                                        mode='markers',
                                                        marker=dict(
                                                            size=8,
                                                            color='blue',
                                                            opacity=0.8
                                                        ),
                                                        name='Test Locations',
                                                        hovertemplate='<b>Test Location</b><br>' +
                                                                     'Lat: %{lat:.6f}<br>' +
                                                                     'Lon: %{lon:.6f}<extra></extra>'
                                                    ))
                                                
                                                # Add chainage reference markers
                                                if 'Chainage' in coord_data.columns:
                                                    chainage_data = coord_data[['Chainage', lat_col, lon_col]].drop_duplicates()
                                                    chainage_data = chainage_data.sort_values('Chainage')
                                                    
                                                    min_ch = chainage_data['Chainage'].min()
                                                    max_ch = chainage_data['Chainage'].max()
                                                    
                                                    interval = 2000  # Every 2000m
                                                    chainage_marks = []
                                                    start_ch = int(min_ch // interval) * interval
                                                    if start_ch < min_ch:
                                                        start_ch += interval
                                                    
                                                    for ch in range(start_ch, int(max_ch) + 1, interval):
                                                        if ch >= min_ch and ch <= max_ch:
                                                            if len(chainage_data) > 1:
                                                                lat_interp = np.interp(ch, chainage_data['Chainage'], chainage_data[lat_col])
                                                                lon_interp = np.interp(ch, chainage_data['Chainage'], chainage_data[lon_col])
                                                                
                                                                if lat_interp > 1000:  # UTM coordinates
                                                                    try:
                                                                        avg_easting = lon_interp
                                                                        if avg_easting < 500000:
                                                                            utm_zone = 'EPSG:32755'
                                                                        else:
                                                                            utm_zone = 'EPSG:32756'
                                                                        
                                                                        utm_crs = pyproj.CRS(utm_zone)
                                                                        wgs84_crs = pyproj.CRS('EPSG:4326')
                                                                        transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
                                                                        lon_wgs84, lat_wgs84 = transformer.transform(lon_interp, lat_interp)
                                                                        
                                                                        chainage_marks.append({
                                                                            'chainage': ch,
                                                                            'lat': lat_wgs84,
                                                                            'lon': lon_wgs84
                                                                        })
                                                                    except:
                                                                        pass
                                                                else:
                                                                    chainage_marks.append({
                                                                        'chainage': ch,
                                                                        'lat': lat_interp,
                                                                        'lon': lon_interp
                                                                    })
                                                    
                                                    # Add chainage markers to map
                                                    if chainage_marks:
                                                        chainage_df = pd.DataFrame(chainage_marks)
                                                        fig.add_trace(go.Scattermapbox(
                                                            lat=chainage_df['lat'],
                                                            lon=chainage_df['lon'],
                                                            mode='markers+text',
                                                            marker=dict(
                                                                size=14,
                                                                color='red',
                                                                symbol='triangle',
                                                                opacity=0.9
                                                            ),
                                                            text=[f"{ch}" for ch in chainage_df['chainage']],
                                                            textposition="top center",
                                                            textfont=dict(size=10, color='red'),
                                                            name='Chainage Markers',
                                                            hovertemplate='<b>Chainage: %{customdata} m</b><br>' +
                                                                         'Lat: %{lat:.6f}<br>' +
                                                                         'Lon: %{lon:.6f}<extra></extra>',
                                                            customdata=chainage_df['chainage']
                                                        ))
                                                
                                                # Update map layout
                                                fig.update_layout(
                                                    mapbox_style="carto-positron",
                                                    height=528,
                                                    margin={"r":0,"t":0,"l":0,"b":0},
                                                    showlegend=False,
                                                    mapbox=dict(
                                                        center=dict(
                                                            lat=map_data['lat'].mean(),
                                                            lon=map_data['lon'].mean()
                                                        ),
                                                        zoom=11
                                                    )
                                                )
                                                
                                                # Create layout with map at 90% width
                                                map_col, spacer_col = st.columns([9, 1])
                                                with map_col:
                                                    st.plotly_chart(fig, use_container_width=True, config={
                                                        'scrollZoom': True,
                                                        'displayModeBar': False
                                                    })
                                            else:
                                                # Fallback to basic map layout at 90% width
                                                map_col, spacer_col = st.columns([9, 1])
                                                with map_col:
                                                    st.map(map_data[['lat', 'lon']])
                                            
                                        except Exception as e:
                                            st.error(f"❌ Could not convert coordinates: {str(e)}")
                                    else:
                                        st.warning("⚠️ pyproj library not available for coordinate conversion.")
                                else:
                                    # Already in WGS84 format - prepare for direct mapping
                                    map_data = map_data.rename(columns={lat_col: 'lat', lon_col: 'lon'})
                                    
                                    # Create layout with map at 90% width
                                    map_col, spacer_col = st.columns([9, 1])
                                    with map_col:
                                        st.map(map_data)
                            else:
                                st.info("No coordinate data available for PSD test locations")
                        except Exception as e:
                            st.warning(f"Could not generate map: {str(e)}")
                    else:
                        st.info("No coordinate columns found for map visualization.")
                    
                    # PSD Distribution Analysis (moved after map)
                    st.markdown("### PSD Distribution")
                    
                    try:
                        # Function to extract test types from columns
                        def get_test_types_from_columns(data):
                            test_columns = [col for col in data.columns if '?' in col]
                            test_types = []
                            for col in test_columns:
                                test_type = col.replace('?', '').strip()
                                if test_type:
                                    test_types.append(test_type)
                            return test_types, test_columns
                        
                        # Function to filter positive values
                        def filter_positive_values(data, column):
                            col_data = data[column]
                            mask = pd.Series([False] * len(data), index=data.index)
                            
                            # Check for positive indicators
                            positive_indicators = ['yes', 'y', '1', 'true', 'x', 'done', 'positive']
                            for value_str in positive_indicators:
                                mask |= (col_data.astype(str).str.lower().str.strip() == value_str)
                                if value_str == '1':
                                    mask |= (col_data.astype(str).str.strip() == '1.0')
                            
                            return data[mask]
                        
                        # Function to render single test chart using original data to count unique tests
                        def render_single_test_chart(test_type, original_filtered_data, bins):
                            st.write(f"**{test_type} Distribution:**")
                            
                            test_col = f"{test_type}?"
                            if test_col in original_filtered_data.columns:
                                # Use original data to count unique tests, not long-format data
                                test_data = original_filtered_data[original_filtered_data[test_col] == 'Y']
                                
                                if not test_data.empty and 'Chainage' in test_data.columns:
                                    # Count unique tests per chainage bin (one count per unique Hole_ID + From_mbgl combination)
                                    test_data_unique = test_data.drop_duplicates(subset=['Hole_ID', 'From_mbgl'])
                                    test_chainage = test_data_unique['Chainage'].dropna()
                                    
                                    if not test_chainage.empty:
                                        hist, bin_edges = np.histogram(test_chainage, bins=bins)
                                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                        
                                        chart_data = pd.DataFrame({
                                            'Chainage': bin_centers.astype(int),
                                            'Test Count': hist.astype(int)
                                        })
                                        
                                        if chart_data['Test Count'].sum() > 0:
                                            st.line_chart(chart_data.set_index('Chainage'))
                                        else:
                                            st.info(f"No {test_type} tests found in chainage bins")
                                    else:
                                        st.info(f"No {test_type} tests with chainage data")
                                else:
                                    st.info(f"No {test_type} test data available")
                            else:
                                st.caption(f"Column '{test_col}' not found")
                        
                        # Check if chainage data is available
                        if 'Chainage' in filtered_psd.columns:
                            available_test_types, test_columns = get_test_types_from_columns(filtered_psd)
                            
                            if len(available_test_types) > 0:
                                chainage_data = filtered_psd['Chainage'].dropna()
                                if not chainage_data.empty:
                                    min_chainage = chainage_data.min()
                                    max_chainage = chainage_data.max()
                                    
                                    # Create fixed interval bins (200m intervals)
                                    bin_interval = 200
                                    bin_start = int(min_chainage // bin_interval) * bin_interval
                                    bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
                                    bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
                                    
                                    # Find PSD-specific test types
                                    psd_test_types = [t for t in available_test_types if 'PSD' in t or 'Particle' in t or 'Sieve' in t or 'Hydrometer' in t]
                                    
                                    if psd_test_types:
                                        # Create charts for PSD test types - each chart at 90% width in separate rows
                                        for i, test_type in enumerate(psd_test_types):
                                            if i > 0:
                                                st.write("")
                                            
                                            # Each chart gets 90% width layout
                                            chart_col, spacer_col = st.columns([9, 1])
                                            
                                            with chart_col:
                                                render_single_test_chart(test_type, filtered_data, bins)
                                    else:
                                        # If no specific PSD tests found, show the first few available - each at 90% width
                                        display_types = available_test_types[:4]  # Show up to 4 test types
                                        for i, test_type in enumerate(display_types):
                                            if i > 0:
                                                st.write("")
                                            
                                            # Each chart gets 90% width layout
                                            chart_col, spacer_col = st.columns([9, 1])
                                            
                                            with chart_col:
                                                render_single_test_chart(test_type, filtered_data, bins)
                                else:
                                    st.info("No chainage data available for distribution analysis")
                            else:
                                st.info("No test data available for distribution analysis")
                        else:
                            st.info("Chainage column not found - cannot create spatial distribution")
                        
                    except Exception as e:
                        st.warning(f"Could not generate PSD distribution chart: {str(e)}")
                    
                    # Add visual separator before plot summary
                    st.divider()
                    
                    # Default Plot Summary Table (Engineering-focused) - show if plot exists
                    if success:
                        st.markdown("**Plot Summary**")
                        try:
                            # Calculate engineering-relevant statistics
                            summary_data = []
                            
                            # Filter data based on active filters for summary
                            summary_df = plot_data.copy()
                            
                            # Apply filters to get the actual plotted data
                            if filter1_by != "None" and filter1_values and filter1_values != ["All"]:
                                filter_col = "Geology_Orgin" if filter1_by == "Geology Origin" else filter1_by
                                if filter_col in summary_df.columns:
                                    summary_df = summary_df[summary_df[filter_col].isin(filter1_values)]
                            
                            if filter2_by != "None" and filter2_values and filter2_values != ["All"]:
                                filter_col = "Geology_Orgin" if filter2_by == "Geology Origin" else filter2_by
                                if filter_col in summary_df.columns:
                                    summary_df = summary_df[summary_df[filter_col].isin(filter2_values)]
                            
                            if not summary_df.empty:
                                # Sample count
                                unique_samples = len(summary_df[['Hole_ID', 'From_mbgl']].drop_duplicates())
                                total_points = len(summary_df)
                                
                                summary_data.extend([
                                    {'Metric': 'Total Samples', 'Value': f"{unique_samples:,}"},
                                    {'Metric': 'Total Data Points', 'Value': f"{total_points:,}"}
                                ])
                                
                                # Depth range
                                if 'From_mbgl' in summary_df.columns:
                                    depth_data = summary_df['From_mbgl'].dropna()
                                    if not depth_data.empty:
                                        depth_range = f"{depth_data.min():.1f} - {depth_data.max():.1f} m"
                                        summary_data.append({'Metric': 'Depth Range', 'Value': depth_range})
                                
                                # Sieve size range
                                if 'Sieve_Size_mm' in summary_df.columns:
                                    sieve_data = summary_df['Sieve_Size_mm'].dropna()
                                    if not sieve_data.empty:
                                        sieve_range = f"{sieve_data.min():.3f} - {sieve_data.max():.0f} mm"
                                        summary_data.append({'Metric': 'Sieve Size Range', 'Value': sieve_range})
                                
                                # Engineering percentiles - simplified
                                if 'Percentage passing (%)' in summary_df.columns and 'Sieve_Size_mm' in summary_df.columns:
                                    try:
                                        # Calculate percentiles for all samples combined
                                        percent_data = summary_df['Percentage passing (%)'].dropna()
                                        if not percent_data.empty:
                                            # Passing percentage statistics
                                            summary_data.append({
                                                'Metric': 'Passing % Range', 
                                                'Value': f"{percent_data.min():.1f} - {percent_data.max():.1f}%"
                                            })
                                        
                                        # Calculate D10, D30, D60 for each sample
                                        sample_stats = []
                                        for (hole_id, depth), group in summary_df.groupby(['Hole_ID', 'From_mbgl']):
                                            if len(group) >= 3:  # Need minimum points for interpolation
                                                try:
                                                    # Sort by sieve size for interpolation
                                                    sorted_group = group.sort_values('Sieve_Size_mm')
                                                    
                                                    # Interpolate D-values
                                                    d10 = np.interp(10, sorted_group['Percentage passing (%)'], sorted_group['Sieve_Size_mm'])
                                                    d30 = np.interp(30, sorted_group['Percentage passing (%)'], sorted_group['Sieve_Size_mm'])
                                                    d60 = np.interp(60, sorted_group['Percentage passing (%)'], sorted_group['Sieve_Size_mm'])
                                                    
                                                    sample_stats.append({'D10': d10, 'D30': d30, 'D60': d60})
                                                except:
                                                    continue  # Skip samples with interpolation issues
                                        
                                        # Calculate statistics for D-values
                                        if len(sample_stats) > 0:
                                            d10_values = [s['D10'] for s in sample_stats]
                                            d30_values = [s['D30'] for s in sample_stats]
                                            d60_values = [s['D60'] for s in sample_stats]
                                            
                                            summary_data.extend([
                                                {'Metric': 'D10 (mm)', 'Value': f"{np.mean(d10_values):.3f} ({np.std(d10_values):.3f})"},
                                                {'Metric': 'D30 (mm)', 'Value': f"{np.mean(d30_values):.3f} ({np.std(d30_values):.3f})"},
                                                {'Metric': 'D60 (mm)', 'Value': f"{np.mean(d60_values):.3f} ({np.std(d60_values):.3f})"}
                                            ])
                                    except:
                                        pass
                                
                                
                                # Active filters summary
                                if filter1_by != "None" and filter1_values:
                                    if len(filter1_values) == 1:
                                        summary_data.append({'Metric': f'Filter: {filter1_by}', 'Value': filter1_values[0]})
                                    else:
                                        summary_data.append({'Metric': f'Filter: {filter1_by}', 'Value': f"{len(filter1_values)} selected"})
                                
                                if filter2_by != "None" and filter2_values:
                                    if len(filter2_values) == 1:
                                        summary_data.append({'Metric': f'Filter: {filter2_by}', 'Value': filter2_values[0]})
                                    else:
                                        summary_data.append({'Metric': f'Filter: {filter2_by}', 'Value': f"{len(filter2_values)} selected"})
                                
                                # Create summary table
                                if summary_data:
                                    summary_df_display = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df_display, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No summary data available")
                            else:
                                st.info("No data available after filtering")
                        except Exception as e:
                            st.warning(f"Could not generate plot summary: {str(e)}")
                    
                    # Show warning only if plot generation failed
                    if not success:
                        st.warning("No plot generated - check data availability")
                        
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
            else:
                st.error("Functions folder not accessible")
                st.info("Check Functions folder and plot_psd.py module")
        else:
            st.warning("No valid PSD data available for plotting")
        
        # Data preview and statistics options underneath plot (standard format)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Show data preview", key="psd_data_preview"):
                # Show relevant columns for PSD analysis
                preview_cols = ['Hole_ID', 'From_mbgl', 'To_mbgl', 'Sieve_Size_mm', 'Percentage passing (%)']
                if color_by and color_by in psd_long_format.columns:
                    preview_cols.append(color_by)
                
                available_cols = [col for col in preview_cols if col in psd_long_format.columns]
                st.dataframe(psd_long_format[available_cols].head(20), use_container_width=True)
                st.caption(f"{len(psd_long_format)} total records")
        
        with col2:
            if st.checkbox("Show statistics", key="psd_statistics", disabled=True):
                try:
                    # Use filtered data that matches what's actually plotted
                    valid_data = filtered_psd.dropna(subset=['Sieve_Size_mm', 'Percentage passing (%)'])
                    
                    if not valid_data.empty:
                        stats_data = []
                        
                        # Calculate sample-wise statistics (each hole/depth combination)
                        sample_d10_values = []
                        sample_d30_values = []
                        sample_d50_values = []
                        sample_d60_values = []
                        sample_cu_values = []
                        sample_cc_values = []
                        sample_fine_content = []  # % passing 0.075mm
                        
                        for (hole_id, depth), group in valid_data.groupby(['Hole_ID', 'From_mbgl']):
                            if len(group) >= 3:  # Need minimum points for interpolation
                                try:
                                    # Sort by sieve size
                                    sorted_group = group.sort_values('Sieve_Size_mm')
                                    
                                    # Interpolate D-values
                                    d10 = np.interp(10, sorted_group['Percentage passing (%)'], sorted_group['Sieve_Size_mm'])
                                    d30 = np.interp(30, sorted_group['Percentage passing (%)'], sorted_group['Sieve_Size_mm'])
                                    d50 = np.interp(50, sorted_group['Percentage passing (%)'], sorted_group['Sieve_Size_mm'])
                                    d60 = np.interp(60, sorted_group['Percentage passing (%)'], sorted_group['Sieve_Size_mm'])
                                    
                                    sample_d10_values.append(d10)
                                    sample_d30_values.append(d30)
                                    sample_d50_values.append(d50)
                                    sample_d60_values.append(d60)
                                    
                                    # Calculate Cu and Cc
                                    if d10 > 0:
                                        cu = d60 / d10
                                        sample_cu_values.append(cu)
                                        
                                        if d60 > 0:
                                            cc = (d30 ** 2) / (d60 * d10)
                                            sample_cc_values.append(cc)
                                    
                                    # Calculate fine content (% passing 0.075mm sieve)
                                    fine_percent = np.interp(0.075, sorted_group['Sieve_Size_mm'], sorted_group['Percentage passing (%)'])
                                    sample_fine_content.append(fine_percent)
                                    
                                except:
                                    continue  # Skip samples with interpolation issues
                        
                        # Calculate statistics on sample-wise values
                        if sample_d10_values:
                            stats_data.extend([
                                {'Statistic': 'Samples Analyzed', 'Value': f"{len(sample_d10_values)}"},
                                {'Statistic': 'D10 Mean (mm)', 'Value': f"{np.mean(sample_d10_values):.3f}"},
                                {'Statistic': 'D10 Std Dev (mm)', 'Value': f"{np.std(sample_d10_values):.3f}"},
                                {'Statistic': 'D30 Mean (mm)', 'Value': f"{np.mean(sample_d30_values):.3f}"},
                                {'Statistic': 'D30 Std Dev (mm)', 'Value': f"{np.std(sample_d30_values):.3f}"},
                                {'Statistic': 'D50 Mean (mm)', 'Value': f"{np.mean(sample_d50_values):.3f}"},
                                {'Statistic': 'D50 Std Dev (mm)', 'Value': f"{np.std(sample_d50_values):.3f}"},
                                {'Statistic': 'D60 Mean (mm)', 'Value': f"{np.mean(sample_d60_values):.3f}"},
                                {'Statistic': 'D60 Std Dev (mm)', 'Value': f"{np.std(sample_d60_values):.3f}"},
                            ])
                        
                        if sample_cu_values:
                            stats_data.extend([
                                {'Statistic': 'Cu Mean', 'Value': f"{np.mean(sample_cu_values):.2f}"},
                                {'Statistic': 'Cu Std Dev', 'Value': f"{np.std(sample_cu_values):.2f}"},
                            ])
                        
                        if sample_cc_values:
                            stats_data.extend([
                                {'Statistic': 'Cc Mean', 'Value': f"{np.mean(sample_cc_values):.3f}"},
                                {'Statistic': 'Cc Std Dev', 'Value': f"{np.std(sample_cc_values):.3f}"},
                            ])
                        
                        if sample_fine_content:
                            stats_data.extend([
                                {'Statistic': 'Fine Content Mean (%)', 'Value': f"{np.mean(sample_fine_content):.1f}"},
                                {'Statistic': 'Fine Content Std Dev (%)', 'Value': f"{np.std(sample_fine_content):.1f}"},
                            ])
                        
                        # Gradation classification statistics
                        if sample_cu_values:
                            well_graded_count = sum(1 for cu in sample_cu_values if cu > 4)
                            uniform_graded_count = len(sample_cu_values) - well_graded_count
                            
                            stats_data.extend([
                                {'Statistic': 'Well-graded Samples', 'Value': f"{well_graded_count} ({100*well_graded_count/len(sample_cu_values):.1f}%)"},
                                {'Statistic': 'Uniform-graded Samples', 'Value': f"{uniform_graded_count} ({100*uniform_graded_count/len(sample_cu_values):.1f}%)"},
                            ])
                        
                        if not stats_data:
                            stats_data.append({'Statistic': 'Status', 'Value': 'Insufficient data for sample-wise analysis'})
                        
                        # Create statistics table
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No valid PSD data for statistics")
                        
                except Exception as e:
                    st.error(f"Error calculating statistics: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in PSD analysis: {str(e)}")
        st.write("Full error details:")
        st.exception(e)
