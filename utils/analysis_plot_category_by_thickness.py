"""
Thickness Analysis Module

This module handles thickness distribution analysis for geological formations,
following the Jupyter notebook workflow exactly.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Import Functions from Functions folder
try:
    import sys
    import os
    functions_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Functions')
    if functions_path not in sys.path:
        sys.path.insert(0, functions_path)
    
    from plot_category_by_thickness import plot_category_by_thickness
    HAS_FUNCTIONS = True
except ImportError as e:
    HAS_FUNCTIONS = False
    print(f"Warning: Could not import Functions: {e}")

# Import dashboard utilities
try:
    from .dashboard_site import store_spatial_plot
    HAS_DASHBOARD = True
except ImportError:
    try:
        from dashboard_site import store_spatial_plot
        HAS_DASHBOARD = True
    except ImportError:
        HAS_DASHBOARD = False

# Import spatial utilities
try:
    from .common_utility_tool import calculate_map_zoom_and_center
except ImportError:
    try:
        from common_utility_tool import calculate_map_zoom_and_center
    except ImportError:
        def calculate_map_zoom_and_center(lat_data, lon_data):
            """Fallback function if import fails"""
            return 10, {'lat': lat_data.mean(), 'lon': lon_data.mean()}


def load_bh_interpretation_data() -> Optional[pd.DataFrame]:
    """
    Load BH_Interpretation data from session state only.
    
    Returns:
        pd.DataFrame or None: BH_Interpretation data if available
    """
    try:
        # Only get from session state (uploaded data)
        if HAS_STREAMLIT and hasattr(st.session_state, 'bh_data'):
            return st.session_state.bh_data
        
        return None
            
    except Exception as e:
        if HAS_STREAMLIT:
            st.error(f"Error loading BH_Interpretation data: {str(e)}")
        return None


def get_grouping_columns(data):
    """Get all potential grouping columns with smart pattern matching"""
    grouping_patterns = [
        'geology', 'geology_orgin', 'geology origin', 'geologyorgin', 'geology_origin',
        'map_symbol', 'map symbol', 'mapsymbol', 'map_sym', 'symbol',
        'consistency', 'consist', 'strength', 'cohesion', 'condition',
        'report', 'project', 'reference', 'ref',
        'type', 'material_type', 'material type', 'materialtype', 'mat_type'
    ]
    potential_cols = []
    for col in data.columns:
        col_lower = col.lower().replace('(', '').replace(')', '').replace('-', '_').replace(' ', '_')
        # Check for exact matches and partial matches
        for pattern in grouping_patterns:
            if pattern in col_lower:
                potential_cols.append(col)
                break
    return list(set(potential_cols))  # Remove duplicates


def get_category_columns(data):
    """Get all potential category columns with smart pattern matching and flexibility"""
    # First, get columns matching known category patterns
    category_patterns = [
        'consistency', 'consist', 'strength', 'condition',
        'rock_class', 'rock class', 'rockclass', 'rock_type', 'rock type',
        'material_type', 'material type', 'materialtype', 'mat_type',
        'map_symbol', 'map symbol', 'mapsymbol', 'map_sym', 'symbol',
        'formation', 'layer', 'stratum', 'unit', 'classification', 'class',
        'category', 'group', 'type', 'kind', 'grade', 'zone', 'level'
    ]
    
    pattern_matched_cols = []
    all_potential_cols = []
    
    for col in data.columns:
        col_lower = col.lower().replace('(', '').replace(')', '').replace('-', '_').replace(' ', '_')
        
        # Add all columns except obvious numeric/system columns
        if not any(exclude in col_lower for exclude in ['thickness', 'depth', 'from_', 'to_', 'id', 'proportion', 'percent']):
            all_potential_cols.append(col)
        
        # Check for pattern matches (these will be prioritized)
        for pattern in category_patterns:
            if pattern in col_lower:
                pattern_matched_cols.append(col)
                break
    
    # Combine lists with pattern-matched columns first, then all other potential columns
    pattern_matched_cols = list(set(pattern_matched_cols))
    remaining_cols = [col for col in all_potential_cols if col not in pattern_matched_cols]
    
    return pattern_matched_cols + remaining_cols  # Pattern matches first, then others


def get_value_columns(data):
    """Get all potential value columns with smart pattern matching and flexibility"""
    # First, get columns matching known value patterns
    value_patterns = [
        'thickness', 'thick', 'depth', 'height', 'width',
        'proportion', 'percent', 'percentage', 'pct', '%',
        'value', 'amount', 'quantity', 'measure', 'count', 'sum', 'total'
    ]
    
    pattern_matched_cols = []
    all_numeric_cols = []
    
    for col in data.columns:
        col_lower = col.lower().replace('(', '').replace(')', '').replace('-', '_').replace(' ', '_')
        
        # Add all numeric columns (excluding obvious ID columns)
        try:
            if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                if not any(exclude in col_lower for exclude in ['id', 'hole', 'from_', 'to_', 'mbgl']):
                    all_numeric_cols.append(col)
        except:
            pass
        
        # Check for pattern matches (these will be prioritized)
        for pattern in value_patterns:
            if pattern in col_lower:
                pattern_matched_cols.append(col)
                break
        
        # Also check for columns ending with specific patterns
        if any(col_lower.endswith(suffix) for suffix in ['__%', '_percent', '_proportion']):
            pattern_matched_cols.append(col)
    
    # Combine lists with pattern-matched columns first, then all other numeric columns
    pattern_matched_cols = list(set(pattern_matched_cols))
    remaining_numeric = [col for col in all_numeric_cols if col not in pattern_matched_cols]
    
    return pattern_matched_cols + remaining_numeric  # Pattern matches first, then other numeric


def process_thickness_data(bh_data: pd.DataFrame, formation: str) -> pd.DataFrame:
    """
    Process thickness data for a specific geological formation exactly like Jupyter notebook.
    
    Args:
        bh_data: BH_Interpretation DataFrame
        formation: Geological formation code (e.g., 'Tos', 'Rjbw', 'Rin', 'Dcf')
        
    Returns:
        pd.DataFrame: Processed thickness data with proportions
    """
    try:
        # Extract formation data exactly like Jupyter notebook
        formation_data = bh_data.groupby('Geology_Orgin').get_group(formation)
        
        # Calculate thickness proportions using pivot_table exactly like Jupyter notebook
        thickness_summary = formation_data.pivot_table(
            values='Thickness', 
            index="Consistency", 
            aggfunc='sum'
        ).reset_index()
        
        # Add thickness proportion column exactly like Jupyter notebook
        total_thickness = thickness_summary['Thickness'].sum()
        thickness_summary.insert(
            loc=2, 
            column='thickness_proportion_%', 
            value=(thickness_summary['Thickness'] / total_thickness) * 100
        )
        
        return thickness_summary
        
    except KeyError as e:
        if HAS_STREAMLIT:
            st.error(f"Formation '{formation}' not found in data")
        return pd.DataFrame()
    except Exception as e:
        if HAS_STREAMLIT:
            st.error(f"Error processing thickness data: {str(e)}")
        return pd.DataFrame()


def render_thickness_analysis_tab(filtered_data: pd.DataFrame):
    """
    Render the Thickness Analysis tab with comprehensive parameter coverage.
    Following Jupyter notebook logic as golden standard.
    
    Args:
        filtered_data: Filtered laboratory data
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render tab.")
        return
    
    try:
        if filtered_data.empty:
            st.warning("No data available with current filters.")
            return
        
        # Try to load BH_Interpretation data
        bh_data = load_bh_interpretation_data()
        
        if bh_data is not None and not bh_data.empty:
            
            # Process data to add calculated columns BEFORE determining available columns
            # Add thickness proportion calculation if it doesn't exist
            if 'thickness_proportion_%' not in bh_data.columns and 'Thickness' in bh_data.columns:
                # Calculate proportions by group (geology/formation)
                if 'Geology_Orgin' in bh_data.columns:
                    # Group by Geology_Orgin and calculate proportions within each group
                    for group in bh_data['Geology_Orgin'].dropna().unique():
                        mask = bh_data['Geology_Orgin'] == group
                        group_data = bh_data[mask]
                        total_thickness = group_data['Thickness'].sum()
                        if total_thickness > 0:
                            bh_data.loc[mask, 'thickness_proportion_%'] = (group_data['Thickness'] / total_thickness) * 100
                        else:
                            bh_data.loc[mask, 'thickness_proportion_%'] = 0
                else:
                    # Fallback: calculate proportions for entire dataset
                    total_thickness = bh_data['Thickness'].sum()
                    if total_thickness > 0:
                        bh_data['thickness_proportion_%'] = (bh_data['Thickness'] / total_thickness) * 100
                    else:
                        bh_data['thickness_proportion_%'] = 0
            
            # Get all columns for maximum flexibility
            all_columns = list(bh_data.columns)
            
            # For grouping: Allow all columns, but put text columns first
            text_columns = [col for col in all_columns if bh_data[col].dtype == 'object']
            numeric_columns = [col for col in all_columns if col not in text_columns]
            
            # Build grouping columns list with smart ordering
            available_grouping_cols = []
            # Priority columns first
            priority_grouping = ['Geology_Orgin', 'Map_symbol', 'Report', 'Type']
            for col in priority_grouping:
                if col in text_columns:
                    available_grouping_cols.append(col)
                    text_columns.remove(col)
            # Then other text columns
            available_grouping_cols.extend(text_columns)
            # Then numeric columns
            available_grouping_cols.extend(numeric_columns)
            
            # For categories: Similar approach but prioritize different columns
            available_category_cols = []
            # Priority columns first
            priority_category = ['Consistency', 'Rock_Class', 'Material_Type', 'Map_symbol']
            remaining_cols = all_columns.copy()
            for col in priority_category:
                if col in remaining_cols:
                    available_category_cols.append(col)
                    remaining_cols.remove(col)
            # Then all other columns (excluding obvious numeric ones)
            exclude_from_category = ['From_mbgl', 'To_mbgl', 'Thickness', 'thickness_proportion_%']
            for col in remaining_cols:
                if col not in exclude_from_category:
                    available_category_cols.append(col)
            
            # For values: Use smart detection but include all numeric columns
            available_value_cols = get_value_columns(bh_data)
            if not available_value_cols:
                # Fallback - include all numeric columns
                available_value_cols = [col for col in bh_data.columns 
                                      if bh_data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
            
            if not available_grouping_cols:
                st.warning("No suitable grouping columns found in BH_Interpretation data.")
                return
            
            # Main parameter box (essential parameters) - following standard pattern
            with st.expander("Plot Parameters", expanded=True):
                # Enhanced helper functions for parsing tuple inputs
                def parse_tuple(input_str, default, param_name="parameter", min_val=None, max_val=None):
                    """Enhanced tuple parsing with validation and helpful error messages"""
                    try:
                        if not input_str or 'auto' in input_str.lower():
                            return None
                        
                        # Handle different input formats: "(12,8)", "12,8", "12 8"
                        cleaned = input_str.strip().replace('(', '').replace(')', '').replace(' ', ',')
                        # Split by comma or space, filter empty strings
                        parts = [x.strip() for x in cleaned.replace(',', ' ').split() if x.strip()]
                        
                        if len(parts) != 2:
                            st.warning(f"Invalid {param_name} format: '{input_str}'. Expected 2 values like '(12, 8)' or 'auto'. Using default.")
                            return default
                        
                        values = [float(x) for x in parts]
                        
                        # Validation checks
                        if min_val is not None:
                            if any(v < min_val for v in values):
                                st.warning(f"{param_name} values must be >= {min_val}. Got: {values}. Using default.")
                                return default
                        
                        if max_val is not None:
                            if any(v > max_val for v in values):
                                st.warning(f"{param_name} values must be <= {max_val}. Got: {values}. Using default.")
                                return default
                        
                        # Special validation for figsize
                        if param_name.lower() == "figsize":
                            if values[0] <= 0 or values[1] <= 0:
                                st.warning(f"Figure size must be positive. Got: {values}. Using default.")
                                return default
                            if values[0] > 50 or values[1] > 50:
                                st.warning(f"Figure size seems unusually large: {values}. Using default.")
                                return default
                        
                        return tuple(values)
                        
                    except ValueError as e:
                        st.warning(f"Could not parse {param_name}: '{input_str}'. Expected numbers like '(12, 8)'. Using default.")
                        return default
                    except Exception as e:
                        st.warning(f"Error parsing {param_name}: {str(e)}. Using default.")
                        return default
                
                def parse_list(input_str, param_name="list", available_options=None):
                    """Enhanced list parsing with validation"""
                    try:
                        if not input_str or input_str.strip() == '':
                            return None
                        
                        # Split by comma and clean up
                        items = [x.strip() for x in input_str.split(',') if x.strip()]
                        
                        # Validate against available options if provided
                        if available_options:
                            invalid_items = [item for item in items if item not in available_options]
                            if invalid_items:
                                st.warning(f"Invalid {param_name} items: {invalid_items}. Available options: {available_options[:10]}{'...' if len(available_options) > 10 else ''}")
                                # Return only valid items
                                valid_items = [item for item in items if item in available_options]
                                return valid_items if valid_items else None
                        
                        return items
                        
                    except Exception as e:
                        st.warning(f"Error parsing {param_name}: {str(e)}. Expected comma-separated values.")
                        return None
                
                # Row 1: Essential Data & Flexible Grouping
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    # Smart grouping column selection with priority
                    grouping_default_index = 0
                    if 'Geology_Orgin' in available_grouping_cols:
                        grouping_default_index = available_grouping_cols.index('Geology_Orgin')
                    
                    group_by_col = st.selectbox(
                        "Group By Column",
                        available_grouping_cols,
                        index=grouping_default_index,
                        key="thickness_group_by_col",
                        help="Column to group thickness data by. All columns are available. Default: Geology_Orgin"
                    )
                
                # Get available groups for selected column (handle mixed data types)
                try:
                    unique_groups = bh_data[group_by_col].dropna().unique()
                    # Convert to strings for sorting to handle mixed data types
                    available_groups = sorted([str(x) for x in unique_groups])
                except Exception as e:
                    # Fallback: use unsorted list
                    available_groups = [str(x) for x in bh_data[group_by_col].dropna().unique()]
                
                # Add "All" option at the beginning
                group_options = ["All"] + available_groups
                
                with col2:
                    selected_group = st.selectbox(
                        f"Selected {group_by_col}",
                        group_options,
                        index=0,  # "All" will be default
                        key="thickness_selected_group",
                        help="Select 'All' to analyze all groups together, or choose a specific group for focused analysis"
                    )
                
                with col3:
                    # Smart category column selection with priority
                    category_default_index = 0
                    if 'Consistency' in available_category_cols:
                        category_default_index = available_category_cols.index('Consistency')
                    
                    category_col = st.selectbox(
                        "Category Column",
                        available_category_cols,
                        index=category_default_index,
                        key="thickness_category_col",
                        help="Column for x-axis categories. All columns are available. Default: Consistency"
                    )
                
                with col4:
                    # Smart value column selection with priority
                    value_default_index = 0
                    if "thickness_proportion_%" in available_value_cols:
                        value_default_index = available_value_cols.index("thickness_proportion_%")
                    elif "Thickness" in available_value_cols:
                        value_default_index = available_value_cols.index("Thickness")
                    
                    value_col = st.selectbox(
                        "Value Column",
                        available_value_cols,
                        index=value_default_index,
                        key="thickness_value_col",
                        help="Column containing numeric values to plot. Options: thickness_proportion_% (percentages), Thickness (absolute values), or other measurement columns"
                    )
                
                with col5:
                    show_legend = st.selectbox(
                        "Show Legend",
                        [False, True],
                        index=0,
                        key="thickness_show_legend",
                        help="Whether to display the legend showing category colors"
                    )
                
                # Row 2: Plot Configuration
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    figsize_str = st.text_input("figsize (w, h)", value="(15, 7)", key="thickness_figsize",
                        help="Figure size as (width, height) in inches. Example: '(12, 8)' for larger plots")
                
                with col2:
                    xlim_str = st.text_input("xlim (min, max)", value="(auto, auto)", key="thickness_xlim",
                        help="X-axis limits. Use 'auto' for automatic or specify: '(0, 100)'")
                
                with col3:
                    ylim_str = st.text_input("ylim (min, max)", value="(auto, auto)", key="thickness_ylim",
                        help="Y-axis limits. Use 'auto' for automatic or specify: '(0, 50)'")
                
                with col4:
                    title = st.text_input("title", value="", key="thickness_title",
                        help="Custom plot title. Leave empty for automatic title generation")
                
                with col5:
                    title_suffix = st.text_input("title_suffix", value="", key="thickness_title_suffix",
                        help="Additional text to append to title")
                
                # Row 3: Visual Style Basics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    bar_width = st.number_input("bar_width", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="thickness_bar_width")
                
                with col2:
                    bar_alpha = st.number_input("bar_alpha", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="thickness_bar_alpha")
                
                with col3:
                    rotation = st.number_input("rotation", min_value=0, max_value=90, value=0, step=15, key="thickness_rotation")
                
                with col4:
                    percentage_decimal_places = st.number_input("decimal_places", min_value=0, max_value=3, value=1, key="thickness_decimal_places")
                
                with col5:
                    legend_bbox = st.text_input(
                        "Legend BBox (x,y)",
                        value="",
                        key="thickness_legend_bbox",
                        help="Legend position coordinates: Leave empty for automatic best location, or specify: '1.05,1' (right), '0.5,-0.1' (bottom center), '0,1' (top left)"
                    )
                
                # Row 4: Font Sizing
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    title_fontsize = st.number_input("title_fontsize", min_value=8, max_value=24, value=14, key="thickness_title_fontsize")
                
                with col2:
                    label_fontsize = st.number_input("label_fontsize", min_value=8, max_value=20, value=12, key="thickness_label_fontsize")
                
                with col3:
                    tick_fontsize = st.number_input("tick_fontsize", min_value=6, max_value=16, value=11, key="thickness_tick_fontsize")
                
                with col4:
                    legend_fontsize = st.number_input("legend_fontsize", min_value=8, max_value=16, value=10, key="thickness_legend_fontsize")
                
                with col5:
                    value_label_fontsize = st.number_input("value_label_fontsize", min_value=6, max_value=14, value=10, key="thickness_value_label_fontsize")
            
            # Advanced Parameters Section (separate from main expander)
            with st.expander("Advanced Parameters", expanded=False):
                
                # Advanced Row 1: Visual Controls & Styling
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    bar_edgecolor = st.text_input(
                        "Bar Edge Color",
                        value="black",
                        key="thickness_edge_color",
                        help="Color for bar edges (e.g., black, blue, #FF0000)"
                    )
                with adv_col2:
                    bar_linewidth = st.number_input(
                        "Bar Line Width",
                        min_value=0.0,
                        max_value=3.0,
                        value=0.6,
                        step=0.1,
                        key="thickness_line_width",
                        help="Width of bar edge lines"
                    )
                with adv_col3:
                    plot_style = st.selectbox(
                        "Plot Style",
                        ["seaborn-v0_8-whitegrid", "default", "classic", "seaborn-v0_8-colorblind"],
                        index=0,
                        key="thickness_plot_style",
                        help="Overall plot styling theme"
                    )
                with adv_col4:
                    bar_hatch = st.selectbox(
                        "Bar Hatch Pattern",
                        ["None", "//", "\\\\", "|||", "---", "+++", "xxx", "ooo", "...", "***"],
                        index=0,
                        key="thickness_bar_hatch",
                        help="Pattern for bar filling (useful for accessibility)"
                    )
                with adv_col5:
                    dpi = st.number_input(
                        "Save DPI",
                        min_value=150,
                        max_value=600,
                        value=300,
                        step=50,
                        key="thickness_save_dpi",
                        help="Resolution for saved plots"
                    )
                
                # Advanced Row 2: Grid & Legend Controls
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    grid_axis = st.selectbox(
                        "Grid Axis",
                        ["y", "x", "both"],
                        index=0,
                        key="thickness_grid_axis",
                        help="Which axes to show grid lines on"
                    )
                with adv_col2:
                    grid_linestyle = st.selectbox(
                        "Grid Line Style",
                        ["--", "-", ":", "-."],
                        index=0,
                        key="thickness_grid_linestyle",
                        help="Style of grid lines"
                    )
                with adv_col3:
                    grid_alpha = st.number_input(
                        "Grid Alpha",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.35,
                        step=0.05,
                        key="thickness_grid_alpha",
                        help="Transparency of grid lines"
                    )
                with adv_col4:
                    legend_loc = st.selectbox(
                        "Legend Location",
                        ["best", "upper right", "upper left", "lower left", "lower right"],
                        index=0,
                        key="thickness_legend_loc",
                        help="Default legend position when bbox not specified"
                    )
                with adv_col5:
                    colors_str = st.text_input(
                        "Custom Colors",
                        value="",
                        key="thickness_colors",
                        help="Custom colors: red,blue,green OR #FF0000,#0000FF,#00FF00 (comma-separated)"
                    )
                
                # Advanced Row 3: Custom Labels & Ordering
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    xlabel_custom = st.text_input(
                        "Custom X-Label",
                        value="",
                        key="thickness_xlabel_custom",
                        help="Override default x-axis label. Leave empty for auto"
                    )
                with adv_col2:
                    ylabel_custom = st.text_input(
                        "Custom Y-Label",
                        value="",
                        key="thickness_ylabel_custom",
                        help="Override default y-axis label. Leave empty for auto"
                    )
                with adv_col3:
                    category_order_str = st.text_input(
                        "Category Order",
                        value="",
                        key="thickness_category_order",
                        help="Custom order: 1a,2a,3a,4a,5a (comma-separated). Leave empty for auto"
                    )
                with adv_col4:
                    legend_order_str = st.text_input(
                        "Legend Order",
                        value="",
                        key="thickness_legend_order",
                        help="Custom legend order: 5a,4a,3a,2a,1a (comma-separated). Leave empty for same as x-axis"
                    )
                with adv_col5:
                    title_fontweight = st.selectbox(
                        "Title Font Weight",
                        ["bold", "normal", "light", "heavy"],
                        index=0,
                        key="thickness_title_fontweight",
                        help="Font weight for plot title"
                    )
                
                # Advanced Row 4: Font Styling
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    label_fontweight = st.selectbox(
                        "Label Font Weight",
                        ["bold", "normal", "light", "heavy"],
                        index=0,
                        key="thickness_label_fontweight",
                        help="Font weight for axis labels"
                    )
                with adv_col2:
                    show_grid = st.selectbox(
                        "Show Grid",
                        [True, False],
                        index=0,
                        key="thickness_show_grid",
                        help="Whether to display grid lines"
                    )
                with adv_col3:
                    show_percentage_labels = st.selectbox(
                        "Show Value Labels",
                        [True, False],
                        index=0,
                        key="thickness_show_labels",
                        help="Whether to show value labels on bars"
                    )
                with adv_col4:
                    x_axis_sort = st.selectbox(
                        "X-axis Sort",
                        ["alphabetical", "ascending", "descending"],
                        index=0,
                        key="thickness_x_sort",
                        help="How to sort categories on x-axis"
                    )
                with adv_col5:
                    st.write("")  # Empty placeholder for alignment
            
            # Filter data based on selected group
            if selected_group == "All":
                filtered_bh_data = bh_data.copy()
            else:
                try:
                    # Try direct comparison first
                    filtered_bh_data = bh_data[bh_data[group_by_col] == selected_group]
                except Exception:
                    # Fallback: string comparison for mixed data types
                    filtered_bh_data = bh_data[bh_data[group_by_col].astype(str) == str(selected_group)]
            
            if filtered_bh_data.empty:
                st.warning(f"No data available for selected {group_by_col}: {selected_group}")
                return
            
            # Group by category column and calculate thickness statistics
            try:
                thickness_data = filtered_bh_data.groupby(category_col)['Thickness'].sum().reset_index()
                
                # Add proportion calculation
                total_thickness = thickness_data['Thickness'].sum()
                thickness_data['thickness_proportion_%'] = (thickness_data['Thickness'] / total_thickness) * 100
                
                if thickness_data.empty:
                    st.warning(f"No thickness data available for {selected_group}")
                    return
                
                # Parse parameter inputs with enhanced validation
                base_figsize = parse_tuple(figsize_str, (15, 7), "figsize", min_val=1)
                
                # Apply sidebar plot size if available
                try:
                    if hasattr(st.session_state, 'plot_display_settings') and 'width_percentage' in st.session_state.plot_display_settings:
                        width_pct = st.session_state.plot_display_settings['width_percentage']
                        # Scale the width based on percentage (30-100% range maps to 0.3-1.0 scale)
                        width_scale = width_pct / 100.0
                        # Apply scaling to base figure width
                        scaled_width = base_figsize[0] * width_scale
                        figsize = (scaled_width, base_figsize[1])
                    else:
                        figsize = base_figsize
                except:
                    figsize = base_figsize
                xlim = parse_tuple(xlim_str, None, "xlim")
                ylim = parse_tuple(ylim_str, None, "ylim", min_val=0)
                
                # Get available categories for validation
                available_categories = thickness_data[category_col].astype(str).tolist() if not thickness_data.empty else []
                category_order = parse_list(category_order_str, "category order", available_categories)
                legend_order = parse_list(legend_order_str, "legend order", available_categories)
                colors = parse_list(colors_str, "colors") if colors_str else None
                
                # Dynamic group name mapping with fallback to hardcoded
                def get_dynamic_group_names(data, group_col):
                    """Dynamically detect group name mappings in the data"""
                    group_names = {}
                    
                    # Get unique values from the data
                    unique_values = data[group_col].dropna().unique()
                    
                    for value in unique_values:
                        # Use the value as-is for now, could add mapping logic later
                        group_names[value] = str(value)
                    
                    return group_names
                
                group_names = get_dynamic_group_names(bh_data, group_by_col)
                
                # Get group full name with fallback
                group_full_name = group_names.get(selected_group, str(selected_group))
                
                # Handle bar hatch (convert "None" to None)
                bar_hatch_final = None if bar_hatch == "None" else bar_hatch
                
                # Parse legend bbox (x,y coordinates only)
                if legend_bbox.strip() == "":
                    # Empty input = automatic best location
                    legend_style_dict = {
                        'loc': legend_loc
                    }
                else:
                    try:
                        bbox_parts = [float(x.strip()) for x in legend_bbox.split(',')]
                        if len(bbox_parts) == 2:
                            legend_bbox_to_anchor = tuple(bbox_parts)
                        else:
                            legend_bbox_to_anchor = (1.05, 1)
                    except:
                        legend_bbox_to_anchor = (1.05, 1)
                    
                    # Create legend style dictionary with bbox_to_anchor
                    legend_style_dict = {
                        'bbox_to_anchor': legend_bbox_to_anchor,
                        'loc': 'upper left'
                    }
                
                # Determine ylabel based on selected value column
                if 'proportion' in value_col.lower() or '%' in value_col:
                    ylabel = 'Distribution (%)'
                elif 'thickness' in value_col.lower():
                    ylabel = 'Thickness (m)'
                else:
                    ylabel = value_col.replace('_', ' ').title()
                
                # Handle custom labels
                final_xlabel = xlabel_custom if xlabel_custom.strip() else 'Rock Class Unit'
                final_ylabel = ylabel_custom if ylabel_custom.strip() else ylabel
                
                # Generate final title
                if title:
                    final_title = title
                else:
                    # Format: "Distribution of [category column] by thickness" for All, or "Distribution of [category column] by thickness - [category]" for specific
                    if selected_group == "All":
                        final_title = f"Distribution of {category_col} by thickness"
                    else:
                        final_title = f"Distribution of {category_col} by thickness - {selected_group}"
                
                if title_suffix:
                    final_title += f" {title_suffix}"
                
                # Create thickness distribution plot
                try:
                    # Import matplotlib at function level to avoid scope issues
                    import matplotlib.pyplot as plt
                    
                    # Clear any existing figures first
                    plt.close('all')
                    
                    # Create the plot using Functions folder with comprehensive parameters
                    # Temporarily patch plt.close to prevent figure from being closed
                    import matplotlib.pyplot as plt
                    original_close = plt.close
                    saved_fig = None
                    
                    def patched_close(fig=None):
                        # Save the figure before it gets closed
                        nonlocal saved_fig
                        if fig is None:
                            saved_fig = plt.gcf()
                        else:
                            saved_fig = fig
                        # Don't actually close the figure
                        pass
                    
                    # Apply the patch
                    plt.close = patched_close
                    
                    try:
                        plot_category_by_thickness(
                            df=thickness_data,
                            value_col=value_col,
                            category_col=category_col,
                            title=final_title,
                            figsize=figsize,
                            xlim=xlim,
                            ylim=ylim,
                            xlabel=final_xlabel,
                            ylabel=final_ylabel,
                            category_order=category_order,
                            x_axis_sort=x_axis_sort,
                            legend_order=legend_order,
                            colors=colors,
                            bar_hatch=bar_hatch_final,
                            plot_style=plot_style if 'plot_style' in locals() else None,
                            tick_fontsize=tick_fontsize,
                            title_fontsize=title_fontsize,
                            title_fontweight=title_fontweight,
                            xlabel_fontsize=label_fontsize,
                            ylabel_fontsize=label_fontsize,
                            label_fontweight=label_fontweight,
                            legend_fontsize=legend_fontsize,
                            value_label_fontsize=value_label_fontsize,
                            bar_alpha=bar_alpha,
                            bar_width=bar_width,
                            bar_edgecolor=bar_edgecolor,
                            bar_linewidth=bar_linewidth,
                            rotation=rotation,
                            show_legend=show_legend,
                            show_grid=show_grid,
                            grid_axis=grid_axis,
                            legend_style=legend_style_dict,
                            show_percentage_labels=show_percentage_labels,
                            percentage_decimal_places=percentage_decimal_places,
                            show_plot=False
                        )
                    finally:
                        # Restore the original close function
                        plt.close = original_close
                    
                    # Use the saved figure or get current figure
                    current_fig = saved_fig or plt.gcf()
                    
                    # Display the plot with Streamlit using sidebar width control
                    if current_fig and current_fig.get_axes():
                        # Apply sidebar plot size setting to display (left-aligned)
                        try:
                            if hasattr(st.session_state, 'plot_display_settings') and 'width_percentage' in st.session_state.plot_display_settings:
                                width_pct = st.session_state.plot_display_settings['width_percentage']
                                # Create columns to control width (left-aligned)
                                if width_pct < 100:
                                    right_margin = 100 - width_pct
                                    col1, col2 = st.columns([width_pct, right_margin])
                                    with col1:
                                        st.pyplot(current_fig, use_container_width=True)
                                else:
                                    st.pyplot(current_fig, use_container_width=True)
                            else:
                                st.pyplot(current_fig)
                        except:
                            st.pyplot(current_fig)
                        success = True
                    else:
                        st.error("Failed to create plot - no figure available")
                        success = False
                    
                    if success:
                        # Store plot for dashboard
                        try:
                            current_fig = plt.gcf()
                            if current_fig and current_fig.get_axes():
                                import io
                                buf = io.BytesIO()
                                current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                store_spatial_plot('thickness_distribution', buf)
                        except Exception as e:
                            pass
                        
                        # Simple download button with figure reference
                        from .plot_download_simple import create_simple_download_button
                        create_simple_download_button("thickness_distribution", "thickness", fig=current_fig)
                        
                        # Enhanced plot summary
                        st.divider()
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{category_col} Distribution Summary**")
                            
                            # Show processed data summary
                            if not thickness_data.empty:
                                display_cols = [category_col, 'Thickness', 'thickness_proportion_%']
                                display_data = thickness_data[display_cols].copy()
                                display_data['thickness_proportion_%'] = display_data['thickness_proportion_%'].round(2)
                                st.dataframe(display_data, use_container_width=True)
                                st.caption(f"{len(thickness_data)} {category_col.lower()} classes")
                        
                        with col2:
                            st.markdown("**Data Source Information**")
                            st.write(f"• **Group By**: {group_by_col}")
                            st.write(f"• **Selected**: {selected_group}")
                            st.write(f"• **Category**: {category_col}")
                            st.write(f"• **Value**: {value_col}")
                            st.write(f"• **Total Records**: {len(filtered_bh_data)}")
                            st.write(f"• **Categories**: {len(thickness_data)}")
                            
                            if 'thickness_proportion_%' in thickness_data.columns:
                                total_thickness = thickness_data['Thickness'].sum()
                                st.write(f"• **Total {value_col}**: {total_thickness:.2f}")
                    else:
                        st.warning("Plot function completed but no plot was displayed")
                        
                except Exception as e:
                    st.error(f"Error creating thickness plot: {str(e)}")
                    st.exception(e)
                
            except Exception as e:
                st.error(f"Error processing thickness data: {str(e)}")
                st.exception(e)
                
        else:
            st.warning("BH_Interpretation data not available. Please ensure BH_Interpretation.xlsx is in the Input folder.")
            st.info("The BH_Interpretation data is required for thickness analysis and geological interpretation.")
            
    except Exception as e:
        st.error(f"Error in thickness analysis: {str(e)}")
        st.exception(e)


def render_thickness_analysis_map_visualization(bh_data: pd.DataFrame, filtered_data: pd.DataFrame) -> None:
    """
    Render map visualization for thickness analysis data.
    """
    if not HAS_PLOTLY or not HAS_PYPROJ:
        st.info("Map visualization requires plotly and pyproj packages")
        return
    
    st.subheader("Thickness Analysis Site Locations")
    
    # Check for coordinate columns
    coord_columns = []
    for col in ['Easting', 'Northing', 'X', 'Y', 'Longitude', 'Latitude']:
        if col in filtered_data.columns:
            coord_columns.append(col)
    
    if len(coord_columns) >= 2:
        try:
            # Determine coordinate system
            easting_col = next((col for col in ['Easting', 'X'] if col in coord_columns), coord_columns[0])
            northing_col = next((col for col in ['Northing', 'Y'] if col in coord_columns), coord_columns[1])
            
            # Create map data
            map_data = filtered_data[[easting_col, northing_col, 'Hole_ID', 'Geology_Orgin']].dropna()
            
            if not map_data.empty:
                # Convert UTM to WGS84 if needed
                try:
                    # Assume UTM Zone 56S (common for Queensland, Australia)
                    transformer = pyproj.Transformer.from_crs('EPSG:28356', 'EPSG:4326', always_xy=True)
                    lon, lat = transformer.transform(map_data[easting_col].values, map_data[northing_col].values)
                    
                    map_data['Longitude'] = lon
                    map_data['Latitude'] = lat
                except:
                    # If conversion fails, assume already in lat/lon
                    map_data['Longitude'] = map_data[easting_col]
                    map_data['Latitude'] = map_data[northing_col]
                
                # Calculate optimal zoom and center
                zoom_level, center = calculate_map_zoom_and_center(map_data['Latitude'], map_data['Longitude'])
                
                # Create plotly map
                fig = px.scatter_mapbox(
                    map_data,
                    lat='Latitude',
                    lon='Longitude',
                    hover_name='Hole_ID',
                    hover_data=['Geology_Orgin'],
                    color='Geology_Orgin',
                    title="Borehole Locations for Thickness Analysis",
                    mapbox_style='open-street-map',
                    height=500,
                    zoom=zoom_level,
                    center={'lat': center['lat'], 'lon': center['lon']}
                )
                
                fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)
                
                # Location statistics
                st.write(f"**Map Statistics:** {len(map_data)} boreholes shown")
                
        except Exception as e:
            st.info(f"Could not create map: {str(e)}")
    else:
        st.info("No coordinate data available for mapping")


def render_thickness_analysis_plot_summary(thickness_data: pd.DataFrame, formation: str) -> None:
    """
    Render enhanced plot summary with engineering interpretations for thickness analysis.
    """
    st.subheader("Thickness Analysis Summary")
    
    if thickness_data.empty:
        st.warning("No thickness data available for summary")
        return
    
    # Calculate summary statistics
    total_thickness = thickness_data['Thickness'].sum()
    mean_thickness = thickness_data['Thickness'].mean()
    median_thickness = thickness_data['Thickness'].median()
    std_thickness = thickness_data['Thickness'].std()
    
    # Create summary DataFrame
    summary_data = []
    summary_data.extend([
        {'Parameter': 'Formation', 'Value': formation, 'Unit': '-'},
        {'Parameter': 'Total Thickness', 'Value': f"{total_thickness:.2f}", 'Unit': 'm'},
        {'Parameter': 'Number of Rock Classes', 'Value': str(len(thickness_data)), 'Unit': '-'},
        {'Parameter': 'Mean Thickness', 'Value': f"{mean_thickness:.2f}", 'Unit': 'm'},
        {'Parameter': 'Median Thickness', 'Value': f"{median_thickness:.2f}", 'Unit': 'm'},
        {'Parameter': 'Standard Deviation', 'Value': f"{std_thickness:.2f}", 'Unit': 'm'},
        {'Parameter': 'Coefficient of Variation', 'Value': f"{(std_thickness/mean_thickness)*100:.1f}", 'Unit': '%'}
    ])
    
    # Add rock class breakdown
    st.write("**Formation Thickness Summary:**")
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Rock class distribution
    st.write("**Rock Class Distribution:**")
    class_data = []
    for _, row in thickness_data.iterrows():
        class_data.append({
            'Rock Class': row['Consistency'],
            'Thickness (m)': f"{row['Thickness']:.2f}",
            'Proportion (%)': f"{row['thickness_proportion_%']:.1f}"
        })
    
    class_df = pd.DataFrame(class_data)
    st.dataframe(class_df, use_container_width=True, hide_index=True)
    
    # Engineering interpretation
    st.write("**Engineering Interpretation:**")
    interpretation_text = []
    
    # Formation assessment
    if total_thickness > 20:
        interpretation_text.append(f"• Substantial {formation} formation thickness ({total_thickness:.1f}m)")
    elif total_thickness > 10:
        interpretation_text.append(f"• Moderate {formation} formation thickness ({total_thickness:.1f}m)")
    else:
        interpretation_text.append(f"• Limited {formation} formation thickness ({total_thickness:.1f}m)")
    
    # Variability assessment
    cv = (std_thickness/mean_thickness)*100
    if cv > 50:
        interpretation_text.append(f"• High thickness variability (CV = {cv:.1f}%)")
    elif cv > 25:
        interpretation_text.append(f"• Moderate thickness variability (CV = {cv:.1f}%)")
    else:
        interpretation_text.append(f"• Low thickness variability (CV = {cv:.1f}%)")
    
    # Dominant rock class
    dominant_class = thickness_data.loc[thickness_data['Thickness'].idxmax(), 'Consistency']
    max_proportion = thickness_data['thickness_proportion_%'].max()
    interpretation_text.append(f"• Dominant rock class: {dominant_class} ({max_proportion:.1f}%)")
    
    for text in interpretation_text:
        st.write(text)