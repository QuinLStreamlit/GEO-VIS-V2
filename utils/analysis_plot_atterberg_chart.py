"""
Atterberg Limits Analysis Module

This module handles Atterberg limits data processing, analysis, and visualization for geotechnical applications.
"""

import pandas as pd
import numpy as np
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
except ImportError:
    HAS_PYPROJ = False

try:
    from .data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_id_columns_from_data
    from .plot_defaults import get_default_parameters, get_color_schemes
    from .dashboard_materials import store_material_plot
    HAS_PLOTTING_UTILS = True
except ImportError:
    # For standalone testing
    from data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_id_columns_from_data
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
    
    from plot_atterberg_chart import plot_atterberg_chart
    HAS_FUNCTIONS = True
except ImportError as e:
    HAS_FUNCTIONS = False
    print(f"Warning: Could not import Functions: {e}")


def extract_atterberg_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Atterberg limits test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: Atterberg-specific dataframe
    """
    # Use dynamic ID columns detection by passing dataframe to get_standard_id_columns
    id_columns = get_standard_id_columns(df)
    atterberg_columns = extract_test_columns(df, 'Atterberg')
    
    if not atterberg_columns:
        raise ValueError("No Atterberg data columns found")
    
    return create_test_dataframe(df, 'Atterberg', id_columns, atterberg_columns)


def get_atterberg_columns(atterberg_df: pd.DataFrame) -> Dict[str, str]:
    """
    Identify liquid limit and plasticity index columns from Atterberg dataframe.
    
    Args:
        atterberg_df: Atterberg dataframe
        
    Returns:
        Dict[str, str]: Dictionary with 'll_col' and 'pi_col' keys
    """
    # Common column patterns for Liquid Limit
    ll_patterns = ['LL (%)', 'LL(%)', 'Liquid_Limit', 'LL', 'liquid_limit']
    # Common column patterns for Plasticity Index  
    pi_patterns = ['PI (%)', 'PI(%)', 'Plasticity_Index', 'PI', 'plasticity_index']
    
    ll_col = None
    pi_col = None
    
    # Find LL column
    for pattern in ll_patterns:
        if pattern in atterberg_df.columns:
            ll_col = pattern
            break
    
    # Find PI column
    for pattern in pi_patterns:
        if pattern in atterberg_df.columns:
            pi_col = pattern
            break
    
    # If exact matches not found, look for partial matches
    if ll_col is None:
        for col in atterberg_df.columns:
            if any(pattern.lower() in col.lower() for pattern in ['ll', 'liquid']):
                ll_col = col
                break
    
    if pi_col is None:
        for col in atterberg_df.columns:
            if any(pattern.lower() in col.lower() for pattern in ['pi', 'plasticity']):
                pi_col = col
                break
    
    return {'ll_col': ll_col, 'pi_col': pi_col}


def classify_soil_type(ll: float, pi: float) -> str:
    """
    Classify soil type based on Atterberg limits using USCS classification.
    
    Args:
        ll: Liquid Limit
        pi: Plasticity Index
        
    Returns:
        str: Soil classification
    """
    if pd.isna(ll) or pd.isna(pi):
        return 'Unknown'
    
    # A-line equation: PI = 0.73 * (LL - 20)
    a_line_pi = 0.73 * (ll - 20) if ll > 20 else 4
    
    # U-line equation: PI = 0.9 * (LL - 8)  
    u_line_pi = 0.9 * (ll - 8) if ll > 8 else 7
    
    # Classification logic
    if ll < 50:  # Low plasticity
        if pi < 4:
            return 'CL-ML'
        elif pi > a_line_pi and pi < u_line_pi:
            return 'CL'
        elif pi < a_line_pi:
            return 'ML'
        else:
            return 'CL or OL'
    else:  # High plasticity (LL >= 50)
        if pi > a_line_pi:
            return 'CH'
        else:
            return 'MH'


def calculate_atterberg_statistics(atterberg_df: pd.DataFrame, ll_col: str, pi_col: str) -> pd.DataFrame:
    """
    Calculate Atterberg limits statistics and soil classifications.
    
    Args:
        atterberg_df: Atterberg dataframe
        ll_col: Liquid limit column name
        pi_col: Plasticity index column name
        
    Returns:
        pd.DataFrame: Statistics dataframe with classifications
    """
    stats_list = []
    
    for _, row in atterberg_df.iterrows():
        if pd.notna(row.get(ll_col)) and pd.notna(row.get(pi_col)):
            ll_value = float(row[ll_col])
            pi_value = float(row[pi_col])
            
            # Calculate derived properties
            pl_value = ll_value - pi_value  # Plastic Limit
            
            # Classify soil
            soil_type = classify_soil_type(ll_value, pi_value)
            
            stats_list.append({
                'Hole_ID': row.get('Hole_ID', ''),
                'From_mbgl': row.get('From_mbgl', np.nan),
                'To_mbgl': row.get('To_mbgl', np.nan),
                'Geology_Orgin': row.get('Geology_Orgin', ''),
                'Consistency': row.get('Consistency', ''),
                'LL': ll_value,
                'PI': pi_value,
                'PL': pl_value,
                'Soil_Classification': soil_type
            })
    
    return pd.DataFrame(stats_list)


def create_plasticity_chart(atterberg_df: pd.DataFrame, ll_col: str, pi_col: str, parameters: Dict[str, Any]):
    """
    Create Atterberg plasticity chart with A-line and U-line.
    
    Args:
        atterberg_df: Atterberg dataframe
        ll_col: Liquid limit column name
        pi_col: Plasticity index column name
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
    if group_by == 'geology' and 'Geology_Orgin' in atterberg_df.columns:
        group_col = 'Geology_Orgin'
    elif group_by == 'consistency' and 'Consistency' in atterberg_df.columns:
        group_col = 'Consistency'
    else:
        group_col = 'Hole_ID'
    
    # Plot data points by groups
    grouped = atterberg_df.groupby(group_col)
    color_idx = 0
    
    for group_val, group_data in grouped:
        # Skip groups with missing LL or PI data
        valid_data = group_data.dropna(subset=[ll_col, pi_col])
        if valid_data.empty:
            continue
            
        color = colors[color_idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=valid_data[ll_col],
            y=valid_data[pi_col],
            mode='markers',
            name=str(group_val),
            marker=dict(
                size=parameters.get('marker_size', 8),
                color=color,
                opacity=parameters.get('alpha', 0.8),
                line=dict(width=1, color='black')
            ),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         f'{ll_col}: %{{x:.1f}}%<br>' +
                         f'{pi_col}: %{{y:.1f}}%<br>' +
                         '<extra></extra>'
        ))
        
        color_idx += 1
    
    # Add A-line and U-line if requested
    if parameters.get('show_a_line', True):
        x_range = parameters.get('x_range', (0, 100))
        ll_range = np.linspace(20, x_range[1], 100)
        a_line_pi = 0.73 * (ll_range - 20)
        
        fig.add_trace(go.Scatter(
            x=ll_range,
            y=a_line_pi,
            mode='lines',
            name='A-line',
            line=dict(color='black', width=2, dash='solid'),
            hovertemplate='A-line: PI = 0.73(LL-20)<extra></extra>'
        ))
        
        # Add horizontal line at PI=4 for LL < 25.48
        fig.add_trace(go.Scatter(
            x=[x_range[0], 25.48],
            y=[4, 4],
            mode='lines',
            name='A-line extension',
            line=dict(color='black', width=2, dash='solid'),
            showlegend=False,
            hovertemplate='A-line: PI = 4<extra></extra>'
        ))
    
    if parameters.get('show_u_line', True):
        x_range = parameters.get('x_range', (0, 100))
        ll_range = np.linspace(8, x_range[1], 100)
        u_line_pi = 0.9 * (ll_range - 8)
        
        fig.add_trace(go.Scatter(
            x=ll_range,
            y=u_line_pi,
            mode='lines',
            name='U-line',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='U-line: PI = 0.9(LL-8)<extra></extra>'
        ))
        
        # Add horizontal line at PI=7 for LL < 16.33
        fig.add_trace(go.Scatter(
            x=[x_range[0], 16.33],
            y=[7, 7],
            mode='lines',
            name='U-line extension',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False,
            hovertemplate='U-line: PI = 7<extra></extra>'
        ))
    
    # Add vertical lines at LL=50
    if parameters.get('show_classification_zones', True):
        y_range = parameters.get('y_range', (0, 60))
        fig.add_trace(go.Scatter(
            x=[50, 50],
            y=[0, y_range[1]],
            mode='lines',
            name='LL=50 line',
            line=dict(color='black', width=1, dash='dot'),
            showlegend=False,
            hovertemplate='LL = 50<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=parameters.get('title', 'Plasticity Chart (Atterberg Limits)'),
        xaxis_title=parameters.get('x_label', 'Liquid Limit (%)'),
        yaxis_title=parameters.get('y_label', 'Plasticity Index (%)'),
        width=parameters.get('figure_size', (10, 8))[0] * 100,
        height=parameters.get('figure_size', (10, 8))[1] * 100,
        showlegend=parameters.get('legend', True),
        hovermode='closest'
    )
    
    # Set axes ranges
    x_range = parameters.get('x_range', (0, 100))
    y_range = parameters.get('y_range', (0, 60))
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)
    
    # Add grid
    if parameters.get('grid', True):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def render_atterberg_analysis_tab(filtered_data: pd.DataFrame):
    """
    Render the Atterberg limits analysis tab in Streamlit.
    Uses original plotting functions from Functions folder exactly as in Jupyter notebook.
    
    Args:
        filtered_data: Filtered laboratory data
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render tab.")
        return
    
    try:
        # Extract Atterberg data exactly like Jupyter notebook
        atterberg_data = extract_atterberg_data(filtered_data)
        
        if atterberg_data.empty:
            st.warning("No Atterberg limits data available with current filters.")
            return
        
        
        # Get column mappings
        column_info = get_atterberg_columns(atterberg_data)
        ll_col = column_info['ll_col']
        pi_col = column_info['pi_col']
        
        if not ll_col or not pi_col:
            st.error("Could not identify Liquid Limit and Plasticity Index columns")
            st.info("Expected columns: 'LL (%)', 'PI (%)', or similar naming patterns")
            
            # Debug - show available columns
            with st.expander("Debug: Available Atterberg Columns"):
                st.write("Available columns:")
                for col in atterberg_data.columns:
                    sample_vals = atterberg_data[col].dropna().head(3).tolist()
                    st.write(f"- `{col}`: {sample_vals}")
            return
        
        # Helper function for parsing tuple inputs
        def parse_tuple(input_str, default):
            try:
                # Remove parentheses and split by comma
                cleaned = input_str.strip().replace('(', '').replace(')', '')
                values = [float(x.strip()) for x in cleaned.split(',')]
                return tuple(values) if len(values) == 2 else default
            except:
                return default
        
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
        
        # Get standard ID columns for parameter selection using dynamic detection
        from .data_processing import get_standard_id_columns
        standard_id_columns = get_standard_id_columns(atterberg_data)
        
        # Helper function to find smart defaults
        def find_column(patterns, columns, default=""):
            for pattern in patterns:
                for col in columns:
                    if pattern.lower() in col.lower():
                        return col
            return default
        
        # Find smart defaults for all columns
        hole_id_default = find_column(["hole_id", "hole", "borehole"], standard_id_columns, "Hole_ID")
        depth_default = find_column(["from_mbgl", "depth", "from"], standard_id_columns, "From_mbgl")
        consistency_default = find_column(["consistency", "consist"], atterberg_data.columns, "Consistency")
        geology_default = find_column(["geology_orgin", "geology", "geological"], atterberg_data.columns, "Geology_Orgin")
        
        # Smart defaults for LL and PI columns with broader pattern matching
        def get_ll_columns():
            """Get all potential LL columns with smart pattern matching"""
            ll_patterns = ['ll', 'liquid_limit', 'liquid limit', 'liquidlimit', 'l.l', 'l_l']
            potential_cols = []
            for col in atterberg_data.columns:
                col_lower = col.lower()
                # Check for exact matches and partial matches
                for pattern in ll_patterns:
                    if pattern in col_lower:
                        potential_cols.append(col)
                        break
                # Also check for LL with numbers (LL1, LL2, LL3, etc.)
                if 'll' in col_lower and any(char.isdigit() for char in col_lower):
                    potential_cols.append(col)
            return list(set(potential_cols))  # Remove duplicates
        
        def get_pi_columns():
            """Get all potential PI columns with smart pattern matching"""
            pi_patterns = ['pi', 'plasticity_index', 'plasticity index', 'plasticityindex', 'p.i', 'p_i', 'plastic_index']
            potential_cols = []
            for col in atterberg_data.columns:
                col_lower = col.lower()
                # Check for exact matches and partial matches
                for pattern in pi_patterns:
                    if pattern in col_lower:
                        potential_cols.append(col)
                        break
                # Also check for PI with numbers (PI1, PI2, PI3, etc.)
                if 'pi' in col_lower and any(char.isdigit() for char in col_lower):
                    potential_cols.append(col)
            return list(set(potential_cols))  # Remove duplicates
        
        # Get available LL and PI columns
        available_ll_cols = get_ll_columns()
        available_pi_cols = get_pi_columns()
        
        # Find best defaults - prefer exact matches, then with numbers
        ll_default = None
        pi_default = None
        
        # Priority order for LL: LL, LL (%), LL1, LL2, etc.
        ll_priority = ['LL (%)', 'LL', 'LL(%)', 'Liquid_Limit', 'Liquid Limit']
        for preferred in ll_priority:
            if preferred in available_ll_cols:
                ll_default = preferred
                break
        if not ll_default and available_ll_cols:
            ll_default = available_ll_cols[0]
        
        # Priority order for PI: PI, PI (%), PI1, PI2, etc.
        pi_priority = ['PI (%)', 'PI', 'PI(%)', 'Plasticity_Index', 'Plasticity Index'] 
        for preferred in pi_priority:
            if preferred in available_pi_cols:
                pi_default = preferred
                break
        if not pi_default and available_pi_cols:
            pi_default = available_pi_cols[0]
        
        # Helper function to get available values for filter types
        def get_filter_options(filter_type):
            if filter_type == "Geology Origin":
                return sorted(atterberg_data['Geology_Orgin'].dropna().unique()) if 'Geology_Orgin' in atterberg_data.columns else []
            elif filter_type == "Consistency":
                return sorted(atterberg_data['Consistency'].dropna().unique()) if 'Consistency' in atterberg_data.columns else []
            elif filter_type == "Hole ID":
                return sorted(atterberg_data['Hole_ID'].dropna().unique()) if 'Hole_ID' in atterberg_data.columns else []
            elif filter_type == "Report":
                return sorted(atterberg_data['Report'].dropna().unique()) if 'Report' in atterberg_data.columns else []
            else:
                return []
        
        # Display parameter inputs in organized layout (matching PSD style)
        with st.expander("Plot Parameters", expanded=True):
            # Row 1: Core Data Selection
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                # LL Column selection with fallback to all columns if no matches found
                if available_ll_cols:
                    ll_col_options = available_ll_cols
                    ll_default_index = ll_col_options.index(ll_default) if ll_default in ll_col_options else 0
                else:
                    # Fallback to all numeric columns if no LL patterns found
                    ll_col_options = [col for col in atterberg_data.columns if atterberg_data[col].dtype in ['float64', 'int64']]
                    ll_default_index = 0
                    st.warning("No LL columns found with standard naming. Please select from available numeric columns.")
                
                ll_col = st.selectbox("LL Column", ll_col_options,
                    index=ll_default_index, key="atterberg_ll_col", 
                    help="Liquid Limit column (supports LL, LL1, LL2, LL(%), Liquid_Limit, etc.)")
                    
            with col2:
                # PI Column selection with fallback to all columns if no matches found  
                if available_pi_cols:
                    pi_col_options = available_pi_cols
                    pi_default_index = pi_col_options.index(pi_default) if pi_default in pi_col_options else 0
                else:
                    # Fallback to all numeric columns if no PI patterns found
                    pi_col_options = [col for col in atterberg_data.columns if atterberg_data[col].dtype in ['float64', 'int64']]
                    pi_default_index = 0
                    st.warning("No PI columns found with standard naming. Please select from available numeric columns.")
                
                pi_col = st.selectbox("PI Column", pi_col_options,
                    index=pi_default_index, key="atterberg_pi_col", 
                    help="Plasticity Index column (supports PI, PI1, PI2, PI(%), Plasticity_Index, etc.)")
            with col3:
                filter1_by = st.selectbox("Filter 1 By", ["None", "Geology Origin", "Consistency", "Hole ID", "Report"], index=1, key="atterberg_filter1_by", help="Select first filter type")
            with col4:
                if filter1_by == "None":
                    filter1_values = []
                    st.selectbox("Filter 1 Value", ["All"], index=0, disabled=True, key="atterberg_filter1_value_disabled", help="Select filter type first")
                else:
                    filter1_options = get_filter_options(filter1_by)
                    filter1_dropdown_options = ["All"] + filter1_options
                    filter1_selection = st.selectbox(f"{filter1_by}", filter1_dropdown_options, index=0, key="atterberg_filter1_value", help=f"Select {filter1_by.lower()} value")
                    if filter1_selection == "All":
                        filter1_values = filter1_options
                    else:
                        filter1_values = [filter1_selection]
            with col5:
                filter2_by = st.selectbox("Filter 2 By", ["None", "Geology Origin", "Consistency", "Hole ID", "Report"], index=0, key="atterberg_filter2_by", help="Select second filter type")
            with col6:
                if filter2_by == "None":
                    filter2_values = []
                    st.selectbox("Filter 2 Value", ["All"], index=0, disabled=True, key="atterberg_filter2_value_disabled", help="Select filter type first")
                else:
                    filter2_options = get_filter_options(filter2_by)
                    filter2_dropdown_options = ["All"] + filter2_options
                    filter2_selection = st.selectbox(f"{filter2_by}", filter2_dropdown_options, index=0, key="atterberg_filter2_value", help=f"Select {filter2_by.lower()} value")
                    if filter2_selection == "All":
                        filter2_values = filter2_options
                    else:
                        filter2_values = [filter2_selection]
            
            # Row 2: Color and Basic Plot Settings
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                category_col = st.selectbox("Color by", ["Consistency", "Geology_Orgin", "Hole_ID", "Report", None], index=0, key="atterberg_color_by", help="Group data by color")
            with col2:
                xlim_str = st.text_input("X-axis limits", value="(0, 100)", key="atterberg_xlim", help="X-axis limits as (min, max)")
            with col3:
                ylim_str = st.text_input("Y-axis limits", value="(0, 80)", key="atterberg_ylim", help="Y-axis limits as (min, max)")
            with col4:
                figsize_str = st.text_input("Figure size", value="(8, 6)", key="atterberg_figsize", help="Figure size as (width, height)")
            with col5:
                title = st.text_input("Title", value="", key="atterberg_title", help="Custom plot title")
            with col6:
                title_fontsize = st.number_input("Title font size", min_value=10, max_value=24, value=14, key="atterberg_title_fontsize", help="Title font size")
        
        # Advanced Parameters Expander (matching PSD style)
        with st.expander("Advanced Parameters", expanded=False):
            # Row 1: Marker Appearance
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                marker_size = st.number_input("Marker size", min_value=10, max_value=100, value=40, key="atterberg_marker_size", help="Size of data point markers")
            with col2:
                marker_alpha = st.number_input("Marker alpha", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="atterberg_marker_alpha", help="Marker transparency (0=transparent, 1=opaque)")
            with col3:
                marker_edge_lw = st.number_input("Marker edge width", min_value=0.0, max_value=3.0, value=0.5, step=0.1, key="atterberg_marker_edge", help="Width of marker border lines")
            with col4:
                dpi = st.number_input("DPI", min_value=72, max_value=600, value=300, key="atterberg_dpi", help="Resolution for saved plots")
            with col5:
                include_background = st.selectbox("Include reference lines", ["Yes", "No"], index=0, key="atterberg_include_background", help="Show A-line and U-line reference")
            with col6:
                include_zone_labels = st.selectbox("Include zone labels", ["Yes", "No"], index=0, key="atterberg_include_zone_labels", help="Show soil classification zone labels")
            
            # Row 2: Legend and Font Settings
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                show_legend = st.selectbox("Show legend", ["Yes", "No"], index=0, key="atterberg_show_legend", help="Display legend for color categories")
            with col2:
                legend_fontsize = st.number_input("Legend font size", min_value=6, max_value=16, value=9, key="atterberg_legend_fontsize", help="Size of legend text")
            with col3:
                legend_title_fontsize = st.number_input("Legend title font size", min_value=6, max_value=16, value=10, key="atterberg_legend_title_fontsize", help="Size of legend title")
            with col4:
                label_fontsize = st.number_input("Axis label font size", min_value=8, max_value=20, value=12, key="atterberg_label_fontsize", help="Size of axis labels")
            with col5:
                tick_fontsize = st.number_input("Tick font size", min_value=6, max_value=16, value=10, key="atterberg_tick_fontsize", help="Size of axis tick labels")
            with col6:
                zone_label_fontsize = st.number_input("Zone label font size", min_value=6, max_value=14, value=8, key="atterberg_zone_label_fontsize", help="Size of soil zone labels")
            
            # Row 3: Legend Positioning and Color Control
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                legend_bbox = st.text_input("Legend position", value="best", key="atterberg_legend_bbox", help="Legend position ('best' or tuple like '(1.02, 1)')")
            with col2:
                exclude_reference_from_legend = st.selectbox("Exclude A-line/U-line from legend", ["Yes", "No"], index=0, key="atterberg_exclude_ref_legend", help="Remove reference lines from legend")
            with col3:
                marker_color = st.text_input("Marker color", value="", key="atterberg_marker_color", help="Single color or comma-separated list (leave empty for auto colors)")
            with col4:
                marker_shape = st.text_input("Marker shape", value="", key="atterberg_marker_shape", help="Single shape or comma-separated list (leave empty for auto shapes)")
            with col5:
                pass  # Empty for balance
            with col6:
                pass  # Empty for balance
                
        # Parse inputs
        xlim = parse_tuple(xlim_str, (0, 100))
        ylim = parse_tuple(ylim_str, (0, 80))
        figsize = parse_tuple(figsize_str, (8, 6))
        # Handle legend position - can be 'best' or a tuple
        if legend_bbox.strip().lower() == 'best':
            legend_bbox_to_anchor = None
            legend_loc = 'best'
        else:
            legend_bbox_to_anchor = parse_tuple(legend_bbox, None)
            legend_loc = 'upper left' if legend_bbox_to_anchor else 'best'
        
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
        
        # Set dynamic title suffix based on applied filters
        title_suffix = generate_title_suffix()
        
        # Process marker color and shape parameters
        if marker_color.strip():
            marker_color_param = [c.strip() for c in marker_color.split(',') if c.strip()]
            if len(marker_color_param) == 1:
                marker_color_param = marker_color_param[0]
        else:
            marker_color_param = None
            
        if marker_shape.strip():
            marker_shape_param = [s.strip() for s in marker_shape.split(',') if s.strip()]
            if len(marker_shape_param) == 1:
                marker_shape_param = marker_shape_param[0]
        else:
            marker_shape_param = None
        
        # Handle None selection for category_col
        if category_col == "None":
            category_col = None
            
        # Handle default blue color when color_by is None (matching PSD tab behavior)
        if category_col is None and marker_color_param is None:
            marker_color_param = '#1f77b4'  # Default matplotlib blue
            
        # Apply dual filtering (matching PSD style)
        filtered_atterberg = atterberg_data.copy()
        
        # Apply Filter 1
        if filter1_by != "None" and filter1_values:
            if filter1_by == "Geology Origin" and 'Geology_Orgin' in filtered_atterberg.columns:
                filtered_atterberg = filtered_atterberg[filtered_atterberg['Geology_Orgin'].isin(filter1_values)]
            elif filter1_by == "Consistency" and 'Consistency' in filtered_atterberg.columns:
                filtered_atterberg = filtered_atterberg[filtered_atterberg['Consistency'].isin(filter1_values)]
            elif filter1_by == "Hole ID" and 'Hole_ID' in filtered_atterberg.columns:
                filtered_atterberg = filtered_atterberg[filtered_atterberg['Hole_ID'].isin(filter1_values)]
            elif filter1_by == "Report" and 'Report' in filtered_atterberg.columns:
                filtered_atterberg = filtered_atterberg[filtered_atterberg['Report'].isin(filter1_values)]
        
        # Apply Filter 2  
        if filter2_by != "None" and filter2_values:
            if filter2_by == "Geology Origin" and 'Geology_Orgin' in filtered_atterberg.columns:
                filtered_atterberg = filtered_atterberg[filtered_atterberg['Geology_Orgin'].isin(filter2_values)]
            elif filter2_by == "Consistency" and 'Consistency' in filtered_atterberg.columns:
                filtered_atterberg = filtered_atterberg[filtered_atterberg['Consistency'].isin(filter2_values)]
            elif filter2_by == "Hole ID" and 'Hole_ID' in filtered_atterberg.columns:
                filtered_atterberg = filtered_atterberg[filtered_atterberg['Hole_ID'].isin(filter2_values)]
            elif filter2_by == "Report" and 'Report' in filtered_atterberg.columns:
                filtered_atterberg = filtered_atterberg[filtered_atterberg['Report'].isin(filter2_values)]
        
            
        # Store data summary for display later (underneath plots)
        geology_counts = None
        if 'Geology_Orgin' in filtered_atterberg.columns:
            geology_counts = filtered_atterberg['Geology_Orgin'].value_counts()
            
        # Create main plot using filtered data
        
        if not filtered_atterberg.empty:
            valid_data = filtered_atterberg.dropna(subset=[ll_col, pi_col])
            
            if HAS_FUNCTIONS:
                # Always generate plot for immediate display and dashboard storage
                try:
                    # Clear any existing figures first
                    plt.close('all')
                    
                    # Call function directly from Functions folder with ALL parameters
                    plot_atterberg_chart(
                        df=filtered_atterberg,
                        ll_col=ll_col,
                        pi_col=pi_col,
                        category_col=category_col,
                        title=title if title else None,
                        title_suffix=title_suffix if title_suffix else None,
                        xlim=xlim,
                        ylim=ylim,
                        figsize=figsize,
                        marker_color=marker_color_param,
                        marker_shape=marker_shape_param,
                        marker_size=marker_size,
                        marker_alpha=marker_alpha,
                        marker_edge_lw=marker_edge_lw,
                        show_legend=(show_legend == "Yes") and (category_col is not None),
                        include_background=(include_background == "Yes"),
                        include_zone_labels=(include_zone_labels == "Yes"),
                        title_fontsize=title_fontsize,
                        label_fontsize=label_fontsize,
                        tick_fontsize=tick_fontsize,
                        legend_fontsize=legend_fontsize,
                        legend_title_fontsize=legend_title_fontsize,
                        zone_label_fontsize=zone_label_fontsize,
                        dpi=dpi,
                        legend_loc=legend_loc,
                        legend_bbox_to_anchor=legend_bbox_to_anchor,
                        show_plot=False,
                        save_plot=False,
                        close_plot=False,
                        output_filepath=None
                    )
                    
                    # Handle A-line/U-line legend exclusion if requested
                    if exclude_reference_from_legend == "Yes":
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            ax = current_fig.get_axes()[0]
                            handles, labels = ax.get_legend_handles_labels()
                            
                            # Filter out A-line and U-line from legend
                            filtered_handles = []
                            filtered_labels = []
                            for handle, label in zip(handles, labels):
                                if label not in ('A-line', 'U-line'):
                                    filtered_handles.append(handle)
                                    filtered_labels.append(label)
                            
                            # Recreate legend without reference lines if legend should be shown
                            if (show_legend == "Yes") and (category_col is not None) and filtered_handles:
                                ax.legend(handles=filtered_handles, labels=filtered_labels,
                                         title=category_col.replace('_', ' ').title() if category_col else '',
                                         fontsize=legend_fontsize, title_fontsize=legend_title_fontsize,
                                         loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor,
                                         frameon=True, facecolor='white', framealpha=0.75, edgecolor='darkgrey',
                                         markerscale=0.9)
                    
                    # Capture and display the figure in Streamlit with size control
                    current_fig = plt.gcf()
                    if current_fig and current_fig.get_axes():
                        # Use the same plot size control as PSD tab
                        from .plotting_utils import display_plot_with_size_control
                        display_plot_with_size_control(current_fig)
                        success = True
                    else:
                        success = False
                    
                    if success:
                        # Store the plot for Materials Dashboard
                        try:
                            # Get the current figure for storage
                            current_fig = plt.gcf()
                            if current_fig and current_fig.get_axes():
                                # Save figure to buffer for dashboard
                                import io
                                buf = io.BytesIO()
                                current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                store_material_plot('atterberg_chart', buf)
                        except Exception as e:
                            pass  # Don't break the main functionality if storage fails
                        
                        # Simple download button with figure reference
                        from .plot_download_simple import create_simple_download_button
                        create_simple_download_button("atterberg_chart", "main", fig=current_fig)
                    else:
                        st.warning("No plot generated - check data availability")
                        
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
            else:
                st.error("❌ Functions folder not accessible")
                st.info("Check Functions folder and plot_atterberg_chart.py module")
        else:
            st.warning("No valid Atterberg data available for plotting")
        
        # Atterberg Test Distribution and Map Visualization (consistent with PSD layout)
        st.markdown("### Atterberg Test Distribution")
        
        # Check for coordinate data and display map
        if HAS_PYPROJ:
            # Use dynamic ID columns detection to find coordinate columns
            id_columns = get_id_columns_from_data(filtered_atterberg)
            
            # Precise coordinate matching - find northing/easting columns
            def is_coordinate_column(column_name, keywords):
                col_clean = column_name.lower().replace('(', '').replace(')', '').replace('_', '').replace(' ', '').replace('-', '')
                return any(col_clean == keyword or col_clean.startswith(keyword) for keyword in keywords)
            
            northing_keywords = ['northing', 'north', 'latitude', 'lat', 'y']
            potential_lat_cols = [col for col in id_columns if is_coordinate_column(col, northing_keywords)]
            
            easting_keywords = ['easting', 'east', 'longitude', 'lon', 'x'] 
            potential_lon_cols = [col for col in id_columns if is_coordinate_column(col, easting_keywords)]
            
            if potential_lat_cols and potential_lon_cols:
                lat_col = potential_lat_cols[0]
                lon_col = potential_lon_cols[0]
                
                # Get coordinate data from Atterberg test locations
                try:
                    # Get unique sample locations from Atterberg data
                    sample_locations = filtered_atterberg[['Hole_ID', 'From_mbgl']].drop_duplicates()
                    
                    # Merge with original data to get coordinates and chainage
                    merge_cols = ['Hole_ID', 'From_mbgl', lat_col, lon_col]
                    if 'Chainage' in filtered_atterberg.columns:
                        merge_cols.append('Chainage')
                    coord_data = sample_locations.merge(
                        filtered_atterberg[merge_cols], 
                        on=['Hole_ID', 'From_mbgl'], 
                        how='left'
                    ).dropna(subset=[lat_col, lon_col])
                    
                    if not coord_data.empty and len(coord_data) > 0:
                        # Prepare map data
                        map_data = coord_data.copy()
                        
                        # Convert UTM to WGS84 for mapping
                        if coord_data[lat_col].max() > 1000:  # UTM coordinates
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
                                
                                # Display enhanced map with Atterberg test locations
                                if HAS_PLOTLY:
                                    import plotly.express as px
                                    import plotly.graph_objects as go
                                    
                                    fig = go.Figure()
                                    
                                    # Add Atterberg test location points
                                    if 'Chainage' in coord_data.columns:
                                        map_data_with_chainage = map_data.copy()
                                        map_data_with_chainage['chainage'] = coord_data['Chainage'].values
                                        
                                        fig.add_trace(go.Scattermapbox(
                                            lat=map_data_with_chainage['lat'],
                                            lon=map_data_with_chainage['lon'],
                                            mode='markers',
                                            marker=dict(
                                                size=8,
                                                color='green',  # Different color for Atterberg tests
                                                opacity=0.8
                                            ),
                                            name='Atterberg Test Locations',
                                            customdata=map_data_with_chainage['chainage'],
                                            hovertemplate='<b>Atterberg Test Location</b><br>' +
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
                                                color='green',
                                                opacity=0.8
                                            ),
                                            name='Atterberg Test Locations',
                                            hovertemplate='<b>Atterberg Test Location</b><br>' +
                                                         'Lat: %{lat:.6f}<br>' +
                                                         'Lon: %{lon:.6f}<extra></extra>'
                                        ))
                                    
                                    # Add chainage reference markers (same logic as PSD)
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
                                        height=528,  # Same as PSD
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
                                    # Fallback to basic map
                                    map_col, spacer_col = st.columns([9, 1])
                                    with map_col:
                                        st.map(map_data[['lat', 'lon']])
                            except Exception as e:
                                st.error(f"❌ Could not convert coordinates: {str(e)}")
                        else:
                            st.warning("⚠️ WGS84 coordinates not supported yet.")
                    else:
                        st.info("No coordinate data available for Atterberg test locations")
                except Exception as e:
                    st.warning(f"Could not generate map: {str(e)}")
            else:
                st.info("No coordinate columns found for map visualization.")
        
        # Atterberg Test Distribution Charts
        try:
            if 'Chainage' in filtered_data.columns:
                # Get Atterberg test types from column names (excluding WPI and shrink swell)
                def get_atterberg_test_types(data):
                    test_columns = [col for col in data.columns if '?' in col]
                    atterberg_types = []
                    for col in test_columns:
                        test_type = col.replace('?', '').strip()
                        # Include Atterberg-related tests but exclude WPI and shrink swell
                        if any(keyword in test_type.lower() for keyword in ['atterberg', 'liquid', 'plastic', 'll', 'pl', 'pi']):
                            # Exclude WPI and shrink swell related tests
                            if not any(excluded in test_type.lower() for excluded in ['wpi', 'shrink', 'swell']):
                                atterberg_types.append(test_type)
                    return atterberg_types, [f"{t}?" for t in atterberg_types]
                
                def render_atterberg_test_chart(test_type, original_filtered_data, bins):
                    st.write(f"**{test_type} Distribution:**")
                    
                    test_col = f"{test_type}?"
                    if test_col in original_filtered_data.columns:
                        test_data = original_filtered_data[original_filtered_data[test_col] == 'Y']
                        
                        if not test_data.empty and 'Chainage' in test_data.columns:
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
                
                # Get Atterberg test types
                atterberg_test_types, test_columns = get_atterberg_test_types(filtered_data)
                
                if len(atterberg_test_types) > 0:
                    chainage_data = filtered_data['Chainage'].dropna()
                    if not chainage_data.empty:
                        min_chainage = chainage_data.min()
                        max_chainage = chainage_data.max()
                        
                        # Create fixed interval bins (200m intervals)
                        bin_interval = 200
                        bin_start = int(min_chainage // bin_interval) * bin_interval
                        bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
                        bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
                        
                        # Create charts for Atterberg test types - each chart at 90% width
                        for i, test_type in enumerate(atterberg_test_types):
                            if i > 0:
                                st.write("")
                            
                            # Each chart gets 90% width layout (same as map)
                            chart_col, spacer_col = st.columns([9, 1])
                            
                            with chart_col:
                                render_atterberg_test_chart(test_type, filtered_data, bins)
                    else:
                        st.info("No chainage data available for distribution analysis")
                else:
                    st.info("No Atterberg test data available for distribution analysis")
            else:
                st.warning("Chainage column not found - cannot create spatial distribution")
        
        except Exception as e:
            st.warning(f"Could not generate Atterberg distribution chart: {str(e)}")
        
        # Plot Summary (moved here after test distribution)
        if 'success' in locals() and success and 'filtered_atterberg' in locals() and not filtered_atterberg.empty:
            st.markdown("### Plot Summary")
            summary_data = []
            
            # Get the valid data from the filtered dataset
            valid_data = filtered_atterberg.dropna(subset=[ll_col, pi_col])
            
            # Basic plot information
            summary_data.extend([
                {'Parameter': 'Plot Type', 'Value': 'Atterberg Limits Chart'},
                {'Parameter': 'Total Samples', 'Value': f"{len(filtered_atterberg):,}"},
                {'Parameter': 'Valid Data Points', 'Value': f"{len(valid_data):,}"},
                {'Parameter': 'X-axis Range', 'Value': f"{xlim[0]} - {xlim[1]}"},
                {'Parameter': 'Y-axis Range', 'Value': f"{ylim[0]} - {ylim[1]}"},
                {'Parameter': 'Figure Size', 'Value': f"{figsize[0]} x {figsize[1]} inches"}
            ])
            
            # Data range information
            if not valid_data.empty:
                ll_values = valid_data[ll_col]
                pi_values = valid_data[pi_col]
                
                summary_data.extend([
                    {'Parameter': 'LL Data Range', 'Value': f"{ll_values.min():.1f} - {ll_values.max():.1f}%"},
                    {'Parameter': 'PI Data Range', 'Value': f"{pi_values.min():.1f} - {pi_values.max():.1f}%"},
                    {'Parameter': 'LL Mean', 'Value': f"{ll_values.mean():.1f}%"},
                    {'Parameter': 'PI Mean', 'Value': f"{pi_values.mean():.1f}%"}
                ])
            
            # Filtering information
            if filter1_by != "None" and filter1_values:
                all_options = get_filter_options(filter1_by)
                if filter1_values != all_options:
                    summary_data.append({
                        'Parameter': f'Filter 1 ({filter1_by})',
                        'Value': f"{', '.join(filter1_values[:3])}{'...' if len(filter1_values) > 3 else ''}"
                    })
            
            if filter2_by != "None" and filter2_values:
                all_options = get_filter_options(filter2_by)
                if filter2_values != all_options:
                    summary_data.append({
                        'Parameter': f'Filter 2 ({filter2_by})',
                        'Value': f"{', '.join(filter2_values[:3])}{'...' if len(filter2_values) > 3 else ''}"
                    })
            
            # Color grouping information
            if category_col and category_col != "None":
                unique_categories = valid_data[category_col].unique()
                summary_data.append({
                    'Parameter': f'Color Groups ({category_col})',
                    'Value': f"{len(unique_categories)} categories"
                })
            
            # Create summary table
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Add visual separator before data options (consistent with PSD layout)
        st.divider()
        
        # Data preview and statistics options underneath plot (standard format)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Show data preview", key="atterberg_data_preview"):
                # Show relevant columns for Atterberg analysis
                preview_cols = ['Hole_ID', 'From_mbgl', 'To_mbgl', ll_col, pi_col]
                if category_col and category_col in filtered_atterberg.columns:
                    preview_cols.append(category_col)
                
                available_cols = [col for col in preview_cols if col in filtered_atterberg.columns]
                st.dataframe(filtered_atterberg[available_cols].head(20), use_container_width=True)
                st.caption(f"{len(filtered_atterberg)} total records")
        
        with col2:
            if st.checkbox("Show statistics", key="atterberg_statistics"):
                try:
                    valid_data = filtered_atterberg.dropna(subset=[ll_col, pi_col])
                    
                    if not valid_data.empty:
                        stats_data = []
                        
                        # Basic Atterberg statistics
                        ll_values = valid_data[ll_col]
                        pi_values = valid_data[pi_col]
                        pl_values = ll_values - pi_values  # Calculate PL
                        
                        stats_data.extend([
                            {'Parameter': 'Total Tests', 'Value': f"{len(valid_data):,}"},
                            {'Parameter': 'Mean LL', 'Value': f"{ll_values.mean():.1f}%"},
                            {'Parameter': 'Mean PI', 'Value': f"{pi_values.mean():.1f}%"},
                            {'Parameter': 'Mean PL', 'Value': f"{pl_values.mean():.1f}%"},
                            {'Parameter': 'LL Range', 'Value': f"{ll_values.min():.0f} - {ll_values.max():.0f}%"},
                            {'Parameter': 'PI Range', 'Value': f"{pi_values.min():.0f} - {pi_values.max():.0f}%"}
                        ])
                        
                        # Depth range
                        if 'From_mbgl' in valid_data.columns:
                            depth_data = valid_data['From_mbgl'].dropna()
                            if not depth_data.empty:
                                depth_range = f"{depth_data.min():.1f} - {depth_data.max():.1f} m"
                                stats_data.append({'Parameter': 'Depth Range', 'Value': depth_range})
                        
                        # Geology/category distribution
                        if category_col and category_col in valid_data.columns:
                            geo_counts = valid_data[category_col].value_counts()
                            for geo, count in geo_counts.head(5).items():
                                percentage = (count / len(valid_data)) * 100
                                stats_data.append({
                                    'Parameter': f'{geo}',
                                    'Value': f"{count} ({percentage:.1f}%)"
                                })
                        
                        # Soil classification distribution
                        classifications = []
                        for _, row in valid_data.iterrows():
                            ll_val = row[ll_col]
                            pi_val = row[pi_col]
                            soil_class = classify_soil_type(ll_val, pi_val)
                            classifications.append(soil_class)
                        
                        class_counts = pd.Series(classifications).value_counts()
                        for soil_class, count in class_counts.head(3).items():
                            percentage = (count / len(valid_data)) * 100
                            stats_data.append({
                                'Parameter': f'{soil_class}',
                                'Value': f"{count} ({percentage:.1f}%)"
                            })
                        
                        # Create statistics table
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No valid Atterberg data for statistics")
                        
                except Exception as e:
                    st.error(f"Error calculating statistics: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in Atterberg analysis: {str(e)}")
        st.write("Full error details:")
        st.exception(e)
