"""
UCS (Unconfined Compressive Strength) Analysis Module

This module handles UCS data processing, analysis, and visualization for geotechnical applications.
Uses original plotting functions from Functions folder exactly as in Jupyter notebook.
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

try:
    from .data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_id_columns_from_data
    from .plot_defaults import get_default_parameters, get_color_schemes
    from .dashboard_rock import store_rock_plot
    from .plotting_utils import display_plot_with_size_control
    HAS_PLOTTING_UTILS = True
except ImportError:
    # For standalone testing
    from data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_id_columns_from_data
    from plot_defaults import get_default_parameters, get_color_schemes
    try:
        from dashboard_rock import store_rock_plot
        from plotting_utils import display_plot_with_size_control
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
    
    from plot_UCS_vs_depth import plot_UCS_vs_depth
    from plot_UCS_Is50 import plot_UCS_Is50
    from plot_by_chainage import plot_by_chainage
    HAS_FUNCTIONS = True
except ImportError as e:
    HAS_FUNCTIONS = False
    print(f"Warning: Could not import Functions: {e}")


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


def extract_ucs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract UCS test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: UCS-specific dataframe
    """
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


def render_ucs_vs_depth_analysis(filtered_data: pd.DataFrame) -> pd.DataFrame:
    """
    Render comprehensive UCS vs Depth analysis with standardized parameter structure.
    Follows exact structure of enhanced Emerson/PSD/Atterberg/SPT modules.
    
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
        
        # Extract UCS data
        ucs_data = extract_ucs_data(filtered_data)
        
        if ucs_data.empty:
            st.warning("No UCS data available with current filters.")
            return
        
        # Get standard ID columns for UI controls
        standard_id_columns = get_id_columns_from_data(ucs_data)
        
        # Get category columns and default values
        category_columns = [col for col in standard_id_columns if col in ucs_data.columns]
        if not category_columns:
            category_columns = list(ucs_data.columns)[:10]
        
        geology_index = 0
        if "Geology_Orgin" in category_columns:
            geology_index = category_columns.index("Geology_Orgin")
        
        # Get available UCS columns
        ucs_columns = extract_test_columns(filtered_data, 'UCS')
        if not ucs_columns:
            ucs_columns = ['UCS (MPa)']  # fallback
        
        # Set default UCS column
        default_ucs_idx = 0
        if 'UCS (MPa)' in ucs_columns:
            default_ucs_idx = ucs_columns.index('UCS (MPa)')
        
        # Initialize filter variables with default values
        filter1_by = "None"
        filter1_values = []
        filter2_by = "None" 
        filter2_values = []
        category_by = category_columns[geology_index] if category_columns else "Geology_Orgin"
        
        # Helper function to get available values for filter types
        def get_filter_options(filter_type):
            if filter_type == "Geology Origin":
                return sorted(ucs_data['Geology_Orgin'].dropna().unique()) if 'Geology_Orgin' in ucs_data.columns else []
            elif filter_type == "Consistency":
                return sorted(ucs_data['Consistency'].dropna().unique()) if 'Consistency' in ucs_data.columns else []
            elif filter_type == "Hole ID":
                return sorted(ucs_data['Hole_ID'].dropna().unique()) if 'Hole_ID' in ucs_data.columns else []
            elif filter_type == "Report":
                return sorted(ucs_data['Report'].dropna().unique()) if 'Report' in ucs_data.columns else []
            else:
                return []
        
        # Parameters box - comprehensive parameter controls matching enhanced modules
        with st.expander("**PARAMETERS**", expanded=False):
            # Row 1: Core Data Selection
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                depth_col = st.selectbox(
                    "Depth Column",
                    ['From_mbgl', 'To_mbgl'],
                    index=0,
                    key="ucs_depth_col"
                )
            with col2:
                ucs_col = st.selectbox(
                    "UCS Column",
                    ucs_columns,
                    index=default_ucs_idx,
                    key="ucs_col"
                )
            with col3:
                use_log_scale = st.selectbox(
                    "use_log_scale",
                    [True, False],
                    index=0,
                    key="ucs_log_scale"
                )
            with col4:
                invert_yaxis = st.selectbox(
                    "invert_yaxis",
                    [True, False],
                    index=0,
                    key="ucs_invert_y"
                )
            with col5:
                pass
            with col6:
                pass
            
            # Row 2: Data Filtering & Category
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                filter1_by = st.selectbox(
                    "Filter 1 By",
                    ["None", "Geology Origin", "Consistency", "Hole ID", "Report"],
                    index=1,
                    key="ucs_filter1_by"
                )
            with col2:
                if filter1_by == "None":
                    filter1_values = []
                    st.selectbox("Filter 1 Value", ["All"], index=0, disabled=True, key="ucs_filter1_value_disabled")
                else:
                    filter1_options = get_filter_options(filter1_by)
                    filter1_dropdown_options = ["All"] + filter1_options
                    filter1_selection = st.selectbox(f"{filter1_by}", filter1_dropdown_options, index=0, key="ucs_filter1_value")
                    if filter1_selection == "All":
                        filter1_values = filter1_options
                    else:
                        filter1_values = [filter1_selection]
            
            with col3:
                filter2_by = st.selectbox(
                    "Filter 2 By",
                    ["None", "Geology Origin", "Consistency", "Hole ID", "Report"],
                    index=0,
                    key="ucs_filter2_by"
                )
            with col4:
                if filter2_by == "None":
                    filter2_values = []
                    st.selectbox("Filter 2 Value", ["All"], index=0, disabled=True, key="ucs_filter2_value_disabled")
                else:
                    filter2_options = get_filter_options(filter2_by)
                    filter2_dropdown_options = ["All"] + filter2_options
                    filter2_selection = st.selectbox(f"{filter2_by}", filter2_dropdown_options, index=0, key="ucs_filter2_value")
                    if filter2_selection == "All":
                        filter2_values = filter2_options
                    else:
                        filter2_values = [filter2_selection]
            with col5:
                category_by = st.selectbox(
                    "Category By",
                    category_columns,
                    index=geology_index,
                    key="ucs_category_by"
                )
            with col6:
                pass
            
            # Row 3: Plot Configuration
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                figsize_str = st.text_input("figsize (w, h)", value="(9, 6)", key="ucs_figsize")
            with col2:
                xlim_str = st.text_input("xlim (min, max)", value="(0.6, 400)", key="ucs_xlim")
            with col3:
                ylim_str = st.text_input("ylim (min, max)", value="(0, 29)", key="ucs_ylim")
            with col4:
                title = st.text_input("title", value="", key="ucs_title")
            with col5:
                title_suffix = st.text_input("title_suffix", value="", key="ucs_title_suffix")
            with col6:
                ytick_interval = st.number_input("ytick_interval", min_value=0.5, max_value=10.0, value=5.0, step=0.5, key="ucs_ytick_interval")
            
            # Row 4: Visual Style
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                show_strength_boundaries = st.selectbox(
                    "show_strength_boundaries",
                    [True, False],
                    index=0,
                    key="ucs_show_boundaries"
                )
            with col2:
                show_strength_indicators = st.selectbox(
                    "show_strength_indicators",
                    [True, False],
                    index=0,
                    key="ucs_show_indicators"
                )
            with col3:
                strength_indicator_position = st.number_input("strength_indicator_position", min_value=0.1, max_value=0.9, value=0.85, step=0.05, key="ucs_indicator_pos")
            with col4:
                marker_size = st.number_input("marker_size", min_value=20, max_value=100, value=40, step=5, key="ucs_marker_size")
            with col5:
                marker_alpha = st.number_input("marker_alpha", min_value=0.3, max_value=1.0, value=0.7, step=0.05, key="ucs_marker_alpha")
            with col6:
                show_legend = st.selectbox(
                    "show_legend",
                    [True, False],
                    index=0,
                    key="ucs_show_legend"
                )
            
            # Row 5: Text Formatting
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                title_fontsize = st.number_input("title_fontsize", min_value=8, max_value=24, value=14, key="ucs_title_fontsize")
            with col2:
                label_fontsize = st.number_input("label_fontsize", min_value=8, max_value=20, value=12, key="ucs_label_fontsize")
            with col3:
                tick_fontsize = st.number_input("tick_fontsize", min_value=6, max_value=16, value=10, key="ucs_tick_fontsize")
            with col4:
                legend_fontsize = st.number_input("legend_fontsize", min_value=6, max_value=16, value=11, key="ucs_legend_fontsize")
            with col5:
                marker_edge_lw = st.number_input("marker_edge_lw", min_value=0.0, max_value=2.0, value=0.5, step=0.1, key="ucs_marker_edge_lw")
            with col6:
                pass
        
        # Apply filters to data
        filtered_ucs = ucs_data.copy()
        
        # Apply Filter 1
        if filter1_by != "None" and filter1_values:
            if filter1_by == "Geology Origin":
                filtered_ucs = filtered_ucs[filtered_ucs['Geology_Orgin'].isin(filter1_values)]
            elif filter1_by == "Consistency":
                filtered_ucs = filtered_ucs[filtered_ucs['Consistency'].isin(filter1_values)]
            elif filter1_by == "Hole ID":
                filtered_ucs = filtered_ucs[filtered_ucs['Hole_ID'].isin(filter1_values)]
            elif filter1_by == "Report" and 'Report' in filtered_ucs.columns:
                filtered_ucs = filtered_ucs[filtered_ucs['Report'].isin(filter1_values)]
        
        # Apply Filter 2
        if filter2_by != "None" and filter2_values:
            if filter2_by == "Geology Origin":
                filtered_ucs = filtered_ucs[filtered_ucs['Geology_Orgin'].isin(filter2_values)]
            elif filter2_by == "Consistency":
                filtered_ucs = filtered_ucs[filtered_ucs['Consistency'].isin(filter2_values)]
            elif filter2_by == "Hole ID":
                filtered_ucs = filtered_ucs[filtered_ucs['Hole_ID'].isin(filter2_values)]
            elif filter2_by == "Report" and 'Report' in filtered_ucs.columns:
                filtered_ucs = filtered_ucs[filtered_ucs['Report'].isin(filter2_values)]
        
        # Generate dynamic title suffix based on applied filters
        def generate_title_suffix():
            suffix_parts = []
            
            # Add Filter 1 to suffix if applied (and not "All")
            if filter1_by != "None" and filter1_values:
                all_options = get_filter_options(filter1_by)
                if filter1_values != all_options:
                    if len(filter1_values) == 1:
                        suffix_parts.append(f"{filter1_values[0]}")
                    elif len(filter1_values) <= 3:
                        suffix_parts.append(f"{', '.join(filter1_values)}")
                    else:
                        suffix_parts.append(f"{filter1_by} (Multiple)")
            
            # Add Filter 2 to suffix if applied (and not "All")
            if filter2_by != "None" and filter2_values:
                all_options = get_filter_options(filter2_by)
                if filter2_values != all_options:
                    if len(filter2_values) == 1:
                        suffix_parts.append(f"{filter2_values[0]}")
                    elif len(filter2_values) <= 3:
                        suffix_parts.append(f"{', '.join(filter2_values)}")
                    else:
                        suffix_parts.append(f"{filter2_by} (Multiple)")
            
            return " - ".join(suffix_parts) if suffix_parts else None
        
        # Determine final title and suffix
        dynamic_suffix = generate_title_suffix()
        if not title:
            final_title = None
            final_title_suffix = dynamic_suffix or title_suffix
        else:
            final_title = title
            final_title_suffix = title_suffix
        
        if filtered_ucs.empty:
            st.warning("No data available after applying filters.")
            return pd.DataFrame()
            
        if HAS_FUNCTIONS:
            try:
                # Clear any existing figures first
                plt.close('all')
                
                # Parse parameters from text inputs
                figsize = parse_tuple(figsize_str, (9, 6))
                xlim = parse_tuple(xlim_str, (0.6, 400)) if xlim_str != "auto" else None
                ylim = parse_tuple(ylim_str, (0, 29)) if ylim_str != "auto" else None
                
                plot_UCS_vs_depth(
                    df=filtered_ucs,
                    depth_col=depth_col,
                    ucs_col=ucs_col,
                    category_col=category_by if show_legend else None,
                    xlim=xlim,
                    ylim=ylim,
                    ytick_interval=ytick_interval,
                    invert_yaxis=invert_yaxis,
                    use_log_scale=use_log_scale,
                    show_strength_indicators=show_strength_indicators,
                    show_strength_boundaries=show_strength_boundaries,
                    strength_indicator_position=strength_indicator_position,
                    marker_size=marker_size,
                    marker_alpha=marker_alpha,
                    marker_edge_lw=marker_edge_lw,
                    show_legend=show_legend,
                    title=final_title,
                    title_suffix=final_title_suffix,
                    figsize=figsize,
                    title_fontsize=title_fontsize,
                    label_fontsize=label_fontsize,
                    tick_fontsize=tick_fontsize,
                    legend_fontsize=legend_fontsize,
                    show_plot=False,
                    close_plot=False
                )
                
                # Display plot with size control
                current_fig = plt.gcf()
                display_plot_with_size_control(current_fig)
                
                # Store the plot for Rock Dashboard
                try:
                    if current_fig and current_fig.get_axes():
                        import io
                        buf = io.BytesIO()
                        current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        store_rock_plot('ucs_depth', buf)
                except Exception as e:
                    pass
                
                # Simple download button with figure reference
                from .plot_download_simple import create_simple_download_button
                create_simple_download_button("ucs_vs_depth", "main", fig=current_fig)
                    
            except Exception as e:
                st.error(f"Error creating UCS vs Depth plot: {str(e)}")
        else:
            st.error("❌ Functions folder not accessible")
            st.info("Check Functions folder and UCS plotting modules")
    
    except Exception as e:
        st.error(f"Error in UCS depth analysis: {str(e)}")
        st.error("Please check that your data contains valid UCS columns")
        return pd.DataFrame()
    
    # Return the filtered UCS data for other visualizations
    return filtered_ucs


def render_ucs_is50_correlation_analysis(filtered_data: pd.DataFrame):
    """
    Render comprehensive UCS vs Is50 correlation analysis with multi-dataset support.
    Follows enhanced structure and implements full Jupyter notebook workflow.
    
    Args:
        filtered_data: Filtered laboratory data
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render tab.")
        return
    
    try:
        # Extract UCS data
        ucs_data = extract_ucs_data(filtered_data)
        
        if ucs_data.empty:
            st.warning("No UCS data available with current filters.")
            return
        
        # Check if UCS data contains Is50 columns
        is50_axial_cols = [col for col in ucs_data.columns if 'Is50a' in col and 'MPa' in col]
        is50_diametral_cols = [col for col in ucs_data.columns if 'Is50d' in col and 'MPa' in col]
        has_is50_data = bool(is50_axial_cols or is50_diametral_cols)
        
        if not has_is50_data:
            st.warning("No Is50 data available for correlation analysis.")
            st.info("Is50 (Point Load Index) data required for UCS correlation analysis.")
            return
        
        # Available geological formations
        available_geologies = get_available_geologies(ucs_data)
        category_columns = ['Geology_Orgin']
        if 'Consistency' in ucs_data.columns:
            category_columns.append('Consistency')
        
        # Helper function for parsing tuple inputs (same as UCS vs Depth)
        def parse_tuple(input_str, default):
            try:
                cleaned = input_str.strip().replace('(', '').replace(')', '')
                values = [float(x.strip()) for x in cleaned.split(',')]
                return tuple(values) if len(values) == 2 else default
            except:
                return default

        # Helper function to get filter options
        def get_filter_options(filter_type):
            if filter_type == "Geology Origin":
                return sorted(ucs_data['Geology_Orgin'].dropna().unique().tolist())
            elif filter_type == "Consistency":
                return sorted(ucs_data['Consistency'].dropna().unique().tolist())
            elif filter_type == "Hole ID":
                return sorted(ucs_data['Hole_ID'].dropna().unique().tolist())
            elif filter_type == "Report" and 'Report' in ucs_data.columns:
                return sorted(ucs_data['Report'].dropna().unique().tolist())
            return []

        with st.expander("**PARAMETERS**", expanded=False):
            # Row 1: Filter Configuration and Core Data Selection
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                filter1_by = st.selectbox(
                    "Filter 1 By",
                    ["None", "Geology Origin", "Consistency", "Hole ID", "Report"],
                    index=0,
                    key="ucs_is50_filter1_by"
                )
            with col2:
                if filter1_by == "None":
                    filter1_values = []
                    st.selectbox("Filter 1 Value", ["All"], index=0, disabled=True, key="ucs_is50_filter1_value_disabled")
                else:
                    filter1_options = get_filter_options(filter1_by)
                    filter1_dropdown_options = ["All"] + filter1_options
                    filter1_selection = st.selectbox(f"{filter1_by}", filter1_dropdown_options, index=0, key="ucs_is50_filter1_value")
                    if filter1_selection == "All":
                        filter1_values = filter1_options
                    else:
                        filter1_values = [filter1_selection]
            
            with col3:
                filter2_by = st.selectbox(
                    "Filter 2 By",
                    ["None", "Geology Origin", "Consistency", "Hole ID", "Report"],
                    index=0,
                    key="ucs_is50_filter2_by"
                )
            with col4:
                if filter2_by == "None":
                    filter2_values = []
                    st.selectbox("Filter 2 Value", ["All"], index=0, disabled=True, key="ucs_is50_filter2_value_disabled")
                else:
                    filter2_options = get_filter_options(filter2_by)
                    filter2_dropdown_options = ["All"] + filter2_options
                    filter2_selection = st.selectbox(f"{filter2_by}", filter2_dropdown_options, index=0, key="ucs_is50_filter2_value")
                    if filter2_selection == "All":
                        filter2_values = filter2_options
                    else:
                        filter2_values = [filter2_selection]
            
            with col5:
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Overall Correlation", "By Geological Unit", "Axial vs Diametral", "Custom Analysis"],
                    index=0,
                    key="ucs_is50_analysis_type"
                )
            with col6:
                category_by = st.selectbox(
                    "Category By",
                    category_columns,
                    index=0,
                    key="ucs_is50_category_by"
                )
            
            # Row 2: Data Selection
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                ucs_column = st.selectbox(
                    "UCS Column",
                    ['UCS (MPa)'],
                    index=0,
                    key="ucs_is50_ucs_col"
                )
            with col2:
                show_trendlines = st.selectbox(
                    "show_trendlines",
                    [True, False],
                    index=0,
                    key="ucs_is50_show_trendlines"
                )
            with col3:
                show_equations = st.selectbox(
                    "show_equations",
                    [True, False],
                    index=0,
                    key="ucs_is50_show_equations"
                )
            with col4:
                show_r_squared = st.selectbox(
                    "show_r_squared",
                    [True, False],
                    index=0,
                    key="ucs_is50_show_r_squared"
                )
            with col5:
                show_legend = st.selectbox(
                    "show_legend",
                    [True, False],
                    index=0,
                    key="ucs_is50_show_legend"
                )
            with col6:
                pass
            
            # Row 3: Plot Configuration
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                figsize_str = st.text_input("figsize (w, h)", value="(8, 6)", key="ucs_is50_figsize")
            with col2:
                xlim_str = st.text_input("xlim (min, max)", value="(0, 10)", key="ucs_is50_xlim")
            with col3:
                ylim_str = st.text_input("ylim (min, max)", value="(0, 140)", key="ucs_is50_ylim")
            with col4:
                title = st.text_input("title", value="", key="ucs_is50_title")
            with col5:
                title_suffix = st.text_input("title_suffix", value="", key="ucs_is50_title_suffix")
            with col6:
                xtick_interval = st.number_input("xtick_interval", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="ucs_is50_xtick_interval")
            
            # Row 4: Visual Style
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                scatter_size = st.number_input("scatter_size", min_value=20, max_value=80, value=35, step=5, key="ucs_is50_scatter_size")
            with col2:
                scatter_alpha = st.number_input("scatter_alpha", min_value=0.3, max_value=1.0, value=0.65, step=0.05, key="ucs_is50_scatter_alpha")
            with col3:
                ytick_interval = st.number_input("ytick_interval", min_value=5.0, max_value=50.0, value=20.0, step=5.0, key="ucs_is50_ytick_interval")
            with col4:
                legend_fontsize = st.number_input("legend_fontsize", min_value=8, max_value=16, value=11, key="ucs_is50_legend_fontsize")
            with col5:
                pass
            with col6:
                pass
            
            # Row 5: Text Formatting
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                title_fontsize = st.number_input("title_fontsize", min_value=8, max_value=24, value=14, key="ucs_is50_title_fontsize")
            with col2:
                label_fontsize = st.number_input("label_fontsize", min_value=8, max_value=20, value=12, key="ucs_is50_label_fontsize")
            with col3:
                tick_fontsize = st.number_input("tick_fontsize", min_value=6, max_value=16, value=10, key="ucs_is50_tick_fontsize")
            with col4:
                pass
            with col5:
                pass
            with col6:
                dpi = st.number_input(
                    "dpi",
                    min_value=150,
                    max_value=600,
                    value=300,
                    step=50,
                    key="ucs_is50_dpi"
                )
        
        # Apply filters to data
        filtered_ucs = ucs_data.copy()
        
        # Apply Filter 1
        if filter1_by != "None" and filter1_values:
            if filter1_by == "Geology Origin":
                filtered_ucs = filtered_ucs[filtered_ucs['Geology_Orgin'].isin(filter1_values)]
            elif filter1_by == "Consistency":
                filtered_ucs = filtered_ucs[filtered_ucs['Consistency'].isin(filter1_values)]
            elif filter1_by == "Hole ID":
                filtered_ucs = filtered_ucs[filtered_ucs['Hole_ID'].isin(filter1_values)]
            elif filter1_by == "Report" and 'Report' in filtered_ucs.columns:
                filtered_ucs = filtered_ucs[filtered_ucs['Report'].isin(filter1_values)]
        
        # Apply Filter 2
        if filter2_by != "None" and filter2_values:
            if filter2_by == "Geology Origin":
                filtered_ucs = filtered_ucs[filtered_ucs['Geology_Orgin'].isin(filter2_values)]
            elif filter2_by == "Consistency":
                filtered_ucs = filtered_ucs[filtered_ucs['Consistency'].isin(filter2_values)]
            elif filter2_by == "Hole ID":
                filtered_ucs = filtered_ucs[filtered_ucs['Hole_ID'].isin(filter2_values)]
            elif filter2_by == "Report" and 'Report' in filtered_ucs.columns:
                filtered_ucs = filtered_ucs[filtered_ucs['Report'].isin(filter2_values)]
        
        # Generate dynamic title suffix based on applied filters
        def generate_title_suffix():
            suffix_parts = []
            
            # Add Filter 1 to suffix if applied (and not "All")
            if filter1_by != "None" and filter1_values:
                all_options = get_filter_options(filter1_by)
                if filter1_values != all_options:
                    if len(filter1_values) == 1:
                        suffix_parts.append(f"{filter1_values[0]}")
                    elif len(filter1_values) <= 3:
                        suffix_parts.append(f"{', '.join(filter1_values)}")
                    else:
                        suffix_parts.append(f"{filter1_by} (Multiple)")
            
            # Add Filter 2 to suffix if applied (and not "All")
            if filter2_by != "None" and filter2_values:
                all_options = get_filter_options(filter2_by)
                if filter2_values != all_options:
                    if len(filter2_values) == 1:
                        suffix_parts.append(f"{filter2_values[0]}")
                    elif len(filter2_values) <= 3:
                        suffix_parts.append(f"{', '.join(filter2_values)}")
                    else:
                        suffix_parts.append(f"{filter2_by} (Multiple)")
            
            return " - ".join(suffix_parts) if suffix_parts else None
        
        # Determine final title and suffix
        dynamic_suffix = generate_title_suffix()
        if not title:
            final_title = None
            final_title_suffix = dynamic_suffix or title_suffix
        else:
            final_title = title
            final_title_suffix = title_suffix
        
        if filtered_ucs.empty:
            st.warning("No data remains after applying filters. Please adjust your filter criteria.")
            return
        
        # Prepare datasets based on analysis type
        datasets = []
        title = "UCS vs Is50 Correlation"
        
        if analysis_type == "Overall Correlation":
            if is50_axial_cols:
                datasets.append({
                    'data_df': filtered_ucs,
                    'x_col': is50_axial_cols[0],
                    'y_col': ucs_column,
                    'label': 'Overall Correlation'
                })
                
        elif analysis_type == "By Geological Unit":
            if is50_axial_cols and category_by in filtered_ucs.columns:
                datasets.append({
                    'data_df': filtered_ucs,
                    'x_col': is50_axial_cols[0],
                    'y_col': ucs_column
                })
                
        elif analysis_type == "Axial vs Diametral":
            if is50_axial_cols:
                datasets.append({
                    'data_df': filtered_ucs,
                    'x_col': is50_axial_cols[0],
                    'y_col': ucs_column,
                    'label': 'Axial',
                    'color': 'orange',
                    'marker': 'o'
                })
            if is50_diametral_cols:
                datasets.append({
                    'data_df': filtered_ucs,
                    'x_col': is50_diametral_cols[0],
                    'y_col': ucs_column,
                    'label': 'Diametral',
                    'color': 'blue',
                    'marker': 's'
                })
                
        elif analysis_type == "Custom Analysis":
            # Advanced multi-dataset analysis like Jupyter notebook examples
            if geological_unit != "All" and is50_axial_cols and is50_diametral_cols:
                # Split by Is50 threshold for detailed analysis
                threshold = 4.6
                for col_type, col_name in [("axial", is50_axial_cols[0]), ("diametral", is50_diametral_cols[0])]:
                    if col_name in filtered_ucs.columns:
                        less_data = filtered_ucs[filtered_ucs[col_name] <= threshold]
                        greater_data = filtered_ucs[filtered_ucs[col_name] > threshold]
                        
                        if not less_data.empty:
                            datasets.append({
                                'data_df': less_data,
                                'x_col': col_name,
                                'y_col': ucs_column,
                                'label': f'{col_type.title()} (<{threshold})'
                            })
                        if not greater_data.empty:
                            datasets.append({
                                'data_df': greater_data,
                                'x_col': col_name,
                                'y_col': ucs_column,
                                'label': f'{col_type.title()} (>{threshold})'
                            })
            else:
                st.warning("Custom analysis requires geological unit selection and both axial/diametral Is50 data.")
                return
        
        if not datasets:
            st.warning("No datasets available for correlation analysis.")
            return
            
        if HAS_FUNCTIONS:
            try:
                # Clear any existing figures first
                plt.close('all')
                
                # Parse tuple inputs (following enhanced pattern)
                xlim = parse_tuple(xlim_str, (0, 10))
                ylim = parse_tuple(ylim_str, (0, 140))
                figsize = parse_tuple(figsize_str, (8, 6))
                
                final_title = f"{title} - {title_suffix}" if title_suffix != "All Geological Units" else title
                
                plot_UCS_Is50(
                    datasets=datasets,
                    title=final_title,
                    category_by=category_by if analysis_type == "By Geological Unit" and show_legend else None,
                    xlim=xlim,
                    ylim=ylim,
                    xtick_interval=xtick_interval,
                    ytick_interval=ytick_interval,
                    show_trendlines=show_trendlines,
                    show_equations=show_equations,
                    show_legend=show_legend,
                    scatter_size=scatter_size,
                    scatter_alpha=scatter_alpha,
                    legend_fontsize=legend_fontsize,
                    figsize=figsize,
                    show_plot=False,
                    close_plot=False
                )
                
                # Display plot with size control
                current_fig = plt.gcf()
                display_plot_with_size_control(current_fig)
                
                # Store the plot for Rock Dashboard
                try:
                    current_fig = plt.gcf()
                    if current_fig and current_fig.get_axes():
                        import io
                        buf = io.BytesIO()
                        current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        store_rock_plot('ucs_is50_correlation', buf)
                except Exception as e:
                    pass
                
                # Simple download button
                from .plot_download_simple import create_simple_download_button
                create_simple_download_button("ucs_vs_is50", "main")
                
                # Enhanced plot summary with correlation analysis and strength classifications
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**UCS vs Is50 Correlation Summary**")
                    
                    summary_data = []
                    
                    # Basic data statistics
                    if not filtered_ucs.empty:
                        # UCS statistics
                        if ucs_column in filtered_ucs.columns:
                            ucs_values = filtered_ucs[ucs_column].dropna()
                            if not ucs_values.empty:
                                summary_data.extend([
                                    {'Parameter': 'Total UCS Tests', 'Value': f"{len(ucs_values):,}"},
                                    {'Parameter': 'UCS Mean', 'Value': f"{ucs_values.mean():.1f} MPa"},
                                    {'Parameter': 'UCS Range', 'Value': f"{ucs_values.min():.1f} - {ucs_values.max():.1f} MPa"},
                                    {'Parameter': 'UCS Std Dev', 'Value': f"{ucs_values.std():.1f} MPa"}
                                ])
                                
                                # UCS strength classification assessment (following enhanced pattern)
                                total_ucs_tests = len(ucs_values)
                                very_low = len(ucs_values[ucs_values < 2])
                                low = len(ucs_values[(ucs_values >= 2) & (ucs_values < 6)])
                                medium = len(ucs_values[(ucs_values >= 6) & (ucs_values < 20)])
                                high = len(ucs_values[(ucs_values >= 20) & (ucs_values < 60)])
                                very_high = len(ucs_values[(ucs_values >= 60) & (ucs_values < 200)])
                                extremely_high = len(ucs_values[ucs_values >= 200])
                                
                                summary_data.extend([
                                    {'Parameter': 'Very Low Strength (<2 MPa)', 'Value': f"{very_low} ({(very_low/total_ucs_tests)*100:.1f}%)"},
                                    {'Parameter': 'Low Strength (2-6 MPa)', 'Value': f"{low} ({(low/total_ucs_tests)*100:.1f}%)"},
                                    {'Parameter': 'Medium Strength (6-20 MPa)', 'Value': f"{medium} ({(medium/total_ucs_tests)*100:.1f}%)"},
                                    {'Parameter': 'High Strength (20-60 MPa)', 'Value': f"{high} ({(high/total_ucs_tests)*100:.1f}%)"},
                                    {'Parameter': 'Very High Strength (60-200 MPa)', 'Value': f"{very_high} ({(very_high/total_ucs_tests)*100:.1f}%)"},
                                    {'Parameter': 'Extremely High (>200 MPa)', 'Value': f"{extremely_high} ({(extremely_high/total_ucs_tests)*100:.1f}%)"}
                                ])
                        
                        # Is50 statistics
                        if is50_axial_cols:
                            for col in is50_axial_cols:
                                if col in filtered_ucs.columns:
                                    is50_values = filtered_ucs[col].dropna()
                                    if not is50_values.empty:
                                        summary_data.extend([
                                            {'Parameter': f'Total {col} Tests', 'Value': f"{len(is50_values):,}"},
                                            {'Parameter': f'{col} Mean', 'Value': f"{is50_values.mean():.2f} MPa"},
                                            {'Parameter': f'{col} Range', 'Value': f"{is50_values.min():.2f} - {is50_values.max():.2f} MPa"}
                                        ])
                        
                        # Correlation analysis (following enhanced pattern)
                        if is50_axial_cols and ucs_column in filtered_ucs.columns:
                            for col in is50_axial_cols:
                                if col in filtered_ucs.columns:
                                    # Calculate correlation for paired data
                                    paired_data = filtered_ucs[[ucs_column, col]].dropna()
                                    if len(paired_data) > 1:
                                        correlation = paired_data[ucs_column].corr(paired_data[col])
                                        r_squared = correlation**2
                                        
                                        # Calculate UCS/Is50 ratio
                                        ratio_mean = (paired_data[ucs_column] / paired_data[col]).mean()
                                        
                                        summary_data.extend([
                                            {'Parameter': f'UCS vs {col} Correlation (R)', 'Value': f"{correlation:.3f}"},
                                            {'Parameter': f'UCS vs {col} R²', 'Value': f"{r_squared:.3f}"},
                                            {'Parameter': f'UCS vs {col} Paired Samples', 'Value': f"{len(paired_data)}"},
                                            {'Parameter': f'UCS/{col} Ratio (Mean)', 'Value': f"{ratio_mean:.1f}"}
                                        ])
                                        
                                        # Correlation strength interpretation
                                        if r_squared > 0.7:
                                            corr_strength = "Strong"
                                        elif r_squared > 0.4:
                                            corr_strength = "Moderate"
                                        else:
                                            corr_strength = "Weak"
                                        
                                        summary_data.append({
                                            'Parameter': f'UCS vs {col} Correlation Strength',
                                            'Value': corr_strength
                                        })
                        
                        # Geological distribution if available
                        if 'Geology_Orgin' in filtered_ucs.columns:
                            geology_counts = filtered_ucs['Geology_Orgin'].value_counts()
                            summary_data.append({
                                'Parameter': 'Geological Units Tested',
                                'Value': f"{len(geology_counts)} types"
                            })
                            for geology, count in geology_counts.head(3).items():
                                percentage = (count / len(filtered_ucs)) * 100
                                summary_data.append({
                                    'Parameter': f'  {geology}',
                                    'Value': f"{count} ({percentage:.1f}%)"
                                })
                        
                        # Depth distribution analysis
                        if 'From_mbgl' in filtered_ucs.columns:
                            depth_data = filtered_ucs['From_mbgl'].dropna()
                            if not depth_data.empty:
                                shallow = (depth_data <= 5).sum()
                                medium_depth = ((depth_data > 5) & (depth_data <= 15)).sum()
                                deep = (depth_data > 15).sum()
                                
                                summary_data.extend([
                                    {'Parameter': 'Shallow Depth (≤5m)', 'Value': f"{shallow} tests"},
                                    {'Parameter': 'Medium Depth (5-15m)', 'Value': f"{medium_depth} tests"},
                                    {'Parameter': 'Deep (>15m)', 'Value': f"{deep} tests"}
                                ])
                    
                    # Create summary table
                    if summary_data:
                        summary_df_display = pd.DataFrame(summary_data)
                        st.dataframe(summary_df_display, use_container_width=True, hide_index=True)
                    else:
                        st.info("No summary data available")
                
                with col2:
                    st.markdown("**UCS vs Is50 Correlation Guidelines**")
                    st.write("**Typical UCS/Is50 Ratios:**")
                    st.write("• **Intact Rock**: 15-25 (strong rock)")
                    st.write("• **Weathered Rock**: 10-15 (weaker rock)")
                    st.write("• **Highly Weathered**: 5-10 (very weak)")
                    st.write("")
                    st.write("**Correlation Strength:**")
                    st.write("• **Strong**: R² > 0.7 (reliable prediction)")
                    st.write("• **Moderate**: R² 0.4-0.7 (fair prediction)")
                    st.write("• **Weak**: R² < 0.4 (poor prediction)")
                    st.write("")
                    st.write("**Is50 Test Types:**")
                    st.write("• **Axial**: Load parallel to core axis")
                    st.write("• **Diametral**: Load perpendicular to axis")
                    st.write("• **Axial vs Diametral**: Axial typically 20% higher")
                    st.write("")
                    st.write("**Rock Strength Classification:**")
                    st.write("• **Very Low**: <2 MPa (very weak rock)")
                    st.write("• **Low**: 2-6 MPa (weak rock)")
                    st.write("• **Medium**: 6-20 MPa (moderately strong)")
                    st.write("• **High**: 20-60 MPa (strong rock)")
                    st.write("• **Very High**: 60-200 MPa (very strong)")
                    st.write("• **Extremely High**: >200 MPa (extremely strong)")
                
                # Add map visualization (following enhanced tab pattern)
                render_ucs_is50_map_visualization(filtered_ucs)
                
                # Add test distribution by chainage (following enhanced tab pattern)
                render_ucs_is50_test_distribution(filtered_ucs, filtered_data)
                
                # Add data preview and statistics sections (following enhanced tab pattern)
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.checkbox("Show data preview", key="ucs_is50_data_preview"):
                        # Show relevant columns for UCS vs Is50 analysis
                        preview_cols = ['Hole_ID', 'From_mbgl', 'To_mbgl', ucs_column]
                        
                        # Add Is50 columns to preview
                        if is50_axial_cols:
                            preview_cols.extend(is50_axial_cols)
                        if is50_diametral_cols:
                            preview_cols.extend(is50_diametral_cols)
                        
                        # Add geological and location columns if available
                        additional_cols = ['Geology_Orgin', 'Consistency', 'Chainage']
                        for col in additional_cols:
                            if col in filtered_ucs.columns:
                                preview_cols.append(col)
                        
                        # Filter to only available columns
                        available_cols = [col for col in preview_cols if col in filtered_ucs.columns]
                        
                        if available_cols:
                            st.dataframe(filtered_ucs[available_cols].head(20), use_container_width=True)
                            st.caption(f"{len(filtered_ucs)} total records")
                        else:
                            st.info("No data columns available for preview")
                
                with col2:
                    if st.checkbox("Show detailed statistics", key="ucs_is50_statistics"):
                        if not filtered_ucs.empty:
                            stats_data = []
                            
                            # UCS statistics
                            if ucs_column in filtered_ucs.columns:
                                ucs_values = filtered_ucs[ucs_column].dropna()
                                if not ucs_values.empty:
                                    stats_data.extend([
                                        {'Parameter': 'UCS Samples', 'Value': f"{len(ucs_values)}"},
                                        {'Parameter': 'UCS Mean', 'Value': f"{ucs_values.mean():.1f} MPa"},
                                        {'Parameter': 'UCS Std Dev', 'Value': f"{ucs_values.std():.1f} MPa"},
                                        {'Parameter': 'UCS Min', 'Value': f"{ucs_values.min():.1f} MPa"},
                                        {'Parameter': 'UCS Max', 'Value': f"{ucs_values.max():.1f} MPa"},
                                        {'Parameter': 'UCS 25th %ile', 'Value': f"{np.percentile(ucs_values, 25):.1f} MPa"},
                                        {'Parameter': 'UCS 50th %ile', 'Value': f"{np.percentile(ucs_values, 50):.1f} MPa"},
                                        {'Parameter': 'UCS 75th %ile', 'Value': f"{np.percentile(ucs_values, 75):.1f} MPa"}
                                    ])
                            
                            # Is50 axial statistics
                            if is50_axial_cols:
                                for col in is50_axial_cols:
                                    if col in filtered_ucs.columns:
                                        is50_values = filtered_ucs[col].dropna()
                                        if not is50_values.empty:
                                            stats_data.extend([
                                                {'Parameter': f'{col} Samples', 'Value': f"{len(is50_values)}"},
                                                {'Parameter': f'{col} Mean', 'Value': f"{is50_values.mean():.2f} MPa"},
                                                {'Parameter': f'{col} Std Dev', 'Value': f"{is50_values.std():.2f} MPa"},
                                                {'Parameter': f'{col} Min', 'Value': f"{is50_values.min():.2f} MPa"},
                                                {'Parameter': f'{col} Max', 'Value': f"{is50_values.max():.2f} MPa"}
                                            ])
                            
                            # Is50 diametral statistics
                            if is50_diametral_cols:
                                for col in is50_diametral_cols:
                                    if col in filtered_ucs.columns:
                                        is50_values = filtered_ucs[col].dropna()
                                        if not is50_values.empty:
                                            stats_data.extend([
                                                {'Parameter': f'{col} Samples', 'Value': f"{len(is50_values)}"},
                                                {'Parameter': f'{col} Mean', 'Value': f"{is50_values.mean():.2f} MPa"},
                                                {'Parameter': f'{col} Std Dev', 'Value': f"{is50_values.std():.2f} MPa"},
                                                {'Parameter': f'{col} Min', 'Value': f"{is50_values.min():.2f} MPa"},
                                                {'Parameter': f'{col} Max', 'Value': f"{is50_values.max():.2f} MPa"}
                                            ])
                            
                            # Correlation statistics (if both UCS and Is50 data available)
                            if is50_axial_cols and ucs_column in filtered_ucs.columns:
                                for col in is50_axial_cols:
                                    if col in filtered_ucs.columns:
                                        # Calculate correlation for paired data
                                        paired_data = filtered_ucs[[ucs_column, col]].dropna()
                                        if len(paired_data) > 1:
                                            correlation = paired_data[ucs_column].corr(paired_data[col])
                                            stats_data.append({
                                                'Parameter': f'UCS vs {col} Correlation (R)',
                                                'Value': f"{correlation:.3f}"
                                            })
                                            stats_data.append({
                                                'Parameter': f'UCS vs {col} R²',
                                                'Value': f"{correlation**2:.3f}"
                                            })
                                            stats_data.append({
                                                'Parameter': f'UCS vs {col} Paired Samples',
                                                'Value': f"{len(paired_data)}"
                                            })
                            
                            # Geological distribution if available
                            if 'Geology_Orgin' in filtered_ucs.columns:
                                geo_counts = filtered_ucs['Geology_Orgin'].value_counts()
                                stats_data.append({
                                    'Parameter': 'Geological Units',
                                    'Value': f"{len(geo_counts)} types"
                                })
                                for geo_type, count in geo_counts.head(3).items():
                                    stats_data.append({
                                        'Parameter': f'  {geo_type}',
                                        'Value': f"{count} samples"
                                    })
                            
                            # Create statistics table
                            if stats_data:
                                stats_df = pd.DataFrame(stats_data)
                                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No statistical data available")
                        else:
                            st.info("No UCS vs Is50 data available for detailed statistics")
                        
            except Exception as e:
                st.error(f"Error creating UCS vs Is50 plot: {str(e)}")
        else:
            st.error("❌ Functions folder not accessible")
            st.info("Check Functions folder and UCS plotting modules")
    
    except Exception as e:
        st.error(f"Error in UCS vs Is50 analysis: {str(e)}")
        st.error("Please check that your data contains valid UCS and Is50 columns")


def render_ucs_is50_map_visualization(ucs_data: pd.DataFrame):
    """
    Render UCS vs Is50 test location map visualization (following enhanced tab pattern).
    """
    try:
        st.markdown("### Test Locations Map")
        
        # Check for coordinate data and display map
        if HAS_PYPROJ:
            # Use dynamic ID columns detection to find coordinate columns
            id_columns = get_id_columns_from_data(ucs_data)
            
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
                
                # Get coordinate data from UCS test locations
                try:
                    # Get unique sample locations from UCS data
                    sample_locations = ucs_data[['Hole_ID', 'From_mbgl']].drop_duplicates()
                    
                    # Merge with coordinate data including Is50 columns for hover info
                    merge_cols = ['Hole_ID', 'From_mbgl', lat_col, lon_col]
                    if 'Chainage' in ucs_data.columns:
                        merge_cols.append('Chainage')
                    
                    # Include Is50 columns for hover info
                    is50_cols = [col for col in ucs_data.columns if 'Is50' in col]
                    merge_cols.extend(is50_cols)
                    
                    coord_data = sample_locations.merge(
                        ucs_data[merge_cols], 
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
                                
                                # Display enhanced map with UCS vs Is50 test locations
                                if HAS_PLOTLY:
                                    # Prepare hover data including Is50 values
                                    hover_data_dict = {'From_mbgl': True}
                                    if 'Chainage' in map_data.columns:
                                        hover_data_dict['Chainage'] = True
                                    for col in is50_cols:
                                        if col in map_data.columns:
                                            hover_data_dict[col] = True
                                    
                                    # Calculate optimal zoom and center
                                    zoom_level, center = calculate_map_zoom_and_center(map_data['lat'], map_data['lon'])
                                    
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat='lat',
                                        lon='lon',
                                        hover_name='Hole_ID',
                                        hover_data=hover_data_dict,
                                        color_discrete_sequence=['orange'],
                                        zoom=zoom_level,
                                        center=center,
                                        height=400,
                                        title=f"UCS vs Is50 Test Locations ({len(coord_data)} locations)"
                                    )
                                    fig.update_layout(mapbox_style="carto-positron")
                                    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
                                    # Enable zoom and scroll functionality with 90% width layout
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
                                st.warning(f"Could not convert coordinates: {str(e)}")
                        else:
                            # Already in lat/lon format
                            map_data['lat'] = coord_data[lat_col]
                            map_data['lon'] = coord_data[lon_col]
                            
                            if HAS_PLOTLY:
                                # Prepare hover data including Is50 values
                                hover_data_dict = {'From_mbgl': True}
                                if 'Chainage' in map_data.columns:
                                    hover_data_dict['Chainage'] = True
                                for col in is50_cols:
                                    if col in map_data.columns:
                                        hover_data_dict[col] = True
                                
                                # Calculate optimal zoom and center
                                zoom_level, center = calculate_map_zoom_and_center(map_data['lat'], map_data['lon'])
                                
                                fig = px.scatter_mapbox(
                                    map_data,
                                    lat='lat',
                                    lon='lon',
                                    hover_name='Hole_ID',
                                    hover_data=hover_data_dict,
                                    color_discrete_sequence=['orange'],
                                    zoom=zoom_level,
                                    center=center,
                                    height=400,
                                    title=f"UCS vs Is50 Test Locations ({len(coord_data)} locations)"
                                )
                                fig.update_layout(mapbox_style="carto-positron")
                                fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
                                # Enable zoom and scroll functionality with 90% width layout
                                map_col, spacer_col = st.columns([9, 1])
                                with map_col:
                                    st.plotly_chart(fig, use_container_width=True, config={
                                        'scrollZoom': True,
                                        'displayModeBar': False
                                    })
                            else:
                                st.info("Map visualization requires Plotly (coordinate data available)")
                            
                            st.caption(f"Found {len(coord_data)} UCS vs Is50 test locations with coordinates")
                    else:
                        st.info("No coordinate data found for UCS vs Is50 test locations")
                except Exception as e:
                    st.warning(f"Could not process coordinates: {str(e)}")
            else:
                st.info("No coordinate columns detected in the data")
        else:
            st.info("Map visualization requires pyproj library")
            
    except Exception as e:
        st.error(f"Error creating map visualization: {str(e)}")


def render_ucs_is50_test_distribution(ucs_data: pd.DataFrame, original_filtered_data: pd.DataFrame):
    """
    Render UCS vs Is50 test distribution by chainage (following enhanced tab pattern).
    """
    try:
        st.markdown("### Test Distribution by Chainage")
        
        # Check if chainage data is available
        if 'Chainage' not in original_filtered_data.columns:
            st.info("Chainage data not available for test distribution analysis")
            return
            
        chainage_data = original_filtered_data['Chainage'].dropna()
        if chainage_data.empty:
            st.info("No chainage data available for test distribution analysis")
            return
        
        # Create chainage bins (200m intervals following enhanced pattern)
        bin_interval = 200
        min_chainage = chainage_data.min()
        max_chainage = chainage_data.max()
        
        bin_start = int(min_chainage // bin_interval) * bin_interval
        bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
        bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
        
        # Helper function for rendering individual test charts (following enhanced pattern)
        def render_single_test_chart(test_type, test_col_name):
            st.write(f"**{test_type} Distribution:**")
            
            if test_col_name in original_filtered_data.columns:
                # Use original data to count unique tests, not filtered data
                test_data = original_filtered_data[original_filtered_data[test_col_name] == 'Y']
                
                if not test_data.empty and 'Chainage' in test_data.columns:
                    # Count unique tests per chainage bin (one count per unique Hole_ID + From_mbgl combination)
                    test_data_unique = test_data.drop_duplicates(subset=['Hole_ID', 'From_mbgl'])
                    test_chainage = test_data_unique['Chainage'].dropna()
                    
                    if not test_chainage.empty:
                        # NUMPY HISTOGRAM IMPLEMENTATION (following enhanced pattern)
                        hist, bin_edges = np.histogram(test_chainage, bins=bins)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        chart_data = pd.DataFrame({
                            'Chainage': bin_centers.astype(int),
                            'Test Count': hist.astype(int)
                        })
                        
                        if chart_data['Test Count'].sum() > 0:
                            # 90% width layout following enhanced pattern
                            chart_col, spacer_col = st.columns([9, 1])
                            with chart_col:
                                st.line_chart(chart_data.set_index('Chainage'))
                            
                            # Show summary statistics
                            total_tests = chart_data['Test Count'].sum()
                            max_tests_per_bin = chart_data['Test Count'].max()
                            st.caption(f"Total {test_type} tests: {total_tests}, Max per 200m bin: {max_tests_per_bin}")
                        else:
                            st.info(f"No {test_type} tests found in chainage bins")
                    else:
                        st.info(f"No chainage data available for {test_type} tests")
                else:
                    st.info(f"No {test_type} test data found")
            else:
                st.info(f"{test_type} test indicator column not found")
        
        # Render UCS test distribution
        render_single_test_chart("UCS", "UCS?")
        
        # Render Is50 test distributions (check for different Is50 test types)
        is50_test_types = [
            ("Is50 Axial", "Is50a?"),
            ("Is50 Diametral", "Is50d?"),
            ("Is50", "Is50?")  # Generic Is50 test indicator
        ]
        
        for test_name, test_col in is50_test_types:
            if test_col in original_filtered_data.columns:
                render_single_test_chart(test_name, test_col)
        
        # Combined UCS + Is50 distribution (tests that have both)
        st.write("**Combined UCS + Is50 Distribution:**")
        if 'UCS?' in original_filtered_data.columns:
            # Find samples that have both UCS and any Is50 test
            is50_indicators = [col for col in original_filtered_data.columns if col.startswith('Is50') and col.endswith('?')]
            
            if is50_indicators:
                # Create combined filter for samples with both UCS and any Is50 test
                ucs_filter = original_filtered_data['UCS?'] == 'Y'
                is50_filter = original_filtered_data[is50_indicators].eq('Y').any(axis=1)
                combined_filter = ucs_filter & is50_filter
                
                combined_data = original_filtered_data[combined_filter]
                
                if not combined_data.empty and 'Chainage' in combined_data.columns:
                    # Count unique combined tests per chainage bin
                    combined_data_unique = combined_data.drop_duplicates(subset=['Hole_ID', 'From_mbgl'])
                    combined_chainage = combined_data_unique['Chainage'].dropna()
                    
                    if not combined_chainage.empty:
                        # NUMPY HISTOGRAM IMPLEMENTATION
                        hist, bin_edges = np.histogram(combined_chainage, bins=bins)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        chart_data = pd.DataFrame({
                            'Chainage': bin_centers.astype(int),
                            'Test Count': hist.astype(int)
                        })
                        
                        if chart_data['Test Count'].sum() > 0:
                            # 90% width layout following enhanced pattern
                            chart_col, spacer_col = st.columns([9, 1])
                            with chart_col:
                                st.line_chart(chart_data.set_index('Chainage'))
                            
                            # Show summary statistics
                            total_tests = chart_data['Test Count'].sum()
                            max_tests_per_bin = chart_data['Test Count'].max()
                            st.caption(f"Total UCS+Is50 combined tests: {total_tests}, Max per 200m bin: {max_tests_per_bin}")
                        else:
                            st.info("No combined UCS+Is50 tests found in chainage bins")
                    else:
                        st.info("No chainage data available for combined UCS+Is50 tests")
                else:
                    st.info("No combined UCS+Is50 test data found")
            else:
                st.info("No Is50 test indicators found for combined analysis")
        else:
            st.info("UCS test indicator column not found for combined analysis")
        
    except Exception as e:
        st.error(f"Error creating test distribution visualization: {str(e)}")


def render_ucs_analysis_tab(filtered_data: pd.DataFrame):
    """
    Render UCS vs Depth analysis tab.
    This tab is specifically for UCS vs Depth analysis only.
    
    Args:
        filtered_data: Filtered laboratory data
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render tab.")
        return
    
    # Extract UCS data to check availability
    try:
        ucs_data = extract_ucs_data(filtered_data)
    except:
        st.error("No UCS data available in the current dataset.")
        return
    
    if ucs_data.empty:
        st.warning("No UCS data available with current filters.")
        return
    
    # Render UCS vs Depth analysis (this function handles its own filtering)
    filtered_ucs_data = render_ucs_vs_depth_analysis(filtered_data)
    
    # Get the filtered UCS data for other visualizations
    if filtered_ucs_data is None or filtered_ucs_data.empty:
        try:
            filtered_ucs_data = extract_ucs_data(filtered_data)
        except:
            st.error("No UCS data available for visualizations.")
            return
        
    if filtered_ucs_data.empty:
        st.warning("No UCS data available for additional visualizations.")
        return
    
    # Add map visualization if coordinate data is available
    if HAS_PLOTLY and any(col in filtered_ucs_data.columns for col in ['Chainage', 'Easting', 'Northing']):
        render_ucs_map_visualization(filtered_ucs_data)
        
    # Add test distribution by chainage
    if 'Chainage' in filtered_ucs_data.columns:
        render_ucs_test_distribution(filtered_ucs_data)
        
    # Add plot summary
    render_ucs_plot_summary(filtered_ucs_data)
    
    # Data preview and statistics options (matching PSD/Atterberg/SPT/Emerson layout)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Show data preview", key="ucs_data_preview"):
            # Show relevant columns for UCS analysis (remove duplicate Geology_Orgin)
            preview_cols = ['Hole_ID', 'From_mbgl', 'To_mbgl', 'UCS (MPa)', 'Geology_Orgin']
            
            available_cols = [col for col in preview_cols if col in filtered_ucs_data.columns]
            # Remove duplicates while preserving order
            available_cols = list(dict.fromkeys(available_cols))
            
            st.dataframe(filtered_ucs_data[available_cols].head(20), use_container_width=True)
            st.caption(f"{len(filtered_ucs_data)} total records")
    
    with col2:
        if st.checkbox("Show detailed statistics", key="ucs_statistics"):
            if not filtered_ucs_data.empty and 'UCS (MPa)' in filtered_ucs_data.columns:
                # Calculate detailed UCS statistics for advanced users
                ucs_values = filtered_ucs_data['UCS (MPa)'].dropna()
                stats_data = []
                
                # Advanced statistics
                if not ucs_values.empty:
                    # Percentiles
                    percentiles = [10, 25, 50, 75, 90]
                    for p in percentiles:
                        value = np.percentile(ucs_values, p)
                        stats_data.append({
                            'Parameter': f'{p}th Percentile',
                            'Value': f"{value:.1f} MPa"
                        })
                    
                    # Additional statistics
                    stats_data.extend([
                        {'Parameter': 'Coefficient of Variation', 'Value': f"{(ucs_values.std()/ucs_values.mean())*100:.1f}%"},
                        {'Parameter': 'Skewness', 'Value': f"{ucs_values.skew():.2f}"},
                        {'Parameter': 'Kurtosis', 'Value': f"{ucs_values.kurtosis():.2f}"}
                    ])
                    
                    # UCS strength classification details
                    very_low = len(ucs_values[ucs_values < 2])
                    low = len(ucs_values[(ucs_values >= 2) & (ucs_values < 6)])
                    medium = len(ucs_values[(ucs_values >= 6) & (ucs_values < 20)])
                    high = len(ucs_values[(ucs_values >= 20) & (ucs_values < 60)])
                    very_high = len(ucs_values[(ucs_values >= 60) & (ucs_values < 200)])
                    extremely_high = len(ucs_values[ucs_values >= 200])
                    
                    total = len(ucs_values)
                    stats_data.extend([
                        {'Parameter': 'Very Low Strength (<2 MPa)', 'Value': f"{very_low} ({(very_low/total)*100:.1f}%)"},
                        {'Parameter': 'Low Strength (2-6 MPa)', 'Value': f"{low} ({(low/total)*100:.1f}%)"},
                        {'Parameter': 'Medium Strength (6-20 MPa)', 'Value': f"{medium} ({(medium/total)*100:.1f}%)"},
                        {'Parameter': 'High Strength (20-60 MPa)', 'Value': f"{high} ({(high/total)*100:.1f}%)"},
                        {'Parameter': 'Very High Strength (60-200 MPa)', 'Value': f"{very_high} ({(very_high/total)*100:.1f}%)"},
                        {'Parameter': 'Extremely High Strength (>200 MPa)', 'Value': f"{extremely_high} ({(extremely_high/total)*100:.1f}%)"}
                    ])
                
                # Create detailed statistics table
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.info("No UCS data available for detailed statistics")

def render_ucs_map_visualization(ucs_data: pd.DataFrame):
    """
    Render UCS test location map visualization (following Emerson/PSD/Atterberg style).
    """
    try:
        st.markdown("### Test Locations Map")
        
        # Check for coordinate data and display map
        if HAS_PYPROJ:
            # Use dynamic ID columns detection to find coordinate columns
            id_columns = get_id_columns_from_data(ucs_data)
            
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
                
                # Get coordinate data from UCS test locations
                try:
                    # Get unique sample locations from UCS data
                    sample_locations = ucs_data[['Hole_ID', 'From_mbgl']].drop_duplicates()
                    
                    # Merge with coordinate data
                    merge_cols = ['Hole_ID', 'From_mbgl', lat_col, lon_col]
                    if 'Chainage' in ucs_data.columns:
                        merge_cols.append('Chainage')
                    coord_data = sample_locations.merge(
                        ucs_data[merge_cols], 
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
                                
                                # Display enhanced map with UCS test locations
                                if HAS_PLOTLY:
                                    # Calculate dynamic zoom and center
                                    zoom_level, center_point = calculate_map_zoom_and_center(lats, lons)
                                    
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat='lat',
                                        lon='lon',
                                        hover_name='Hole_ID',
                                        hover_data={'From_mbgl': True, 'Chainage': True} if 'Chainage' in map_data.columns else {'From_mbgl': True},
                                        color_discrete_sequence=['blue'],
                                        zoom=zoom_level,
                                        center=center_point,
                                        height=400,
                                        title=f"UCS Test Locations ({len(coord_data)} locations)"
                                    )
                                    fig.update_layout(mapbox_style="carto-positron")
                                    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
                                    # Enable zoom and scroll functionality with 90% width layout
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
                                st.warning(f"Could not convert coordinates: {str(e)}")
                        else:
                            # Already in lat/lon format
                            map_data['lat'] = coord_data[lat_col]
                            map_data['lon'] = coord_data[lon_col]
                            
                            if HAS_PLOTLY:
                                # Calculate dynamic zoom and center
                                zoom_level, center_point = calculate_map_zoom_and_center(coord_data[lat_col], coord_data[lon_col])
                                
                                fig = px.scatter_mapbox(
                                    map_data,
                                    lat='lat',
                                    lon='lon',
                                    hover_name='Hole_ID',
                                    hover_data={'From_mbgl': True, 'Chainage': True} if 'Chainage' in map_data.columns else {'From_mbgl': True},
                                    color_discrete_sequence=['blue'],
                                    zoom=zoom_level,
                                    center=center_point,
                                    height=400,
                                    title=f"UCS Test Locations ({len(coord_data)} locations)"
                                )
                                fig.update_layout(mapbox_style="carto-positron")
                                fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
                                # Enable zoom and scroll functionality with 90% width layout
                                map_col, spacer_col = st.columns([9, 1])
                                with map_col:
                                    st.plotly_chart(fig, use_container_width=True, config={
                                        'scrollZoom': True,
                                        'displayModeBar': False
                                    })
                            else:
                                st.info("Map visualization requires Plotly (coordinate data available)")
                            
                            st.caption(f"Found {len(coord_data)} UCS test locations with coordinates")
                    else:
                        st.info("No coordinate data found for UCS test locations")
                except Exception as e:
                    st.warning(f"Could not process coordinates: {str(e)}")
            else:
                st.info("No coordinate columns detected in the data")
        else:
            st.info("Map visualization requires pyproj library")
            
    except Exception as e:
        st.error(f"Error creating map visualization: {str(e)}")

def render_ucs_test_distribution(ucs_data: pd.DataFrame):
    """
    Render UCS test distribution by chainage (following Emerson/PSD/Atterberg style).
    """
    try:
        # UCS Test Distribution by Chainage (directly after map like PSD/Atterberg/SPT)
        # Function to extract test types from columns
        def get_test_types_from_columns(data):
            test_columns = [col for col in data.columns if '?' in col]
            test_types = []
            for col in test_columns:
                test_type = col.replace('?', '').strip()
                if test_type:
                    test_types.append(test_type)
            return test_types, test_columns

        # Function to render single test chart
        def render_single_test_chart(test_type, original_filtered_data, bins):
            st.write(f"**{test_type} Distribution:**")
            
            test_col = f"{test_type}?"
            if test_col in original_filtered_data.columns:
                # Use original data to count unique tests, not filtered data
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
        if 'Chainage' in ucs_data.columns:
            available_test_types, test_columns = get_test_types_from_columns(ucs_data)
            
            if len(available_test_types) > 0:
                chainage_data = ucs_data['Chainage'].dropna()
                if not chainage_data.empty:
                    min_chainage = chainage_data.min()
                    max_chainage = chainage_data.max()
                    
                    # Create fixed interval bins (200m intervals)
                    bin_interval = 200
                    bin_start = int(min_chainage // bin_interval) * bin_interval
                    bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
                    bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
                    
                    # Find UCS-specific test types
                    ucs_test_types = [t for t in available_test_types if 'UCS' in t or 'ucs' in t.lower()]
                    
                    if ucs_test_types:
                        # Create charts for UCS test types - each chart at 90% width in separate rows
                        for i, test_type in enumerate(ucs_test_types):
                            if i > 0:
                                st.write("")
                            
                            # Each chart gets 90% width layout
                            chart_col, spacer_col = st.columns([9, 1])
                            
                            with chart_col:
                                render_single_test_chart(test_type, ucs_data, bins)
                    else:
                        # If no specific UCS tests found, show the first few available - each at 90% width
                        display_types = available_test_types[:4]  # Show up to 4 test types
                        for i, test_type in enumerate(display_types):
                            if i > 0:
                                st.write("")
                            
                            # Each chart gets 90% width layout
                            chart_col, spacer_col = st.columns([9, 1])
                            
                            with chart_col:
                                render_single_test_chart(test_type, ucs_data, bins)
                else:
                    st.info("No chainage data available for distribution analysis")
            else:
                st.info("No test data available for distribution analysis")
        else:
            st.info("Chainage column not found - cannot create spatial distribution")
            
    except Exception as e:
        st.warning(f"Could not generate UCS distribution chart: {str(e)}")

def render_ucs_plot_summary(ucs_data: pd.DataFrame):
    """
    Render comprehensive UCS analysis summary (following Emerson/PSD/Atterberg style).
    """
    try:
        # Add visual separator before plot summary
        st.divider()
        
        # Plot Summary (matching PSD/Atterberg/SPT/Emerson structure)
        st.markdown("**Plot Summary**")
        try:
            # Calculate engineering-relevant UCS statistics
            summary_data = []
            
            # Use the actual plotted data for summary
            if not ucs_data.empty and 'UCS (MPa)' in ucs_data.columns:
                ucs_values = ucs_data['UCS (MPa)'].dropna()
                
                if not ucs_values.empty:
                    # Basic UCS statistics
                    summary_data.extend([
                        {'Parameter': 'Total UCS Tests', 'Value': f"{len(ucs_values):,}"},
                        {'Parameter': 'Mean UCS', 'Value': f"{ucs_values.mean():.1f} MPa"},
                        {'Parameter': 'Median UCS', 'Value': f"{ucs_values.median():.1f} MPa"},
                        {'Parameter': 'Standard Deviation', 'Value': f"{ucs_values.std():.1f} MPa"},
                        {'Parameter': 'Range (Min-Max)', 'Value': f"{ucs_values.min():.1f} - {ucs_values.max():.1f} MPa"}
                    ])
                    
                    # Depth information
                    if 'From_mbgl' in ucs_data.columns:
                        depth_data = ucs_data['From_mbgl'].dropna()
                        if not depth_data.empty:
                            summary_data.append({
                                'Parameter': 'Depth Range (m)', 
                                'Value': f"{depth_data.min():.1f} - {depth_data.max():.1f}"
                            })
                    
                    # Rock strength classification assessment
                    very_low = len(ucs_values[ucs_values < 2])
                    low = len(ucs_values[(ucs_values >= 2) & (ucs_values < 6)])
                    medium = len(ucs_values[(ucs_values >= 6) & (ucs_values < 20)])
                    high = len(ucs_values[(ucs_values >= 20) & (ucs_values < 60)])
                    very_high = len(ucs_values[(ucs_values >= 60) & (ucs_values < 200)])
                    extremely_high = len(ucs_values[ucs_values >= 200])
                    total_tests = len(ucs_values)
                    
                    summary_data.extend([
                        {'Parameter': 'Very Low Strength (<2 MPa)', 'Value': f"{very_low} ({(very_low/total_tests)*100:.1f}%)"},
                        {'Parameter': 'Low Strength (2-6 MPa)', 'Value': f"{low} ({(low/total_tests)*100:.1f}%)"},
                        {'Parameter': 'Medium Strength (6-20 MPa)', 'Value': f"{medium} ({(medium/total_tests)*100:.1f}%)"},
                        {'Parameter': 'High Strength (20-60 MPa)', 'Value': f"{high} ({(high/total_tests)*100:.1f}%)"},
                        {'Parameter': 'Very High Strength (60-200 MPa)', 'Value': f"{very_high} ({(very_high/total_tests)*100:.1f}%)"},
                        {'Parameter': 'Extremely High Strength (>200 MPa)', 'Value': f"{extremely_high} ({(extremely_high/total_tests)*100:.1f}%)"}
                    ])
                    
                    # Geology origin breakdown if available
                    if 'Geology_Orgin' in ucs_data.columns:
                        geology_counts = ucs_data['Geology_Orgin'].value_counts()
                        for geology, count in geology_counts.items():
                            percentage = (count / len(ucs_data)) * 100
                            summary_data.append({
                                'Parameter': f'{geology} Tests',
                                'Value': f"{count} ({percentage:.1f}%)"
                            })
                    
                    # Depth category analysis
                    if 'From_mbgl' in ucs_data.columns:
                        depth_data = ucs_data['From_mbgl'].dropna()
                        if not depth_data.empty:
                            shallow = (depth_data <= 5).sum()
                            medium_depth = ((depth_data > 5) & (depth_data <= 15)).sum()
                            deep = (depth_data > 15).sum()
                            
                            summary_data.extend([
                                {'Parameter': 'Shallow Depth (≤5m)', 'Value': f"{shallow} tests"},
                                {'Parameter': 'Medium Depth (5-15m)', 'Value': f"{medium_depth} tests"},
                                {'Parameter': 'Deep (>15m)', 'Value': f"{deep} tests"}
                            ])
            
            # Create summary table
            if summary_data:
                summary_df_display = pd.DataFrame(summary_data)
                st.dataframe(summary_df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No summary data available")
        except Exception as e:
            st.warning(f"Could not generate plot summary: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error creating plot summary: {str(e)}")

def render_ucs_correlation_summary(ucs_data: pd.DataFrame):
    """
    Render correlation-specific analysis summary.
    """
    try:
        st.subheader("Correlation Analysis Summary")
        
        # Check for Is50 data availability
        is50_axial_cols = [col for col in ucs_data.columns if 'Is50a' in col and 'MPa' in col]
        is50_diametral_cols = [col for col in ucs_data.columns if 'Is50d' in col and 'MPa' in col]
        
        corr_col1, corr_col2 = st.columns(2)
        
        with corr_col1:
            st.markdown("**Available Correlations**")
            if is50_axial_cols:
                axial_data = ucs_data[is50_axial_cols[0]].dropna()
                st.write(f"Is50 Axial: {len(axial_data)} tests")
                if not axial_data.empty:
                    st.write(f"Range: {axial_data.min():.2f} - {axial_data.max():.2f} MPa")
                    
            if is50_diametral_cols:
                diametral_data = ucs_data[is50_diametral_cols[0]].dropna()
                st.write(f"Is50 Diametral: {len(diametral_data)} tests")
                if not diametral_data.empty:
                    st.write(f"Range: {diametral_data.min():.2f} - {diametral_data.max():.2f} MPa")
                    
            # Calculate correlation pairs
            if is50_axial_cols and 'UCS (MPa)' in ucs_data.columns:
                correlation_data = ucs_data[['UCS (MPa)', is50_axial_cols[0]]].dropna()
                st.write(f"Valid Correlation Pairs: {len(correlation_data)}")
        
        with corr_col2:
            st.markdown("**Correlation Guidelines**")
            st.write("• **UCS vs Is50 Ratio**: Typically 15-25 for intact rock")
            st.write("• **Strong Correlation**: R² > 0.7")
            st.write("• **Moderate Correlation**: R² 0.4-0.7")
            st.write("• **Weak Correlation**: R² < 0.4")
            st.write("• **Axial vs Diametral**: Axial typically 20% higher")
            
    except Exception as e:
        st.error(f"Error creating correlation summary: {str(e)}")

# Keep legacy functions for backward compatibility
def render_ucs_depth_tab(filtered_data: pd.DataFrame):
    """
    Legacy function - redirects to comprehensive UCS analysis.
    """
    render_ucs_analysis_tab(filtered_data)
    
def render_ucs_is50_tab(filtered_data: pd.DataFrame):
    """
    Legacy function - redirects to Is50 correlation analysis.
    """
    render_ucs_is50_correlation_analysis(filtered_data)