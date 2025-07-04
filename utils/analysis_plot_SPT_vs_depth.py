"""
SPT (Standard Penetration Test) Analysis Module

This module handles SPT data processing, analysis, and visualization for geotechnical applications.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Optional imports for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

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
except ImportError:
    # For standalone testing
    from data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_id_columns_from_data
    from plot_defaults import get_default_parameters, get_color_schemes

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

# Import Functions from Functions folder
try:
    import sys
    import os
    functions_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Functions')
    if functions_path not in sys.path:
        sys.path.insert(0, functions_path)
    
    from plot_SPT_vs_depth import plot_SPT_vs_depth
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


def extract_spt_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract SPT test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: SPT-specific dataframe
    """
    id_columns = get_id_columns_from_data(df)
    spt_columns = extract_test_columns(df, 'SPT')
    
    if not spt_columns:
        raise ValueError("No SPT data columns found")
    
    return create_test_dataframe(df, 'SPT', id_columns, spt_columns)


def get_spt_columns(spt_df: pd.DataFrame) -> Dict[str, str]:
    """
    Identify SPT N-value and depth columns from SPT dataframe.
    Based on original Jupyter notebook which uses 'SPT N Value' column.
    
    Args:
        spt_df: SPT dataframe
        
    Returns:
        Dict[str, str]: Dictionary with 'n_value_col' and 'depth_col' keys
    """
    # Primary column patterns based on original Jupyter notebook
    n_value_patterns = ['SPT N Value', 'SPT N', 'SPT_N', 'N_value', 'N60', 'N Value']
    depth_patterns = ['From_mbgl', 'Depth_m', 'From_m', 'Depth (m)', 'From (mbgl)', 'Depth']
    
    n_value_col = None
    depth_col = 'From_mbgl'  # Default from standard ID columns
    
    # Find N-value column - prioritize exact matches from original notebook
    for pattern in n_value_patterns:
        if pattern in spt_df.columns:
            n_value_col = pattern
            break
    
    # Find depth column (use From_mbgl if available from standard columns)
    if 'From_mbgl' in spt_df.columns:
        depth_col = 'From_mbgl'
    else:
        for pattern in depth_patterns:
            if pattern in spt_df.columns:
                depth_col = pattern
                break
    
    return {'n_value_col': n_value_col, 'depth_col': depth_col}


def classify_spt_consistency(n_value: float, soil_type: str = 'cohesive') -> str:
    """
    Classify soil consistency/density based on SPT N-value.
    
    Args:
        n_value: SPT N-value
        soil_type: 'cohesive' or 'granular'
        
    Returns:
        str: Consistency/density classification
    """
    if pd.isna(n_value):
        return 'Unknown'
    
    if soil_type.lower() == 'cohesive':
        # Cohesive soil consistency
        if n_value < 2:
            return 'Very Soft'
        elif n_value < 4:
            return 'Soft'
        elif n_value < 8:
            return 'Firm'
        elif n_value < 15:
            return 'Stiff'
        elif n_value < 30:
            return 'Very Stiff'
        else:
            return 'Hard'
    else:
        # Granular soil relative density
        if n_value < 4:
            return 'Very Loose'
        elif n_value < 10:
            return 'Loose'
        elif n_value < 30:
            return 'Medium Dense'
        elif n_value < 50:
            return 'Dense'
        else:
            return 'Very Dense'


def calculate_corrected_n_values(spt_df: pd.DataFrame, n_col: str, depth_col: str) -> pd.DataFrame:
    """
    Calculate corrected SPT N-values (N60, N1_60).
    
    Args:
        spt_df: SPT dataframe
        n_col: N-value column name
        depth_col: Depth column name
        
    Returns:
        pd.DataFrame: Dataframe with corrected N-values
    """
    result_df = spt_df.copy()
    
    # Standard corrections
    # Rod energy correction (Er = 60%)
    # Borehole diameter correction (Cb = 1.0 for 65-115mm)
    # Sampling method correction (Cs = 1.0 for standard sampler)
    # Liner correction (Cr = 1.0 for no liner)
    
    # Simple N60 correction (assuming 60% energy efficiency)
    if n_col in result_df.columns:
        result_df['N60'] = result_df[n_col] * (60/75)  # Assuming 75% efficiency
        
        # Overburden pressure correction (simplified)
        if depth_col in result_df.columns:
            # Simplified overburden correction factor
            # sigma_v0 = depth * 19 kN/m3 (average unit weight)
            sigma_v0 = result_df[depth_col] * 19  # kN/m2
            cn = np.minimum(2.0, (95.76 / sigma_v0) ** 0.5)  # Overburden correction factor
            result_df['N1_60'] = result_df['N60'] * cn
        else:
            result_df['N1_60'] = result_df['N60']
    
    return result_df



def render_spt_analysis_tab(filtered_data: pd.DataFrame):
    """
    Render the SPT analysis tab with comprehensive parameters and map visualization.
    
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
                cleaned = input_str.strip().replace('(', '').replace(')', '')
                values = [float(x.strip()) for x in cleaned.split(',')]
                return tuple(values) if len(values) == 2 else default
            except:
                return default
        
        # Extract SPT data
        spt_data = extract_spt_data(filtered_data)
        
        if spt_data.empty:
            st.warning("No SPT data available with current filters.")
            return
        
        # Check for required columns (depth and material type are always required, SPT column varies)
        required_base_cols = ['From_mbgl', 'Material Type']
        missing_cols = [col for col in required_base_cols if col not in spt_data.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return
        
        # Get smart defaults and helper functions
        # Get standard ID columns
        standard_id_columns = [col for col in get_standard_id_columns(spt_data) if col in spt_data.columns]
        if not standard_id_columns:
            standard_id_columns = list(spt_data.columns)[:5]  # Use first 5 columns as fallback
        
        # Smart column detection helper
        def find_column(patterns, columns, default=""):
            for pattern in patterns:
                for col in columns:
                    if pattern.lower() in col.lower():
                        return col
            return default
        
        # Get all available categorical columns for filtering (both ID columns and SPT-specific columns)
        def get_available_filter_columns():
            """Get all categorical columns available for filtering"""
            categorical_cols = []
            
            # Add standard ID columns that exist in the data
            id_cols_mapping = {
                "Geology Origin": "Geology_Orgin",
                "Consistency": "Consistency", 
                "Hole ID": "Hole_ID",
                "Report": "Report",
                "Material Type": "Material Type"
            }
            
            for display_name, col_name in id_cols_mapping.items():
                if col_name in spt_data.columns:
                    categorical_cols.append(display_name)
            
            # Add any additional categorical columns from SPT data
            for col in spt_data.columns:
                # Skip if already included via ID columns
                if col in id_cols_mapping.values():
                    continue
                    
                # Include categorical columns (non-numeric or mixed)
                if (spt_data[col].dtype == 'object' or 
                    spt_data[col].dtype.name == 'category' or
                    (spt_data[col].dtype in ['float64', 'int64'] and spt_data[col].nunique() <= 20)):
                    
                    # Exclude obvious numeric/system columns
                    if not any(pattern in col.lower() for pattern in ['mbgl', 'depth', 'from', 'to', 'id', 'lat', 'lon', 'north', 'east']):
                        categorical_cols.append(col)
            
            return categorical_cols
        
        # Helper function to get available values for any filter column
        def get_filter_options(filter_type):
            """Get unique values for any filter column"""
            # Map display names to actual column names
            column_mapping = {
                "Geology Origin": "Geology_Orgin",
                "Consistency": "Consistency",
                "Hole ID": "Hole_ID", 
                "Report": "Report",
                "Material Type": "Material Type"
            }
            
            # Get actual column name
            actual_col = column_mapping.get(filter_type, filter_type)
            
            if actual_col in spt_data.columns:
                unique_vals = spt_data[actual_col].dropna().unique()
                # Convert to string and sort
                return sorted([str(val) for val in unique_vals])
            else:
                return []
        
        # Display parameter inputs in organized layout (matching PSD/Atterberg style)
        with st.expander("Plot Parameters", expanded=True):
            # Row 1: Core Data Selection
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                depth_col = st.selectbox("Depth Column", standard_id_columns,
                    index=standard_id_columns.index("From_mbgl") if "From_mbgl" in standard_id_columns else 0,
                    key="spt_depth_col", help="Column containing depth values")
            with col2:
                # Smart SPT column detection with comprehensive pattern matching
                def get_spt_columns():
                    """Get all potential SPT columns with smart pattern matching"""
                    spt_patterns = [
                        'spt', 'spt n', 'spt_n', 'spt-n', 'spt value', 'spt_value', 'sptvalue',
                        'n value', 'n_value', 'nvalue', 'n-value', 'n60', 'n_60',
                        'standard penetration', 'standard_penetration', 'standardpenetration',
                        'blow count', 'blow_count', 'blowcount', 'blows', 'blow',
                        'penetration test', 'penetration_test', 'penetrationtest'
                    ]
                    potential_cols = []
                    for col in spt_data.columns:
                        col_lower = col.lower().replace(' ', '').replace('_', '').replace('-', '')
                        # Check for exact matches and partial matches
                        for pattern in spt_patterns:
                            pattern_clean = pattern.replace(' ', '').replace('_', '').replace('-', '')
                            if pattern_clean in col_lower:
                                potential_cols.append(col)
                                break
                        # Also check for SPT/N with numbers (SPT1, SPT2, N1, N2, etc.)
                        if any(term in col_lower for term in ['spt', 'n']) and any(char.isdigit() for char in col_lower):
                            potential_cols.append(col)
                    return list(set(potential_cols))  # Remove duplicates
                
                # Get available SPT columns
                available_spt_cols = get_spt_columns()
                
                # Find best default - prefer standard names over numbered variants
                spt_default = None
                spt_priority = ['SPT N Value', 'SPT N', 'SPT', 'N Value', 'N60', 'SPT_N', 'Standard_Penetration']
                for preferred in spt_priority:
                    if preferred in available_spt_cols:
                        spt_default = preferred
                        break
                if not spt_default and available_spt_cols:
                    spt_default = available_spt_cols[0]
                
                # SPT Column selection with fallback to all columns if no matches found
                if available_spt_cols:
                    spt_col_options = available_spt_cols
                    spt_default_index = spt_col_options.index(spt_default) if spt_default in spt_col_options else 0
                else:
                    # Fallback to all numeric columns if no SPT patterns found
                    spt_col_options = [col for col in spt_data.columns if spt_data[col].dtype in ['float64', 'int64']]
                    spt_default_index = 0
                    st.warning("No SPT columns found with standard naming. Please select from available numeric columns.")
                
                spt_col = st.selectbox("SPT Column", spt_col_options,
                    index=spt_default_index, key="spt_value_col", 
                    help="SPT N-Value column (supports SPT, SPT1, SPT2, N60, N_Value, Standard_Penetration, etc.)")
            with col3:
                material_type = st.selectbox("Material Type", ["All", "Cohesive", "Granular"], index=0, key="spt_material_type", help="Material classification for strength indicators")
            with col4:
                # Get dynamic filter options
                available_filter_cols = get_available_filter_columns()
                filter1_options = ["None"] + available_filter_cols
                default_index = 1 if "Geology Origin" in available_filter_cols else 0
                filter1_by = st.selectbox("Filter 1 By", filter1_options, index=default_index, key="spt_filter1_by", help="Select first filter type")
            with col5:
                if filter1_by == "None":
                    filter1_values = []
                    st.selectbox("Filter 1 Value", ["All"], index=0, disabled=True, key="spt_filter1_value_disabled", help="Select filter type first")
                else:
                    filter1_options = get_filter_options(filter1_by)
                    filter1_dropdown_options = ["All"] + filter1_options
                    filter1_selection = st.selectbox(f"{filter1_by}", filter1_dropdown_options, index=0, key="spt_filter1_value", help=f"Select {filter1_by.lower()} value")
                    if filter1_selection == "All":
                        filter1_values = filter1_options
                    else:
                        filter1_values = [filter1_selection]
            with col6:
                filter2_options = ["None"] + available_filter_cols
                filter2_by = st.selectbox("Filter 2 By", filter2_options, index=0, key="spt_filter2_by", help="Select second filter type")
            
            # Row 2: Color and Basic Plot Settings
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                if filter2_by == "None":
                    filter2_values = []
                    st.selectbox("Filter 2 Value", ["All"], index=0, disabled=True, key="spt_filter2_value_disabled", help="Select filter type first")
                else:
                    filter2_options = get_filter_options(filter2_by)
                    filter2_dropdown_options = ["All"] + filter2_options
                    filter2_selection = st.selectbox(f"{filter2_by}", filter2_dropdown_options, index=0, key="spt_filter2_value", help=f"Select {filter2_by.lower()} value")
                    if filter2_selection == "All":
                        filter2_values = filter2_options
                    else:
                        filter2_values = [filter2_selection]
            with col2:
                # Category columns should be from ID columns (geological/classification columns)
                category_columns = [col for col in standard_id_columns if col in spt_data.columns]
                if not category_columns:
                    category_columns = list(spt_data.columns)[:10]  # Use first 10 columns as fallback
                
                geology_index = 0
                if "Geology_Orgin" in category_columns:
                    geology_index = category_columns.index("Geology_Orgin")
                
                category_col = st.selectbox("Color by", category_columns, index=geology_index, key="spt_category_col", help="Group data by color")
            with col3:
                xlim_str = st.text_input("X-axis limits", value="(0, 51)", key="spt_xlim", help="X-axis limits as (min, max)")
            with col4:
                ylim_str = st.text_input("Y-axis limits", value="(0, 10)", key="spt_ylim", help="Y-axis limits as (min, max)")
            with col5:
                figsize_str = st.text_input("Figure size", value="(9, 6)", key="spt_figsize", help="Figure size as (width, height)")
            with col6:
                title = st.text_input("Title", value="", key="spt_title", help="Custom plot title")
        
        # Advanced Parameters Expander (matching PSD/Atterberg style)
        with st.expander("Advanced Parameters", expanded=False):
            # Row 1: Visual Customization
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                marker_size = st.number_input("Marker size", min_value=10, max_value=100, value=40, key="spt_marker_size", help="Size of data point markers")
            with col2:
                strength_position = st.number_input("Strength indicator position", min_value=0.0, max_value=1.0, value=0.85, step=0.05, key="spt_strength_position", help="Vertical position of strength labels")
            with col3:
                show_legend_option = st.selectbox("Show legend", ["Yes", "No"], index=0, key="spt_show_legend", help="Display legend for color categories")
                show_legend = show_legend_option == "Yes"
            with col4:
                legend_position = st.text_input("Legend position", value="(1.02, 1)", key="spt_legend_position", help="Legend position ('best' or tuple like '(1.02, 1)')")
            with col5:
                xtick_interval = st.number_input("X tick interval", min_value=1, max_value=20, value=10, key="spt_xtick", help="X-axis tick spacing")
            with col6:
                ytick_interval = st.number_input("Y tick interval", min_value=0.5, max_value=5.0, value=1.0, step=0.5, key="spt_ytick", help="Y-axis tick spacing")
            
            # Row 2: Font Settings
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                title_fontsize = st.number_input("Title font size", min_value=10, max_value=24, value=14, key="spt_title_font", help="Size of plot title")
            with col2:
                axis_label_fontsize = st.number_input("Axis label font size", min_value=8, max_value=20, value=12, key="spt_axis_font", help="Size of axis labels")
            with col3:
                legend_fontsize = st.number_input("Legend font size", min_value=6, max_value=16, value=10, key="spt_legend_font", help="Size of legend text")
            with col4:
                pass  # Empty for balance
            with col5:
                pass  # Empty for balance
            with col6:
                pass  # Empty for balance
        
        # Apply filters to data
        filtered_spt = spt_data.copy()
        
        # Apply Material Type filter first
        if material_type != "All":
            filtered_spt = filtered_spt[filtered_spt['Material Type'] == material_type].copy()
        
        # Helper function to apply dynamic filtering
        def apply_filter(data, filter_by, filter_values):
            """Apply filter dynamically to any column"""
            if filter_by == "None" or not filter_values:
                return data
                
            # Map display names to actual column names
            column_mapping = {
                "Geology Origin": "Geology_Orgin",
                "Consistency": "Consistency",
                "Hole ID": "Hole_ID",
                "Report": "Report", 
                "Material Type": "Material Type"
            }
            
            # Get actual column name
            actual_col = column_mapping.get(filter_by, filter_by)
            
            if actual_col in data.columns:
                return data[data[actual_col].astype(str).isin([str(val) for val in filter_values])]
            else:
                st.warning(f"Column '{actual_col}' not found in data. Skipping filter.")
                return data
        
        # Apply Filter 1
        filtered_spt = apply_filter(filtered_spt, filter1_by, filter1_values)
        
        # Apply Filter 2
        filtered_spt = apply_filter(filtered_spt, filter2_by, filter2_values)
        
        if filtered_spt.empty:
            st.warning("No data remains after applying filters. Please adjust your filter criteria.")
            return
        
        # Validate selected SPT column has data
        if spt_col not in filtered_spt.columns:
            st.error(f"Selected SPT column '{spt_col}' not found in data. Available columns with potential SPT data: {[col for col in filtered_spt.columns if any(term in col.lower() for term in ['spt', 'n', 'value', 'blow'])]}")
            st.info("💡 **Tip**: If your dataset uses different column names, select the appropriate column from the 'SPT Column' dropdown above.")
            return
        
        spt_data_count = filtered_spt[spt_col].notna().sum()
        if spt_data_count == 0:
            st.warning(f"Selected SPT column '{spt_col}' has no data. Please select a different column with SPT N-values.")
            return
        
        
        # Parse tuple inputs
        xlim = parse_tuple(xlim_str, (0, 51))
        ylim = parse_tuple(ylim_str, (0, 10))
        figsize = parse_tuple(figsize_str, (8, 6))
        
        # Parse legend position - support both string and tuple formats
        try:
            # Try to parse as tuple first (e.g., "(1.02, 1)")
            legend_bbox = parse_tuple(legend_position, (1.02, 1))
            legend_loc = "upper left"  # Default loc when using bbox coordinates
            legend_style = {'bbox_to_anchor': legend_bbox}
        except:
            # If parsing fails, treat as string position
            legend_loc = legend_position.strip() if legend_position.strip() else "best"
            legend_style = None
        
        # Main plotting section - full width
        if HAS_FUNCTIONS and HAS_MATPLOTLIB:
            try:
                # Clear any existing figures first
                if HAS_MATPLOTLIB:
                    plt.close('all')
                
                
                success = False
                
                # Determine which data to plot based on material type
                if material_type == "Cohesive":
                    spt_filtered = filtered_spt[filtered_spt['Material Type'] == 'Cohesive']
                    data_description = "Cohesive material"
                elif material_type == "Granular":
                    spt_filtered = filtered_spt[filtered_spt['Material Type'] == 'Granular']
                    data_description = "Granular material"
                else:  # material_type == "All"
                    spt_filtered = filtered_spt
                    data_description = "all materials"
                
                # Check if we have data to plot
                if spt_filtered.empty:
                    st.warning(f"No {data_description} data available")
                    success = False
                else:
                    try:
                        # Create title with suffix based on material type
                        final_title = title if title else None
                        if not title:  # Only add suffix if user hasn't provided custom title
                            final_title = None  # Let function use default title
                            title_suffix = material_type if material_type != "All" else "All Materials"
                        else:
                            title_suffix = None
                        
                        # Call unified SPT plotting function
                        plot_SPT_vs_depth(
                            spt_filtered, 
                            depth_col='From_mbgl', 
                            spt_col=spt_col,
                            material_type=material_type,
                            category_col=category_col, 
                            title=final_title,
                            title_suffix=title_suffix,
                            strength_indicator_position=strength_position, 
                            legend_loc=legend_loc,
                            legend_style=legend_style,
                            xlim=xlim, 
                            ylim=ylim, 
                            ytick_interval=ytick_interval, 
                            xtick_interval=xtick_interval,
                            figsize=figsize,
                            show_legend=show_legend,
                            show_plot=False,
                            close_plot=False
                        )
                        
                        # Display the plot
                        if HAS_MATPLOTLIB:
                            current_figs = plt.get_fignums()
                            if current_figs:
                                current_fig = plt.figure(current_figs[-1])
                                # Use the display function that respects sidebar width control
                                from .plotting_utils import display_plot_with_size_control
                                display_plot_with_size_control(current_fig)
                            
                        success = True
                        
                    except Exception as e:
                        st.error(f"Error creating {material_type} SPT plot: {str(e)}")
                        success = False
                
                
                if success:
                    # Store the plot for Site Dashboard
                    try:
                        if HAS_MATPLOTLIB:
                            current_fig = plt.gcf()
                            if current_fig and current_fig.get_axes():
                                import io
                                buf = io.BytesIO()
                                current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                plot_name = f'spt_analysis_{material_type.lower()}'
                                store_spatial_plot(plot_name, buf)
                    except Exception as e:
                        pass  # Don't break main functionality if storage fails
                    
                    # Simple download button with figure reference
                    from .plot_download_simple import create_simple_download_button
                    create_simple_download_button(f"spt_analysis_{material_type.lower()}", "main", fig=current_fig)
                else:
                    st.warning("No plot generated - check data availability")
                    
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
        else:
            st.error("Functions folder not accessible")
            st.info("Check Functions folder and SPT plotting modules")
        
        # Map visualization (matching PSD/Atterberg style)
        st.markdown("### Test Locations Map")
        
        # Check for coordinate data and display map
        if HAS_PYPROJ:
            # Use dynamic ID columns detection to find coordinate columns
            id_columns = get_id_columns_from_data(filtered_spt)
            
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
                
                # Get coordinate data from SPT test locations
                try:
                    # Get unique sample locations from SPT data
                    sample_locations = filtered_spt[['Hole_ID', 'From_mbgl']].drop_duplicates()
                    
                    # Merge with coordinate data
                    merge_cols = ['Hole_ID', 'From_mbgl', lat_col, lon_col]
                    if 'Chainage' in filtered_spt.columns:
                        merge_cols.append('Chainage')
                    coord_data = sample_locations.merge(
                        filtered_spt[merge_cols], 
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
                                
                                # Create map using Plotly
                                if HAS_PLOTLY:
                                    # Calculate optimal zoom and center
                                    zoom_level, center = calculate_map_zoom_and_center(map_data['lat'], map_data['lon'])
                                    
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat='lat',
                                        lon='lon',
                                        hover_name='Hole_ID',
                                        hover_data={'From_mbgl': True, 'Chainage': True} if 'Chainage' in map_data.columns else {'From_mbgl': True},
                                        color_discrete_sequence=['blue'],
                                        zoom=zoom_level,
                                        center=center,
                                        height=400,
                                        title=f"SPT Test Locations ({len(coord_data)} locations)"
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
                                    
                                # Show coordinate data summary
                                st.caption(f"Found {len(coord_data)} SPT test locations with coordinates")
                            except Exception as e:
                                st.warning(f"Coordinate conversion failed: {str(e)}")
                                st.caption(f"Raw coordinate data available: {len(coord_data)} locations")
                        else:
                            # Already in lat/lon format
                            map_data['lat'] = coord_data[lat_col]
                            map_data['lon'] = coord_data[lon_col]
                            
                            if HAS_PLOTLY:
                                # Calculate optimal zoom and center
                                zoom_level, center = calculate_map_zoom_and_center(map_data['lat'], map_data['lon'])
                                
                                fig = px.scatter_mapbox(
                                    map_data,
                                    lat='lat',
                                    lon='lon',
                                    hover_name='Hole_ID',
                                    hover_data={'From_mbgl': True, 'Chainage': True} if 'Chainage' in map_data.columns else {'From_mbgl': True},
                                    color_discrete_sequence=['blue'],
                                    zoom=zoom_level,
                                    center=center,
                                    height=400,
                                    title=f"SPT Test Locations ({len(coord_data)} locations)"
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
                            
                            st.caption(f"Found {len(coord_data)} SPT test locations with coordinates")
                    else:
                        st.info("No coordinate data found for SPT test locations")
                except Exception as e:
                    st.warning(f"Could not process coordinates: {str(e)}")
            else:
                st.info("No coordinate columns detected in the data")
        else:
            st.info("Map visualization requires pyproj library")
        
        # SPT Test Distribution by Chainage (directly after map like PSD/Atterberg)
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
            if 'Chainage' in filtered_spt.columns:
                available_test_types, test_columns = get_test_types_from_columns(filtered_spt)
                
                if len(available_test_types) > 0:
                    chainage_data = filtered_spt['Chainage'].dropna()
                    if not chainage_data.empty:
                        min_chainage = chainage_data.min()
                        max_chainage = chainage_data.max()
                        
                        # Create fixed interval bins (200m intervals)
                        bin_interval = 200
                        bin_start = int(min_chainage // bin_interval) * bin_interval
                        bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
                        bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
                        
                        # Find SPT-specific test types
                        spt_test_types = [t for t in available_test_types if 'SPT' in t or 'Standard' in t or 'Penetration' in t]
                        
                        if spt_test_types:
                            # Create charts for SPT test types - each chart at 90% width in separate rows
                            for i, test_type in enumerate(spt_test_types):
                                if i > 0:
                                    st.write("")
                                
                                # Each chart gets 90% width layout
                                chart_col, spacer_col = st.columns([9, 1])
                                
                                with chart_col:
                                    render_single_test_chart(test_type, filtered_spt, bins)
                        else:
                            # If no specific SPT tests found, show the first few available - each at 90% width
                            display_types = available_test_types[:4]  # Show up to 4 test types
                            for i, test_type in enumerate(display_types):
                                if i > 0:
                                    st.write("")
                                
                                # Each chart gets 90% width layout
                                chart_col, spacer_col = st.columns([9, 1])
                                
                                with chart_col:
                                    render_single_test_chart(test_type, filtered_spt, bins)
                    else:
                        st.info("No chainage data available for distribution analysis")
                else:
                    st.info("No test data available for distribution analysis")
            else:
                st.info("Chainage column not found - cannot create spatial distribution")
                
        except Exception as e:
            st.warning(f"Could not generate SPT distribution chart: {str(e)}")
        
        # SPT Test Distribution (matching PSD/Atterberg style)
        st.markdown("### SPT Test Distribution")
        
        # Material type distribution
        if 'Material Type' in filtered_spt.columns:
            material_counts = filtered_spt['Material Type'].value_counts()
            if not material_counts.empty:
                if HAS_PLOTLY:
                    fig = px.bar(
                        x=material_counts.index,
                        y=material_counts.values,
                        title="SPT Tests by Material Type",
                        labels={'x': 'Material Type', 'y': 'Number of Tests'}
                    )
                    fig.update_layout(height=300, margin={"r":0,"t":30,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("**Material Type Distribution:**")
                    for material, count in material_counts.items():
                        st.write(f"- {material}: {count} tests")
        
        # Geology distribution
        if 'Geology_Orgin' in filtered_spt.columns:
            geology_counts = filtered_spt['Geology_Orgin'].value_counts()
            if not geology_counts.empty:
                if HAS_PLOTLY:
                    fig = px.pie(
                        values=geology_counts.values,
                        names=geology_counts.index,
                        title="SPT Tests by Geology Origin"
                    )
                    fig.update_layout(height=400, margin={"r":0,"t":30,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("**Geology Origin Distribution:**")
                    for geology, count in geology_counts.items():
                        st.write(f"- {geology}: {count} tests")
        
        # Add visual separator before plot summary
        st.divider()
        
        # Plot Summary (matching PSD/Atterberg structure)
        if success:
            st.markdown("**Plot Summary**")
            try:
                # Calculate engineering-relevant SPT statistics
                summary_data = []
                
                # Use the actual plotted data for summary
                if not filtered_spt.empty and spt_col in filtered_spt.columns:
                    spt_values = filtered_spt[spt_col].dropna()
                    
                    if not spt_values.empty:
                        # Basic SPT statistics
                        summary_data.extend([
                            {'Parameter': 'Total SPT Tests', 'Value': f"{len(spt_values):,}"},
                            {'Parameter': 'Mean N-Value', 'Value': f"{spt_values.mean():.1f}"},
                            {'Parameter': 'Median N-Value', 'Value': f"{spt_values.median():.1f}"},
                            {'Parameter': 'Standard Deviation', 'Value': f"{spt_values.std():.1f}"},
                            {'Parameter': 'Range (Min-Max)', 'Value': f"{spt_values.min():.0f} - {spt_values.max():.0f}"}
                        ])
                        
                        # Depth information
                        if 'From_mbgl' in filtered_spt.columns:
                            depth_data = filtered_spt['From_mbgl'].dropna()
                            if not depth_data.empty:
                                summary_data.append({
                                    'Parameter': 'Depth Range (m)', 
                                    'Value': f"{depth_data.min():.1f} - {depth_data.max():.1f}"
                                })
                        
                        # Material type breakdown
                        if 'Material Type' in filtered_spt.columns:
                            material_counts = filtered_spt['Material Type'].value_counts()
                            for material, count in material_counts.items():
                                percentage = (count / len(filtered_spt)) * 100
                                summary_data.append({
                                    'Parameter': f'{material} Tests',
                                    'Value': f"{count} ({percentage:.1f}%)"
                                })
                        
                        # SPT consistency classification summary (for cohesive soils)
                        if material_type == "Cohesive":
                            # Define cohesive consistency classes
                            consistency_ranges = [
                                (0, 2, 'Very Soft'),
                                (2, 4, 'Soft'), 
                                (4, 8, 'Firm'),
                                (8, 15, 'Stiff'),
                                (15, 30, 'Very Stiff'),
                                (30, float('inf'), 'Hard')
                            ]
                            
                            for min_val, max_val, class_name in consistency_ranges:
                                if max_val == float('inf'):
                                    count = len(spt_values[spt_values >= min_val])
                                else:
                                    count = len(spt_values[(spt_values >= min_val) & (spt_values < max_val)])
                                
                                if count > 0:
                                    percentage = (count / len(spt_values)) * 100
                                    summary_data.append({
                                        'Parameter': f'{class_name} Consistency',
                                        'Value': f"{count} ({percentage:.1f}%)"
                                    })
                        
                        elif material_type == "Granular":
                            # Define granular density classes
                            density_ranges = [
                                (0, 4, 'Very Loose'),
                                (4, 10, 'Loose'),
                                (10, 30, 'Medium Dense'),
                                (30, 50, 'Dense'),
                                (50, float('inf'), 'Very Dense')
                            ]
                            
                            for min_val, max_val, class_name in density_ranges:
                                if max_val == float('inf'):
                                    count = len(spt_values[spt_values >= min_val])
                                else:
                                    count = len(spt_values[(spt_values >= min_val) & (spt_values < max_val)])
                                
                                if count > 0:
                                    percentage = (count / len(spt_values)) * 100
                                    summary_data.append({
                                        'Parameter': f'{class_name} Density',
                                        'Value': f"{count} ({percentage:.1f}%)"
                                    })
                
                # Create summary table
                if summary_data:
                    summary_df_display = pd.DataFrame(summary_data)
                    st.dataframe(summary_df_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No summary data available")
            except Exception as e:
                st.warning(f"Could not generate plot summary: {str(e)}")
        
        # Data preview and statistics options (matching PSD/Atterberg layout)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Show data preview", key="spt_data_preview"):
                # Show relevant columns for SPT analysis
                preview_cols = ['Hole_ID', 'From_mbgl', 'To_mbgl', spt_col, 'Material Type']
                if 'Geology_Orgin' in filtered_spt.columns:
                    preview_cols.append('Geology_Orgin')
                
                available_cols = [col for col in preview_cols if col in filtered_spt.columns]
                st.dataframe(filtered_spt[available_cols].head(20), use_container_width=True)
                st.caption(f"{len(filtered_spt)} total records")
        
        with col2:
            if st.checkbox("Show detailed statistics", key="spt_statistics"):
                if not filtered_spt.empty and spt_col in filtered_spt.columns:
                    # Calculate detailed SPT statistics for advanced users
                    spt_values = filtered_spt[spt_col].dropna()
                    stats_data = []
                    
                    # Advanced statistics
                    if not spt_values.empty:
                        # Percentiles
                        percentiles = [10, 25, 50, 75, 90]
                        for p in percentiles:
                            value = np.percentile(spt_values, p)
                            stats_data.append({
                                'Parameter': f'{p}th Percentile',
                                'Value': f"{value:.1f}"
                            })
                        
                        # Additional statistics
                        stats_data.extend([
                            {'Parameter': 'Coefficient of Variation', 'Value': f"{(spt_values.std()/spt_values.mean())*100:.1f}%"},
                            {'Parameter': 'Skewness', 'Value': f"{spt_values.skew():.2f}"},
                            {'Parameter': 'Kurtosis', 'Value': f"{spt_values.kurtosis():.2f}"}
                        ])
                    
                    # Create detailed statistics table
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No SPT data available for detailed statistics")
    
    except Exception as e:
        st.error(f"Error in SPT analysis: {str(e)}")