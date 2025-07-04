"""
Spatial Analysis Module

This module handles spatial data analysis and visualization for geotechnical applications.
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
    from .data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_numerical_properties_smart
    from .plot_defaults import get_default_parameters, get_color_schemes
    from .dashboard_site import store_spatial_plot
    HAS_PLOTTING_UTILS = True
except ImportError:
    # For standalone testing
    from data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_numerical_properties_smart
    from plot_defaults import get_default_parameters, get_color_schemes
    try:
        from dashboard_site import store_spatial_plot
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
    
    from plot_by_chainage import plot_by_chainage
    from plot_category_by_thickness import plot_category_by_thickness
    from plot_engineering_property_vs_depth import plot_engineering_property_vs_depth
    from plot_emerson_by_origin import plot_emerson_by_origin
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


def extract_emerson_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Emerson test data from the main dataframe.
    
    Args:
        df: Main laboratory data DataFrame
        
    Returns:
        pd.DataFrame: Emerson-specific dataframe
    """
    id_columns = get_standard_id_columns(df)
    emerson_columns = extract_test_columns(df, 'Emerson')
    
    if not emerson_columns:
        return pd.DataFrame()
    
    return create_test_dataframe(df, 'Emerson', id_columns, emerson_columns)


def get_numerical_properties(df: pd.DataFrame, include_spatial: bool = False) -> List[str]:
    """
    Get list of numerical properties suitable for spatial analysis using smart detection.
    
    Args:
        df: DataFrame to analyze
        include_spatial: Whether to include spatial columns (Chainage, coordinates)
        
    Returns:
        List[str]: List of numerical column names organized by property type
    """
    return get_numerical_properties_smart(df, include_spatial=include_spatial)


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


def load_bh_interpretation_data() -> Optional[pd.DataFrame]:
    """
    Load BH_Interpretation data from Input folder or session state.
    
    Returns:
        pd.DataFrame or None: BH_Interpretation data if available
    """
    try:
        # First check if BH data is in session state (uploaded via file uploader)
        if HAS_STREAMLIT and hasattr(st, 'session_state') and hasattr(st.session_state, 'bh_data'):
            if st.session_state.bh_data is not None:
                return st.session_state.bh_data
        
        # Try to load from Input folder (same as Jupyter notebook)
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bh_path = os.path.join(current_dir, 'Input', 'BH_Interpretation.xlsx')
        
        if os.path.exists(bh_path):
            bh_data = pd.read_excel(bh_path)
            return bh_data
        
        return None
        
    except Exception as e:
        if HAS_STREAMLIT:
            st.warning(f"Could not load BH_Interpretation data: {str(e)}")
        return None


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


def render_spatial_analysis_tab(filtered_data: pd.DataFrame):
    """
    Render the spatial analysis tab in Streamlit.
    Uses original plotting functions from Functions folder exactly as in Jupyter notebook.
    
    Args:
        filtered_data: Filtered laboratory data
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render tab.")
        return
        
    st.header("Spatial Analysis")
    
    if filtered_data is None or filtered_data.empty:
        st.warning("No data available for spatial analysis.")
        return
    
    # Check for spatial columns
    spatial_columns = [col for col in filtered_data.columns 
                      if any(term in col.upper() for term in ['CHAINAGE', 'EASTING', 'NORTHING', 'X', 'Y'])]
    
    if not spatial_columns:
        st.warning("No spatial coordinate columns detected in current dataset")
        return
    
    
    # Show chainage range if available
    if 'Chainage' in filtered_data.columns:
        chainage_data = filtered_data['Chainage'].dropna()
        if not chainage_data.empty:
            chainage_range = chainage_data.agg(['min', 'max', 'count'])
            st.write(f"**Chainage Range:** {chainage_range['min']:.0f} - {chainage_range['max']:.0f} m ({chainage_range['count']} records)")
    
    # Main spatial analysis tabs
    spatial_tab1, spatial_tab2, spatial_tab3 = st.tabs([
        "Property vs Chainage", "Property vs Depth", "Thickness Analysis"
    ])
    
    with spatial_tab1:
        
        if 'Chainage' in filtered_data.columns and HAS_FUNCTIONS:
            # Get numerical properties
            numerical_props = get_numerical_properties(filtered_data)
            
            if numerical_props:
                # Property selection
                selected_property = st.selectbox(
                    "Select Property:",
                    numerical_props,
                    index=0,
                    help="Choose engineering property to plot along chainage",
                    key="spatial_property"
                )
                
                # Standardized parameter box (5-row × 5-column structure)
                with st.container():
                    st.markdown("### Plot Parameters")
                    
                    # Row 1: Core selections
                    row1_cols = st.columns(5)
                    with row1_cols[0]:
                        category_options = ["Geology_Orgin", "Consistency", "Type", "None"]
                        category_by = st.selectbox(
                            "Group by",
                            category_options,
                            index=0,
                            help="Group data points by category",
                            key="spatial_category"
                        )
                    with row1_cols[1]:
                        color_options = ["From_mbgl", "Geology_Orgin", "Consistency", "None"]
                        color_by = st.selectbox(
                            "Color by",
                            color_options,
                            index=0,
                            help="Color data points by parameter",
                            key="spatial_color"
                        )
                    with row1_cols[2]:
                        figsize_str = st.text_input(
                            "Figure Size",
                            value="(14, 7)",
                            key="chainage_figsize",
                            help="Format: (width, height)"
                        )
                    with row1_cols[3]:
                        colormap = st.selectbox(
                            "Colormap",
                            options=['viridis', 'plasma', 'inferno', 'magma', 'tab10', 'Set1', 'Set2', 'Set3'],
                            index=0,
                            key="chainage_colormap"
                        )
                    with row1_cols[4]:
                        marker_style = st.selectbox(
                            "Marker Style",
                            options=['o', 's', '^', 'v', 'D', 'p', '*', 'h'],
                            index=0,
                            key="chainage_marker_style"
                        )
                    
                    # Row 2: Scatter styling
                    row2_cols = st.columns(5)
                    with row2_cols[0]:
                        marker_size = st.number_input(
                            "Scatter Size",
                            min_value=10,
                            max_value=500,
                            value=50,
                            step=10,
                            key="chainage_marker_size"
                        )
                    with row2_cols[1]:
                        marker_alpha = st.number_input(
                            "Scatter Alpha",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.8,
                            step=0.1,
                            format="%.1f",
                            key="chainage_marker_alpha"
                        )
                    with row2_cols[2]:
                        line_width = st.number_input(
                            "Line Width",
                            min_value=0.1,
                            max_value=10.0,
                            value=1.0,
                            step=0.1,
                            format="%.1f",
                            key="chainage_line_width"
                        )
                    with row2_cols[3]:
                        marker_edge_lw = st.number_input(
                            "Edge Width",
                            min_value=0.0,
                            max_value=5.0,
                            value=0.5,
                            step=0.1,
                            format="%.1f",
                            key="chainage_marker_edge"
                        )
                    with row2_cols[4]:
                        connect_points = st.checkbox("Connect Points", value=False, key="chainage_connect_points")
                    
                    # Row 3: Font styling
                    row3_cols = st.columns(5)
                    with row3_cols[0]:
                        title_fontsize = st.number_input(
                            "Title Font Size",
                            min_value=8,
                            max_value=32,
                            value=14,
                            step=1,
                            key="chainage_title_fontsize"
                        )
                    with row3_cols[1]:
                        label_fontsize = st.number_input(
                            "Label Font Size",
                            min_value=6,
                            max_value=28,
                            value=12,
                            step=1,
                            key="chainage_label_fontsize"
                        )
                    with row3_cols[2]:
                        tick_fontsize = st.number_input(
                            "Tick Font Size",
                            min_value=6,
                            max_value=24,
                            value=10,
                            step=1,
                            key="chainage_tick_fontsize"
                        )
                    with row3_cols[3]:
                        legend_fontsize = st.number_input(
                            "Legend Font Size",
                            min_value=6,
                            max_value=24,
                            value=11,
                            step=1,
                            key="chainage_legend_fontsize"
                        )
                    with row3_cols[4]:
                        zone_label_fontsize = st.number_input(
                            "Zone Font Size",
                            min_value=6,
                            max_value=20,
                            value=10,
                            step=1,
                            key="chainage_zone_fontsize"
                        )
                    
                    # Row 4: Display options
                    row4_cols = st.columns(5)
                    with row4_cols[0]:
                        show_grid = st.checkbox("Show Grid", value=True, key="chainage_show_grid")
                    with row4_cols[1]:
                        show_legend = st.checkbox("Show Legend", value=True, key="chainage_show_legend")
                    with row4_cols[2]:
                        show_colorbar = st.checkbox("Show Colorbar", value=True, key="chainage_show_colorbar")
                    with row4_cols[3]:
                        legend_outside = st.checkbox("Legend Outside", value=True, key="chainage_legend_outside")
                    with row4_cols[4]:
                        show_zone_boundaries = st.checkbox("Zone Boundaries", value=True, key="chainage_zone_boundaries")
                    
                    # Row 5: Filter controls
                    row5_cols = st.columns(5)
                    with row5_cols[0]:
                        filter1_by = st.selectbox(
                            "Filter 1 By",
                            ["None", "Geology Origin", "Consistency", "Hole ID", "Depth Range"],
                            index=0,
                            key="chainage_filter1_by"
                        )
                    with row5_cols[1]:
                        # Always show text input, but change placeholder and help based on filter type
                        filter1_enabled = filter1_by != "None"
                        
                        # Determine placeholder and help text based on filter type
                        if filter1_by == "Geology Origin":
                            placeholder = "e.g., Alluvium, Clay, Sand"
                            help_text = "Enter geology types separated by commas" if filter1_enabled else "Select 'Geology Origin' filter first"
                        elif filter1_by == "Consistency":
                            placeholder = "e.g., Soft, Medium, Stiff"
                            help_text = "Enter consistency types separated by commas" if filter1_enabled else "Select 'Consistency' filter first"
                        elif filter1_by == "Hole ID":
                            placeholder = "e.g., BH01, BH02, BH03"
                            help_text = "Enter hole IDs separated by commas" if filter1_enabled else "Select 'Hole ID' filter first"
                        elif filter1_by == "Depth Range":
                            placeholder = "e.g., 0,10"
                            help_text = "Enter depth range as min,max" if filter1_enabled else "Select 'Depth Range' filter first"
                        else:
                            placeholder = "Select filter type first"
                            help_text = "Choose a filter type from dropdown above"
                        
                        filter1_values = st.text_input(
                            "Filter 1 Values",
                            value="",
                            key="chainage_filter1_values",
                            disabled=not filter1_enabled,
                            placeholder=placeholder,
                            help=help_text
                        )
                    with row5_cols[2]:
                        filter2_by = st.selectbox(
                            "Filter 2 By",
                            ["None", "Geology Origin", "Consistency", "Hole ID", "Property Range"],
                            index=0,
                            key="chainage_filter2_by"
                        )
                    with row5_cols[3]:
                        # Always show text input, but change placeholder and help based on filter type
                        filter2_enabled = filter2_by != "None"
                        
                        # Determine placeholder and help text based on filter type
                        if filter2_by == "Geology Origin":
                            placeholder = "e.g., Alluvium, Clay, Sand"
                            help_text = "Enter geology types separated by commas" if filter2_enabled else "Select 'Geology Origin' filter first"
                        elif filter2_by == "Consistency":
                            placeholder = "e.g., Soft, Medium, Stiff"
                            help_text = "Enter consistency types separated by commas" if filter2_enabled else "Select 'Consistency' filter first"
                        elif filter2_by == "Hole ID":
                            placeholder = "e.g., BH01, BH02, BH03"
                            help_text = "Enter hole IDs separated by commas" if filter2_enabled else "Select 'Hole ID' filter first"
                        elif filter2_by == "Property Range":
                            placeholder = "e.g., 10,50"
                            help_text = "Enter property range as min,max" if filter2_enabled else "Select 'Property Range' filter first"
                        else:
                            placeholder = "Select filter type first"
                            help_text = "Choose a filter type from dropdown above"
                        
                        filter2_values = st.text_input(
                            "Filter 2 Values",
                            value="",
                            key="chainage_filter2_values",
                            disabled=not filter2_enabled,
                            placeholder=placeholder,
                            help=help_text
                        )
                    with row5_cols[4]:
                        pass
                    
                    # Advanced options section (collapsed)
                    with st.expander("Advanced Zone & Line Options"):
                        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                        with adv_col1:
                            show_zone_labels = st.checkbox("Zone Labels", value=True, key="chainage_zone_labels")
                        with adv_col2:
                            use_custom_zones = st.checkbox("Custom Zones", value=False, key="chainage_custom_zones")
                        with adv_col3:
                            # Always show zone orientation, but disable if not using custom zones
                            zone_orientation = st.selectbox(
                                "Zone Orientation",
                                options=['vertical', 'horizontal'],
                                index=0,
                                key="chainage_zone_orientation",
                                disabled=not use_custom_zones,
                                help="Enable 'Custom Zones' to modify orientation"
                            )
                        with adv_col4:
                            line_style = st.selectbox(
                                "Line Style",
                                options=['-', '--', '-.', ':'],
                                index=0,
                                key="chainage_line_style"
                            )
                        
                        # Always show zones text area, but disable if custom zones not enabled
                        zones_text = st.text_area(
                            "Zone Dictionary (JSON format)",
                            value='{\n"Zone 1": (21300, 26300),\n"Zone 2": (26300, 32770),\n"Zone 3": (32770, 37100),\n"Zone 4": (37100, 41120)\n}',
                            key="chainage_zones_dict",
                            help="Enable 'Custom Zones' to modify zone dictionary" if not use_custom_zones else "Format: {\"name\": (start, end)}",
                            height=100,
                            disabled=not use_custom_zones
                        )
                
                # Filter data for plotting
                plot_data = filter_valid_chainage_data(filtered_data, selected_property)
                
                # Apply Filter 1
                if filter1_by != "None" and filter1_values and filter1_values.strip():
                    if filter1_by == "Geology Origin":
                        # Parse comma-separated values
                        filter_list = [x.strip() for x in filter1_values.split(',') if x.strip()]
                        if filter_list:
                            plot_data = plot_data[plot_data['Geology_Orgin'].isin(filter_list)]
                    elif filter1_by == "Consistency":
                        # Parse comma-separated values
                        filter_list = [x.strip() for x in filter1_values.split(',') if x.strip()]
                        if filter_list:
                            plot_data = plot_data[plot_data['Consistency'].isin(filter_list)]
                    elif filter1_by == "Hole ID":
                        # Parse comma-separated values
                        filter_list = [x.strip() for x in filter1_values.split(',') if x.strip()]
                        if filter_list:
                            plot_data = plot_data[plot_data['Hole_ID'].isin(filter_list)]
                    elif filter1_by == "Depth Range":
                        try:
                            depth_range = [float(x.strip()) for x in filter1_values.split(',')]
                            if len(depth_range) == 2:
                                min_depth, max_depth = depth_range
                                plot_data = plot_data[(plot_data['From_mbgl'] >= min_depth) & (plot_data['From_mbgl'] <= max_depth)]
                        except:
                            pass  # Invalid depth range format
                
                # Apply Filter 2
                if filter2_by != "None" and filter2_values and filter2_values.strip():
                    if filter2_by == "Geology Origin":
                        # Parse comma-separated values
                        filter_list = [x.strip() for x in filter2_values.split(',') if x.strip()]
                        if filter_list:
                            plot_data = plot_data[plot_data['Geology_Orgin'].isin(filter_list)]
                    elif filter2_by == "Consistency":
                        # Parse comma-separated values
                        filter_list = [x.strip() for x in filter2_values.split(',') if x.strip()]
                        if filter_list:
                            plot_data = plot_data[plot_data['Consistency'].isin(filter_list)]
                    elif filter2_by == "Hole ID":
                        # Parse comma-separated values
                        filter_list = [x.strip() for x in filter2_values.split(',') if x.strip()]
                        if filter_list:
                            plot_data = plot_data[plot_data['Hole_ID'].isin(filter_list)]
                    elif filter2_by == "Property Range":
                        try:
                            prop_range = [float(x.strip()) for x in filter2_values.split(',')]
                            if len(prop_range) == 2:
                                min_prop, max_prop = prop_range
                                plot_data = plot_data[(plot_data[selected_property] >= min_prop) & (plot_data[selected_property] <= max_prop)]
                        except:
                            pass  # Invalid property range format
                
                if not plot_data.empty:
                    # Generate dynamic title suffix based on applied filters
                    title_suffix_parts = []
                    if filter1_by != "None" and filter1_values:
                        if filter1_by == "Depth Range":
                            title_suffix_parts.append(f"Depth: {filter1_values}m")
                        elif isinstance(filter1_values, list) and len(filter1_values) <= 3:
                            title_suffix_parts.append(f"{filter1_by}: {', '.join(map(str, filter1_values))}")
                        elif isinstance(filter1_values, list):
                            title_suffix_parts.append(f"{filter1_by}: {len(filter1_values)} items")
                        else:
                            title_suffix_parts.append(f"{filter1_by}: {filter1_values}")
                    
                    if filter2_by != "None" and filter2_values:
                        if filter2_by == "Property Range":
                            title_suffix_parts.append(f"{selected_property}: {filter2_values}")
                        elif isinstance(filter2_values, list) and len(filter2_values) <= 3:
                            title_suffix_parts.append(f"{filter2_by}: {', '.join(map(str, filter2_values))}")
                        elif isinstance(filter2_values, list):
                            title_suffix_parts.append(f"{filter2_by}: {len(filter2_values)} items")
                        else:
                            title_suffix_parts.append(f"{filter2_by}: {filter2_values}")
                    
                    dynamic_title_suffix = " | ".join(title_suffix_parts)
                    plot_title = f"{selected_property} along chainage"
                    if dynamic_title_suffix:
                        plot_title += f" | {dynamic_title_suffix}"
                    
                    st.info(f"Plotting {len(plot_data)} records with valid {selected_property} and Chainage data")
                    
                    # Axis limits
                    col1, col2 = st.columns(2)
                    with col1:
                        xlim_min = st.number_input("Chainage min (m)", value=float(plot_data['Chainage'].min()), key="chainage_min")
                        xlim_max = st.number_input("Chainage max (m)", value=float(plot_data['Chainage'].max()), key="chainage_max")
                    
                    with col2:
                        ylim_min = st.number_input("Y-axis min", value=float(plot_data[selected_property].min() * 0.9), key="chainage_ylim_min")
                        ylim_max = st.number_input("Y-axis max", value=float(plot_data[selected_property].max() * 1.1), key="chainage_ylim_max")
                    
                    # Parse figsize and zones
                    try:
                        figsize = eval(figsize_str) if figsize_str else (14, 7)
                    except:
                        figsize = (14, 7)
                    
                    # Handle zone setup
                    if use_custom_zones:
                        try:
                            zonage = eval(zones_text) if zones_text else None
                        except:
                            st.warning("Invalid zone dictionary format. Using default zones.")
                            zonage = {
                                "Zone 1": (21300, 26300),
                                "Zone 2": (26300, 32770), 
                                "Zone 3": (32770, 37100),
                                "Zone 4": (37100, 41120)
                            }
                    else:
                        # Default zones from Jupyter notebook examples
                        zonage = {
                            "Zone 1": (21300, 26300),
                            "Zone 2": (26300, 32770), 
                            "Zone 3": (32770, 37100),
                            "Zone 4": (37100, 41120)
                        }
                    
                    # Create plot exactly like Jupyter notebook
                    try:
                        # Clear any existing figures first
                        plt.close('all')
                        
                        plot_by_chainage(
                            df=plot_data,
                            chainage_col='Chainage',
                            property_col=selected_property,
                            category_by_col=category_by if category_by != "None" else None,
                            color_by_col=color_by if color_by != "None" else None,
                            figsize=figsize,
                            xlim=(xlim_min, xlim_max),
                            ylim=(ylim_min, ylim_max),
                            title=plot_title,
                            classification_zones=zonage if show_zone_boundaries else None,
                            zone_orientation=zone_orientation,
                            show_zone_boundaries=show_zone_boundaries,
                            show_zone_labels=show_zone_labels,
                            legend_outside=legend_outside,
                            legend_position='right' if legend_outside else 'best',
                            colormap=colormap,
                            marker_size=marker_size,
                            marker_alpha=marker_alpha,
                            marker_style=marker_style,
                            marker_edge_lw=marker_edge_lw,
                            line_width=line_width,
                            line_style=line_style,
                            connect_points=connect_points,
                            show_grid=show_grid,
                            show_legend=show_legend,
                            show_colorbar=show_colorbar,
                            title_fontsize=title_fontsize,
                            label_fontsize=label_fontsize,
                            tick_fontsize=tick_fontsize,
                            legend_fontsize=legend_fontsize,
                            zone_label_fontsize=zone_label_fontsize,
                            show_plot=False,
                            close_plot=False
                        )
                        
                        # Capture and display the figure in Streamlit with sidebar size control
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            # Use plotting utils for sidebar size control
                            try:
                                from .plotting_utils import display_plot_with_size_control
                                display_plot_with_size_control(current_fig, key_suffix="chainage")
                            except ImportError:
                                # Fallback to standard display
                                st.pyplot(current_fig, use_container_width=True)
                            success = True
                        else:
                            success = False
                        
                        if success:
                            # Store the plot for Site Dashboard
                            try:
                                # Get the current figure for storage
                                current_fig = plt.gcf()
                                if current_fig and current_fig.get_axes():
                                    # Save figure to buffer for dashboard
                                    import io
                                    buf = io.BytesIO()
                                    current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                    buf.seek(0)
                                    store_spatial_plot('property_vs_chainage', buf)
                            except Exception as e:
                                pass  # Don't break the main functionality if storage fails
                            st.success("✅ Chainage plot created successfully!")
                            
                            # Simple download button with figure reference
                            from .plot_download_simple import create_simple_download_button
                            create_simple_download_button("property_vs_chainage", "chainage", fig=current_fig)
                        else:
                            st.warning("Plot function completed but no plot was displayed")
                            
                    except Exception as e:
                        st.error(f"Error creating chainage plot: {str(e)}")
                        st.exception(e)
                else:
                    st.warning(f"No valid data points found for {selected_property} vs Chainage")
            else:
                st.warning("No numerical properties found for spatial analysis")
        else:
            st.error("Chainage column or plotting utilities not available")
    
    with spatial_tab2:
        """
        Property vs Depth Analysis
        
        This tab creates professional scatter plots showing engineering properties versus depth
        with comprehensive customization options and classification zones support.
        """
        
        if HAS_FUNCTIONS:
            # Smart column selection functions
            def get_depth_columns(data):
                """Get all potential depth columns with smart pattern matching"""
                depth_patterns = [
                    'from_mbgl', 'from mbgl', 'depth', 'depth_m', 'from_m', 
                    'depth (m)', 'from (mbgl)', 'from', 'z', 'elevation',
                    'rl', 'level', 'depth_top', 'top_depth', 'sample_depth'
                ]
                potential_cols = []
                for col in data.columns:
                    col_lower = col.lower().replace('_', ' ').replace('-', ' ')
                    for pattern in depth_patterns:
                        if pattern in col_lower:
                            potential_cols.append(col)
                            break
                
                # Ensure From_mbgl is first if it exists
                if 'From_mbgl' in potential_cols:
                    potential_cols.remove('From_mbgl')
                    potential_cols.insert(0, 'From_mbgl')
                
                return potential_cols if potential_cols else ['From_mbgl']
            
            # Get numerical properties
            numerical_props = get_numerical_properties(filtered_data)
            depth_columns = get_depth_columns(filtered_data)
            
            # Get categorical columns for grouping
            categorical_cols = []
            for col in filtered_data.columns:
                if (filtered_data[col].dtype == 'object' or 
                    filtered_data[col].nunique() <= 20):
                    categorical_cols.append(col)
            
            if numerical_props:
                # Helper function for parsing tuple inputs
                def parse_tuple(tuple_str, default):
                    """Parse tuple string with error handling"""
                    try:
                        if tuple_str.strip().startswith('(') and tuple_str.strip().endswith(')'):
                            tuple_content = tuple_str.strip()[1:-1]
                            values = [float(x.strip()) for x in tuple_content.split(',')]
                            return tuple(values)
                        else:
                            values = [float(x.strip()) for x in tuple_str.split(',')]
                            return tuple(values)
                    except:
                        return default
                
                # Plot Parameters expander (matching other tabs)
                with st.expander("Plot Parameters", expanded=True):
                    # Row 1: Data Selection
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        selected_property = st.selectbox(
                            "Property Column",
                            numerical_props,
                            index=0,
                            help="Select engineering property to plot vs depth",
                            key="depth_property"
                        )
                    
                    with col2:
                        depth_col = st.selectbox(
                            "Depth Column",
                            depth_columns,
                            index=0,
                            help="Select depth column (typically From_mbgl)",
                            key="depth_column"
                        )
                    
                    with col3:
                        category_options = ["None"] + categorical_cols
                        category_by = st.selectbox(
                            "Category By",
                            category_options,
                            index=category_options.index("Geology_Orgin") if "Geology_Orgin" in category_options else 0,
                            help="Group data points by category",
                            key="depth_category"
                        )
                    
                    with col4:
                        title = st.text_input(
                            "Title",
                            value="",
                            help="Custom plot title (leave empty for auto)",
                            key="depth_title"
                        )
                    
                    with col5:
                        title_suffix = st.text_input(
                            "Title Suffix",
                            value="",
                            help="Additional text to append to title",
                            key="depth_title_suffix"
                        )
                    
                    # Row 2: Axis Configuration
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        xlim_min = st.number_input(
                            "X-axis Min",
                            value=None,
                            help="Minimum value for property axis (auto if empty)",
                            key="depth_xlim_min"
                        )
                    
                    with col2:
                        xlim_max = st.number_input(
                            "X-axis Max",
                            value=None,
                            help="Maximum value for property axis (auto if empty)",
                            key="depth_xlim_max"
                        )
                    
                    with col3:
                        ylim_min = st.number_input(
                            "Y-axis Min (Depth)",
                            value=0.0,
                            help="Minimum depth value",
                            key="depth_ylim_min"
                        )
                    
                    with col4:
                        ylim_max_default = float(filtered_data[depth_col].max() + 5) if depth_col in filtered_data.columns else 50.0
                        ylim_max = st.number_input(
                            "Y-axis Max (Depth)",
                            value=ylim_max_default,
                            help="Maximum depth value",
                            key="depth_ylim_max"
                        )
                    
                    with col5:
                        ytick_interval = st.number_input(
                            "Y-tick Interval",
                            value=2.5,
                            step=0.5,
                            help="Spacing between depth axis ticks",
                            key="depth_ytick"
                        )
                    
                    # Row 3: Plot Configuration
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        figsize_str = st.text_input(
                            "Figure Size",
                            value="(9, 6)",
                            help="Figure size as (width, height)",
                            key="depth_figsize"
                        )
                    
                    with col2:
                        xtick_interval = st.number_input(
                            "X-tick Interval",
                            value=None,
                            help="Spacing between property axis ticks (auto if empty)",
                            key="depth_xtick"
                        )
                    
                    with col3:
                        use_log_scale = st.selectbox(
                            "Log Scale",
                            ["No", "Yes"],
                            index=0,
                            help="Use logarithmic scale for property axis",
                            key="depth_log_scale"
                        )
                    
                    with col4:
                        invert_yaxis = st.selectbox(
                            "Invert Y-axis",
                            ["Yes", "No"],
                            index=0,
                            help="Depth increases downward (standard)",
                            key="depth_invert_y"
                        )
                    
                    with col5:
                        show_legend = st.selectbox(
                            "Show Legend",
                            ["Yes", "No"],
                            index=0,
                            help="Display legend for categories",
                            key="depth_show_legend"
                        )
                    
                    # Row 4: Visual Styling
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        marker_size = st.number_input(
                            "Marker Size",
                            min_value=10,
                            max_value=200,
                            value=40,
                            help="Size of data points",
                            key="depth_marker_size"
                        )
                    
                    with col2:
                        marker_alpha = st.number_input(
                            "Marker Alpha",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.7,
                            step=0.1,
                            help="Transparency of markers",
                            key="depth_marker_alpha"
                        )
                    
                    with col3:
                        marker_edge_lw = st.number_input(
                            "Marker Edge Width",
                            min_value=0.0,
                            max_value=3.0,
                            value=0.5,
                            step=0.1,
                            help="Width of marker borders",
                            key="depth_marker_edge"
                        )
                    
                    with col4:
                        xlabel = st.text_input(
                            "X-axis Label",
                            value="",
                            help="Custom x-axis label (auto if empty)",
                            key="depth_xlabel"
                        )
                    
                    with col5:
                        ylabel = st.text_input(
                            "Y-axis Label", 
                            value="",
                            help="Custom y-axis label (auto if empty)",
                            key="depth_ylabel"
                        )
                
                # Advanced Parameters expander
                with st.expander("Advanced Parameters", expanded=False):
                    # Row 1: Font Sizes
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        title_fontsize = st.number_input(
                            "Title Font Size",
                            min_value=8,
                            max_value=24,
                            value=14,
                            help="Font size for plot title",
                            key="depth_title_fontsize"
                        )
                    
                    with col2:
                        label_fontsize = st.number_input(
                            "Label Font Size",
                            min_value=8,
                            max_value=20,
                            value=12,
                            help="Font size for axis labels",
                            key="depth_label_fontsize"
                        )
                    
                    with col3:
                        tick_fontsize = st.number_input(
                            "Tick Font Size",
                            min_value=6,
                            max_value=16,
                            value=10,
                            help="Font size for axis ticks",
                            key="depth_tick_fontsize"
                        )
                    
                    with col4:
                        legend_fontsize = st.number_input(
                            "Legend Font Size",
                            min_value=6,
                            max_value=16,
                            value=11,
                            help="Font size for legend",
                            key="depth_legend_fontsize"
                        )
                    
                    with col5:
                        zone_label_fontsize = st.number_input(
                            "Zone Label Font Size",
                            min_value=6,
                            max_value=16,
                            value=10,
                            help="Font size for zone labels",
                            key="depth_zone_label_fontsize"
                        )
                    
                    # Row 2: Zone Configuration
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        show_zone_boundaries = st.selectbox(
                            "Show Zone Boundaries",
                            ["No", "Yes"],
                            index=0,
                            help="Display classification zone boundaries",
                            key="depth_show_zones"
                        )
                    
                    with col2:
                        show_zone_labels = st.selectbox(
                            "Show Zone Labels",
                            ["No", "Yes"],
                            index=0,
                            help="Display zone classification labels",
                            key="depth_show_zone_labels"
                        )
                    
                    with col3:
                        zone_label_position = st.number_input(
                            "Zone Label Position",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.15,
                            step=0.05,
                            help="Horizontal position of zone labels (0=left, 1=right)",
                            key="depth_zone_label_pos"
                        )
                    
                    with col4:
                        plot_style = st.selectbox(
                            "Plot Style",
                            ["seaborn-v0_8-whitegrid", "default", "classic", "seaborn-v0_8-colorblind"],
                            index=0,
                            help="Overall plot styling theme",
                            key="depth_plot_style"
                        )
                    
                    with col5:
                        legend_loc = st.selectbox(
                            "Legend Location",
                            ["best", "upper right", "upper left", "lower left", "lower right", "center"],
                            index=0,
                            help="Legend position on plot",
                            key="depth_legend_loc"
                        )
                    
                    # Row 3: Advanced Options
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        dpi = st.number_input(
                            "DPI",
                            min_value=150,
                            max_value=600,
                            value=300,
                            step=50,
                            help="Resolution for saved plots",
                            key="depth_dpi"
                        )
                    
                    with col2:
                        property_name = st.text_input(
                            "Property Name",
                            value="",
                            help="Override property name in title/labels",
                            key="depth_property_name"
                        )
                    
                    with col3:
                        # Empty for now
                        pass
                    
                    with col4:
                        # Empty for now
                        pass
                    
                    with col5:
                        # Empty for now
                        pass
                
                # Parse inputs
                figsize = parse_tuple(figsize_str, (9, 6))
                xlim = (xlim_min, xlim_max) if xlim_min is not None or xlim_max is not None else None
                ylim = (ylim_min, ylim_max)
                
                # Filter data for plotting
                depth_data = filtered_data.dropna(subset=[depth_col, selected_property])
                
                if not depth_data.empty:
                    st.info(f"Plotting {len(depth_data)} records with valid {selected_property} and depth data")
                    
                    # Create plot with all parameters
                    try:
                        # Clear any existing figures first
                        plt.close('all')
                        
                        # Generate title
                        if not title:
                            title = f"{selected_property} vs Depth"
                        
                        plot_engineering_property_vs_depth(
                            df=depth_data,
                            property_col=selected_property,
                            depth_col=depth_col,
                            title=title,
                            title_suffix=title_suffix if title_suffix else None,
                            property_name=property_name if property_name else selected_property,
                            figsize=figsize,
                            category_col=category_by if category_by != "None" else None,
                            xlim=xlim,
                            ylim=ylim,
                            xtick_interval=xtick_interval if xtick_interval else None,
                            ytick_interval=ytick_interval,
                            xlabel=xlabel if xlabel else None,
                            ylabel=ylabel if ylabel else None,
                            invert_yaxis=(invert_yaxis == "Yes"),
                            use_log_scale=(use_log_scale == "Yes"),
                            show_zone_boundaries=(show_zone_boundaries == "Yes"),
                            show_zone_labels=(show_zone_labels == "Yes"),
                            zone_label_position=zone_label_position,
                            show_plot=False,
                            show_legend=(show_legend == "Yes"),
                            dpi=dpi,
                            marker_size=marker_size,
                            marker_alpha=marker_alpha,
                            marker_edge_lw=marker_edge_lw,
                            title_fontsize=title_fontsize,
                            label_fontsize=label_fontsize,
                            tick_fontsize=tick_fontsize,
                            legend_fontsize=legend_fontsize,
                            zone_label_fontsize=zone_label_fontsize,
                            plot_style=plot_style if plot_style != "default" else None,
                            legend_loc=legend_loc,
                            close_plot=False
                        )
                        
                        # Capture and display the figure in Streamlit with size control
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            # Use display function that respects sidebar width control
                            from .plotting_utils import display_plot_with_size_control
                            display_plot_with_size_control(current_fig)
                            success = True
                        else:
                            success = False
                        
                        if success:
                            # Store the plot for Site Dashboard
                            try:
                                # Get the current figure for storage
                                current_fig = plt.gcf()
                                if current_fig and current_fig.get_axes():
                                    # Save figure to buffer for dashboard
                                    import io
                                    buf = io.BytesIO()
                                    current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                    buf.seek(0)
                                    store_spatial_plot('property_vs_depth', buf)
                            except Exception as e:
                                pass  # Don't break the main functionality if storage fails
                            st.success("✅ Depth plot created successfully!")
                            
                            # Simple download button with figure reference
                            from .plot_download_simple import create_simple_download_button
                            create_simple_download_button("property_vs_depth", "depth", fig=current_fig)
                        else:
                            st.warning("Plot function completed but no plot was displayed")
                            
                    except Exception as e:
                        st.error(f"Error creating depth plot: {str(e)}")
                        st.exception(e)
                else:
                    st.warning(f"No valid data points found for {selected_property} vs {depth_col}")
            else:
                st.warning("No numerical properties found for depth analysis")
        else:
            st.error("Plotting utilities not available")
    
    with spatial_tab3:
        """
        Rock Class Thickness Distribution Analysis
        
        This tab analyzes the thickness distribution of different rock classes within geological formations,
        exactly as implemented in the Jupyter notebook workflow.
        """
        
        # Try to load BH_Interpretation data
        bh_data = load_bh_interpretation_data()
        
        if bh_data is not None and not bh_data.empty:
            st.success(f"✅ BH_Interpretation data loaded: {len(bh_data)} records")
            
            # Show available geological formations
            available_formations = bh_data['Geology_Orgin'].value_counts()
            st.write("**Available Geological Formations:**")
            for formation, count in available_formations.items():
                st.write(f"- {formation}: {count} records")
            
            # Formation selection
            formation_options = list(available_formations.index)
            selected_formation = st.selectbox(
                "Select Geological Formation:",
                formation_options,
                index=0,
                help="Choose formation for thickness distribution analysis"
            )
            
            # Formation name mapping (from Jupyter notebook)
            formation_names = {
                'Tos': 'Sunnybank Formation',
                'Rjbw': 'Woogaroo Subgroup', 
                'Rin': 'Tingalpa Formation',
                'Dcf': 'Neranleigh Fernvale Beds',
                'RS_XW': 'Brisbane Tuff',
                'ALLUVIUM': 'Alluvial Deposits',
                'FILL': 'Fill Material',
                'TOPSOIL': 'Topsoil',
                'Toc': 'Older Alluvium',
                'ASPHALT': 'Asphalt Pavement'
            }
            
            formation_full_name = formation_names.get(selected_formation, selected_formation)
            
            if HAS_FUNCTIONS:
                # Process thickness data exactly like Jupyter notebook
                try:
                    thickness_data = process_thickness_data(bh_data, selected_formation)
                    
                    if not thickness_data.empty:
                        # Main parameter box (essential parameters) - following standard pattern
                        with st.expander("Plot Parameters", expanded=True):
                            # Helper function for parsing tuple inputs (following standardized pattern)
                            def parse_tuple(input_str, default):
                                try:
                                    cleaned = input_str.strip().replace('(', '').replace(')', '')
                                    values = [float(x.strip()) for x in cleaned.split(',')]
                                    return tuple(values) if len(values) == 2 else default
                                except:
                                    return default
                            
                            # Row 1: Data Display Options
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                # Get available value columns from thickness data
                                numerical_cols = thickness_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                                value_col_options = [col for col in numerical_cols if col in ['Thickness', 'thickness_proportion_%']]
                                if not value_col_options:
                                    value_col_options = numerical_cols
                                
                                value_column = st.selectbox(
                                    "Value Column",
                                    value_col_options,
                                    index=0 if 'thickness_proportion_%' in value_col_options else 0,
                                    help="Select which value to plot",
                                    key="thickness_value_column"
                                )
                            with col2:
                                x_axis_sort = st.selectbox(
                                    "X-axis Sort",
                                    ["alphabetical", "ascending", "descending"],
                                    index=0,
                                    key="thickness_x_sort"
                                )
                            with col3:
                                show_legend = st.selectbox(
                                    "Show Legend",
                                    [False, True],
                                    index=0,
                                    key="thickness_show_legend"
                                )
                            with col4:
                                show_grid = st.selectbox(
                                    "Show Grid",
                                    [True, False],
                                    index=0,
                                    key="thickness_show_grid"
                                )
                            with col5:
                                show_percentage_labels = st.selectbox(
                                    "Show Value Labels",
                                    [True, False],
                                    index=0,
                                    key="thickness_show_labels"
                                )
                            
                            # Row 2: Plot Configuration
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                figsize_str = st.text_input("figsize (w, h)", value="(9, 6)", key="thickness_figsize")
                            with col2:
                                xlim_str = st.text_input("xlim (min, max)", value="(auto, auto)", key="thickness_xlim")
                            with col3:
                                ylim_str = st.text_input("ylim (min, max)", value="(auto, auto)", key="thickness_ylim")
                            with col4:
                                title = st.text_input("title", value="", key="thickness_title")
                            with col5:
                                title_suffix = st.text_input("title_suffix", value="", key="thickness_title_suffix")
                            
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
                                st.write("")  # Empty placeholder for alignment
                            with adv_col3:
                                st.write("")  # Empty placeholder for alignment  
                            with adv_col4:
                                st.write("")  # Empty placeholder for alignment
                            with adv_col5:
                                st.write("")  # Empty placeholder for alignment
                        
                        # Parse all parameters (main + advanced)
                        figsize = parse_tuple(figsize_str, (9, 6))
                        
                        # Handle xlim and ylim - support "auto" values
                        if xlim_str.strip().lower() in ["(auto, auto)", "auto", ""]:
                            xlim = None
                        else:
                            xlim = parse_tuple(xlim_str, None)
                        
                        if ylim_str.strip().lower() in ["(auto, auto)", "auto", ""]:
                            ylim = None
                        else:
                            ylim = parse_tuple(ylim_str, None)
                        
                        # Parse advanced parameters
                        # Handle custom category and legend ordering
                        category_order = None
                        if category_order_str.strip():
                            try:
                                category_order = [x.strip() for x in category_order_str.split(',') if x.strip()]
                            except:
                                category_order = None
                        
                        legend_order = None
                        if legend_order_str.strip():
                            try:
                                legend_order = [x.strip() for x in legend_order_str.split(',') if x.strip()]
                            except:
                                legend_order = None
                        
                        # Handle custom colors
                        colors = None
                        if colors_str.strip():
                            try:
                                colors = [x.strip() for x in colors_str.split(',') if x.strip()]
                            except:
                                colors = None
                        
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
                        
                        # Handle custom labels
                        final_xlabel = xlabel_custom if xlabel_custom.strip() else 'Rock Class Unit'
                        final_ylabel = ylabel_custom if ylabel_custom.strip() else ylabel
                        
                        # Generate final title
                        if title:
                            final_title = title
                        else:
                            final_title = f"Distribution of Rock Class by Thickness of {formation_full_name} ({selected_formation})"
                        
                        if title_suffix:
                            final_title += f" - {title_suffix}"
                        
                        # Determine ylabel based on selected value column
                        if 'proportion' in value_column.lower() or '%' in value_column:
                            ylabel = 'Distribution (%)'
                        elif 'thickness' in value_column.lower():
                            ylabel = 'Thickness (m)'
                        else:
                            ylabel = value_column.replace('_', ' ').title()
                        
                        # Create thickness distribution plot exactly like Jupyter notebook
                        try:
                            # Import matplotlib at function level to avoid scope issues
                            import matplotlib.pyplot as plt
                            
                            # Clear any existing figures first
                            plt.close('all')
                            
                            # Create the plot using Functions folder with comprehensive parameters
                            plot_category_by_thickness(
                                df=thickness_data,
                                value_col=value_column,
                                category_col='Consistency',
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
                                show_plot=False,
                                close_plot=False
                            )
                            
                            # Capture and display the figure with size control
                            current_fig = plt.gcf()
                            if current_fig and current_fig.get_axes():
                                # Use plotting utils for sidebar size control
                                try:
                                    from .plotting_utils import display_plot_with_size_control
                                    display_plot_with_size_control(current_fig)
                                except ImportError:
                                    # Fallback to standard display
                                    st.pyplot(current_fig, use_container_width=True)
                                success = True
                            else:
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
                                
                                # Enhanced plot summary with engineering interpretations (following standardized pattern)
                                st.divider()
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Thickness Distribution Summary**")
                                    
                                    summary_data = []
                                    
                                    # Basic data statistics
                                    if not thickness_data.empty:
                                        total_thickness = thickness_data['Thickness'].sum()
                                        thickness_values = thickness_data['Thickness']
                                        
                                        summary_data.extend([
                                            {'Parameter': 'Formation', 'Value': f"{formation_full_name} ({selected_formation})"},
                                            {'Parameter': 'Rock Classes', 'Value': f"{len(thickness_data)}"},
                                            {'Parameter': 'Total Thickness', 'Value': f"{total_thickness:.2f} m"},
                                            {'Parameter': 'Average Thickness', 'Value': f"{thickness_values.mean():.2f} m"},
                                            {'Parameter': 'Thickness Range', 'Value': f"{thickness_values.min():.2f} - {thickness_values.max():.2f} m"}
                                        ])
                                        
                                        # Rock class distribution
                                        for _, row in thickness_data.iterrows():
                                            summary_data.append({
                                                'Parameter': f'{row["Consistency"]} Thickness',
                                                'Value': f"{row['Thickness']:.2f} m ({row['thickness_proportion_%']:.1f}%)"
                                            })
                                    
                                    # Create summary table
                                    if summary_data:
                                        summary_df_display = pd.DataFrame(summary_data)
                                        st.dataframe(summary_df_display, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No summary data available")
                                
                                with col2:
                                    st.markdown("**Rock Class Interpretation Guidelines**")
                                    st.write("**Rock Strength Classes:**")
                                    st.write("• **1a, 1b**: Extremely strong rock (>250 MPa)")
                                    st.write("• **2a, 2b**: Very strong rock (100-250 MPa)")
                                    st.write("• **3a, 3b**: Strong rock (50-100 MPa)")
                                    st.write("• **4a, 4b**: Moderately strong rock (25-50 MPa)")
                                    st.write("• **5a, 5b**: Weak rock (5-25 MPa)")
                                    st.write("")
                                    st.write("**Engineering Significance:**")
                                    st.write("• **Thickness Distribution**: Critical for foundation design")
                                    st.write("• **Rock Class Variation**: Affects excavatability and stability")
                                    st.write("• **Proportion Analysis**: Guides construction methodology")
                                    st.write("• **Formation Consistency**: Important for risk assessment")
                            else:
                                st.warning("Plot function completed but no plot was displayed")
                                
                        except Exception as e:
                            st.error(f"Error creating thickness plot: {str(e)}")
                    else:
                        st.warning(f"No thickness data available for {selected_formation}")
                        
                except Exception as e:
                    st.error(f"Error processing thickness data: {str(e)}")
            else:
                st.error("❌ Functions folder not accessible")
                st.info("Check Functions folder and spatial plotting modules")
        else:
            st.info("🚧 Thickness analysis requires BH_Interpretation data")
            st.markdown("""
            **Expected Features (from Jupyter notebook):**
            - Distribution of Rock Class by Thickness (Sunnybank Formation - Tos)
            - Distribution of Rock Class by Thickness (Woogaroo Subgroup - Rjbw) 
            - Distribution of Rock Class by Thickness (Tingalpa Formation - Rin)
            - Distribution of Rock Class by Thickness (Neranleigh Fernvale Beds - Dcf)
            
            **Data Requirements:**
            - BH_Interpretation.xlsx file with thickness data
            - Upload this file using the sidebar file uploader
            """)
    
    # Export section
    
    # Individual plots now have direct download buttons
    
    # Data export
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Spatial Data", key="spatial_export_data"):
            # Filter to spatial data only
            spatial_data = filtered_data.dropna(subset=['Chainage']) if 'Chainage' in filtered_data.columns else filtered_data
            csv = spatial_data.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Spatial CSV",
                data=csv,
                file_name=f"spatial_analysis_data_{timestamp}.csv",
                mime="text/csv",
                key="spatial_download_csv_button"
            )
    
    with col2:
        # Export thickness data if available
        bh_data = load_bh_interpretation_data()
        if bh_data is not None and not bh_data.empty:
            if st.button("📊 Export Thickness Data", key="thickness_export_data"):
                csv = bh_data.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Download BH Interpretation CSV",
                    data=csv,
                    file_name=f"BH_Interpretation_data_{timestamp}.csv",
                    mime="text/csv",
                    key="thickness_download_csv_button"
                )


def render_property_chainage_tab(filtered_data: pd.DataFrame):
    """
    Render the Engineering Properties vs Chainage analysis tab.
    
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
        
        
        if HAS_FUNCTIONS and 'Chainage' in filtered_data.columns:
            # Get numerical properties
            numerical_props = get_numerical_properties(filtered_data)
            
            if numerical_props:
                # Enhanced parameter box with 5-row × 5-column structure (following enhanced pattern)
                with st.expander("Plot Parameters", expanded=True):
                    # Helper function for parsing tuple inputs (following enhanced pattern)
                    def parse_tuple(input_str, default):
                        try:
                            cleaned = input_str.strip().replace('(', '').replace(')', '')
                            values = [float(x.strip()) for x in cleaned.split(',')]
                            return tuple(values) if len(values) == 2 else default
                        except:
                            return default
                    
                    # Helper function to get zone names from config
                    def get_zone_names_from_config(config_str):
                        try:
                            import json
                            zone_config = json.loads(config_str)
                            return list(zone_config.keys())
                        except:
                            return ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
                    
                    # First, get the zone configuration to use in filters
                    zone_config_str = st.session_state.get("property_chainage_zone_config", '{\n"Zone 1": [21300, 26300],\n"Zone 2": [26300, 32770],\n"Zone 3": [32770, 37100],\n"Zone 4": [37100, 41120]\n}')
                    available_zones = get_zone_names_from_config(zone_config_str)
                    
                    # Row 1: Property Selection and Basic Options
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        selected_property = st.selectbox(
                            "Property",
                            numerical_props,
                            key="property_chainage_property"
                        )
                    with col2:
                        category_by = st.selectbox(
                            "Category By",
                            ["Geology_Orgin", "Consistency", "Material", "None"],
                            index=0,
                            key="property_chainage_category_by"
                        )
                    with col3:
                        color_by = st.selectbox(
                            "Color By",
                            ["From_mbgl", "Geology_Orgin", "Consistency", "None"],
                            index=0,
                            key="property_chainage_color_by"
                        )
                    with col4:
                        show_zone_boundaries = st.selectbox(
                            "Zone Boundaries",
                            [True, False],
                            index=0,
                            key="property_chainage_show_zones"
                        )
                    with col5:
                        show_zone_labels = st.selectbox(
                            "Zone Labels",
                            [True, False],
                            index=0,
                            key="property_chainage_show_labels"
                        )
                    
                    # Row 2: Filtering Options
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        filter1_by = st.selectbox(
                            "Filter 1 By",
                            ["None", "Geology Origin", "Consistency", "Hole ID", "Zone"],
                            index=0,
                            key="property_chainage_filter1_by"
                        )
                    with col2:
                        # Always show filter input, greyed out when disabled
                        filter1_enabled = filter1_by != "None"
                        
                        if filter1_by == "Geology Origin" and 'Geology_Orgin' in filtered_data.columns:
                            filter1_values = st.multiselect(
                                "Filter 1 Values",
                                sorted(filtered_data['Geology_Orgin'].dropna().unique()),
                                key="property_chainage_filter1_values",
                                disabled=not filter1_enabled
                            )
                        elif filter1_by == "Consistency" and 'Consistency' in filtered_data.columns:
                            filter1_values = st.multiselect(
                                "Filter 1 Values",
                                sorted(filtered_data['Consistency'].dropna().unique()),
                                key="property_chainage_filter1_values",
                                disabled=not filter1_enabled
                            )
                        elif filter1_by == "Hole ID":
                            filter1_values = st.multiselect(
                                "Filter 1 Values",
                                sorted(filtered_data['Hole_ID'].dropna().unique()),
                                key="property_chainage_filter1_values",
                                disabled=not filter1_enabled
                            )
                        elif filter1_by == "Zone":
                            filter1_values = st.multiselect(
                                "Filter 1 Values",
                                available_zones,
                                key="property_chainage_filter1_values",
                                disabled=not filter1_enabled
                            )
                        else:
                            # Always show empty multiselect when None selected, but disabled
                            filter1_values = st.multiselect(
                                "Filter 1 Values",
                                [],
                                key="property_chainage_filter1_values",
                                disabled=True
                            )
                    
                    with col3:
                        filter2_by = st.selectbox(
                            "Filter 2 By",
                            ["None", "Geology Origin", "Consistency", "Hole ID", "Depth Range"],
                            index=0,
                            key="property_chainage_filter2_by"
                        )
                    with col4:
                        # Always show filter input, greyed out when disabled
                        filter2_enabled = filter2_by != "None"
                        
                        if filter2_by == "Geology Origin" and 'Geology_Orgin' in filtered_data.columns:
                            filter2_values = st.multiselect(
                                "Filter 2 Values",
                                sorted(filtered_data['Geology_Orgin'].dropna().unique()),
                                key="property_chainage_filter2_values",
                                disabled=not filter2_enabled
                            )
                        elif filter2_by == "Consistency" and 'Consistency' in filtered_data.columns:
                            filter2_values = st.multiselect(
                                "Filter 2 Values",
                                sorted(filtered_data['Consistency'].dropna().unique()),
                                key="property_chainage_filter2_values",
                                disabled=not filter2_enabled
                            )
                        elif filter2_by == "Hole ID":
                            filter2_values = st.multiselect(
                                "Filter 2 Values",
                                sorted(filtered_data['Hole_ID'].dropna().unique()),
                                key="property_chainage_filter2_values",
                                disabled=not filter2_enabled
                            )
                        elif filter2_by == "Depth Range":
                            filter2_values = st.text_input(
                                "Depth Range (min,max)",
                                value="0,50",
                                key="property_chainage_filter2_values",
                                disabled=not filter2_enabled
                            )
                        else:
                            # Always show empty multiselect when None selected, but disabled
                            filter2_values = st.multiselect(
                                "Filter 2 Values",
                                [],
                                key="property_chainage_filter2_values",
                                disabled=True
                            )
                    with col5:
                        pass
                    
                    # Row 3: Plot Configuration
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        figsize_str = st.text_input("figsize (w, h)", value="(13, 6)", key="property_chainage_figsize")
                    with col2:
                        xlim_str = st.text_input("xlim (min, max)", value="(20000, 45000)", key="property_chainage_xlim")
                    with col3:
                        ylim_str = st.text_input("ylim (min, max)", value="(auto, auto)", key="property_chainage_ylim")
                    with col4:
                        title = st.text_input("title", value="", key="property_chainage_title")
                    with col5:
                        title_suffix = st.text_input("title_suffix", value="", key="property_chainage_title_suffix")
                    
                    # Row 4: Visual Style
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        scatter_size = st.number_input("scatter_size", min_value=20, max_value=100, value=40, step=5, key="property_chainage_scatter_size")
                    with col2:
                        scatter_alpha = st.number_input("scatter_alpha", min_value=0.3, max_value=1.0, value=0.7, step=0.05, key="property_chainage_scatter_alpha")
                    with col3:
                        line_width = st.number_input("line_width", min_value=0.5, max_value=3.0, value=1.0, step=0.1, key="property_chainage_line_width")
                    with col4:
                        legend_fontsize = st.number_input("legend_fontsize", min_value=8, max_value=16, value=10, key="property_chainage_legend_fontsize")
                    with col5:
                        legend_bbox = st.text_input(
                            "Legend BBox (x,y)",
                            value="",
                            key="property_chainage_legend_bbox",
                            help="Legend position coordinates: Leave empty for automatic best location, or specify: '1.05,1' (right), '0.5,-0.1' (bottom center), '0,1' (top left)"
                        )
                    
                    # Row 5: Text Formatting
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        title_fontsize = st.number_input("title_fontsize", min_value=8, max_value=24, value=14, key="property_chainage_title_fontsize")
                    with col2:
                        label_fontsize = st.number_input("label_fontsize", min_value=8, max_value=20, value=12, key="property_chainage_label_fontsize")
                    with col3:
                        tick_fontsize = st.number_input("tick_fontsize", min_value=6, max_value=16, value=10, key="property_chainage_tick_fontsize")
                    with col4:
                        zone_orientation = st.selectbox(
                            "Zone Orientation",
                            ["vertical", "horizontal"],
                            index=0,
                            key="property_chainage_zone_orientation2"
                        )
                    with col5:
                        pass
                    
                    # Row 6: Zone Configuration
                    st.markdown("**Zone Configuration**")
                    zone_config_str = st.text_area(
                        "Define Project Zones",
                        value='{\n"Zone 1": [21300, 26300],\n"Zone 2": [26300, 32770],\n"Zone 3": [32770, 37100],\n"Zone 4": [37100, 41120]\n}',
                        height=120,
                        key="property_chainage_zone_config",
                        help="Define zones: Zone Name -> [start_chainage, end_chainage]. You can add/remove/modify zones as needed."
                    )
                
                # Parse zone configuration
                try:
                    import json
                    zone_config_dict = json.loads(zone_config_str)
                    # Convert lists to tuples for consistency with existing code
                    zonage = {name: tuple(bounds) for name, bounds in zone_config_dict.items()}
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    st.error(f"Invalid zone configuration JSON: {str(e)}")
                    # Fallback to default zones
                    zonage = {
                        "Zone 1": (21300, 26300),
                        "Zone 2": (26300, 32770),
                        "Zone 3": (32770, 37100),
                        "Zone 4": (37100, 41120)
                    }
                
                # Apply filters to data (following enhanced pattern)
                valid_data = filtered_data.dropna(subset=['Chainage', selected_property])
                
                # Apply Filter 1
                if filter1_by != "None" and filter1_values:
                    if filter1_by == "Geology Origin":
                        valid_data = valid_data[valid_data['Geology_Orgin'].isin(filter1_values)]
                    elif filter1_by == "Consistency":
                        valid_data = valid_data[valid_data['Consistency'].isin(filter1_values)]
                    elif filter1_by == "Hole ID":
                        valid_data = valid_data[valid_data['Hole_ID'].isin(filter1_values)]
                    elif filter1_by == "Zone":
                        # Filter by project zones (use dynamic zonage from configuration)
                        zone_filter = pd.Series(False, index=valid_data.index)
                        for zone_name in filter1_values:
                            if zone_name in zonage:
                                start, end = zonage[zone_name]
                                zone_filter |= (valid_data['Chainage'] >= start) & (valid_data['Chainage'] <= end)
                        valid_data = valid_data[zone_filter]
                
                # Apply Filter 2
                if filter2_by != "None" and filter2_values:
                    if filter2_by == "Geology Origin":
                        valid_data = valid_data[valid_data['Geology_Orgin'].isin(filter2_values)]
                    elif filter2_by == "Consistency":
                        valid_data = valid_data[valid_data['Consistency'].isin(filter2_values)]
                    elif filter2_by == "Hole ID":
                        valid_data = valid_data[valid_data['Hole_ID'].isin(filter2_values)]
                    elif filter2_by == "Depth Range":
                        try:
                            depth_range = [float(x.strip()) for x in filter2_values.split(',')]
                            if len(depth_range) == 2:
                                min_depth, max_depth = depth_range
                                valid_data = valid_data[(valid_data['From_mbgl'] >= min_depth) & (valid_data['From_mbgl'] <= max_depth)]
                        except:
                            pass  # Invalid depth range format
                
                # Generate dynamic title suffix based on applied filters
                title_suffix_parts = []
                if filter1_by != "None" and filter1_values:
                    if isinstance(filter1_values, list) and len(filter1_values) <= 3:
                        title_suffix_parts.append(f"{filter1_by}: {', '.join(map(str, filter1_values))}")
                    else:
                        title_suffix_parts.append(f"{filter1_by}: {len(filter1_values)} items")
                
                if filter2_by != "None" and filter2_values:
                    if filter2_by == "Depth Range":
                        title_suffix_parts.append(f"Depth: {filter2_values}m")
                    elif isinstance(filter2_values, list) and len(filter2_values) <= 3:
                        title_suffix_parts.append(f"{filter2_by}: {', '.join(map(str, filter2_values))}")
                    else:
                        title_suffix_parts.append(f"{filter2_by}: {len(filter2_values)} items")
                
                dynamic_title_suffix = " | ".join(title_suffix_parts) if title_suffix_parts else ""
                
                if not valid_data.empty:
                    try:
                        # Import matplotlib at function level to avoid scope issues
                        import matplotlib.pyplot as plt
                        
                        # Use the dynamic zones from configuration (zonage already defined above)
                        # Clear any existing figures first
                        plt.close('all')
                        
                        # Parse enhanced parameters (following enhanced pattern)
                        figsize = parse_tuple(figsize_str, (13, 6))
                        xlim = parse_tuple(xlim_str, (20000, 45000))
                        
                        # Handle ylim - support "auto" values
                        if ylim_str.strip().lower() in ["(auto, auto)", "auto", ""]:
                            ylim = None
                        else:
                            ylim = parse_tuple(ylim_str, None)
                        
                        # Handle category_by and color_by None values
                        category_by_col = None if category_by == "None" else category_by
                        color_by_col = None if color_by == "None" else color_by
                        
                        # Parse legend bbox (x,y coordinates only)
                        if legend_bbox.strip() == "":
                            # Empty input = automatic best location
                            legend_style_dict = {
                                'loc': 'best'  # Let matplotlib find the best location
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
                        
                        # Generate final title
                        if title:
                            final_title = title
                        else:
                            final_title = f"{selected_property} along Chainage"
                        
                        if title_suffix:
                            final_title += f" - {title_suffix}"
                        elif dynamic_title_suffix:
                            final_title += f" - {dynamic_title_suffix}"
                        
                        # Create enhanced plot using the original plotting function with all new parameters
                        plot_by_chainage(
                            df=valid_data,
                            chainage_col='Chainage',
                            property_col=selected_property,
                            category_by_col=category_by_col,
                            color_by_col=color_by_col,
                            figsize=figsize,
                            xlim=xlim,
                            ylim=ylim,
                            classification_zones=zonage,
                            zone_orientation=zone_orientation,
                            show_zone_boundaries=show_zone_boundaries,
                            show_zone_labels=show_zone_labels,
                            title=final_title,
                            marker_size=scatter_size,
                            marker_alpha=scatter_alpha,
                            line_width=line_width,
                            title_fontsize=title_fontsize,
                            label_fontsize=label_fontsize,
                            tick_fontsize=tick_fontsize,
                            legend_fontsize=legend_fontsize,
                            legend_style=legend_style_dict,
                            show_plot=False,
                            close_plot=False
                        )
                        
                        # Capture and display the figure with size control
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            # Use plotting utils for sidebar size control
                            try:
                                from .plotting_utils import display_plot_with_size_control
                                display_plot_with_size_control(current_fig)
                            except ImportError:
                                # Fallback to standard display
                                st.pyplot(current_fig, use_container_width=True)
                            success = True
                        else:
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
                                    store_spatial_plot('property_chainage', buf)
                            except Exception as e:
                                pass
                            
                            # Simple download button with figure reference
                            from .plot_download_simple import create_simple_download_button
                            create_simple_download_button("property_vs_chainage_tab", "tab", fig=current_fig)
                            
                            # Add map visualization before summary (following enhanced tab pattern)
                            render_property_chainage_map_visualization(valid_data, selected_property)
                            
                            # Enhanced plot summary with engineering interpretations (following enhanced pattern)
                            st.divider()
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Property vs Chainage Summary**")
                                
                                summary_data = []
                                
                                # Basic data statistics
                                if not valid_data.empty:
                                    property_values = valid_data[selected_property].dropna()
                                    chainage_values = valid_data['Chainage'].dropna()
                                    
                                    if not property_values.empty:
                                        summary_data.extend([
                                            {'Parameter': 'Total Data Points', 'Value': f"{len(valid_data):,}"},
                                            {'Parameter': f'{selected_property} Mean', 'Value': f"{property_values.mean():.2f}"},
                                            {'Parameter': f'{selected_property} Range', 'Value': f"{property_values.min():.2f} - {property_values.max():.2f}"},
                                            {'Parameter': f'{selected_property} Std Dev', 'Value': f"{property_values.std():.2f}"},
                                            {'Parameter': 'Chainage Range', 'Value': f"{chainage_values.min():.0f} - {chainage_values.max():.0f} m"}
                                        ])
                                        
                                        # Zone distribution analysis (following enhanced pattern)
                                        for zone_name, (start, end) in zonage.items():
                                            zone_data = valid_data[(valid_data['Chainage'] >= start) & (valid_data['Chainage'] <= end)]
                                            if not zone_data.empty:
                                                zone_property_values = zone_data[selected_property].dropna()
                                                if not zone_property_values.empty:
                                                    summary_data.extend([
                                                        {'Parameter': f'{zone_name} Count', 'Value': f"{len(zone_data)}"},
                                                        {'Parameter': f'{zone_name} {selected_property} Mean', 'Value': f"{zone_property_values.mean():.2f}"},
                                                        {'Parameter': f'{zone_name} {selected_property} Range', 'Value': f"{zone_property_values.min():.2f} - {zone_property_values.max():.2f}"}
                                                    ])
                                        
                                        # Geological distribution if available
                                        if 'Geology_Orgin' in valid_data.columns:
                                            geology_counts = valid_data['Geology_Orgin'].value_counts()
                                            summary_data.append({
                                                'Parameter': 'Geological Units',
                                                'Value': f"{len(geology_counts)} types"
                                            })
                                            for geology, count in geology_counts.head(3).items():
                                                percentage = (count / len(valid_data)) * 100
                                                summary_data.append({
                                                    'Parameter': f'  {geology}',
                                                    'Value': f"{count} ({percentage:.1f}%)"
                                                })
                                        
                                        # Depth distribution analysis
                                        if 'From_mbgl' in valid_data.columns:
                                            depth_data = valid_data['From_mbgl'].dropna()
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
                                st.markdown("**Engineering Interpretation Guidelines**")
                                st.write("**Project Zones:**")
                                # Dynamic zone display based on configuration
                                for zone_name, (start, end) in zonage.items():
                                    length_km = (end - start) / 1000
                                    st.write(f"• **{zone_name}**: Ch. {start:,.0f}-{end:,.0f}m ({length_km:.1f}km)")
                                st.write("")
                                if selected_property in ['UCS (MPa)', 'Is50a (MPa)', 'Is50d (MPa)']:
                                    st.write("**Rock Strength Guidelines:**")
                                    st.write("• **<2 MPa**: Very weak rock")
                                    st.write("• **2-6 MPa**: Weak rock") 
                                    st.write("• **6-20 MPa**: Moderately strong")
                                    st.write("• **20-60 MPa**: Strong rock")
                                    st.write("• **>60 MPa**: Very strong rock")
                                elif selected_property in ['SPT N Value']:
                                    st.write("**SPT Guidelines:**")
                                    st.write("• **<4**: Very loose/soft")
                                    st.write("• **4-10**: Loose/soft")
                                    st.write("• **10-30**: Medium dense/firm")
                                    st.write("• **30-50**: Dense/stiff")
                                    st.write("• **>50**: Very dense/hard")
                                elif selected_property in ['CBR (%)']:
                                    st.write("**CBR Guidelines:**")
                                    st.write("• **<3%**: Poor subgrade")
                                    st.write("• **3-7%**: Fair subgrade")
                                    st.write("• **7-20%**: Good subgrade")
                                    st.write("• **>20%**: Excellent subgrade")
                                
                    except Exception as e:
                        st.error(f"Error creating Property vs Chainage plot: {str(e)}")
                else:
                    st.warning(f"No valid data points found for {selected_property} vs Chainage")
            else:
                st.warning("No numerical properties found for chainage analysis")
        else:
            st.warning("Chainage column or plotting utilities not available")
        
        # Data preview and statistics options underneath plot
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Show data preview", key="property_chainage_data_preview"):
                if 'Chainage' in filtered_data.columns and 'selected_property' in locals():
                    preview_cols = ['Hole_ID', 'From_mbgl', 'Chainage', selected_property]
                    if 'Geology_Orgin' in filtered_data.columns:
                        preview_cols.append('Geology_Orgin')
                    
                    available_cols = [col for col in preview_cols if col in filtered_data.columns]
                    preview_data = filtered_data[available_cols].dropna()
                    st.dataframe(preview_data.head(20), use_container_width=True)
                    st.caption(f"{len(preview_data)} total records")
                else:
                    st.info("No data available for preview")
        
        with col2:
            if st.checkbox("Show statistics", key="property_chainage_statistics"):
                if 'Chainage' in filtered_data.columns and 'selected_property' in locals():
                    valid_data = filtered_data.dropna(subset=['Chainage', selected_property])
                    if not valid_data.empty:
                        property_values = valid_data[selected_property]
                        chainage_values = valid_data['Chainage']
                        
                        stats_data = []
                        stats_data.extend([
                            {'Parameter': 'Total Data Points', 'Value': f"{len(valid_data):,}"},
                            {'Parameter': f'Mean {selected_property}', 'Value': f"{property_values.mean():.2f}"},
                            {'Parameter': f'{selected_property} Range', 'Value': f"{property_values.min():.2f} - {property_values.max():.2f}"},
                            {'Parameter': 'Chainage Range', 'Value': f"{chainage_values.min():.0f} - {chainage_values.max():.0f} m"}
                        ])
                        
                        # Zone distribution (use dynamic zones from configuration)
                        
                        for zone_name, (start, end) in zonage.items():
                            zone_data = valid_data[(valid_data['Chainage'] >= start) & (valid_data['Chainage'] <= end)]
                            if not zone_data.empty:
                                stats_data.append({
                                    'Parameter': f'{zone_name} Count',
                                    'Value': f"{len(zone_data):,}"
                                })
                        
                        # Create statistics table
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No valid data for statistics")
                else:
                    st.info("No data available for statistics")
    
    except Exception as e:
        st.error(f"Error in Property vs Chainage analysis: {str(e)}")


def render_property_chainage_map_visualization(data: pd.DataFrame, selected_property: str):
    """
    Render Property vs Chainage test location map visualization (following enhanced tab pattern).
    """
    try:
        st.markdown("### Test Locations Map")
        
        # Check for coordinate data and display map
        if HAS_PYPROJ and HAS_PLOTLY:
            # Use dynamic ID columns detection to find coordinate columns
            try:
                from .data_processing import get_id_columns_from_data
                id_columns = get_id_columns_from_data(data)
            except:
                # Fallback to standard coordinate detection
                id_columns = [col for col in data.columns if any(keyword in col.lower() for keyword in ['north', 'east', 'lat', 'lon', 'x', 'y'])]
            
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
                
                # Get coordinate data from property test locations
                try:
                    # Get unique sample locations from property data
                    sample_locations = data[['Hole_ID', 'From_mbgl']].drop_duplicates()
                    
                    # Merge with coordinate data including property values for hover info
                    merge_cols = ['Hole_ID', 'From_mbgl', lat_col, lon_col, 'Chainage', selected_property]
                    
                    # Add geological columns if available
                    if 'Geology_Orgin' in data.columns:
                        merge_cols.append('Geology_Orgin')
                    
                    available_merge_cols = [col for col in merge_cols if col in data.columns]
                    
                    coord_data = sample_locations.merge(
                        data[available_merge_cols], 
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
                                
                                # Display enhanced map with property test locations
                                if HAS_PLOTLY:
                                    # Prepare hover data including property values
                                    hover_data_dict = {'From_mbgl': True, 'Chainage': True}
                                    if selected_property in map_data.columns:
                                        hover_data_dict[selected_property] = True
                                    if 'Geology_Orgin' in map_data.columns:
                                        hover_data_dict['Geology_Orgin'] = True
                                    
                                    # Calculate optimal zoom and center
                                    zoom_level, center = calculate_map_zoom_and_center(map_data['lat'], map_data['lon'])
                                    
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat='lat',
                                        lon='lon',
                                        hover_name='Hole_ID',
                                        hover_data=hover_data_dict,
                                        color_discrete_sequence=['green'],
                                        zoom=zoom_level,
                                        center=center,
                                        height=400,
                                        title=f"{selected_property} Test Locations ({len(coord_data)} locations)"
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
                                # Prepare hover data including property values
                                hover_data_dict = {'From_mbgl': True, 'Chainage': True}
                                if selected_property in map_data.columns:
                                    hover_data_dict[selected_property] = True
                                if 'Geology_Orgin' in map_data.columns:
                                    hover_data_dict['Geology_Orgin'] = True
                                
                                # Calculate optimal zoom and center
                                zoom_level, center = calculate_map_zoom_and_center(map_data['lat'], map_data['lon'])
                                
                                fig = px.scatter_mapbox(
                                    map_data,
                                    lat='lat',
                                    lon='lon',
                                    hover_name='Hole_ID',
                                    hover_data=hover_data_dict,
                                    color_discrete_sequence=['green'],
                                    zoom=zoom_level,
                                    center=center,
                                    height=400,
                                    title=f"{selected_property} Test Locations ({len(coord_data)} locations)"
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
                            
                            st.caption(f"Found {len(coord_data)} {selected_property} test locations with coordinates")
                    else:
                        st.info(f"No coordinate data found for {selected_property} test locations")
                except Exception as e:
                    st.warning(f"Could not process coordinates: {str(e)}")
            else:
                st.info("No coordinate columns detected in the data")
        else:
            st.info("Map visualization requires pyproj library")
            
    except Exception as e:
        st.error(f"Error creating map visualization: {str(e)}")


def render_property_depth_tab(filtered_data: pd.DataFrame):
    """
    Render the Property vs Depth analysis tab.
    
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
        
        
        if HAS_FUNCTIONS and 'From_mbgl' in filtered_data.columns:
            # Get numerical properties
            numerical_props = get_numerical_properties(filtered_data)
            
            if numerical_props:
                # Enhanced parameter box with 5-row × 5-column structure (following enhanced pattern)
                with st.expander("Plot Parameters", expanded=True):
                    # Helper function for parsing tuple inputs (following enhanced pattern)
                    def parse_tuple(input_str, default):
                        try:
                            cleaned = input_str.strip().replace('(', '').replace(')', '')
                            values = [float(x.strip()) for x in cleaned.split(',')]
                            return tuple(values) if len(values) == 2 else default
                        except:
                            return default
                    
                    # Row 1: Property Selection and Basic Options
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        selected_property = st.selectbox(
                            "Property",
                            numerical_props,
                            key="property_depth_property"
                        )
                    with col2:
                        show_legend = st.selectbox(
                            "show_legend",
                            [True, False],
                            index=0,
                            key="property_depth_show_legend"
                        )
                    with col3:
                        invert_y_axis = st.selectbox(
                            "invert_y_axis",
                            [True, False],
                            index=0,
                            key="property_depth_invert_y"
                        )
                    with col4:
                        show_grid = st.selectbox(
                            "show_grid",
                            [True, False],
                            index=0,
                            key="property_depth_show_grid"
                        )
                    with col5:
                        pass
                    
                    # Row 2: Filtering Options
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        filter1_by = st.selectbox(
                            "Filter 1 By",
                            ["None", "Geology Origin", "Consistency", "Hole ID", "Depth Range"],
                            index=0,
                            key="property_depth_filter1_by"
                        )
                    with col2:
                        # Always show filter input, but enable/disable based on selection
                        filter1_enabled = filter1_by != "None"
                        
                        if filter1_by == "Geology Origin" and 'Geology_Orgin' in filtered_data.columns:
                            filter1_values = st.multiselect(
                                "Filter 1 Values",
                                sorted(filtered_data['Geology_Orgin'].dropna().unique()) if filter1_enabled else [],
                                key="property_depth_filter1_values",
                                disabled=not filter1_enabled
                            )
                        elif filter1_by == "Consistency" and 'Consistency' in filtered_data.columns:
                            filter1_values = st.multiselect(
                                "Filter 1 Values",
                                sorted(filtered_data['Consistency'].dropna().unique()) if filter1_enabled else [],
                                key="property_depth_filter1_values",
                                disabled=not filter1_enabled
                            )
                        elif filter1_by == "Hole ID":
                            filter1_values = st.multiselect(
                                "Filter 1 Values",
                                sorted(filtered_data['Hole_ID'].dropna().unique()) if filter1_enabled else [],
                                key="property_depth_filter1_values",
                                disabled=not filter1_enabled
                            )
                        elif filter1_by == "Depth Range":
                            filter1_values = st.text_input(
                                "Filter 1 Values",
                                value="0,10" if filter1_enabled else "",
                                key="property_depth_filter1_values",
                                disabled=not filter1_enabled,
                                placeholder="e.g., 0,10"
                            )
                        else:
                            # Default text input for when "None" is selected
                            filter1_values = st.text_input(
                                "Filter 1 Values",
                                value="",
                                key="property_depth_filter1_values",
                                disabled=True,
                                placeholder="Select filter type first"
                            )
                    with col3:
                        filter2_by = st.selectbox(
                            "Filter 2 By",
                            ["None", "Geology Origin", "Consistency", "Hole ID", "Property Range"],
                            index=0,
                            key="property_depth_filter2_by"
                        )
                    with col4:
                        # Always show filter input, but enable/disable based on selection
                        filter2_enabled = filter2_by != "None"
                        
                        if filter2_by == "Geology Origin" and 'Geology_Orgin' in filtered_data.columns:
                            filter2_values = st.multiselect(
                                "Filter 2 Values",
                                sorted(filtered_data['Geology_Orgin'].dropna().unique()) if filter2_enabled else [],
                                key="property_depth_filter2_values",
                                disabled=not filter2_enabled
                            )
                        elif filter2_by == "Consistency" and 'Consistency' in filtered_data.columns:
                            filter2_values = st.multiselect(
                                "Filter 2 Values",
                                sorted(filtered_data['Consistency'].dropna().unique()) if filter2_enabled else [],
                                key="property_depth_filter2_values",
                                disabled=not filter2_enabled
                            )
                        elif filter2_by == "Hole ID":
                            filter2_values = st.multiselect(
                                "Filter 2 Values",
                                sorted(filtered_data['Hole_ID'].dropna().unique()) if filter2_enabled else [],
                                key="property_depth_filter2_values",
                                disabled=not filter2_enabled
                            )
                        elif filter2_by == "Property Range":
                            filter2_values = st.text_input(
                                "Filter 2 Values",
                                value="" if not filter2_enabled else "",
                                key="property_depth_filter2_values",
                                disabled=not filter2_enabled,
                                placeholder="e.g., 10,50"
                            )
                        else:
                            # Default text input for when "None" is selected
                            filter2_values = st.text_input(
                                "Filter 2 Values",
                                value="",
                                key="property_depth_filter2_values",
                                disabled=True,
                                placeholder="Select filter type first"
                            )
                    with col5:
                        color_by = st.selectbox(
                            "Category By",
                            ["Geology_Orgin", "Consistency", "Hole_ID", "None"],
                            index=0,
                            key="property_depth_color_by"
                        )
                    
                    # Row 3: Plot Configuration
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        figsize_str = st.text_input("figsize (w, h)", value="(8, 6)", key="property_depth_figsize")
                    with col2:
                        xlim_str = st.text_input("xlim (min, max)", value="(auto, auto)", key="property_depth_xlim")
                    with col3:
                        ylim_str = st.text_input("ylim (min, max)", value="(auto, auto)", key="property_depth_ylim")
                    with col4:
                        title = st.text_input("title", value="", key="property_depth_title")
                    with col5:
                        title_suffix = st.text_input("title_suffix", value="", key="property_depth_title_suffix")
                    
                    # Row 4: Visual Style
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        scatter_size = st.number_input("scatter_size", min_value=20, max_value=100, value=50, step=5, key="property_depth_scatter_size")
                    with col2:
                        scatter_alpha = st.number_input("scatter_alpha", min_value=0.3, max_value=1.0, value=0.7, step=0.05, key="property_depth_scatter_alpha")
                    with col3:
                        line_width = st.number_input("line_width", min_value=0.5, max_value=3.0, value=1.0, step=0.1, key="property_depth_line_width")
                    with col4:
                        legend_fontsize = st.number_input("legend_fontsize", min_value=8, max_value=16, value=10, key="property_depth_legend_fontsize")
                    with col5:
                        marker_style = st.selectbox(
                            "Marker Style",
                            ["o", "s", "^", "v", "D", "*"],
                            index=0,
                            key="property_depth_marker_style"
                        )
                    
                    # Row 5: Text Formatting and Export
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        title_fontsize = st.number_input("title_fontsize", min_value=6, max_value=32, value=14, key="property_depth_title_fontsize")
                    with col2:
                        label_fontsize = st.number_input("label_fontsize", min_value=6, max_value=28, value=12, key="property_depth_label_fontsize")
                    with col3:
                        tick_fontsize = st.number_input("tick_fontsize", min_value=4, max_value=24, value=10, key="property_depth_tick_fontsize")
                    with col4:
                        legend_fontsize_input = st.number_input("legend_fontsize", min_value=4, max_value=24, value=11, key="property_depth_legend_fontsize_input")
                    with col5:
                        pass
                
                # Apply filters to data (following enhanced pattern)
                valid_data = filtered_data.dropna(subset=['From_mbgl', selected_property])
                
                # Apply Filter 1
                if filter1_by != "None" and filter1_values:
                    if filter1_by == "Geology Origin":
                        valid_data = valid_data[valid_data['Geology_Orgin'].isin(filter1_values)]
                    elif filter1_by == "Consistency":
                        valid_data = valid_data[valid_data['Consistency'].isin(filter1_values)]
                    elif filter1_by == "Hole ID":
                        valid_data = valid_data[valid_data['Hole_ID'].isin(filter1_values)]
                    elif filter1_by == "Depth Range":
                        try:
                            depth_range = [float(x.strip()) for x in filter1_values.split(',')]
                            if len(depth_range) == 2:
                                min_depth, max_depth = depth_range
                                valid_data = valid_data[(valid_data['From_mbgl'] >= min_depth) & (valid_data['From_mbgl'] <= max_depth)]
                        except:
                            pass  # Invalid depth range format
                
                # Apply Filter 2
                if filter2_by != "None" and filter2_values:
                    if filter2_by == "Geology Origin":
                        valid_data = valid_data[valid_data['Geology_Orgin'].isin(filter2_values)]
                    elif filter2_by == "Consistency":
                        valid_data = valid_data[valid_data['Consistency'].isin(filter2_values)]
                    elif filter2_by == "Hole ID":
                        valid_data = valid_data[valid_data['Hole_ID'].isin(filter2_values)]
                    elif filter2_by == "Property Range":
                        try:
                            if filter2_values.strip():
                                prop_range = [float(x.strip()) for x in filter2_values.split(',')]
                                if len(prop_range) == 2:
                                    min_prop, max_prop = prop_range
                                    valid_data = valid_data[(valid_data[selected_property] >= min_prop) & (valid_data[selected_property] <= max_prop)]
                        except:
                            pass  # Invalid property range format
                
                # Generate dynamic title suffix based on applied filters
                title_suffix_parts = []
                if filter1_by != "None" and filter1_values:
                    if filter1_by == "Depth Range":
                        title_suffix_parts.append(f"Depth: {filter1_values}m")
                    elif isinstance(filter1_values, list) and len(filter1_values) <= 3:
                        title_suffix_parts.append(f"{filter1_by}: {', '.join(map(str, filter1_values))}")
                    else:
                        title_suffix_parts.append(f"{filter1_by}: {len(filter1_values)} items")
                
                if filter2_by != "None" and filter2_values:
                    if filter2_by == "Property Range":
                        title_suffix_parts.append(f"{selected_property}: {filter2_values}")
                    elif isinstance(filter2_values, list) and len(filter2_values) <= 3:
                        title_suffix_parts.append(f"{filter2_by}: {', '.join(map(str, filter2_values))}")
                    else:
                        title_suffix_parts.append(f"{filter2_by}: {len(filter2_values)} items")
                
                dynamic_title_suffix = " | ".join(title_suffix_parts) if title_suffix_parts else ""
                
                if not valid_data.empty:
                    try:
                        # Clear any existing figures first
                        plt.close('all')
                        
                        # Parse enhanced parameters (following enhanced pattern)
                        figsize = parse_tuple(figsize_str, (8, 10))
                        
                        # Handle xlim and ylim - support "auto" values
                        if xlim_str.strip().lower() in ["(auto, auto)", "auto", ""]:
                            xlim = None
                        else:
                            xlim = parse_tuple(xlim_str, None)
                            
                        if ylim_str.strip().lower() in ["(auto, auto)", "auto", ""]:
                            ylim = None
                        else:
                            ylim = parse_tuple(ylim_str, None)
                        
                        # Handle color_by None values
                        category_col = None if color_by == "None" else color_by
                        
                        # Generate final title
                        if title:
                            final_title = title
                        else:
                            final_title = f"{selected_property} vs Depth"
                        
                        if title_suffix:
                            final_title += f" - {title_suffix}"
                        elif dynamic_title_suffix:
                            final_title += f" - {dynamic_title_suffix}"
                        
                        # Handle color_by parameter - map to category_col for the plotting function
                        if color_by != "None" and color_by in valid_data.columns:
                            category_col = color_by
                        
                        # Create enhanced plot using the original plotting function with all parameters
                        # Handle grid display
                        grid_style_param = {'linestyle': '--', 'color': 'grey', 'alpha': 0.3} if show_grid else {'alpha': 0}
                        
                        plot_engineering_property_vs_depth(
                            df=valid_data,
                            property_col=selected_property,
                            title=final_title,
                            category_col=category_col,
                            figsize=figsize,
                            xlim=xlim,
                            ylim=ylim,
                            invert_yaxis=invert_y_axis,
                            show_legend=show_legend,
                            marker_size=scatter_size,
                            marker_alpha=scatter_alpha,
                            title_fontsize=title_fontsize,
                            label_fontsize=label_fontsize,
                            tick_fontsize=tick_fontsize,
                            legend_fontsize=legend_fontsize_input,
                            ylabel="Depth (m)",
                            grid_style=grid_style_param,
                            show_plot=False,
                            close_plot=False
                        )
                        
                        # Capture and display the figure in Streamlit with size control
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            # Import display utility
                            try:
                                from .plotting_utils import display_plot_with_size_control
                                display_plot_with_size_control(current_fig)
                            except ImportError:
                                st.pyplot(current_fig, use_container_width=True)
                            success = True
                        else:
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
                                    store_spatial_plot('property_depth', buf)
                            except Exception as e:
                                pass
                            
                            # Simple download button with figure reference
                            from .plot_download_simple import create_simple_download_button
                            create_simple_download_button("property_vs_depth_tab", "tab", fig=current_fig)
                            
                            # Add map visualization (following enhanced tab pattern)
                            render_property_depth_map_visualization(valid_data, selected_property)
                            
                            # Add test distribution chart
                            render_test_distribution_chart(filtered_data, selected_property)
                            
                            # Enhanced plot summary with engineering interpretations (following enhanced pattern)
                            st.divider()
                            
                            # Use 65% width for plot summary, leave 35% empty on the right
                            summary_col, spacer_col = st.columns([65, 35])
                            
                            with summary_col:
                                st.markdown("**Data Summary**")
                                
                                if not valid_data.empty:
                                    property_values = valid_data[selected_property].dropna()
                                    
                                    if not property_values.empty:
                                        # Create clean engineering summary
                                        summary_data = []
                                        
                                        # Dynamic depth zones based on geotechnical engineering principles
                                        depth_values = valid_data['From_mbgl'].dropna()
                                        max_depth = depth_values.max()
                                        
                                        # First layer: Fill/Topsoil (assume 0-1m if can't identify)
                                        fill_depth = 1.0  # Can be made configurable later
                                        
                                        # Remaining depth after fill layer
                                        remaining_depth = max_depth - fill_depth
                                        
                                        # Create zones: Fill + 4-5 percentile-based zones
                                        depth_zones = [("Fill/Topsoil (0-1m)", 0, fill_depth)]
                                        
                                        if remaining_depth > 0:
                                            # Split remaining depth into 4 zones using percentiles
                                            zone_depth = remaining_depth / 4
                                            
                                            zone_boundaries = [
                                                fill_depth + (i * zone_depth) for i in range(5)
                                            ]
                                            
                                            zone_names = [
                                                f"Zone 1 ({zone_boundaries[0]:.1f}-{zone_boundaries[1]:.1f}m)",
                                                f"Zone 2 ({zone_boundaries[1]:.1f}-{zone_boundaries[2]:.1f}m)", 
                                                f"Zone 3 ({zone_boundaries[2]:.1f}-{zone_boundaries[3]:.1f}m)",
                                                f"Zone 4 ({zone_boundaries[3]:.1f}-{zone_boundaries[4]:.1f}m)"
                                            ]
                                            
                                            for i, zone_name in enumerate(zone_names):
                                                start_depth = zone_boundaries[i]
                                                end_depth = zone_boundaries[i + 1] if i < 3 else None
                                                depth_zones.append((zone_name, start_depth, end_depth))
                                        else:
                                            # If total depth <= 1m, just use fill zone
                                            pass
                                        
                                        # Calculate for each depth zone
                                        for zone_name, start, end in depth_zones:
                                            if end is None:  # >20m case
                                                zone_data = valid_data[valid_data['From_mbgl'] >= start]
                                            else:
                                                zone_data = valid_data[(valid_data['From_mbgl'] >= start) & (valid_data['From_mbgl'] < end)]
                                            
                                            zone_props = zone_data[selected_property].dropna()
                                            
                                            if len(zone_props) > 0:
                                                summary_data.append({
                                                    'Depth Zone': zone_name,
                                                    'Tests': len(zone_props),
                                                    'Average': f"{zone_props.mean():.2f}",
                                                    'Range': f"{zone_props.min():.2f} - {zone_props.max():.2f}"
                                                })
                                            else:
                                                summary_data.append({
                                                    'Depth Zone': zone_name,
                                                    'Tests': 0,
                                                    'Average': "No data",
                                                    'Range': "No data"
                                                })
                                        
                                        # Create simple, clean table
                                        if summary_data:
                                            summary_df = pd.DataFrame(summary_data)
                                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                                            
                                            # Geology distribution section removed as requested
                                        else:
                                            st.info("No data available for summary")
                                    else:
                                        st.info("No valid property data found")
                                else:
                                    st.info("No data available")
                                
                    except Exception as e:
                        st.error(f"Error creating Property vs Depth plot: {str(e)}")
                else:
                    st.warning(f"No valid data points found for {selected_property} vs Depth")
            else:
                st.warning("No numerical properties found for depth analysis")
        else:
            st.warning("Depth column or plotting utilities not available")
        
        # Data preview and statistics options underneath plot
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Show data preview", key="property_depth_data_preview"):
                if 'From_mbgl' in filtered_data.columns and 'selected_property' in locals():
                    preview_cols = ['Hole_ID', 'From_mbgl', 'To_mbgl', selected_property]
                    if 'Geology_Orgin' in filtered_data.columns:
                        preview_cols.append('Geology_Orgin')
                    
                    available_cols = [col for col in preview_cols if col in filtered_data.columns]
                    preview_data = filtered_data[available_cols].dropna()
                    st.dataframe(preview_data.head(20), use_container_width=True)
                    st.caption(f"{len(preview_data)} total records")
                else:
                    st.info("No data available for preview")
        
        with col2:
            if st.checkbox("Show statistics", key="property_depth_statistics"):
                if 'From_mbgl' in filtered_data.columns and 'selected_property' in locals():
                    valid_data = filtered_data.dropna(subset=['From_mbgl', selected_property])
                    if not valid_data.empty:
                        property_values = valid_data[selected_property]
                        depth_values = valid_data['From_mbgl']
                        
                        stats_data = []
                        stats_data.extend([
                            {'Parameter': 'Total Data Points', 'Value': f"{len(valid_data):,}"},
                            {'Parameter': f'Mean {selected_property}', 'Value': f"{property_values.mean():.2f}"},
                            {'Parameter': f'{selected_property} Range', 'Value': f"{property_values.min():.2f} - {property_values.max():.2f}"},
                            {'Parameter': 'Depth Range', 'Value': f"{depth_values.min():.1f} - {depth_values.max():.1f} m"}
                        ])
                        
                        # Depth zone distribution
                        depth_ranges = [(0, 5), (5, 10), (10, 20), (20, float('inf'))]
                        for i, (start, end) in enumerate(depth_ranges):
                            if end == float('inf'):
                                zone_data = valid_data[valid_data['From_mbgl'] >= start]
                                zone_name = f"> {start}m"
                            else:
                                zone_data = valid_data[(valid_data['From_mbgl'] >= start) & (valid_data['From_mbgl'] < end)]
                                zone_name = f"{start}-{end}m"
                            
                            if not zone_data.empty:
                                stats_data.append({
                                    'Parameter': f'Depth {zone_name}',
                                    'Value': f"{len(zone_data):,}"
                                })
                        
                        # Create statistics table
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No valid data for statistics")
                else:
                    st.info("No data available for statistics")
    
    except Exception as e:
        st.error(f"Error in Property vs Depth analysis: {str(e)}")


def render_property_depth_map_visualization(data: pd.DataFrame, selected_property: str):
    """
    Render Property vs Depth test location map visualization (following enhanced tab pattern).
    """
    try:
        st.markdown("### Test Locations Map")
        
        # Check for coordinate data and display map
        if HAS_PYPROJ:
            # Use dynamic ID columns detection to find coordinate columns
            try:
                from .data_processing import get_id_columns_from_data
                id_columns = get_id_columns_from_data(data)
            except:
                # Fallback to standard coordinate detection
                id_columns = [col for col in data.columns if any(keyword in col.lower() for keyword in ['north', 'east', 'lat', 'lon', 'x', 'y'])]
            
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
                
                # Get coordinate data from property test locations
                try:
                    # Get unique sample locations from property data
                    sample_locations = data[['Hole_ID', 'From_mbgl']].drop_duplicates()
                    
                    # Merge with coordinate data including property values for hover info
                    merge_cols = ['Hole_ID', 'From_mbgl', lat_col, lon_col, selected_property]
                    
                    # Add additional columns if available
                    if 'Geology_Orgin' in data.columns:
                        merge_cols.append('Geology_Orgin')
                    if 'To_mbgl' in data.columns:
                        merge_cols.append('To_mbgl')
                    
                    available_merge_cols = [col for col in merge_cols if col in data.columns]
                    
                    coord_data = sample_locations.merge(
                        data[available_merge_cols], 
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
                                
                                # Display enhanced map with property test locations
                                if HAS_PLOTLY:
                                    # Prepare hover data including property values and depth
                                    hover_data_dict = {'From_mbgl': True}
                                    if 'To_mbgl' in map_data.columns:
                                        hover_data_dict['To_mbgl'] = True
                                    if selected_property in map_data.columns:
                                        hover_data_dict[selected_property] = True
                                    if 'Geology_Orgin' in map_data.columns:
                                        hover_data_dict['Geology_Orgin'] = True
                                    
                                    # Calculate optimal zoom and center
                                    zoom_level, center = calculate_map_zoom_and_center(map_data['lat'], map_data['lon'])
                                    
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat='lat',
                                        lon='lon',
                                        hover_name='Hole_ID',
                                        hover_data=hover_data_dict,
                                        color_discrete_sequence=['purple'],
                                        zoom=zoom_level,
                                        center=center,
                                        height=400,
                                        title=f"{selected_property} vs Depth Test Locations ({len(coord_data)} locations)"
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
                                # Prepare hover data including property values and depth
                                hover_data_dict = {'From_mbgl': True}
                                if 'To_mbgl' in map_data.columns:
                                    hover_data_dict['To_mbgl'] = True
                                if selected_property in map_data.columns:
                                    hover_data_dict[selected_property] = True
                                if 'Geology_Orgin' in map_data.columns:
                                    hover_data_dict['Geology_Orgin'] = True
                                
                                # Calculate optimal zoom and center
                                zoom_level, center = calculate_map_zoom_and_center(map_data['lat'], map_data['lon'])
                                
                                fig = px.scatter_mapbox(
                                    map_data,
                                    lat='lat',
                                    lon='lon',
                                    hover_name='Hole_ID',
                                    hover_data=hover_data_dict,
                                    color_discrete_sequence=['purple'],
                                    zoom=zoom_level,
                                    center=center,
                                    height=400,
                                    title=f"{selected_property} vs Depth Test Locations ({len(coord_data)} locations)"
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
                            
                            st.caption(f"Found {len(coord_data)} {selected_property} vs depth test locations with coordinates")
                    else:
                        st.info(f"No coordinate data found for {selected_property} vs depth test locations")
                except Exception as e:
                    st.warning(f"Could not process coordinates: {str(e)}")
            else:
                st.info("No coordinate columns detected in the data")
        else:
            st.info("Map visualization requires pyproj library")
            
    except Exception as e:
        st.error(f"Error creating map visualization: {str(e)}")


# Smart column selection functions for thickness analysis
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
            
            # Use smart column detection for available columns (now includes calculated columns)
            available_grouping_cols = get_grouping_columns(bh_data)
            available_category_cols = get_category_columns(bh_data) 
            available_value_cols = get_value_columns(bh_data)
            
            # Fallback to standard columns if smart detection fails
            if not available_grouping_cols:
                available_grouping_cols = [col for col in ['Geology_Orgin', 'Map_symbol', 'Consistency', 'Report', 'Type'] 
                                         if col in bh_data.columns and bh_data[col].notna().any()]
            if not available_category_cols:
                # More flexible fallback - include any non-numeric column that could be categorical
                fallback_cols = [col for col in bh_data.columns if col not in ['Hole_ID', 'From_mbgl', 'To_mbgl', 'Thickness', 'thickness_proportion_%']]
                available_category_cols = fallback_cols if fallback_cols else ['Consistency', 'Rock_Class', 'Material_Type', 'Map_symbol']
            if not available_value_cols:
                # More flexible fallback - include any numeric column that could be a value
                numeric_cols = [col for col in bh_data.columns 
                              if bh_data[col].dtype in ['int64', 'float64', 'int32', 'float32'] 
                              and col not in ['Hole_ID', 'From_mbgl', 'To_mbgl']]
                available_value_cols = numeric_cols if numeric_cols else ["thickness_proportion_%", "Thickness"]
            
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
                        
                        # Handle quoted strings and clean whitespace
                        cleaned_str = input_str.strip()
                        
                        # Split by comma and clean each item
                        items = [x.strip().strip('"\'') for x in cleaned_str.split(',')]
                        items = [x for x in items if x]  # Remove empty strings
                        
                        if not items:
                            return None
                        
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
                        help="Column to group thickness data by. Smart detection supports: Geology_Orgin, Map_symbol, Consistency, Report, Type"
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
                        help="Column for x-axis categories. Can be any column in your data. Common options: Consistency, Rock_Class, Material_Type, Map_symbol, or any other categorical column"
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
                        help="X-axis limits as (min, max). Use 'auto' for automatic scaling or '(0, 10)' format")
                
                with col3:
                    ylim_str = st.text_input("ylim (min, max)", value="(auto, auto)", key="thickness_ylim",
                        help="Y-axis limits as (min, max). Use 'auto' for automatic scaling or '(0, 100)' format")
                
                with col4:
                    title = st.text_input("title", value="", key="thickness_title", 
                        help="Custom plot title. Leave empty for auto-generated title")
                
                with col5:
                    title_suffix = st.text_input("title_suffix", value="", key="thickness_title_suffix",
                        help="Text to append to the plot title (e.g., project name, date)")
                
                # Row 3: Essential Display Options
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    x_axis_sort = st.selectbox(
                        "X-axis Sort",
                        ['smart_consistency', 'alphabetical', 'descending', 'ascending', 'reverse_alphabetical'],
                        index=0,
                        key="thickness_x_axis_sort",
                        help="Smart: consistency order for soil data, alphabetical for numerical. Manual override available in Advanced Parameters"
                    )
                
                with col2:
                    show_grid = st.selectbox("Show Grid", [True, False], index=0, key="thickness_show_grid",
                        help="Whether to display grid lines for easier reading")
                
                with col3:
                    show_percentage_labels = st.selectbox(
                        "Show Value Labels",
                        [True, False],
                        index=0,
                        key="thickness_show_percentage_labels",
                        help="Display numerical values on top of bars"
                    )
                
                with col4:
                    pass  # Empty for future use
                
                with col5:
                    pass  # Empty for future use
            
            # Advanced Parameters Section (separate from main expander)
            with st.expander("Advanced Parameters", expanded=False):
                
                # Advanced Row 1: Bar Styling & Font Sizes
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    bar_width = st.number_input("bar_width", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="thickness_bar_width",
                        help="Width of bars in the chart. 1.0 = full width, 0.5 = half width")
                
                with adv_col2:
                    bar_alpha = st.number_input("bar_alpha", min_value=0.1, max_value=1.0, value=0.8, step=0.05, key="thickness_bar_alpha",
                        help="Transparency of bars. 1.0 = opaque, 0.5 = semi-transparent")
                
                with adv_col3:
                    title_fontsize = st.number_input("title_fontsize", min_value=8, max_value=24, value=14, key="thickness_title_fontsize",
                        help="Font size for the main plot title")
                
                with adv_col4:
                    xlabel_fontsize = st.number_input("xlabel_fontsize", min_value=8, max_value=20, value=12, key="thickness_xlabel_fontsize",
                        help="Font size for the X-axis label")
                
                with adv_col5:
                    ylabel_fontsize = st.number_input("ylabel_fontsize", min_value=8, max_value=20, value=12, key="thickness_ylabel_fontsize",
                        help="Font size for the Y-axis label")
                
                # Advanced Row 2: Additional Font Sizes & Styling
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    tick_fontsize = st.number_input("tick_fontsize", min_value=6, max_value=16, value=11, key="thickness_tick_fontsize",
                        help="Font size for axis tick labels (numbers on axes)")
                
                with adv_col2:
                    legend_fontsize = st.number_input("legend_fontsize", min_value=8, max_value=16, value=10, key="thickness_legend_fontsize",
                        help="Font size for legend text")
                
                with adv_col3:
                    grid_linestyle = st.selectbox("grid_linestyle", ["--", "-", ":", "-."], index=0, key="thickness_grid_linestyle_2",
                        help="Style of grid lines")
                
                with adv_col4:
                    grid_alpha = st.number_input("grid_alpha", min_value=0.1, max_value=1.0, value=0.35, step=0.05, key="thickness_grid_alpha_2",
                        help="Transparency of grid lines")
                
                with adv_col5:
                    legend_bbox_str = st.text_input("legend_bbox", value="", key="thickness_legend_bbox_2",
                        help="Legend position: '1.05,1' (right), '0.5,-0.1' (bottom), leave empty for auto")
                
                # Advanced Row 3: Category & Legend Control
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    category_order_str = st.text_input(
                        "Category Order",
                        value="",
                        key="thickness_category_order",
                        help="Manual category order: H,VSt,F,S,VS,St (comma-separated). Overrides smart sorting. Leave empty for auto"
                    )
                
                with adv_col2:
                    legend_order_str = st.text_input(
                        "Legend Order",
                        value="",
                        key="thickness_legend_order",
                        help="Custom legend order: 5a,4a,3a,2a,1a (comma-separated). Leave empty for same as x-axis"
                    )
                
                with adv_col3:
                    legend_sort = st.selectbox(
                        "Legend Sort",
                        ['same_as_x', 'smart_consistency', 'alphabetical', 'descending', 'ascending', 'reverse_alphabetical'],
                        index=0,
                        key="thickness_legend_sort",
                        help="Legend ordering method. 'same_as_x' uses same order as x-axis"
                    )
                
                with adv_col4:
                    legend_loc = st.selectbox(
                        "Legend Location",
                        ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'],
                        index=0,
                        key="thickness_legend_loc",
                        help="Legend position. 'best' automatically finds optimal location"
                    )
                
                with adv_col5:
                    bar_edgecolor = st.text_input(
                        "Bar Edge Color",
                        value="black",
                        key="thickness_bar_edgecolor",
                        help="Color for bar edges (e.g., black, blue, #FF0000, 'none' for no edges)"
                    )
                
                # Advanced Row 4: Visual Styling & Bar Configuration
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    bar_linewidth = st.number_input(
                        "Bar Line Width",
                        min_value=0.0,
                        max_value=3.0,
                        value=0.6,
                        step=0.1,
                        key="thickness_bar_linewidth",
                        help="Width of bar edge lines. 0 = no edges"
                    )
                
                with adv_col2:
                    bar_hatch = st.text_input(
                        "Bar Hatch Pattern",
                        value="",
                        key="thickness_bar_hatch",
                        help="Hatch pattern: /, \\, |, -, +, x, o, O, ., * (leave empty for none)"
                    )
                
                with adv_col3:
                    rotation = st.number_input(
                        "X-Label Rotation",
                        min_value=0,
                        max_value=90,
                        value=0,
                        step=15,
                        key="thickness_rotation",
                        help="Rotation angle for x-axis labels in degrees"
                    )
                
                with adv_col4:
                    xlabel = st.text_input(
                        "Custom X-Label",
                        value="",
                        key="thickness_xlabel",
                        help="Custom X-axis label. Leave blank for auto-generated label"
                    )
                
                with adv_col5:
                    ylabel = st.text_input(
                        "Custom Y-Label",
                        value="",
                        key="thickness_ylabel",
                        help="Custom Y-axis label. Leave blank for auto-generated label"
                    )
                
                # Advanced Row 5: Colors & Style Configuration
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    colors_str = st.text_input(
                        "Custom Colors",
                        value="",
                        key="thickness_colors",
                        help="Custom colors: red,blue,green OR #FF0000,#0000FF,#00FF00 (comma-separated)"
                    )
                
                with adv_col2:
                    plot_style = st.selectbox(
                        "Plot Style",
                        ['seaborn-v0_8-whitegrid', 'seaborn-v0_8-colorblind', 'classic', 'bmh', 'default'],
                        index=0,
                        key="thickness_plot_style",
                        help="Overall plot styling theme"
                    )
                
                with adv_col3:
                    title_fontweight = st.selectbox(
                        "Title Font Weight",
                        ['bold', 'normal', 'light', 'heavy'],
                        index=0,
                        key="thickness_title_fontweight",
                        help="Font weight for the main plot title"
                    )
                
                with adv_col4:
                    label_fontweight = st.selectbox(
                        "Label Font Weight",
                        ['bold', 'normal', 'light', 'heavy'],
                        index=0,
                        key="thickness_label_fontweight",
                        help="Font weight for axis labels"
                    )
                
                with adv_col5:
                    grid_axis = st.selectbox(
                        "Grid Axis",
                        ['y', 'x', 'both'],
                        index=0,
                        key="thickness_grid_axis",
                        help="Which axes to show grid lines on"
                    )
                
                # Advanced Row 6: Technical Options
                adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
                
                with adv_col1:
                    percentage_decimal_places = st.number_input(
                        "Decimal Places",
                        min_value=0,
                        max_value=3,
                        value=1,
                        key="thickness_percentage_decimal_places",
                        help="Number of decimal places for value labels on bars"
                    )
                
                with adv_col2:
                    value_label_fontsize = st.number_input(
                        "Value Label Font Size",
                        min_value=6,
                        max_value=16,
                        value=10,
                        key="thickness_value_label_fontsize",
                        help="Font size for value labels displayed on bars"
                    )
                
                with adv_col3:
                    show_plot = st.selectbox(
                        "Show Plot",
                        [True, False],
                        index=0,
                        key="thickness_show_plot",
                        help="Whether to display the plot (for batch processing)"
                    )
                
                with adv_col4:
                    pass  # Empty for spacing
                
                with adv_col5:
                    pass  # Empty for spacing
            
            # Process data following Jupyter notebook logic
            try:
                # Filter data by selected group or use all data if "All" is selected
                if selected_group == "All":
                    # Use all available data - no filtering by group
                    filtered_bh_data = bh_data.copy()
                    analysis_scope = f"All {group_by_col}s"
                else:
                    # Filter to specific group (existing logic)
                    try:
                        # Try direct comparison first
                        filtered_bh_data = bh_data[bh_data[group_by_col] == selected_group].copy()
                    except Exception:
                        # Fallback: convert both to strings for comparison
                        filtered_bh_data = bh_data[bh_data[group_by_col].astype(str) == str(selected_group)].copy()
                    
                    analysis_scope = selected_group
                
                if filtered_bh_data.empty:
                    st.warning(f"No data available for {group_by_col} = {selected_group}")
                    return
                
                # Calculate thickness proportions exactly like Jupyter notebook
                thickness_data = filtered_bh_data.pivot_table(
                    values='Thickness', 
                    index=category_col, 
                    aggfunc='sum'
                ).reset_index()
                
                # Add thickness proportion column
                total_thickness = thickness_data['Thickness'].sum()
                thickness_data['thickness_proportion_%'] = (thickness_data['Thickness'] / total_thickness) * 100
                
                if thickness_data.empty:
                    st.warning(f"No thickness data available for {selected_group}")
                    return
                
                # Parse parameter inputs with enhanced validation
                figsize = parse_tuple(figsize_str, (15, 7), "figsize", min_val=1)
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
                    
                    # Try to find corresponding full name columns
                    possible_name_columns = []
                    group_col_lower = group_col.lower()
                    
                    for col in data.columns:
                        col_lower = col.lower()
                        # Look for columns that might contain full names
                        if (col != group_col and 
                            any(keyword in col_lower for keyword in ['name', 'description', 'full', 'long', 'detail', 'formation'])):
                            possible_name_columns.append(col)
                    
                    # If we found potential name columns, create mapping
                    if possible_name_columns:
                        # Use the first matching column
                        name_col = possible_name_columns[0]
                        try:
                            # Create mapping from unique combinations
                            mapping_data = data[[group_col, name_col]].drop_duplicates().dropna()
                            for _, row in mapping_data.iterrows():
                                code = str(row[group_col]).strip()
                                full_name = str(row[name_col]).strip()
                                if code and full_name and code != full_name:
                                    group_names[code] = full_name
                        except Exception:
                            pass  # Failed to create mapping, use fallback
                    
                    # Fallback to hardcoded mappings for common geological formations
                    fallback_names = {
                        'Tos': 'Sunnybank Formation',
                        'Rjbw': 'Woogaroo Subgroup', 
                        'Rin': 'Tingalpa Formation',
                        'Dcf': 'Neranleigh Fernvale Beds',
                        'RS_XW': 'Brisbane Tuff',
                        'ALLUVIUM': 'Alluvial Deposits',
                        'FILL': 'Fill Material',
                        'TOPSOIL': 'Topsoil',
                        'Toc': 'Older Alluvium',
                        'ASPHALT': 'Asphalt Pavement'
                    }
                    
                    # Merge with fallback (dynamic takes precedence)
                    for code, name in fallback_names.items():
                        if code not in group_names:
                            group_names[code] = name
                    
                    return group_names
                
                # Get dynamic group names
                group_names = get_dynamic_group_names(bh_data, group_by_col)
                
                # Generate plot title based on selection
                if title:
                    plot_title = title
                else:
                    if selected_group == "All":
                        plot_title = f"Distribution of {category_col} by Thickness across All {group_by_col}s"
                    else:
                        group_full_name = group_names.get(selected_group, selected_group)
                        plot_title = f"Distribution of {category_col} by Thickness of {group_full_name}"
                
                if title_suffix:
                    plot_title += title_suffix
                
                # Plot generation with comprehensive parameters
                if HAS_FUNCTIONS:
                    try:
                        plot_category_by_thickness(
                            # Essential Data Parameters
                            df=thickness_data,
                            value_col=value_col,
                            category_col=category_col,
                            
                            # Plot Appearance
                            title=plot_title,
                            title_suffix=None,  # Already included in title
                            figsize=figsize,
                            title_fontsize=title_fontsize,
                            title_fontweight=title_fontweight,
                            
                            # Category Options
                            category_order=category_order,
                            x_axis_sort=x_axis_sort,
                            legend_order=legend_order,
                            legend_sort=legend_sort,
                            
                            # Axis Configuration  
                            xlim=xlim,
                            ylim=ylim,
                            xlabel=xlabel if xlabel else None,
                            ylabel=ylabel if ylabel else None,
                            xlabel_fontsize=xlabel_fontsize,
                            ylabel_fontsize=ylabel_fontsize,
                            label_fontweight=label_fontweight,
                            tick_fontsize=tick_fontsize,
                            show_percentage_labels=show_percentage_labels,
                            percentage_decimal_places=percentage_decimal_places,
                            
                            # Display Options
                            show_plot=show_plot,
                            show_legend=show_legend,
                            show_grid=show_grid,
                            grid_axis=grid_axis,
                            legend_fontsize=legend_fontsize,
                            legend_loc=legend_loc,
                            
                            # Visual Customization
                            colors=colors,
                            bar_width=bar_width,
                            bar_alpha=bar_alpha,
                            bar_edgecolor=bar_edgecolor,
                            bar_linewidth=bar_linewidth,
                            bar_hatch=bar_hatch if bar_hatch else None,
                            rotation=rotation,
                            value_label_fontsize=value_label_fontsize,
                            
                            # Advanced Styling
                            plot_style=plot_style if plot_style != 'default' else None
                        )
                        
                        # Display plot in Streamlit with size control
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            # Use the display function that respects sidebar width control
                            from .plotting_utils import display_plot_with_size_control
                            display_plot_with_size_control(current_fig)
                            
                            # Store plot for dashboard
                            try:
                                import io
                                buf = io.BytesIO()
                                current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                store_spatial_plot('thickness_analysis', buf)
                            except Exception as e:
                                pass
                            
                            # Download button
                            try:
                                from .plot_download_simple import create_simple_download_button
                                create_simple_download_button("thickness_analysis_tab", "tab", fig=current_fig)
                            except:
                                pass
                        
                        else:
                            st.warning("Plot generation failed")
                    
                    except Exception as e:
                        st.error(f"Error generating plot: {str(e)}")
                        st.exception(e)
                
                else:
                    st.warning("Plotting functions not available")
                
                # Data preview and statistics (two-column layout)
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.checkbox("Show data preview", key="thickness_data_preview"):
                        st.subheader("Thickness Data Preview")
                        display_cols = [category_col, 'Thickness', 'thickness_proportion_%']
                        display_data = thickness_data[display_cols].copy()
                        display_data['thickness_proportion_%'] = display_data['thickness_proportion_%'].round(2)
                        st.dataframe(display_data, use_container_width=True)
                        st.caption(f"{len(thickness_data)} {category_col.lower()} classes")
                        
                        # Show raw BH data preview
                        st.subheader("BH Interpretation Data Preview")
                        preview_cols = ['Hole_ID', group_by_col, category_col, 'Thickness']
                        available_cols = [col for col in preview_cols if col in bh_data.columns]
                        
                        if selected_group == "All":
                            group_bh_data = filtered_bh_data[available_cols]
                            records_caption = f"{len(group_bh_data)} total records across all {group_by_col}s"
                        else:
                            try:
                                group_bh_data = bh_data[bh_data[group_by_col] == selected_group][available_cols]
                            except Exception:
                                group_bh_data = bh_data[bh_data[group_by_col].astype(str) == str(selected_group)][available_cols]
                            records_caption = f"{len(group_bh_data)} total records for {selected_group}"
                        
                        st.dataframe(group_bh_data.head(20), use_container_width=True)
                        st.caption(records_caption)
                
                with col2:
                    if st.checkbox("Show statistics", key="thickness_statistics"):
                        st.subheader("Thickness Analysis Statistics")
                        
                        # Summary statistics
                        stats_data = []
                        total_thickness = thickness_data['Thickness'].sum()
                        
                        # Analysis scope display
                        if selected_group == "All":
                            scope_display = f"All {group_by_col}s Combined"
                        else:
                            group_full_name = group_names.get(selected_group, selected_group)
                            scope_display = group_full_name
                        
                        stats_data.extend([
                            {'Metric': f'{group_by_col} Scope', 'Value': scope_display},
                            {'Metric': 'Total Thickness', 'Value': f"{total_thickness:.2f} m"},
                            {'Metric': f'Number of {category_col} Classes', 'Value': str(len(thickness_data))},
                            {'Metric': 'Mean Thickness', 'Value': f"{thickness_data['Thickness'].mean():.2f} m"},
                            {'Metric': 'Median Thickness', 'Value': f"{thickness_data['Thickness'].median():.2f} m"},
                            {'Metric': 'Min Thickness', 'Value': f"{thickness_data['Thickness'].min():.2f} m"},
                            {'Metric': 'Max Thickness', 'Value': f"{thickness_data['Thickness'].max():.2f} m"},
                            {'Metric': 'Std Dev', 'Value': f"{thickness_data['Thickness'].std():.2f} m"}
                        ])
                        
                        # Category distribution
                        st.write(f"**{category_col} Distribution:**")
                        for _, row in thickness_data.iterrows():
                            stats_data.append({
                                'Metric': f"Class {row[category_col]}", 
                                'Value': f"{row['thickness_proportion_%']:.1f}% ({row['Thickness']:.2f} m)"
                            })
                        
                        # Create statistics table
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        # Show available groups summary
                        st.write(f"**All Available {group_by_col}:**")
                        try:
                            group_counts = []
                            for group in available_groups:
                                try:
                                    # Try direct comparison
                                    count = len(bh_data[bh_data[group_by_col] == group])
                                except Exception:
                                    # Fallback: string comparison
                                    count = len(bh_data[bh_data[group_by_col].astype(str) == str(group)])
                                group_counts.append(count)
                            
                            available_groups_df = pd.DataFrame({
                                group_by_col: available_groups,
                                'Records': group_counts
                            })
                            st.dataframe(available_groups_df, use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.write(f"Could not generate groups summary: {str(e)}")
            
            except Exception as e:
                st.error(f"Error processing thickness data: {str(e)}")
                st.exception(e)
        
        else:
            st.warning("BH_Interpretation data not available. Please ensure BH_Interpretation.xlsx is in the Input folder.")
    
    except Exception as e:
        st.error(f"Error in Thickness Analysis: {str(e)}")
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


def map_property_to_test_type(property_name: str) -> str:
    """
    Map a property name to its corresponding test type.
    
    Args:
        property_name: Name of the property (e.g., 'CBR (%)', 'UCS (MPa)')
        
    Returns:
        str: Test type name (e.g., 'CBR', 'UCS')
    """
    import re
    
    # Property to test mapping patterns
    test_mapping = {
        'CBR': ['CBR', 'CBR.*%', 'CBR.*SWELL'],
        'UCS': ['UCS', 'UCS.*MPA', 'UNCONFINED'],
        'Atterberg': ['LL', 'PL', 'PI', 'LS', 'LIQUID.*LIMIT', 'PLASTIC.*LIMIT', 'PLASTICITY.*INDEX'],
        'SPT': ['SPT', 'SPT.*N', 'N.*VALUE', 'STANDARD.*PENETRATION'],
        'PSD': ['D10', 'D50', 'D60', 'CU', 'CC', 'PARTICLE.*SIZE', 'GRAIN.*SIZE'],
        'Emerson': ['EMERSON', 'EMERSON.*CLASS', 'DISPERSIVITY'],
        'Is50': ['IS50', 'POINT.*LOAD', 'PLT'],
        'Moisture': ['MC', 'MOISTURE', 'WATER.*CONTENT'],
        'Density': ['DENSITY', 'UNIT.*WEIGHT', 'BULK.*DENSITY']
    }
    
    # Convert property name to uppercase for matching
    prop_upper = property_name.upper()
    
    # Find matching test type
    for test_type, patterns in test_mapping.items():
        for pattern in patterns:
            if re.search(pattern, prop_upper):
                return test_type
    
    # Default fallback - try to extract test type from property name
    if '(' in property_name:
        return property_name.split('(')[0].strip()
    else:
        return property_name


def render_test_distribution_chart(data: pd.DataFrame, selected_property: str):
    """
    Render test distribution chart showing distribution of tests along chainage.
    
    Args:
        data: Full filtered data
        selected_property: Selected property name
    """
    try:
        st.markdown("### Test Distribution")
        
        # Map property to test type
        test_type = map_property_to_test_type(selected_property)
        
        # Find the corresponding test identifier column
        test_identifier_col = f"{test_type}?"
        
        # Check if test identifier column exists
        if test_identifier_col not in data.columns:
            # Try alternative naming patterns
            possible_names = [
                f"{test_type.upper()}?",
                f"{test_type.lower()}?",
                f"{test_type.title()}?"
            ]
            
            for alt_name in possible_names:
                if alt_name in data.columns:
                    test_identifier_col = alt_name
                    break
            else:
                st.info(f"Test identifier column '{test_type}?' not found for {selected_property}")
                return
        
        # Filter data for this test type and chainage
        if 'Chainage' not in data.columns:
            st.info("Chainage data not available for test distribution")
            return
            
        test_data = data[data[test_identifier_col] == 'Y'].copy()
        
        if test_data.empty:
            st.info(f"No {test_type} test data available")
            return
            
        # Create distribution plot
        if HAS_MATPLOTLIB:
            import matplotlib.pyplot as plt
            
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Plot test locations along chainage
            chainage_values = test_data['Chainage'].dropna()
            
            if len(chainage_values) > 0:
                # Calculate nice bin edges based on data range
                min_chainage = chainage_values.min()
                max_chainage = chainage_values.max()
                data_range = max_chainage - min_chainage
                
                # Determine appropriate bin width for nice round numbers
                if data_range <= 500:
                    bin_width = 50  # Use 50m bins for small ranges
                elif data_range <= 2000:
                    bin_width = 100  # Use 100m bins for medium ranges
                elif data_range <= 10000:
                    bin_width = 500  # Use 500m bins for large ranges
                elif data_range <= 50000:
                    bin_width = 1000  # Use 1km bins for very large ranges
                else:
                    bin_width = 2000  # Use 2km bins for huge ranges
                
                # Create bin edges aligned to nice round numbers
                start_bin = (min_chainage // bin_width) * bin_width
                end_bin = ((max_chainage // bin_width) + 1) * bin_width
                bin_edges = list(range(int(start_bin), int(end_bin) + bin_width, bin_width))
                
                # Create histogram with custom bin edges
                counts, bin_edges, patches = ax.hist(
                    chainage_values, 
                    bins=bin_edges, 
                    alpha=0.7, 
                    color='steelblue',
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # Set x-axis ticks to align with bin edges
                ax.set_xticks(bin_edges)
                # Format x-axis labels as clean integers with 90-degree rotation
                ax.set_xticklabels([f'{int(edge)}' for edge in bin_edges], rotation=90, ha='center')
                
                # Add individual test markers
                y_offset = max(counts) * 0.05
                ax.scatter(
                    chainage_values,
                    [y_offset] * len(chainage_values),
                    alpha=0.6,
                    color='red',
                    s=20,
                    marker='|'
                )
                
                ax.set_xlabel('Chainage (m)', fontsize=12)
                ax.set_ylabel('Number of Tests', fontsize=12)
                ax.set_title(f'{test_type} Test Distribution along Chainage ({len(chainage_values)} tests)', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Display in Streamlit with 90% width
                chart_col, spacer_col = st.columns([9, 1])
                with chart_col:
                    st.pyplot(fig, use_container_width=True)
            else:
                st.info(f"No chainage data available for {test_type} tests")
        else:
            st.info("Matplotlib not available for test distribution chart")
            
    except Exception as e:
        st.error(f"Error creating test distribution chart: {str(e)}")