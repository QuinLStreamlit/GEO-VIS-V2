"""
Emerson Class Analysis Module

This module handles Emerson class data analysis and visualization for geotechnical applications.
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
    from .dashboard_materials import store_material_plot
    from .plotting_utils import display_plot_with_size_control
    HAS_PLOTTING_UTILS = True
except ImportError:
    # For standalone testing
    from data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_id_columns_from_data
    from plot_defaults import get_default_parameters, get_color_schemes
    try:
        from dashboard_materials import store_material_plot
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
    
    from plot_emerson_by_origin import plot_emerson_by_origin
    from plot_by_chainage import plot_by_chainage
    HAS_FUNCTIONS = True
except ImportError as e:
    HAS_FUNCTIONS = False
    print(f"Warning: Could not import Functions: {e}")


def get_emerson_columns(data):
    """Get all potential Emerson columns with smart pattern matching"""
    emerson_patterns = [
        'emerson', 'emerson_class', 'emerson class', 'emersonclass', 'emerson_cls',
        'emerson_classification', 'emerson classification', 'emerson test',
        'emerson_test', 'emersontest', 'emerson_value', 'emerson value'
    ]
    potential_cols = []
    for col in data.columns:
        col_lower = col.lower().replace('(', '').replace(')', '').replace('-', '_')
        # Check for exact matches and partial matches
        for pattern in emerson_patterns:
            if pattern in col_lower:
                potential_cols.append(col)
                break
        # Also check for Emerson with numbers (Emerson1, Emerson2, etc.)
        if 'emerson' in col_lower and any(char.isdigit() for char in col_lower):
            potential_cols.append(col)
    return list(set(potential_cols))  # Remove duplicates




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
    id_columns = get_id_columns_from_data(df)
    emerson_columns = extract_test_columns(df, 'Emerson')
    
    if not emerson_columns:
        return pd.DataFrame()
    
    return create_test_dataframe(df, 'Emerson', id_columns, emerson_columns)


def render_emerson_analysis_tab(filtered_data: pd.DataFrame):
    """
    Render the Emerson analysis tab in Streamlit.
    Uses original plotting functions from Functions folder exactly as in Jupyter notebook.
    Follows PSD/Atterberg/SPT structure for consistency.
    
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
        
        # Extract Emerson data exactly like Jupyter notebook
        emerson_data = extract_emerson_data(filtered_data)
        
        if emerson_data.empty:
            st.warning("No Emerson class data available with current filters.")
            return
        
        
        # Get standard ID columns for UI controls
        standard_id_columns = get_id_columns_from_data(emerson_data)
        
        # Get category columns and default values
        category_columns = [col for col in standard_id_columns if col in emerson_data.columns]
        if not category_columns:
            category_columns = list(emerson_data.columns)[:10]
        
        geology_index = 0
        if "Geology_Orgin" in category_columns:
            geology_index = category_columns.index("Geology_Orgin")
        
        # Initialize filter variables with default values
        filter1_by = "None"
        filter1_values = []
        filter2_by = "None" 
        filter2_values = []
        category_by = category_columns[geology_index] if category_columns else "Geology_Orgin"
        
        # Get smart column selections for Emerson column only
        available_emerson_cols = get_emerson_columns(emerson_data)
        
        # Find best default for Emerson column with priority
        emerson_default = None
        emerson_priority = ['Emerson class', 'Emerson', 'Emerson_class', 'emerson', 'emerson_class']
        for preferred in emerson_priority:
            if preferred in available_emerson_cols:
                emerson_default = preferred
                break
        if not emerson_default and available_emerson_cols:
            emerson_default = available_emerson_cols[0]

        # Helper function to get available values for filter types
        def get_filter_options(filter_type):
            if filter_type == "Geology Origin":
                return sorted(emerson_data['Geology_Orgin'].dropna().unique()) if 'Geology_Orgin' in emerson_data.columns else []
            elif filter_type == "Consistency":
                return sorted(emerson_data['Consistency'].dropna().unique()) if 'Consistency' in emerson_data.columns else []
            elif filter_type == "Hole ID":
                return sorted(emerson_data['Hole_ID'].dropna().unique()) if 'Hole_ID' in emerson_data.columns else []
            elif filter_type == "Report":
                return sorted(emerson_data['Report'].dropna().unique()) if 'Report' in emerson_data.columns else []
            elif filter_type == "Map Symbol":
                return sorted(emerson_data['Map_Symbol'].dropna().unique()) if 'Map_Symbol' in emerson_data.columns else []
            else:
                return []

        # Plot Parameters Section
        with st.expander("Plot Parameters", expanded=True):
            # Row 1: Core Data Selection & Basic Settings
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                # Emerson Column selection with fallback
                if available_emerson_cols:
                    emerson_col_options = available_emerson_cols
                    emerson_default_index = emerson_col_options.index(emerson_default) if emerson_default in emerson_col_options else 0
                else:
                    # Fallback to columns containing 'emerson' or all columns
                    emerson_col_options = [col for col in emerson_data.columns if 'emerson' in col.lower()]
                    if not emerson_col_options:
                        emerson_col_options = ['Emerson class'] + list(emerson_data.columns)[:5]
                    emerson_default_index = 0
                    st.warning("No Emerson columns found with standard naming.")
                
                emerson_col = st.selectbox("Emerson Column", emerson_col_options,
                    index=emerson_default_index, key="emerson_col",
                    help="Column containing Emerson class values. Smart detection supports: Emerson, Emerson_class, Emerson1, etc.")
                    
            with col2:
                category_by = st.selectbox(
                    "Category By",
                    category_columns,
                    index=geology_index,
                    key="emerson_category_by",
                    help="Column to group Emerson classes by (e.g., Geology Origin, Consistency)"
                )
            with col3:
                title = st.text_input("title", value="", key="emerson_title",
                    help="Custom plot title. Leave empty for auto-generated title")
            with col4:
                title_suffix = st.text_input("title_suffix", value="", key="emerson_title_suffix",
                    help="Text to append to the plot title (e.g., project name, date)")
            with col5:
                show_legend = st.selectbox(
                    "show_legend",
                    [True, False],
                    index=0,
                    key="emerson_show_legend",
                    help="Whether to display the legend showing category colors"
                )
            
            # Row 2: Data Filtering
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                filter1_by = st.selectbox(
                    "Filter 1 By",
                    ["None", "Geology Origin", "Consistency", "Hole ID", "Report", "Map Symbol"],
                    index=1,
                    key="emerson_filter1_by_v2",
                    help="Primary filter criteria. Select 'None' to disable filtering"
                )
            with col2:
                if filter1_by == "None":
                    filter1_values = []
                    st.selectbox("Filter 1 Value", ["All"], index=0, disabled=True, key="emerson_filter1_value_disabled_v2")
                else:
                    filter1_options = get_filter_options(filter1_by)
                    filter1_dropdown_options = ["All"] + filter1_options
                    filter1_selection = st.selectbox(f"{filter1_by}", filter1_dropdown_options, index=0, key="emerson_filter1_value_v2",
                        help=f"Select specific {filter1_by} value or 'All' to include all options")
                    if filter1_selection == "All":
                        filter1_values = filter1_options
                    else:
                        filter1_values = [filter1_selection]
            
            with col3:
                filter2_by = st.selectbox(
                    "Filter 2 By",
                    ["None", "Geology Origin", "Consistency", "Hole ID", "Report", "Map Symbol"],
                    index=0,
                    key="emerson_filter2_by_v2",
                    help="Secondary filter criteria. Can be combined with Filter 1"
                )
            with col4:
                if filter2_by == "None":
                    filter2_values = []
                    st.selectbox("Filter 2 Value", ["All"], index=0, disabled=True, key="emerson_filter2_value_disabled_v2")
                else:
                    filter2_options = get_filter_options(filter2_by)
                    filter2_dropdown_options = ["All"] + filter2_options
                    filter2_selection = st.selectbox(f"{filter2_by}", filter2_dropdown_options, index=0, key="emerson_filter2_value_v2",
                        help=f"Select specific {filter2_by} value or 'All' to include all options")
                    if filter2_selection == "All":
                        filter2_values = filter2_options
                    else:
                        filter2_values = [filter2_selection]
            
            with col5:
                pass

        # Advanced Parameters Section
        with st.expander("Advanced Parameters", expanded=False):
            # Row 1: Axis & Bar Configuration
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                xlim_str = st.text_input("xlim (min, max)", value="auto", key="emerson_xlim",
                    help="X-axis limits as (min, max). Use 'auto' for automatic scaling or '(0, 5)' format")
            with col2:
                ylim_str = st.text_input("ylim (min, max)", value="auto", key="emerson_ylim",
                    help="Y-axis limits as (min, max). Use 'auto' for automatic scaling or '(0, 100)' format")
            with col3:
                bar_width = st.number_input("bar_width", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="emerson_bar_width",
                    help="Width of bars in the chart. 1.0 = full width, 0.5 = half width")
            with col4:
                bar_alpha = st.number_input("bar_alpha", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="emerson_bar_alpha",
                    help="Transparency of bars. 1.0 = opaque, 0.5 = semi-transparent")
            with col5:
                edge_color = st.selectbox("edge_color", ['black', 'white', 'gray', 'none'], index=0, key="emerson_edge_color",
                    help="Color of bar edges. Use 'none' for no edge lines")
            
            # Row 2: Styling & Legend Configuration
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                cmap_name = st.selectbox(
                    "cmap_name",
                    ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'tab10', 'Set1', 'Set2', 'Set3'],
                    index=0,
                    key="emerson_cmap_name",
                    help="Color scheme for bars. Viridis/plasma = continuous colors, Set1/tab10 = distinct colors"
                )
            with col2:
                legend_loc = st.selectbox(
                    "legend_loc",
                    ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'],
                    index=0,
                    key="emerson_legend_loc",
                    help="Legend anchor position. 'best' automatically finds optimal location"
                )
            with col3:
                legend_bbox_str = st.text_input("legend_bbox_to_anchor", value="(1.01, 1)", key="emerson_legend_bbox",
                    help="Legend position relative to plot. (1.01, 1) = outside right, (0.5, -0.1) = below center")
            with col4:
                title_fontsize = st.number_input("title_fontsize", min_value=8, max_value=24, value=15, key="emerson_title_fontsize",
                    help="Font size for the main plot title")
            with col5:
                label_fontsize = st.number_input("label_fontsize", min_value=8, max_value=20, value=13, key="emerson_label_fontsize",
                    help="Font size for axis labels (X and Y axis titles)")
            
            # Row 3: Font Styling
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                tick_fontsize = st.number_input("tick_fontsize", min_value=6, max_value=16, value=12, key="emerson_tick_fontsize",
                    help="Font size for axis tick labels (numbers on axes)")
            with col2:
                legend_fontsize = st.number_input("legend_fontsize", min_value=6, max_value=16, value=11, key="emerson_legend_fontsize",
                    help="Font size for legend text (category names)")
            with col3:
                legend_title_fontsize = st.number_input("legend_title_fontsize", min_value=6, max_value=18, value=12, key="emerson_legend_title_fontsize",
                    help="Font size for legend title")
            with col4:
                pass
            with col5:
                pass
        
        # Apply filters to data
        filtered_emerson = emerson_data.copy()
        
        # Apply Filter 1
        if filter1_by != "None" and filter1_values:
            if filter1_by == "Geology Origin":
                filtered_emerson = filtered_emerson[filtered_emerson['Geology_Orgin'].isin(filter1_values)]
            elif filter1_by == "Consistency":
                filtered_emerson = filtered_emerson[filtered_emerson['Consistency'].isin(filter1_values)]
            elif filter1_by == "Hole ID":
                filtered_emerson = filtered_emerson[filtered_emerson['Hole_ID'].isin(filter1_values)]
            elif filter1_by == "Report" and 'Report' in filtered_emerson.columns:
                filtered_emerson = filtered_emerson[filtered_emerson['Report'].isin(filter1_values)]
        
        # Apply Filter 2
        if filter2_by != "None" and filter2_values:
            if filter2_by == "Geology Origin":
                filtered_emerson = filtered_emerson[filtered_emerson['Geology_Orgin'].isin(filter2_values)]
            elif filter2_by == "Consistency":
                filtered_emerson = filtered_emerson[filtered_emerson['Consistency'].isin(filter2_values)]
            elif filter2_by == "Hole ID":
                filtered_emerson = filtered_emerson[filtered_emerson['Hole_ID'].isin(filter2_values)]
            elif filter2_by == "Report" and 'Report' in filtered_emerson.columns:
                filtered_emerson = filtered_emerson[filtered_emerson['Report'].isin(filter2_values)]
        
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
            
            return f": {' + '.join(suffix_parts)}" if suffix_parts else ""
        
        # Generate dynamic title suffix
        dynamic_title_suffix = generate_title_suffix()
        
        # Check if filtered data is empty
        if filtered_emerson.empty:
            st.warning("No data remains after applying filters. Please adjust your filter criteria.")
            return
        
        # Parse the tuple inputs
        if xlim_str.lower() == "auto":
            xlim = None
        else:
            xlim = parse_tuple(xlim_str, None)
        if ylim_str.lower() == "auto":
            ylim = None
        else:
            ylim = parse_tuple(ylim_str, None)
        
        # Set default figsize and dpi
        figsize = (9, 6)  # Default figsize for Emerson plots
        dpi = 300  # Default DPI
        
        # Parse legend bbox to anchor
        try:
            legend_bbox_to_anchor = parse_tuple(legend_bbox_str, (1.02, 1))
        except:
            legend_bbox_to_anchor = (1.02, 1)
        
        # Create main plot using filtered data
        if not filtered_emerson.empty:
            if HAS_FUNCTIONS:
                try:
                    # Clear any existing figures first
                    plt.close('all')
                    
                    # Create final title
                    final_title = title if title else None
                    final_title_suffix = title_suffix if title_suffix else None
                    
                    plot_emerson_by_origin(
                        df=filtered_emerson,
                        origin_col=category_by,
                        emerson_col=emerson_col,
                        save_plot=False,
                        show_plot=False,
                        dpi=dpi,
                        xlim=xlim,
                        ylim=ylim,
                        title=final_title,
                        title_suffix=final_title_suffix,
                        show_legend=show_legend,
                        figsize=figsize,
                        cmap_name=cmap_name,
                        bar_width=bar_width,
                        bar_alpha=bar_alpha,
                        edge_color=edge_color,
                        title_fontsize=title_fontsize,
                        label_fontsize=label_fontsize,
                        tick_fontsize=tick_fontsize,
                        legend_fontsize=legend_fontsize,
                        legend_title_fontsize=legend_title_fontsize,
                        legend_loc=legend_loc,
                        legend_bbox_to_anchor=legend_bbox_to_anchor,
                        close_plot=False
                    )
                    
                    # Display the plot with size control
                    if HAS_MATPLOTLIB:
                        current_figs = plt.get_fignums()
                        if current_figs:
                            current_fig = plt.figure(current_figs[-1])
                            # Use the display function that respects sidebar width control
                            display_plot_with_size_control(current_fig)
                    
                    # Store the plot for Materials Dashboard
                    try:
                        if HAS_MATPLOTLIB:
                            current_fig = plt.gcf()
                            if current_fig and current_fig.get_axes():
                                import io
                                buf = io.BytesIO()
                                current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                store_material_plot('emerson_analysis', buf)
                    except Exception as e:
                        pass  # Don't break main functionality if storage fails
                    
                    # Simple download button with figure reference
                    from .plot_download_simple import create_simple_download_button
                    create_simple_download_button("emerson_analysis", "main", fig=current_fig)
                    
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
        else:
            st.warning("No valid Emerson data available for plotting")
            st.info("Check data availability and filter criteria")
        
        # Map visualization (matching PSD/Atterberg/SPT style)
        st.markdown("### Test Locations Map")
        
        # Check for coordinate data and display map
        if HAS_PYPROJ:
            # Use dynamic ID columns detection to find coordinate columns
            id_columns = get_id_columns_from_data(filtered_emerson)
            
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
                
                # Get coordinate data from Emerson test locations
                try:
                    # Get unique sample locations from Emerson data
                    sample_locations = filtered_emerson[['Hole_ID', 'From_mbgl']].drop_duplicates()
                    
                    # Merge with coordinate data
                    merge_cols = ['Hole_ID', 'From_mbgl', lat_col, lon_col]
                    if 'Chainage' in filtered_emerson.columns:
                        merge_cols.append('Chainage')
                    coord_data = sample_locations.merge(
                        filtered_emerson[merge_cols], 
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
                                
                                # Display enhanced map with Emerson test locations
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
                                        title=f"Emerson Test Locations ({len(coord_data)} locations)"
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
                                    title=f"Emerson Test Locations ({len(coord_data)} locations)"
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
                            
                            st.caption(f"Found {len(coord_data)} Emerson test locations with coordinates")
                    else:
                        st.info("No coordinate data found for Emerson test locations")
                except Exception as e:
                    st.warning(f"Could not process coordinates: {str(e)}")
            else:
                st.info("No coordinate columns detected in the data")
        else:
            st.info("Map visualization requires pyproj library")
        
        # Emerson Test Distribution by Chainage (directly after map like PSD/Atterberg/SPT)
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
            if 'Chainage' in filtered_emerson.columns:
                available_test_types, test_columns = get_test_types_from_columns(filtered_emerson)
                
                if len(available_test_types) > 0:
                    chainage_data = filtered_emerson['Chainage'].dropna()
                    if not chainage_data.empty:
                        min_chainage = chainage_data.min()
                        max_chainage = chainage_data.max()
                        
                        # Create fixed interval bins (200m intervals)
                        bin_interval = 200
                        bin_start = int(min_chainage // bin_interval) * bin_interval
                        bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
                        bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
                        
                        # Find Emerson-specific test types
                        emerson_test_types = [t for t in available_test_types if 'Emerson' in t or 'emerson' in t.lower()]
                        
                        if emerson_test_types:
                            # Create charts for Emerson test types - each chart at 90% width in separate rows
                            for i, test_type in enumerate(emerson_test_types):
                                if i > 0:
                                    st.write("")
                                
                                # Each chart gets 90% width layout
                                chart_col, spacer_col = st.columns([9, 1])
                                
                                with chart_col:
                                    render_single_test_chart(test_type, filtered_emerson, bins)
                        else:
                            # If no specific Emerson tests found, show the first few available - each at 90% width
                            display_types = available_test_types[:4]  # Show up to 4 test types
                            for i, test_type in enumerate(display_types):
                                if i > 0:
                                    st.write("")
                                
                                # Each chart gets 90% width layout
                                chart_col, spacer_col = st.columns([9, 1])
                                
                                with chart_col:
                                    render_single_test_chart(test_type, filtered_emerson, bins)
                    else:
                        st.info("No chainage data available for distribution analysis")
                else:
                    st.info("No test data available for distribution analysis")
            else:
                st.info("Chainage column not found - cannot create spatial distribution")
                
        except Exception as e:
            st.warning(f"Could not generate Emerson distribution chart: {str(e)}")
        
        # Emerson Class Distribution (maintain existing functionality)
        st.markdown("### Emerson Class Distribution")
        
        # Class distribution
        if 'Emerson class' in filtered_emerson.columns:
            class_counts = filtered_emerson['Emerson class'].value_counts().sort_index()
            if not class_counts.empty:
                if HAS_PLOTLY:
                    fig = px.bar(
                        x=class_counts.index,
                        y=class_counts.values,
                        title="Emerson Tests by Class",
                        labels={'x': 'Emerson Class', 'y': 'Number of Tests'}
                    )
                    fig.update_layout(height=300, margin={"r":0,"t":30,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("**Emerson Class Distribution:**")
                    for emerson_class, count in class_counts.items():
                        st.write(f"- Class {emerson_class}: {count} tests")
        
        # Geology distribution
        if 'Geology_Orgin' in filtered_emerson.columns:
            geology_counts = filtered_emerson['Geology_Orgin'].value_counts()
            if not geology_counts.empty:
                if HAS_PLOTLY:
                    fig = px.pie(
                        values=geology_counts.values,
                        names=geology_counts.index,
                        title="Emerson Tests by Geology Origin"
                    )
                    fig.update_layout(height=400, margin={"r":0,"t":30,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("**Geology Origin Distribution:**")
                    for geology, count in geology_counts.items():
                        st.write(f"- {geology}: {count} tests")
        
        # Add visual separator before plot summary
        st.divider()
        
        # Plot Summary (matching PSD/Atterberg/SPT structure)
        st.markdown("**Plot Summary**")
        try:
            # Calculate engineering-relevant Emerson statistics
            summary_data = []
            
            # Use the actual plotted data for summary
            if not filtered_emerson.empty and 'Emerson class' in filtered_emerson.columns:
                emerson_values = filtered_emerson['Emerson class'].dropna()
                
                if not emerson_values.empty:
                    # Basic Emerson statistics
                    summary_data.extend([
                        {'Parameter': 'Total Emerson Tests', 'Value': f"{len(emerson_values):,}"},
                        {'Parameter': 'Mean Class', 'Value': f"{emerson_values.mean():.2f}"},
                        {'Parameter': 'Median Class', 'Value': f"{emerson_values.median():.1f}"},
                        {'Parameter': 'Standard Deviation', 'Value': f"{emerson_values.std():.2f}"},
                        {'Parameter': 'Range (Min-Max)', 'Value': f"{emerson_values.min():.0f} - {emerson_values.max():.0f}"}
                    ])
                    
                    # Depth information
                    if 'From_mbgl' in filtered_emerson.columns:
                        depth_data = filtered_emerson['From_mbgl'].dropna()
                        if not depth_data.empty:
                            summary_data.append({
                                'Parameter': 'Depth Range (m)', 
                                'Value': f"{depth_data.min():.1f} - {depth_data.max():.1f}"
                            })
                    
                    # Dispersive risk assessment (Class 4+ indicates dispersive potential)
                    dispersive_count = len(emerson_values[emerson_values >= 4])
                    non_dispersive_count = len(emerson_values[emerson_values < 4])
                    total_tests = len(emerson_values)
                    
                    dispersive_percentage = (dispersive_count / total_tests) * 100 if total_tests > 0 else 0
                    non_dispersive_percentage = (non_dispersive_count / total_tests) * 100 if total_tests > 0 else 0
                    
                    summary_data.extend([
                        {'Parameter': 'Non-Dispersive (Class 1-3)', 'Value': f"{non_dispersive_count} ({non_dispersive_percentage:.1f}%)"},
                        {'Parameter': 'Dispersive Risk (Class 4+)', 'Value': f"{dispersive_count} ({dispersive_percentage:.1f}%)"}
                    ])
                    
                    # Individual class distribution
                    class_counts = emerson_values.value_counts().sort_index()
                    for emerson_class, count in class_counts.items():
                        percentage = (count / total_tests) * 100
                        # Add descriptive labels for each class
                        class_descriptions = {
                            1: 'Non-Dispersive',
                            2: 'Intermediate', 
                            3: 'Intermediate',
                            4: 'Dispersive',
                            5: 'Dispersive',
                            6: 'Dispersive',
                            7: 'Dispersive',
                            8: 'Dispersive'
                        }
                        class_desc = class_descriptions.get(int(emerson_class), 'Unknown')
                        summary_data.append({
                            'Parameter': f'Class {int(emerson_class)} ({class_desc})',
                            'Value': f"{count} ({percentage:.1f}%)"
                        })
                    
                    # Geology origin breakdown if available
                    if 'Geology_Orgin' in filtered_emerson.columns:
                        geology_counts = filtered_emerson['Geology_Orgin'].value_counts()
                        for geology, count in geology_counts.items():
                            percentage = (count / len(filtered_emerson)) * 100
                            summary_data.append({
                                'Parameter': f'{geology} Tests',
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
        
        # Data preview and statistics options (matching PSD/Atterberg/SPT layout)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Show data preview", key="emerson_data_preview"):
                # Show relevant columns for Emerson analysis
                preview_cols = ['Hole_ID', 'From_mbgl', 'To_mbgl', emerson_col, 'Geology_Orgin']
                if 'Geology_Orgin' in filtered_emerson.columns:
                    preview_cols.append('Geology_Orgin')
                
                available_cols = [col for col in preview_cols if col in filtered_emerson.columns]
                st.dataframe(filtered_emerson[available_cols].head(20), use_container_width=True)
                st.caption(f"{len(filtered_emerson)} total records")
        
        with col2:
            if st.checkbox("Show detailed statistics", key="emerson_statistics"):
                if not filtered_emerson.empty and 'Emerson class' in filtered_emerson.columns:
                    # Calculate detailed Emerson statistics for advanced users
                    emerson_values = filtered_emerson['Emerson class'].dropna()
                    stats_data = []
                    
                    # Advanced statistics
                    if not emerson_values.empty:
                        # Percentiles
                        percentiles = [10, 25, 50, 75, 90]
                        for p in percentiles:
                            value = np.percentile(emerson_values, p)
                            stats_data.append({
                                'Parameter': f'{p}th Percentile',
                                'Value': f"{value:.1f}"
                            })
                        
                        # Additional statistics
                        stats_data.extend([
                            {'Parameter': 'Coefficient of Variation', 'Value': f"{(emerson_values.std()/emerson_values.mean())*100:.1f}%"},
                            {'Parameter': 'Skewness', 'Value': f"{emerson_values.skew():.2f}"},
                            {'Parameter': 'Kurtosis', 'Value': f"{emerson_values.kurtosis():.2f}"}
                        ])
                        
                        # Risk assessment details
                        very_high_risk = len(emerson_values[emerson_values >= 6])
                        high_risk = len(emerson_values[(emerson_values >= 4) & (emerson_values < 6)])
                        low_risk = len(emerson_values[emerson_values < 4])
                        
                        total = len(emerson_values)
                        stats_data.extend([
                            {'Parameter': 'Low Risk (Class 1-3)', 'Value': f"{low_risk} ({(low_risk/total)*100:.1f}%)"},
                            {'Parameter': 'High Risk (Class 4-5)', 'Value': f"{high_risk} ({(high_risk/total)*100:.1f}%)"},
                            {'Parameter': 'Very High Risk (Class 6+)', 'Value': f"{very_high_risk} ({(very_high_risk/total)*100:.1f}%)"}
                        ])
                    
                    # Create detailed statistics table
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No Emerson data available for detailed statistics")
    
    except Exception as e:
        st.error(f"Error in Emerson analysis: {str(e)}")
        st.error("Please check that your data contains valid Emerson columns")