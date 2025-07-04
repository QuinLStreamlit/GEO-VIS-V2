"""
Chainage Analysis Module

This module handles engineering property vs chainage analysis for infrastructure projects,
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
    
    from plot_by_chainage import plot_by_chainage
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

# Import data processing utilities
try:
    from .data_processing import get_numerical_properties_smart as get_numerical_properties
    HAS_DATA_PROCESSING = True
except ImportError:
    try:
        from data_processing import get_numerical_properties_smart as get_numerical_properties
        HAS_DATA_PROCESSING = True
    except ImportError:
        HAS_DATA_PROCESSING = False
        # Fallback function
        def get_numerical_properties(df: pd.DataFrame, include_spatial: bool = False) -> List[str]:
            """Get numerical columns that could be used for analysis."""
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ID and depth columns
            exclude_patterns = ['hole_id', 'from_', 'to_', 'depth', 'chainage'] if not include_spatial else ['hole_id']
            return [col for col in numerical_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]

# Import spatial utilities
try:
    from .common_utility_tool import calculate_map_zoom_and_center
    HAS_SPATIAL_UTILS = True
except ImportError:
    try:
        from common_utility_tool import calculate_map_zoom_and_center
        HAS_SPATIAL_UTILS = True
    except ImportError:
        HAS_SPATIAL_UTILS = False
        # Fallback function
        def calculate_map_zoom_and_center(lat_data, lon_data):
            """Calculate appropriate zoom level and center point for map based on data bounds."""
            import numpy as np
            lat_center = float(np.mean(lat_data))
            lon_center = float(np.mean(lon_data))
            max_span = max(float(np.max(lat_data) - np.min(lat_data)), float(np.max(lon_data) - np.min(lon_data)))
            zoom_level = 12 if max_span <= 0.05 else 8 if max_span <= 0.5 else 6 if max_span <= 2 else 4
            return zoom_level, {'lat': lat_center, 'lon': lon_center}


def filter_valid_chainage_data(df: pd.DataFrame, property_col: str) -> pd.DataFrame:
    """
    Filter dataframe to only include rows with valid Chainage and property data.
    
    Args:
        df: Input dataframe
        property_col: Property column name to validate
        
    Returns:
        Filtered dataframe with valid data only
    """
    if 'Chainage' not in df.columns:
        return pd.DataFrame()
    
    # Remove rows with missing essential data
    valid_data = df.dropna(subset=['Chainage', property_col])
    
    # Remove any infinite or invalid values
    valid_data = valid_data[
        np.isfinite(valid_data['Chainage']) & 
        np.isfinite(valid_data[property_col])
    ]
    
    return valid_data


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
                            try:
                                from .plot_download_simple import create_simple_download_button
                                create_simple_download_button("property_vs_chainage_tab", "tab", fig=current_fig)
                            except ImportError:
                                pass
                            
                            # Add map visualization before summary (following enhanced tab pattern)
                            try:
                                render_property_chainage_map_visualization(valid_data, selected_property)
                            except:
                                pass  # Map visualization is optional
                            
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
                        try:
                            for zone_name, (start, end) in zonage.items():
                                zone_data = valid_data[(valid_data['Chainage'] >= start) & (valid_data['Chainage'] <= end)]
                                if not zone_data.empty:
                                    stats_data.append({
                                        'Parameter': f'{zone_name} Count',
                                        'Value': f"{len(zone_data):,}"
                                    })
                        except:
                            pass  # If zonage not defined, skip zone stats
                        
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