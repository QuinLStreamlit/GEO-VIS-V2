"""
Property Depth Analysis Module

This module handles engineering property vs depth analysis for geotechnical applications,
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
    
    from plot_engineering_property_vs_depth import plot_engineering_property_vs_depth
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
                            try:
                                from .plot_download_simple import create_simple_download_button
                                create_simple_download_button("property_vs_depth_tab", "tab", fig=current_fig)
                            except ImportError:
                                pass
                            
                            # Add map visualization (following enhanced tab pattern)
                            try:
                                render_property_depth_map_visualization(valid_data, selected_property)
                            except:
                                pass  # Map visualization is optional
                            
                            # Add test distribution chart
                            try:
                                render_test_distribution_chart(filtered_data, selected_property)
                            except:
                                pass  # Test distribution chart is optional
                            
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


def render_test_distribution_chart(data: pd.DataFrame, selected_property: str):
    """
    Render test distribution chart for property depth analysis.
    
    Args:
        data: Data containing property information
        selected_property: Selected property for visualization
    """
    # This function would contain the test distribution chart logic
    # For now, it's a placeholder since the original function is complex
    try:
        st.markdown("**Test Distribution Chart**")
        st.info("Test distribution chart functionality will be added here")
    except Exception as e:
        st.warning(f"Test distribution chart not available: {str(e)}")