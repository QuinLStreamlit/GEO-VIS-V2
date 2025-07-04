"""
Histogram Analysis Module

This module handles comprehensive histogram analysis using original plotting functions 
from Functions folder exactly as in Jupyter notebook.
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
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    from .data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_numerical_properties_smart, get_id_columns_from_data
    from .common_utility_tool import get_default_parameters, get_color_schemes
    from .dashboard_materials import store_material_plot
    HAS_PLOTTING_UTILS = True
except ImportError:
    # For standalone testing
    try:
        from data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns, get_numerical_properties_smart, get_id_columns_from_data
        from common_utility_tool import get_default_parameters, get_color_schemes
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
    
    from plot_histogram import plot_histogram
    HAS_FUNCTIONS = True
except ImportError as e:
    HAS_FUNCTIONS = False
    print(f"Warning: Could not import histogram Functions: {e}")

# Import shared utilities
try:
    from .common_utility_tool import get_numerical_properties, get_categorical_properties, parse_tuple, find_map_symbol_column
except ImportError:
    try:
        from common_utility_tool import get_numerical_properties, get_categorical_properties, parse_tuple, find_map_symbol_column
    except ImportError:
        # Local fallback implementations
        def get_numerical_properties(df: pd.DataFrame) -> List[str]:
            """Get numerical properties for analysis."""
            return [col for col in df.columns 
                   if df[col].dtype in ['int64', 'float64', 'int32', 'float32'] 
                   and col not in ['Hole_ID', 'From_mbgl', 'To_mbgl']]
        
        def get_categorical_properties(df: pd.DataFrame) -> List[str]:
            """Get categorical properties for analysis."""
            return [col for col in df.columns 
                   if df[col].dtype == 'object' 
                   and col not in ['Hole_ID']]
        
        def parse_tuple(value_str: str, default: tuple) -> tuple:
            """Parse tuple string safely with fallback to default."""
            try:
                if value_str.strip().startswith('(') and value_str.strip().endswith(')'):
                    tuple_content = value_str.strip()[1:-1]
                    values = [float(x.strip()) for x in tuple_content.split(',')]
                    return tuple(values)
                else:
                    values = [float(x.strip()) for x in value_str.split(',')]
                    return tuple(values)
            except:
                return default
        
        def find_map_symbol_column(df: pd.DataFrame) -> Optional[str]:
            """Find map symbol column."""
            for col in df.columns:
                if 'map' in col.lower() and 'symbol' in col.lower():
                    return col
            return None


def get_numerical_properties_smart(df: pd.DataFrame, include_spatial: bool = True) -> List[str]:
    """
    Get list of numerical properties suitable for histogram analysis.
    Uses intelligent property detection with flexible regex patterns.
    
    Args:
        df: DataFrame to analyze
        include_spatial: Whether to include spatial columns
        
    Returns:
        List[str]: List of numerical column names
    """
    # Use the new smart property detection system
    try:
        from .data_processing import get_numerical_properties_smart as smart_props
        return smart_props(df, include_spatial=include_spatial)
    except ImportError:
        try:
            from data_processing import get_numerical_properties_smart as smart_props
            return smart_props(df, include_spatial=include_spatial)
        except ImportError:
            # Fallback to basic implementation
            return get_numerical_properties(df)


def render_comprehensive_histograms_tab(filtered_data: pd.DataFrame):
    """
    Render the comprehensive histograms analysis tab in Streamlit.
    Uses original plotting functions from Functions folder exactly as in Jupyter notebook.
    
    Args:
        filtered_data: Filtered laboratory data
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render tab.")
        return
    
    if not HAS_FUNCTIONS:
        st.error("Functions folder not accessible")
        return
    
    # Get available properties
    numerical_props = get_numerical_properties_smart(filtered_data, include_spatial=True)
    categorical_props = get_categorical_properties(filtered_data)
    
    # Get dynamic ID columns for grouping/filtering
    try:
        from .data_processing import get_id_columns_from_data
        id_columns = get_id_columns_from_data(filtered_data)
    except ImportError:
        from data_processing import get_id_columns_from_data
        id_columns = get_id_columns_from_data(filtered_data)
    
    if not numerical_props:
        st.warning("No numerical properties available.")
        return
    
    # Helper function for tuple parsing
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
    
    # Enhanced parameter box (4-row main + advanced section)
    with st.expander("Plot Parameters", expanded=True):
        
        # Row 1: Basic Column Setup
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Set WPI as default if available, otherwise use first property
            default_properties = []
            for prop in numerical_props:
                if 'WPI' in prop.upper():
                    default_properties = [prop]
                    break
            if not default_properties and numerical_props:
                default_properties = [numerical_props[0]]
            
            selected_properties = st.multiselect(
                "Properties",
                options=numerical_props,
                default=default_properties,
                key="histogram_properties",
                help="Select numerical columns to plot as histograms"
            )
        with col2:
            facet_option = st.selectbox(
                "Facet By",
                options=["None"] + id_columns,
                index=0,
                key="histogram_facet",
                help="Split data into separate subplots by ID column"
            )
            facet_col = None if facet_option == "None" else facet_option
        with col3:
            facet_orientation = st.selectbox(
                "Facet Layout",
                options=['vertical', 'horizontal'],
                index=0,
                key="histogram_facet_orientation",
                help="Arrangement of subplots when using Facet By"
            )
        with col4:
            stack_option = st.selectbox(
                "Stack By",
                options=["None"] + id_columns,
                index=0,
                key="histogram_stack",
                help="Stack histogram bars by ID column"
            )
            stack_col = None if stack_option == "None" else stack_option
        with col5:
            # Default bins mode is Simple (hidden from UI)
            bins_mode = "Simple"  # Always use Simple mode
            
            # Show Plot Style instead
            show_minor_ticks_option = st.selectbox(
                "Minor Ticks",
                options=["No", "Yes"],
                index=0,
                key="histogram_minor_ticks",
                help="Display small tick marks between major ticks"
            )
        
        # Row 2: Filtering with Dynamic Dropdowns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Get available columns for filtering (ID columns + key spatial columns)
        spatial_columns = [col for col in numerical_props if col in ['From_mbgl', 'To_mbgl', 'Chainage']]
        filter_columns = id_columns + spatial_columns
        
        with col1:
            filter1_by = st.selectbox(
                "Filter 1 By",
                options=["None"] + filter_columns,
                index=0,
                key="histogram_filter1_by"
            )
        with col2:
            # Dynamic filter1 values based on selected column
            if filter1_by and filter1_by != "None" and filter1_by in filtered_data.columns:
                if filtered_data[filter1_by].dtype in ['object', 'string']:
                    unique_values = ["Manual Entry"] + sorted(filtered_data[filter1_by].dropna().unique().astype(str).tolist())
                    filter1_dropdown = st.selectbox(
                        "Filter 1 Value",
                        options=unique_values,
                        index=0,
                        key="histogram_filter1_dropdown"
                    )
                    if filter1_dropdown == "Manual Entry":
                        filter1_value = st.text_input(
                            "Custom Value",
                            value="",
                            key="histogram_filter1_custom"
                        )
                    else:
                        filter1_value = filter1_dropdown
                else:
                    filter1_value = st.text_input(
                        "Filter 1 Value",
                        value="",
                        key="histogram_filter1_value",
                        help="Enter numerical value"
                    )
            else:
                filter1_value = st.text_input(
                    "Filter 1 Value",
                    value="",
                    key="histogram_filter1_value"
                )
        with col3:
            filter2_by = st.selectbox(
                "Filter 2 By", 
                options=["None"] + filter_columns,
                index=0,
                key="histogram_filter2_by"
            )
        with col4:
            # Dynamic filter2 values based on selected column
            if filter2_by and filter2_by != "None" and filter2_by in filtered_data.columns:
                if filtered_data[filter2_by].dtype in ['object', 'string']:
                    unique_values = ["Manual Entry"] + sorted(filtered_data[filter2_by].dropna().unique().astype(str).tolist())
                    filter2_dropdown = st.selectbox(
                        "Filter 2 Value",
                        options=unique_values,
                        index=0,
                        key="histogram_filter2_dropdown"
                    )
                    if filter2_dropdown == "Manual Entry":
                        filter2_value = st.text_input(
                            "Custom Value",
                            value="",
                            key="histogram_filter2_custom"
                        )
                    else:
                        filter2_value = filter2_dropdown
                else:
                    filter2_value = st.text_input(
                        "Filter 2 Value",
                        value="",
                        key="histogram_filter2_value",
                        help="Enter numerical value"
                    )
            else:
                filter2_value = st.text_input(
                    "Filter 2 Value",
                    value="",
                    key="histogram_filter2_value"
                )
        with col5:
            show_stats_option = st.selectbox(
                "Show Statistics", 
                options=["No", "Yes"],
                index=1,  # Default to "Yes"
                key="histogram_stats",
                help="Display statistical summary on plots"
            )
            show_stats = show_stats_option == "Yes"
        
        # Row 3: Bins & Tick Configuration
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Simple bins mode only (always)
            bins_count = st.number_input(
                "Bins Count",
                min_value=5,
                max_value=100,
                value=15,
                step=5,
                key="histogram_bins_simple"
            )
            bins_value = bins_count
            
        with col2:
            # X-tick interval control - simple number input
            xtick_interval_input = st.text_input(
                "X-Tick Interval",
                value="",
                key="histogram_xtick_interval",
                help="Leave empty for auto, or enter interval value (e.g., 5, 10, 0.5)"
            )
            
            # Process the input: empty = auto, value = custom interval
            if xtick_interval_input.strip():
                try:
                    xtick_interval = float(xtick_interval_input.strip())
                except ValueError:
                    xtick_interval = None
                    st.error("Invalid number format")
            else:
                xtick_interval = None  # Auto mode
        with col3:
            density_option = st.selectbox(
                "Density", 
                options=["No", "Yes"],
                index=0,  # Default to "No"
                key="histogram_density",
                help="Show probability density instead of frequency counts"
            )
            density = density_option == "Yes"
        with col4:
            alpha_value = st.number_input(
                "Alpha",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.1,
                format="%.1f",
                key="histogram_alpha",
                help="Bar transparency: 0.1=very transparent, 1.0=solid"
            )
        with col5:
            cmap_name = st.selectbox(
                "Color Scheme",
                options=['tab10', 'viridis', 'Set1', 'Set2', 'Set3', 'plasma', 'inferno', 'magma'],
                index=0,
                key="histogram_cmap",
                help="Color palette for grouped histograms"
            )
        
        # Row 4: Display & Layout
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            figsize_str = st.text_input(
                "Figure Size",
                value="(12, 7)",
                key="histogram_figsize",
                help="Plot dimensions in inches: (width, height)"
            )
        with col2:
            subplot_title_option = st.selectbox(
                "Subplot Title",
                options=["Auto", "Property Name", "Custom", "None"],
                index=0,
                key="histogram_subtitle",
                help="How to display subplot titles"
            )
        with col3:
            show_legend_option = st.selectbox(
                "Show Legend", 
                options=["No", "Yes"],
                index=1,  # Default to "Yes"
                key="histogram_legend",
                help="Display legend for grouped data"
            )
            show_legend = show_legend_option == "Yes"
        with col4:
            show_grid_option = st.selectbox(
                "Show Grid", 
                options=["No", "Yes"],
                index=1,  # Default to "Yes"
                key="histogram_grid",
                help="Display background grid lines"
            )
            show_grid = show_grid_option == "Yes"
        with col5:
            # Xtick interval control
            integer_xticks_option = st.selectbox(
                "Integer X-Ticks",
                options=["No", "Yes"],
                index=0,
                key="histogram_integer_xticks",
                help="Force only whole numbers on x-axis labels"
            )
    
    # Compact Advanced Parameters Section
    with st.expander("Advanced Parameters", expanded=False):
        # Row 1: Axis & Limits
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            xlim_str = st.text_input("X-Axis Limits", value="", key="histogram_xlim", help="Set x-axis range. Format: min,max (e.g., 0,100). Leave empty for auto")
        with col2:
            ylim_str = st.text_input("Y-Axis Limits", value="", key="histogram_ylim", help="Set y-axis range. Format: min,max (e.g., 0,50). Leave empty for auto")
        with col3:
            log_scale_option = st.selectbox("Log Scale", options=["None", "X-axis", "Y-axis", "Both"], index=0, key="histogram_log_scale", help="Apply logarithmic scaling to axes")
        with col4:
            bar_width = st.number_input("Bar Width", min_value=0.5, max_value=1.0, value=0.8, step=0.1, format="%.1f", key="histogram_bar_width", help="Width of histogram bars: 0.5=narrow, 1.0=wide")
        
        # Row 2: Grid & Tick Controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            grid_axis = st.selectbox("Grid Axis", options=["y", "x", "both"], index=0, key="histogram_grid_axis", help="Which axes show grid lines")
        with col2:
            grid_linestyle = st.selectbox("Grid Style", options=["--", "-", ":", "-."], index=0, key="histogram_grid_linestyle", help="Line style for grid: --=dashed, -=solid, :=dotted, -.=dash-dot")
        with col3:
            grid_alpha = st.number_input("Grid Alpha", min_value=0.1, max_value=1.0, value=0.4, step=0.1, format="%.1f", key="histogram_grid_alpha", help="Grid transparency: 0.1=faint, 1.0=solid")
        with col4:
            tick_direction = st.selectbox("Tick Direction", options=["out", "in", "inout"], index=0, key="histogram_tick_direction", help="Direction of tick marks relative to plot area")
        
        # Row 3: Stats & Legend Positioning
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stats_location_mode = st.selectbox("Stats Mode", options=["Preset", "Custom"], index=0, key="histogram_stats_mode", help="Use preset locations or custom coordinates")
        with col2:
            if stats_location_mode == "Preset":
                stats_location = st.selectbox("Stats Location", options=["best", "upper right", "upper left", "lower right", "lower left"], index=0, key="histogram_stats_location", help="Where to place statistics box")
            else:
                stats_bbox_str = st.text_input("Stats Bbox (x,y)", value="0.98,0.98", key="histogram_stats_bbox", help="Custom position: x,y coordinates from 0-1")
                stats_location = "upper right"
        with col3:
            legend_location_mode = st.selectbox("Legend Mode", options=["Preset", "Custom"], index=0, key="histogram_legend_mode", help="Use preset locations or custom coordinates")
        with col4:
            if legend_location_mode == "Preset":
                legend_loc = st.selectbox("Legend Location", options=["best", "upper right", "upper left", "lower right", "lower left", "center"], index=0, key="histogram_legend_loc", help="Where to place legend")
            else:
                legend_bbox_str = st.text_input("Legend Bbox (x,y)", value="1.02,1.0", key="histogram_legend_bbox", help="Custom position: x,y coordinates from 0-1")
                legend_loc = "best"
        
        # Row 4: Font Controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fig_title_fontsize = st.number_input("Title Font", min_value=8, max_value=24, value=15, step=1, key="histogram_title_fontsize", help="Font size for main plot title")
        with col2:
            label_fontsize = st.number_input("Label Font", min_value=8, max_value=20, value=13, step=1, key="histogram_label_fontsize", help="Font size for axis labels")
        with col3:
            tick_fontsize = st.number_input("Tick Font", min_value=6, max_value=16, value=12, step=1, key="histogram_tick_fontsize", help="Font size for axis tick numbers")
        with col4:
            legend_fontsize = st.number_input("Legend Font", min_value=6, max_value=16, value=11, step=1, key="histogram_legend_fontsize", help="Font size for legend text")
        
        # Row 5: Subplot & Styling
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sharex_option = st.selectbox("Share X-Axis", options=["No", "Yes"], index=0, key="histogram_sharex", help="Use same x-axis scale across all subplots")
        with col2:
            sharey_option = st.selectbox("Share Y-Axis", options=["No", "Yes"], index=0, key="histogram_sharey", help="Use same y-axis scale across all subplots")
        with col3:
            custom_ylabel = st.text_input("Custom Y-Label", value="", key="histogram_custom_ylabel", help="Override default y-axis label. Leave empty for auto")
        with col4:
            hist_linewidth = st.number_input("Edge Width", min_value=0.0, max_value=3.0, value=0.6, step=0.1, format="%.1f", key="histogram_line_width", help="Thickness of bar outline: 0=no outline, 3=thick outline")
        
        # Row 6: Additional Controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.empty()
        with col2:
            st.empty()
        with col3:
            st.empty()
        with col4:
            st.empty()
    
    # Enhanced parameter parsing and validation
    def parse_axis_limits(limit_str):
        """Parse axis limits from string format 'min,max'"""
        if not limit_str.strip():
            return None
        try:
            parts = [float(x.strip()) for x in limit_str.split(',')]
            if len(parts) == 2:
                return tuple(parts)
        except:
            pass
        return None
    
    def parse_bbox_coords(bbox_str):
        """Parse bbox coordinates from string format 'x,y'"""
        if not bbox_str.strip():
            return None
        try:
            parts = [float(x.strip()) for x in bbox_str.split(',')]
            if len(parts) == 2:
                return tuple(parts)
        except:
            pass
        return None
    
    # Parse enhanced parameters
    xlim = parse_axis_limits(xlim_str) if 'xlim_str' in locals() else None
    ylim = parse_axis_limits(ylim_str) if 'ylim_str' in locals() else None
    
    # Parse bbox coordinates for custom positioning
    stats_bbox = None
    legend_bbox = None
    if 'stats_location_mode' in locals() and stats_location_mode == "Custom Bbox":
        stats_bbox = parse_bbox_coords(stats_bbox_str)
    if 'legend_location_mode' in locals() and legend_location_mode == "Custom Bbox":
        legend_bbox = parse_bbox_coords(legend_bbox_str)
    
    # Convert UI values to function parameters
    log_scale_map = {"None": False, "X-axis": "x", "Y-axis": "y", "Both": "both"}
    log_scale_value = log_scale_map[log_scale_option]
    
    # Convert boolean options
    sharex_value = sharex_option == "Yes"
    sharey_value = sharey_option == "Yes"
    integer_xticks_value = integer_xticks_option == "Yes"
    
    # Use default color for non-grouped plots (hue_col removed)
    use_color_override = not (stack_col or facet_col)
    final_color = "#0072B2" if use_color_override else None
    hue_col = None  # Hue option removed
    
    # Custom ylabel (only if provided)
    final_custom_ylabel = custom_ylabel.strip() if 'custom_ylabel' in locals() and custom_ylabel.strip() else None
    
    # Process bins configuration (Simple mode only)
    final_bins = bins_value if 'bins_value' in locals() else bins_count
    
    # Convert minor ticks and xtick interval options
    show_minor_ticks_value = show_minor_ticks_option == "Yes" if 'show_minor_ticks_option' in locals() else False
    
    # Process xtick_interval input
    if 'xtick_interval_input' in locals() and xtick_interval_input.strip():
        try:
            xtick_interval = float(xtick_interval_input.strip())
        except ValueError:
            xtick_interval = None
            # Note: Error already shown in UI
    else:
        xtick_interval = None  # Auto mode
    
    # Data filtering and processing
    plot_data = filtered_data.copy()
    
    # Apply custom filters
    if filter1_by and filter1_by != 'None' and filter1_value:
        if filter1_by in plot_data.columns:
            if plot_data[filter1_by].dtype in ['object', 'string']:
                plot_data = plot_data[plot_data[filter1_by].astype(str).str.contains(str(filter1_value), case=False, na=False)]
            else:
                try:
                    filter_val = float(filter1_value)
                    plot_data = plot_data[plot_data[filter1_by] == filter_val]
                except ValueError:
                    pass
    
    if filter2_by and filter2_by != 'None' and filter2_value:
        if filter2_by in plot_data.columns:
            if plot_data[filter2_by].dtype in ['object', 'string']:
                plot_data = plot_data[plot_data[filter2_by].astype(str).str.contains(str(filter2_value), case=False, na=False)]
            else:
                try:
                    filter_val = float(filter2_value)
                    plot_data = plot_data[plot_data[filter2_by] == filter_val]
                except ValueError:
                    pass
    
    # Generate title suffix based on applied filters
    title_suffix = ""
    if filter1_by and filter1_by != 'None' and filter1_value:
        title_suffix += f" | {filter1_by}: {filter1_value}"
    if filter2_by and filter2_by != 'None' and filter2_value:
        title_suffix += f" | {filter2_by}: {filter2_value}"
    
    # Parse tuple inputs and integrate with sidebar plot size
    base_figsize = parse_tuple(figsize_str, (12, 7))
    
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
    
    # Generate histograms
    if selected_properties:
        for i, prop in enumerate(selected_properties):
            if prop in plot_data.columns:
                valid_data = plot_data[prop].dropna()
                
                if len(valid_data) == 0:
                    continue
                
                # Add visual separator between different properties
                if i > 0:  # Don't add separator before the first plot
                    st.markdown("---")  # This creates a horizontal line
                    st.markdown("")  # Add some spacing
                
                # Add property title/header with enhanced title
                enhanced_title = f"Distribution of {prop}{title_suffix}"
                st.subheader(f"{prop}")
                
                try:
                    # Use Functions folder plot_histogram exactly as in Jupyter notebook with enhanced parameters
                    if HAS_MATPLOTLIB and HAS_FUNCTIONS:
                        # Clear any existing figures first
                        plt.close('all')
                        
                        plot_histogram(
                            data_df=plot_data,
                            value_cols=prop,
                            facet_col=facet_col,
                            facet_orientation=facet_orientation,
                            hue_col=None,  # Hue functionality removed
                            stack_col=stack_col,
                            bins=final_bins,
                            density=density,
                            kde=False,  # Removed from UI as requested
                            log_scale=log_scale_value,
                            show_stats=show_stats,
                            stats_location=stats_location,
                            integer_xticks=integer_xticks_value,
                            xtick_interval=xtick_interval,  # Added xtick interval control
                            tick_direction=tick_direction,
                            show_minor_ticks=show_minor_ticks_value,  # Added minor ticks control
                            sharex=sharex_value,
                            sharey=sharey_value,
                            xlim=xlim,
                            ylim=ylim,
                            alpha=alpha_value,
                            bar_width=bar_width,
                            color=final_color,
                            cmap_name=cmap_name,
                            hist_edgecolor='black',  # Default black edge color
                            hist_linewidth=hist_linewidth,
                            show_grid=show_grid,
                            grid_axis=grid_axis,
                            grid_linestyle=grid_linestyle,
                            grid_alpha=grid_alpha,
                            show_legend=show_legend,
                            legend_loc=legend_loc,
                            legend_bbox_to_anchor=legend_bbox,  # Enhanced bbox support
                            fig_title_fontsize=fig_title_fontsize,
                            label_fontsize=label_fontsize,
                            tick_fontsize=tick_fontsize,
                            legend_fontsize=legend_fontsize,
                            custom_ylabel=final_custom_ylabel,
                            figsize=figsize,
                            dpi=300,  # Default DPI
                            style='default',  # Default style
                            debug_mode=False,  # Default debug mode off
                            title=f"Histogram Diagram: {prop}{title_suffix}",  # Fixed title format
                            show_plot=False,
                            close_plot=False
                        )
                        
                        # Capture and display the figure in Streamlit with sidebar width control
                        current_fig = plt.gcf()
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
                            
                            # Store plot for dashboard
                            try:
                                import io
                                buf = io.BytesIO()
                                current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                store_material_plot(f'histogram_{prop}', buf)
                            except Exception as e:
                                pass
                        else:
                            success = False
                        
                        # Simple download button with figure reference
                        if success:
                            try:
                                from .plot_download_simple import create_simple_download_button
                                create_simple_download_button(f"histogram_{prop}", f"prop_{i}", fig=current_fig)
                            except ImportError:
                                try:
                                    from plot_download_simple import create_simple_download_button
                                    create_simple_download_button(f"histogram_{prop}", f"prop_{i}", fig=current_fig)
                                except ImportError:
                                    pass
                            
                            # Remove plot summary - statistics now shown via checkbox only
                            pass
                            
                            # Add chainage line chart underneath histogram (Data tab format)
                            if 'Chainage' in plot_data.columns:
                                st.markdown("")  # Add some spacing
                                st.markdown("**Test Distribution**")
                                
                                try:
                                    # Get chainage data for this property
                                    prop_chainage_data = plot_data[['Chainage', prop]].dropna()
                                    
                                    if not prop_chainage_data.empty:
                                        chainage_values = prop_chainage_data['Chainage']
                                        
                                        # Create bins like in Data tab (200m intervals)
                                        min_chainage = chainage_values.min()
                                        max_chainage = chainage_values.max()
                                        bin_interval = 200  # 200m intervals
                                        
                                        # Round down min_chainage and up max_chainage to nearest intervals
                                        bin_start = int(min_chainage // bin_interval) * bin_interval
                                        bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
                                        
                                        # Create bins at fixed intervals
                                        bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
                                        
                                        # Create histogram data
                                        hist, bin_edges = np.histogram(chainage_values, bins=bins)
                                        
                                        # Use bin centers as chainage values for x-axis
                                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                        
                                        # Create chart data
                                        chart_data = pd.DataFrame({
                                            'Chainage (m)': bin_centers.astype(int),
                                            f'{prop} Count': hist.astype(int)
                                        })
                                        
                                        if chart_data[f'{prop} Count'].sum() > 0:
                                            # Use 80% width and add axis labels
                                            col1, col2, col3 = st.columns([1, 8, 1])  # 80% width in middle
                                            with col2:
                                                st.line_chart(
                                                    chart_data.set_index('Chainage (m)'),
                                                    use_container_width=True
                                                )
                                        else:
                                            st.info(f"No {prop} data found in chainage bins")
                                    else:
                                        st.info(f"No {prop} data with chainage values")
                                        
                                except Exception as e:
                                    st.warning(f"Could not generate chainage chart: {str(e)}")
                            else:
                                st.info("Chainage column not available - spatial distribution chart not shown")
                        else:
                            st.warning("No plot generated - check data availability")
                    
                    # Data preview and statistics options underneath each plot
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.checkbox(f"Show data preview", key=f"preview_{prop}"):
                            preview_data = filtered_data[[prop]].dropna()
                            if facet_col and facet_col in filtered_data.columns:
                                preview_data = filtered_data[[prop, facet_col]].dropna()
                            if stack_col and stack_col in filtered_data.columns:
                                preview_data = filtered_data[[prop, stack_col]].dropna() if facet_col is None else filtered_data[[prop, facet_col, stack_col]].dropna()
                            
                            st.dataframe(preview_data.head(20), use_container_width=True)
                            st.caption(f"{len(preview_data)} total records")
                    
                    with col2:
                        if st.checkbox(f"Show statistics", key=f"stats_{prop}"):
                            prop_data = filtered_data[prop].dropna()
                            if not prop_data.empty:
                                # Calculate CV
                                mean_val = prop_data.mean()
                                cv = (prop_data.std() / mean_val) * 100 if mean_val != 0 else 0
                                
                                stats_data = []
                                stats_data.extend([
                                    {'Parameter': 'Total Values', 'Value': f"{len(prop_data):,}"},
                                    {'Parameter': 'Mean', 'Value': f"{mean_val:.2f}"},
                                    {'Parameter': 'Median', 'Value': f"{prop_data.median():.2f}"},
                                    {'Parameter': 'Std Dev', 'Value': f"{prop_data.std():.2f}"},
                                    {'Parameter': 'CV (%)', 'Value': f"{cv:.1f}"},  # Added CV
                                    {'Parameter': 'Min', 'Value': f"{prop_data.min():.2f}"},
                                    {'Parameter': 'Max', 'Value': f"{prop_data.max():.2f}"}
                                ])
                                
                                # Distribution by facet/stack if available
                                if facet_col and facet_col in filtered_data.columns:
                                    facet_counts = filtered_data[facet_col].value_counts()
                                    for facet, count in facet_counts.head(3).items():
                                        percentage = (count / len(filtered_data)) * 100
                                        stats_data.append({
                                            'Parameter': f'{facet}',
                                            'Value': f"{count} ({percentage:.1f}%)"
                                        })
                                
                                # Create statistics table
                                stats_df = pd.DataFrame(stats_data)
                                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No data available for statistics")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def render_histogram_plot_summary(plot_data: pd.DataFrame, property_name: str, facet_col: str = None, stack_col: str = None) -> None:
    """
    Render compact statistical summary for histogram analysis.
    """
    if plot_data.empty or property_name not in plot_data.columns:
        st.warning("No data available for summary")
        return
    
    # Get the property data
    prop_data = plot_data[property_name].dropna()
    
    if prop_data.empty:
        st.warning(f"No valid data for {property_name}")
        return
    
    # Calculate summary statistics
    mean_val = prop_data.mean()
    median_val = prop_data.median()
    std_val = prop_data.std()
    min_val = prop_data.min()
    max_val = prop_data.max()
    count = len(prop_data)
    cv = (std_val/mean_val)*100 if mean_val != 0 else 0
    
    # Display compact statistics in 4-column format
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Count", f"{count:,}")
        st.metric("Min", f"{min_val:.2f}")
    with col2:  
        st.metric("Mean", f"{mean_val:.2f}")
        st.metric("Max", f"{max_val:.2f}")
    with col3:
        st.metric("Median", f"{median_val:.2f}")
        st.metric("Range", f"{max_val - min_val:.2f}")
    with col4:
        st.metric("Std Dev", f"{std_val:.2f}")
        st.metric("CV (%)", f"{cv:.1f}")
    
    # Grouping analysis if facet or stack columns are used
    if facet_col and facet_col in plot_data.columns:
        st.write(f"**Analysis by {facet_col}:**")
        group_stats = plot_data.groupby(facet_col)[property_name].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        group_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max']
        st.dataframe(group_stats, use_container_width=True)
    
    if stack_col and stack_col != facet_col and stack_col in plot_data.columns:
        st.write(f"**Analysis by {stack_col}:**")
        group_stats = plot_data.groupby(stack_col)[property_name].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        group_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max']
        st.dataframe(group_stats, use_container_width=True)