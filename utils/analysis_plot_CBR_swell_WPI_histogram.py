"""
CBR vs WPI Analysis Module

This module handles CBR vs WPI correlation analysis using original plotting functions 
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
    
    from plot_CBR_swell_WPI_histogram import plot_CBR_swell_WPI_histogram
    HAS_FUNCTIONS = True
except ImportError as e:
    HAS_FUNCTIONS = False
    print(f"Warning: Could not import CBR/WPI Functions: {e}")

# Import shared utilities
try:
    from .common_utility_tool import get_numerical_properties, get_categorical_properties, parse_tuple, find_map_symbol_column, detect_cbr_swell_column, detect_wpi_column, get_cbr_swell_column_candidates, get_wpi_column_candidates
except ImportError:
    try:
        from common_utility_tool import get_numerical_properties, get_categorical_properties, parse_tuple, find_map_symbol_column, detect_cbr_swell_column, detect_wpi_column, get_cbr_swell_column_candidates, get_wpi_column_candidates
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
        
        def detect_cbr_swell_column(df: pd.DataFrame) -> Optional[str]:
            """Fallback CBR Swell detection."""
            for col in df.columns:
                col_lower = str(col).lower()
                if 'cbr' in col_lower and 'swell' in col_lower:
                    return col
            return None
        
        def detect_wpi_column(df: pd.DataFrame) -> Optional[str]:
            """Fallback WPI detection."""
            for col in df.columns:
                col_lower = str(col).lower()
                if 'wpi' in col_lower:
                    return col
            return None
        
        def get_cbr_swell_column_candidates(df: pd.DataFrame) -> List[str]:
            """Fallback CBR Swell candidates."""
            candidates = []
            for col in df.columns:
                col_lower = str(col).lower()
                if 'cbr' in col_lower:
                    candidates.append(col)
            return candidates
        
        def get_wpi_column_candidates(df: pd.DataFrame) -> List[str]:
            """Fallback WPI candidates."""
            candidates = []
            for col in df.columns:
                col_lower = str(col).lower()
                if 'wpi' in col_lower:
                    candidates.append(col)
            return candidates


def prepare_cbr_wpi_data(df: pd.DataFrame, depth_cut: Optional[float] = None, 
                        cbr_swell_col: Optional[str] = None, wpi_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Prepare CBR Swell and WPI data following the exact Jupyter notebook workflow.
    
    This function replicates the golden standard data processing:
    1. Smart detection or user-specified CBR Swell and WPI columns
    2. Extract CBR Swell and WPI test datasets separately
    3. Add category columns using golden standard thresholds from Jupyter notebook
    4. Add Cut Category based on depth threshold (if provided)
    5. Select ['Geology_Orgin', 'category', map_symbol, 'Cut_Category'] columns (if available)
    6. Add 'Name' column to identify each dataset (CBR Swell (%) and WPI)
    7. Concatenate into single dataframe for plotting
    
    Args:
        df: Main laboratory data DataFrame
        depth_cut: Optional depth threshold (mbgl) to create Above Cut/Below Cut categories
        cbr_swell_col: User-specified CBR Swell column name (if None, uses smart detection)
        wpi_col: User-specified WPI column name (if None, uses smart detection)
        
    Returns:
        pd.DataFrame: Combined processed dataframe ready for plotting, or None if no data
    """
    try:
        # Get dynamic ID columns
        id_columns = get_id_columns_from_data(df)
    except:
        id_columns = ['Hole_ID', 'Type', 'From_mbgl', 'To_mbgl']
    
    # Find map symbol column
    map_symbol_col = find_map_symbol_column(df)
    
    # Add Cut Category if depth_cut is provided
    if depth_cut is not None and 'From_mbgl' in df.columns:
        df = df.copy()  # Avoid modifying original dataframe
        df['Cut_Category'] = df['From_mbgl'].apply(
            lambda x: 'Above Cut' if pd.notna(x) and x < depth_cut else 'Below Cut' if pd.notna(x) else None
        )
    
    # Step 1: Smart detection or use user-specified columns
    combined_datasets = []
    
    # Handle WPI column detection/selection
    if wpi_col is None:
        wpi_col = detect_wpi_column(df)
    
    if wpi_col and wpi_col in df.columns:
        # Create WPI dataset equivalent to WPI_final.copy() from Jupyter notebook
        wpi_data = df[df[wpi_col].notna()].copy()
        
        if not wpi_data.empty:
            # Add 'category' column using golden standard WPI thresholds (exactly from Jupyter notebook)
            def categorize_wpi(value):
                try:
                    val = float(value)
                    # Golden standard thresholds from Jupyter notebook
                    if val > 4200:
                        return 'Extreme'
                    elif val > 3200:
                        return 'Very high'
                    elif val > 2200:
                        return 'High'
                    elif val > 1200:
                        return 'Moderate'
                    else:
                        return 'Low'
                except:
                    return None
            
            wpi_data['category'] = wpi_data[wpi_col].apply(categorize_wpi)
            wpi_data['Name'] = 'WPI'  # Matches Jupyter notebook
            
            # Keep essential columns following Jupyter notebook pattern
            essential_cols = ['Name', 'category']
            if 'Geology_Orgin' in wpi_data.columns:
                essential_cols.append('Geology_Orgin')
            if map_symbol_col and map_symbol_col in wpi_data.columns:
                essential_cols.append(map_symbol_col)
            if 'Cut_Category' in wpi_data.columns and depth_cut is not None:
                essential_cols.append('Cut_Category')
            
            # Select only columns that exist
            final_cols = [col for col in essential_cols if col in wpi_data.columns]
            wpi_data = wpi_data[final_cols]
            combined_datasets.append(wpi_data)
    
    # Handle CBR Swell column detection/selection
    if cbr_swell_col is None:
        cbr_swell_col = detect_cbr_swell_column(df)
    
    if cbr_swell_col and cbr_swell_col in df.columns:
        # Create CBR Swell dataset equivalent to CBR_GIR from Jupyter notebook
        cbr_data = df[df[cbr_swell_col].notna()].copy()
        
        if not cbr_data.empty:
            # Add 'category' column using golden standard CBR Swell thresholds (exactly from Jupyter notebook)
            def categorize_cbr_swell(value):
                try:
                    val = float(value)
                    # Golden standard CBR Swell thresholds from Jupyter notebook
                    if val > 10:
                        return 'Extreme'
                    elif val > 5:
                        return 'Very high'
                    elif val > 2.5:
                        return 'High'
                    elif val > 0.5:
                        return 'Moderate'
                    else:
                        return 'Low'
                except:
                    return None
            
            cbr_data['category'] = cbr_data[cbr_swell_col].apply(categorize_cbr_swell)
            cbr_data['Name'] = 'CBR Swell (%)'  # Matches Jupyter notebook exactly
            
            # Keep essential columns following Jupyter notebook pattern
            essential_cols = ['Name', 'category']
            if 'Geology_Orgin' in cbr_data.columns:
                essential_cols.append('Geology_Orgin')
            if map_symbol_col and map_symbol_col in cbr_data.columns:
                essential_cols.append(map_symbol_col)
            if 'Cut_Category' in cbr_data.columns and depth_cut is not None:
                essential_cols.append('Cut_Category')
            
            # Select only columns that exist
            final_cols = [col for col in essential_cols if col in cbr_data.columns]
            cbr_data = cbr_data[final_cols]
            combined_datasets.append(cbr_data)
    
    # Combine datasets following Jupyter notebook pattern
    if not combined_datasets:
        return None
    
    final_df = pd.concat(combined_datasets, ignore_index=True)
    return final_df if not final_df.empty else None


def render_cbr_wpi_test_distribution(filtered_data: pd.DataFrame):
    """
    Render test distribution visualization for CBR and WPI.
    Shows spatial distribution of tests across chainage.
    """
    if not HAS_STREAMLIT:
        return
    
    st.subheader("CBR and WPI Test Distribution")
    
    # Check for required columns
    if 'Chainage' not in filtered_data.columns:
        st.warning("Chainage column not found - cannot create spatial distribution")
        return
    
    # Find CBR and WPI columns
    cbr_col = None
    wpi_col = None
    
    for col in filtered_data.columns:
        if 'CBR' in col.upper() and cbr_col is None:
            cbr_col = col
        elif 'WPI' in col.upper() and wpi_col is None:
            wpi_col = col
    
    if not cbr_col and not wpi_col:
        st.warning("No CBR or WPI columns found")
        return
    
    # Create distribution charts
    if cbr_col:
        st.write("**CBR Test Distribution:**")
        cbr_data = filtered_data[filtered_data[cbr_col].notna()]
        if not cbr_data.empty and 'Chainage' in cbr_data.columns:
            chainage_data = cbr_data['Chainage'].dropna()
            if not chainage_data.empty:
                # Create histogram for distribution
                hist_data = pd.DataFrame({
                    'Chainage': chainage_data,
                    'Test_Count': 1
                })
                
                # Group by chainage bins
                hist_data['Chainage_Bin'] = pd.cut(hist_data['Chainage'], bins=20)
                bin_counts = hist_data.groupby('Chainage_Bin').size().reset_index(name='Count')
                bin_counts['Chainage_Center'] = bin_counts['Chainage_Bin'].apply(lambda x: x.mid)
                
                # Display chart
                chart_data = pd.DataFrame({
                    'Chainage': bin_counts['Chainage_Center'],
                    'CBR Tests': bin_counts['Count']
                })
                st.line_chart(chart_data.set_index('Chainage'))
                st.caption(f"Total CBR tests: {len(cbr_data)}")
    
    if wpi_col:
        st.write("**WPI Test Distribution:**")
        wpi_data = filtered_data[filtered_data[wpi_col].notna()]
        if not wpi_data.empty and 'Chainage' in wpi_data.columns:
            chainage_data = wpi_data['Chainage'].dropna()
            if not chainage_data.empty:
                # Create histogram for distribution
                hist_data = pd.DataFrame({
                    'Chainage': chainage_data,
                    'Test_Count': 1
                })
                
                # Group by chainage bins
                hist_data['Chainage_Bin'] = pd.cut(hist_data['Chainage'], bins=20)
                bin_counts = hist_data.groupby('Chainage_Bin').size().reset_index(name='Count')
                bin_counts['Chainage_Center'] = bin_counts['Chainage_Bin'].apply(lambda x: x.mid)
                
                # Display chart
                chart_data = pd.DataFrame({
                    'Chainage': bin_counts['Chainage_Center'],
                    'WPI Tests': bin_counts['Count']
                })
                st.line_chart(chart_data.set_index('Chainage'))
                st.caption(f"Total WPI tests: {len(wpi_data)}")


def render_cbr_wpi_analysis_tab(filtered_data: pd.DataFrame):
    """
    Render the CBR/WPI classification analysis tab in Streamlit.
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
    
    # Get dynamic ID columns for grouping/filtering
    try:
        from .data_processing import get_id_columns_from_data
        id_columns = get_id_columns_from_data(filtered_data)
    except ImportError:
        from data_processing import get_id_columns_from_data
        id_columns = get_id_columns_from_data(filtered_data)

    # Silent smart column detection (no UI changes)
    detected_cbr_swell_col = detect_cbr_swell_column(filtered_data)
    detected_wpi_col = detect_wpi_column(filtered_data)

    # Preliminary data check for UI availability (using smart detection)
    preliminary_data = prepare_cbr_wpi_data(filtered_data, cbr_swell_col=detected_cbr_swell_col, wpi_col=detected_wpi_col)
    if preliminary_data is None:
        st.warning("No CBR Swell or WPI data available.")
        return
    
    # Check what data is available for UI options
    has_cbr = 'CBR Swell (%)' in preliminary_data['Name'].values if 'Name' in preliminary_data.columns else False
    has_wpi = 'WPI' in preliminary_data['Name'].values if 'Name' in preliminary_data.columns else False
    has_expansion = False  # Not applicable in the new format

    # Parameter box with logical grouping
    with st.expander("Plot Parameters", expanded=True):
        
        # Row 1: Core Analysis Settings
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Analysis type options based on available data
            available_options = []
            if has_cbr:
                available_options.append("CBR Swell Categories")
            if has_wpi:
                available_options.append("WPI Categories")
            if has_cbr and has_wpi:
                available_options.append("All Categories (Combined)")
            
            if not available_options:
                st.warning("No CBR or WPI data available for analysis.")
                return
            
            # Set default to Combined if available, otherwise first option
            default_index = 0
            if "All Categories (Combined)" in available_options:
                default_index = available_options.index("All Categories (Combined)")
            
            analysis_type = st.selectbox(
                "Analysis Type",
                options=available_options,
                index=default_index,
                key="cbr_wpi_analysis_type",
                help="Select the type of classification analysis to display"
            )
        with col2:
            depth_cut_str = st.text_input(
                "Depth Cut (mbgl)",
                value="",
                key="cbr_wpi_depth_cut",
                help="Optional: Enter depth threshold to separate Above Cut vs Below Cut categories"
            )
            
            # Convert to float if value provided, otherwise None
            if depth_cut_str.strip():
                try:
                    depth_cut = float(depth_cut_str)
                    if depth_cut <= 0:
                        st.error("Depth cut must be positive")
                        depth_cut = None
                except ValueError:
                    st.error("Please enter a valid number")
                    depth_cut = None
            else:
                depth_cut = None
        with col3:
            # Build stacking options based on processed CBR/WPI dataframe columns
            stack_options = ["None"]
            
            # Add available columns from the processed data (excluding 'Name' and 'category')
            # We know the structure: always has 'Geology_Orgin', conditionally has 'Cut_Category' and map symbol
            available_stack_cols = []
            if 'Geology_Orgin' in preliminary_data.columns:
                available_stack_cols.append('Geology_Orgin')
            
            # Check for map symbol column in original data
            map_symbol_col = find_map_symbol_column(filtered_data)
            if map_symbol_col:
                available_stack_cols.append(map_symbol_col)
            
            if depth_cut is not None:
                available_stack_cols.append('Cut_Category')  # Will be available when data is processed
            
            stack_options.extend(available_stack_cols)
            
            stack_by_option = st.selectbox(
                "Stack By",
                options=stack_options,
                index=0,
                key="cbr_wpi_stack_by",
                help="Stack bars by categorical column from processed data"
            )
        with col4:
            category_order_str = st.text_input(
                "Category Order",
                value="Low,Moderate,High,Very high,Extreme",
                key="cbr_wpi_category_order",
                help="Comma-separated category order"
            )
        with col5:
            facet_order_option = st.selectbox(
                "Facet Order",
                options=["Ascending", "Descending"],
                index=0,
                key="cbr_wpi_facet_order",
                help="Sort order for facet panels"
            )
        
        # Row 2: Data Filtering
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Dynamic filter 1
            filter1_col = st.selectbox(
                "Filter 1 By",
                options=["None"] + id_columns,
                key="cbr_wpi_filter1_col",
                help="Select column to filter data by"
            )
        with col2:
            # Dynamic filter 1 values
            filter1_values = ["All"]
            if filter1_col != "None" and filter1_col in preliminary_data.columns:
                unique_vals = sorted([str(x) for x in preliminary_data[filter1_col].dropna().unique()])
                filter1_values.extend(unique_vals)
            
            filter1_value = st.selectbox(
                "Filter 1 Value",
                options=filter1_values,
                key="cbr_wpi_filter1_value",
                help="Select value to filter by"
            )
        with col3:
            # Dynamic filter 2
            filter2_col = st.selectbox(
                "Filter 2 By",
                options=["None"] + id_columns,
                key="cbr_wpi_filter2_col",
                help="Select second column to filter data by"
            )
        with col4:
            # Dynamic filter 2 values
            filter2_values = ["All"]
            if filter2_col != "None" and filter2_col in preliminary_data.columns:
                unique_vals = sorted([str(x) for x in preliminary_data[filter2_col].dropna().unique()])
                filter2_values.extend(unique_vals)
            
            filter2_value = st.selectbox(
                "Filter 2 Value",
                options=filter2_values,
                key="cbr_wpi_filter2_value",
                help="Select value to filter by"
            )
        with col5:
            st.write("")  # Empty for spacing
        
        # Row 3: Plot Configuration
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            figsize_str = st.text_input(
                "Figure Size",
                value="(15, 7)",
                key="cbr_wpi_figsize",
                help="Format: (width, height) - controls plot dimensions"
            )
        with col2:
            xlim_str = st.text_input(
                "X-Axis Limits",
                value="",
                key="cbr_wpi_xlim",
                help="Format: (min, max) or leave empty for auto"
            )
        with col3:
            ylim_str = st.text_input(
                "Y-Axis Limits",
                value="",
                key="cbr_wpi_ylim",
                help="Format: (min, max) or leave empty for auto"
            )
        with col4:
            title = st.text_input(
                "Custom Title",
                value="",
                key="cbr_wpi_title",
                help="Custom plot title (empty = auto-generated)"
            )
        with col5:
            custom_ylabel = st.text_input(
                "Custom Y-Label",
                value="",
                key="cbr_wpi_custom_ylabel",
                help="Custom Y-axis label (empty = auto)"
            )
        
        # Row 4: Visual Style & Display Options  
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            cmap_name = st.selectbox(
                "Colormap",
                options=["tab10", "tab20", "viridis", "Set1", "Set2", "Set3", "Pastel1", "Dark2"],
                index=0,
                key="cbr_wpi_cmap_name",
                help="Color scheme for categorical data"
            )
        with col2:
            bar_alpha = st.number_input(
                "Alpha",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.05,
                key="cbr_wpi_bar_alpha",
                help="Transparency level (0.1=transparent, 1.0=opaque)"
            )
        with col3:
            show_grid = st.selectbox(
                "Show Grid",
                options=["Yes", "No"],
                index=0,
                key="cbr_wpi_show_grid",
                help="Display grid lines on plots"
            )
        with col4:
            show_legend = st.selectbox(
                "Show Legend",
                options=["Yes", "No"],
                index=0,
                key="cbr_wpi_show_legend",
                help="Display legend when stacking is enabled"
            )
        with col5:
            st.write("")  # Empty for spacing
    
    # Advanced Parameters Section (Separate from main expander to avoid nesting)
    with st.expander("Advanced Parameters", expanded=False):
        
        # Advanced Row 1: Font Controls
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        
        with adv_col1:
            subplot_title_fontsize = st.number_input(
                "Subplot Title Size",
                min_value=8,
                max_value=24,
                value=14,
                step=1,
                key="cbr_wpi_subplot_title_fontsize",
                help="Font size for subplot titles"
            )
        with adv_col2:
            axis_label_fontsize = st.number_input(
                "Axis Label Size",
                min_value=8,
                max_value=20,
                value=13,
                step=1,
                key="cbr_wpi_axis_label_fontsize",
                help="Font size for axis labels"
            )
        with adv_col3:
            tick_fontsize = st.number_input(
                "Tick Font Size",
                min_value=6,
                max_value=16,
                value=12,
                step=1,
                key="cbr_wpi_tick_fontsize",
                help="Font size for axis tick labels"
            )
        with adv_col4:
            legend_fontsize = st.number_input(
                "Legend Font Size",
                min_value=6,
                max_value=16,
                value=11,
                step=1,
                key="cbr_wpi_legend_fontsize",
                help="Font size for legend text"
            )
        
        # Advanced Row 2: Styling Controls
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        
        with adv_col1:
            bar_edgecolor = st.text_input(
                "Bar Edge Color",
                value="black",
                key="cbr_wpi_bar_edgecolor",
                help="Color for bar edges (e.g., black, blue, #FF0000)"
            )
        with adv_col2:
            bar_linewidth = st.number_input(
                "Bar Line Width",
                min_value=0.0,
                max_value=3.0,
                value=0.6,
                step=0.1,
                key="cbr_wpi_bar_linewidth",
                help="Width of bar edge lines"
            )
        with adv_col3:
            grid_axis = st.selectbox(
                "Grid Axis",
                options=["y", "x", "both"],
                index=0,
                key="cbr_wpi_grid_axis",
                help="Which axes to show grid lines on"
            )
        with adv_col4:
            style = st.selectbox(
                "Plot Style",
                options=["seaborn-v0_8-colorblind", "default", "classic", "seaborn-v0_8-whitegrid"],
                index=0,
                key="cbr_wpi_style",
                help="Overall plot styling theme"
            )
        
        # Advanced Row 3: Legend and Layout Controls
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        
        with adv_col1:
            # show_statistics already defined in main parameters
            st.write("")  # Empty column for spacing
        with adv_col2:
            # Use default legend location (controlled by bbox)
            legend_loc = "center left"
            legend_bbox_str = st.text_input(
                "Legend Bbox",
                value="(0.87, 0.5)",
                key="cbr_wpi_legend_bbox",
                help="Precise legend position as (x, y) coordinates"
            )
        with adv_col3:
            bottom_margin = st.number_input(
                "Bottom Margin",
                min_value=0.05,
                max_value=0.3,
                value=0.12,
                step=0.01,
                key="cbr_wpi_bottom_margin",
                help="Bottom margin for plot layout"
            )
        with adv_col4:
            grid_linestyle = st.selectbox(
                "Grid Line Style",
                options=["--", "-", ":", "-."],
                index=0,
                key="cbr_wpi_grid_linestyle",
                help="Style of grid lines"
            )
        
        # Advanced Row 4: Technical Parameters
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        
        with adv_col1:
            figure_dpi = st.number_input(
                "Figure DPI",
                min_value=72,
                max_value=300,
                value=100,
                step=25,
                key="cbr_wpi_figure_dpi",
                help="Display resolution (dots per inch)"
            )
        with adv_col2:
            save_dpi = st.number_input(
                "Save DPI",
                min_value=150,
                max_value=600,
                value=300,
                step=50,
                key="cbr_wpi_save_dpi",
                help="Resolution for saved plots"
            )
        with adv_col3:
            grid_alpha = st.number_input(
                "Grid Alpha",
                min_value=0.1,
                max_value=1.0,
                value=0.4,
                step=0.05,
                key="cbr_wpi_grid_alpha",
                help="Transparency of grid lines"
            )
        with adv_col4:
            # Placeholder - stack column moved to main UI
            st.write("")
        
        # Advanced Row 5: Additional Missing Parameters for 100% Coverage
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        
        with adv_col1:
            yticks_str = st.text_input(
                "Y-Ticks",
                value="",
                key="cbr_wpi_yticks",
                help="Custom Y-axis tick positions as comma-separated values"
            )
        with adv_col2:
            tick_direction = st.selectbox(
                "Tick Direction",
                options=["out", "in", "inout"],
                index=0,
                key="cbr_wpi_tick_direction",
                help="Direction of axis tick marks"
            )
        with adv_col3:
            legend_title_fontsize = st.number_input(
                "Legend Title Size",
                min_value=6,
                max_value=18,
                value=12,
                step=1,
                key="cbr_wpi_legend_title_fontsize",
                help="Font size for legend title text"
            )
        with adv_col4:
            grid_linewidth = st.number_input(
                "Grid Line Width",
                min_value=0.1,
                max_value=2.0,
                value=0.6,
                step=0.1,
                key="cbr_wpi_grid_linewidth",
                help="Width of grid lines"
            )
        
        # Advanced Row 6: Font Weight Controls
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        
        with adv_col1:
            subplot_title_fontweight = st.selectbox(
                "Subplot Title Weight",
                options=["normal", "bold", "light", "heavy"],
                index=1,
                key="cbr_wpi_subplot_title_fontweight",
                help="Font weight for subplot titles"
            )
        with adv_col2:
            axis_label_fontweight = st.selectbox(
                "Axis Label Weight",
                options=["normal", "bold", "light", "heavy"],
                index=1,
                key="cbr_wpi_axis_label_fontweight",
                help="Font weight for axis labels"
            )
        with adv_col3:
            facet_order_option = st.selectbox(
                "Facet Order",
                options=["Ascending", "Descending"],
                index=0,
                key="cbr_wpi_facet_order_advanced",
                help="Sort order for facet panels"
            )
        with adv_col4:
            # Placeholder - removed output filepath control
            st.write("")
    
    # Prepare CBR/WPI data with Cut Category and detected columns (now that depth_cut is defined)
    cbr_wpi_data = prepare_cbr_wpi_data(filtered_data, depth_cut, detected_cbr_swell_col, detected_wpi_col)
    
    if cbr_wpi_data is None:
        st.warning("No CBR or WPI data available.")
        return
    
    
    # Apply data filtering to the processed CBR/WPI data (already in correct format)
    filtered_cbr_wpi_data = cbr_wpi_data.copy()
    
    # Apply filter 1 - Note: cbr_wpi_data now only has ['Name', 'Geology_Orgin', 'category'] columns
    if filter1_col != "None" and filter1_value != "All":
        if filter1_col in filtered_cbr_wpi_data.columns:
            filtered_cbr_wpi_data = filtered_cbr_wpi_data[
                filtered_cbr_wpi_data[filter1_col].astype(str) == filter1_value
            ]
    
    # Apply filter 2
    if filter2_col != "None" and filter2_value != "All":
        if filter2_col in filtered_cbr_wpi_data.columns:
            filtered_cbr_wpi_data = filtered_cbr_wpi_data[
                filtered_cbr_wpi_data[filter2_col].astype(str) == filter2_value
            ]
    
    # Generate analysis - Data is already in correct format from Jupyter notebook workflow
    try:
        # Filter data based on analysis type selection
        if analysis_type == "CBR Swell Categories":
            plot_data = filtered_cbr_wpi_data[filtered_cbr_wpi_data['Name'] == 'CBR Swell (%)'].copy()
            
        elif analysis_type == "WPI Categories":
            plot_data = filtered_cbr_wpi_data[filtered_cbr_wpi_data['Name'] == 'WPI'].copy()
            
        elif analysis_type == "All Categories (Combined)":
            # Show both WPI and CBR categories together (full dataset)
            plot_data = filtered_cbr_wpi_data.copy()
        
        else:
            st.error("Invalid analysis type selected.")
            return
        
        if plot_data.empty:
            st.warning("No valid data available after filtering")
            return
        
        # Use Functions folder plot_CBR_swell_WPI_histogram exactly as in Jupyter notebook
        if HAS_MATPLOTLIB and HAS_FUNCTIONS:
            try:
                # Clear any existing figures first
                plt.close('all')
                
                # Parse parameters
                base_figsize = parse_tuple(figsize_str, (12, 7))
                xlim = parse_tuple(xlim_str, None) if xlim_str.strip() else None
                ylim = parse_tuple(ylim_str, None) if ylim_str.strip() else None
                legend_bbox = parse_tuple(legend_bbox_str, (0.87, 0.5))
                
                # Integrate sidebar plot size control
                figsize = base_figsize
                width_pct = 70  # Default
                if hasattr(st.session_state, 'plot_display_settings') and 'width_percentage' in st.session_state.plot_display_settings:
                    width_pct = st.session_state.plot_display_settings['width_percentage']
                    width_scale = width_pct / 100.0
                    scaled_width = base_figsize[0] * width_scale
                    figsize = (scaled_width, base_figsize[1])
                
                # Parse category order and other parameters
                category_order = None
                if category_order_str.strip():
                    category_order = [cat.strip() for cat in category_order_str.split(',') if cat.strip()]
                
                # Set facet order based on dropdown selection
                facet_order = None
                if facet_order_option == "Descending":
                    # Will be handled in plotting function by reversing the order
                    facet_order = ["descending"]
                # For "Ascending", leave as None to use default ascending order
                
                yticks = None
                if yticks_str.strip():
                    try:
                        yticks = [float(x.strip()) for x in yticks_str.split(',') if x.strip()]
                    except ValueError:
                        yticks = None
                
                # Convert stack by option to boolean and column
                enable_stacking_bool = stack_by_option != "None"
                stack_col = stack_by_option if enable_stacking_bool else None
                # Convert selectbox values to boolean
                show_grid_bool = show_grid == "Yes"
                show_legend_bool = show_legend == "Yes"
                
                # Complete function call with all parameters for 100% coverage
                plot_CBR_swell_WPI_histogram(
                    data_df=plot_data,
                    facet_col='Name',
                    category_col='category',
                    category_order=category_order,
                    facet_order=facet_order,
                    enable_stacking=enable_stacking_bool,
                    stack_col=stack_col,
                    xlim=xlim,
                    ylim=ylim,
                    yticks=yticks,
                    xlabel=custom_ylabel if custom_ylabel.strip() else None,
                    title=title if title.strip() else None,
                    title_suffix=None,
                    figsize=figsize,
                    figure_dpi=figure_dpi,
                    save_dpi=save_dpi,
                    style=style,
                    cmap_name=cmap_name,
                    bar_alpha=bar_alpha,
                    bar_edgecolor=bar_edgecolor,
                    bar_linewidth=bar_linewidth,
                    subplot_title_fontsize=subplot_title_fontsize,
                    subplot_title_fontweight=subplot_title_fontweight,
                    axis_label_fontsize=axis_label_fontsize,
                    axis_label_fontweight=axis_label_fontweight,
                    tick_fontsize=tick_fontsize,
                    tick_direction=tick_direction,
                    legend_fontsize=legend_fontsize,
                    legend_title_fontsize=legend_title_fontsize,
                    show_grid=show_grid_bool,
                    grid_axis=grid_axis,
                    grid_linestyle=grid_linestyle,
                    grid_linewidth=grid_linewidth,
                    grid_alpha=grid_alpha,
                    bottom_margin=bottom_margin,
                    legend_loc=legend_loc,
                    legend_bbox_to_anchor=legend_bbox,
                    show_legend=show_legend_bool,
                    output_filepath=None,  # Removed output filepath control
                    show_plot=False,
                    close_plot=False
                )
                
                # Capture and display the figure in Streamlit with plot size control
                current_fig = plt.gcf()
                if current_fig and current_fig.get_axes():
                    # Left-aligned display based on width percentage
                    if width_pct < 100:
                        right_margin = 100 - width_pct
                        col1, col2 = st.columns([width_pct, right_margin])
                        with col1:
                            st.pyplot(current_fig, use_container_width=True)
                    else:
                        st.pyplot(current_fig, use_container_width=True)
                    success = True
                else:
                    success = False
                
                if success:
                    # Simple download button with figure reference
                    from .plot_download_simple import create_simple_download_button
                    create_simple_download_button(f"cbr_wpi_{analysis_type.lower().replace(' ', '_')}", "main", fig=current_fig)
                else:
                    st.warning("Plot generation failed. Check if the Functions folder is accessible.")
            except Exception as plot_error:
                st.error(f"Error generating plot: {str(plot_error)}")
                st.info("Showing data summary instead:")
                
                # Show basic data summary when plot fails
                summary_data = plot_data['category'].value_counts()
                st.bar_chart(summary_data)
        else:
            st.error("❌ Functions folder not accessible. Check Functions folder and histogram modules.")
        
        
        # Test Distribution section (always show now)
        render_cbr_wpi_test_distribution(filtered_data)
        
        # Add visual separator
        st.divider()
        
        # Controls for statistics and data preview
        col1, col2 = st.columns(2)
        with col1:
            show_statistics_checkbox = st.checkbox("Show Statistics", value=True, key="cbr_wpi_show_stats_checkbox")
        with col2:
            show_data_preview = st.checkbox("Show Data Preview", value=True, key="cbr_wpi_show_preview_checkbox")
        
        # Two-column layout for data preview (left) and statistics (right)
        if show_data_preview or show_statistics_checkbox:
            left_col, right_col = st.columns(2)
            
            # Data preview section (left column)
            with left_col:
                if show_data_preview:
                    st.subheader("Data Preview")
                    # plot_data now only contains ['Name', 'Geology_Orgin', 'category'] columns from Jupyter notebook workflow
                    preview_cols = ['Name', 'Geology_Orgin', 'category']
                    if depth_cut is not None:
                        preview_cols.append('Cut_Category')
                    # Add map symbol if available
                    map_symbol_col = find_map_symbol_column(filtered_data)
                    if map_symbol_col and map_symbol_col in plot_data.columns:
                        preview_cols.append(map_symbol_col)
                    
                    available_cols = [col for col in preview_cols if col in plot_data.columns]
                    st.dataframe(plot_data[available_cols].head(20), use_container_width=True)
            
            # Statistics section (right column)
            with right_col:
                if show_statistics_checkbox:
                    st.subheader("Statistics")
                    if not plot_data.empty:
                        # Create compact statistical summary table
                        stats_data = []
                        
                        # Category distribution
                        category_counts = plot_data['category'].value_counts()
                        
                        # Overview statistics
                        stats_data.extend([
                            {'Stat 1': 'Total Records', 'Value 1': f"{len(plot_data):,}", 
                             'Stat 2': 'Categories', 'Value 2': f"{len(category_counts)}"},
                            {'Stat 1': 'Most Common', 'Value 1': f"{category_counts.index[0]}", 
                             'Stat 2': 'Count', 'Value 2': f"{category_counts.iloc[0]}"}
                        ])
                        
                        # Category breakdown statistics  
                        for i, (category, count) in enumerate(category_counts.head(3).items()):
                            percentage = (count / len(plot_data)) * 100
                            stats_data.append({
                                'Stat 1': f'{category}', 'Value 1': f"{count} ({percentage:.1f}%)", 
                                'Stat 2': '', 'Value 2': ''
                            })
                        
                        # Analysis type specific statistics
                        if analysis_type == "CBR Swell Categories":
                            stats_data.append({
                                'Stat 1': 'Data Type', 'Value 1': 'CBR Swell', 
                                'Stat 2': 'Unit', 'Value 2': 'Percentage (%)'
                            })
                        elif analysis_type == "WPI Categories":
                            stats_data.append({
                                'Stat 1': 'Data Type', 'Value 1': 'WPI', 
                                'Stat 2': 'Unit', 'Value 2': 'Index Value'
                            })
                        elif analysis_type == "All Categories (Combined)":
                            wpi_count = len(plot_data[plot_data['Name'] == 'WPI'])
                            cbr_count = len(plot_data[plot_data['Name'] == 'CBR Swell (%)'])
                            stats_data.append({
                                'Stat 1': 'WPI Records', 'Value 1': f"{wpi_count}", 
                                'Stat 2': 'CBR Records', 'Value 2': f"{cbr_count}"
                            })
                        
                        # Display compact table
                        stats_df = pd.DataFrame(stats_data)
                        if not stats_df.empty:
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No data available for statistics")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")