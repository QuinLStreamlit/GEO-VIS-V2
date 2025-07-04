"""
Batch Export Module - Clean Version

This module provides batch export functionality for generating multiple plots and reports.
Uses original plotting functions from Functions folder exactly as in Jupyter notebook.
"""

import pandas as pd
import numpy as np
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import io

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    from .data_processing import extract_test_columns, create_test_dataframe, get_standard_id_columns
    from .analysis_plot_psd import extract_psd_data, extract_hydrometer_data, convert_psd_to_long_format
    from .analysis_plot_atterberg_chart import extract_atterberg_data
    from .analysis_plot_SPT_vs_depth import extract_spt_data
    from .common_utility_tool import extract_ucs_data, extract_is50_data
    from .analysis_plot_emerson_by_origin import extract_emerson_data
    from .plot_download_simple import create_simple_download_button
    HAS_ANALYSIS_MODULES = True
except ImportError:
    HAS_ANALYSIS_MODULES = False


def generate_timestamp():
    """Generate timestamp for output file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def render_batch_export_tab(filtered_data: pd.DataFrame):
    """
    Render the batch export tab in Streamlit with clean single-page layout.
    
    Args:
        filtered_data: Filtered laboratory data
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render tab.")
        return
        
    if filtered_data is None or filtered_data.empty:
        st.warning("No data available for export.")
        return
    
    # Plot Generation Section
    st.subheader("Batch Plot Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Plot Settings:**")
        plot_format = st.selectbox("Output Format", ["PNG", "PDF", "SVG"], key="plot_format")
        plot_dpi = st.selectbox("Resolution (DPI)", [150, 300, 600], index=1, key="plot_dpi")
        include_titles = st.checkbox("Include Plot Titles", value=True, key="include_titles")
        include_legends = st.checkbox("Include Legends", value=True, key="include_legends")
    
    # Dynamically determine all available plots based on data
    plot_types = []
    
    # Standard plots from individual analysis tabs
    standard_plots = [
        "PSD Analysis by Geology",
        "Atterberg Classification Charts", 
        "SPT vs Depth (Cohesive)",
        "SPT vs Depth (Granular)",
        "UCS vs Depth by Formation",
        "UCS vs Is50 Correlation",
        "Emerson by Geological Origin"
    ]
    plot_types.extend(standard_plots)
    
    # Add test distribution plots dynamically based on available test types
    test_columns = [col.replace('?', '') for col in filtered_data.columns if '?' in col]
    for test_type in test_columns:
        plot_types.append(f"Test Distribution - {test_type}")
    
    # Property vs Depth plots (for each numerical property)
    numerical_props = ['LL (%)', 'PI (%)', 'MC_%', 'UCS (MPa)', 'Is50a (MPa)', 
                      'Is50d (MPa)', 'SPT N Value', 'CBR (%)', 'CBR Swell (%)', 'WPI']
    for prop in numerical_props:
        if prop in filtered_data.columns and filtered_data[prop].notna().sum() > 0:
            plot_types.append(f"Property vs Depth - {prop}")
    
    # Property vs Chainage plots (for each numerical property)
    if 'Chainage' in filtered_data.columns:
        for prop in numerical_props:
            if prop in filtered_data.columns and filtered_data[prop].notna().sum() > 0:
                plot_types.append(f"Property vs Chainage - {prop}")
    
    # Thickness analysis plots (one per formation if BH data available)
    plot_types.append("Thickness Analysis - All Formations")
    
    # Histogram plots
    plot_types.extend([
        "Histogram - LL Distribution",
        "Histogram - PI Distribution", 
        "Histogram - SPT Distribution",
        "Histogram - UCS Distribution"
    ])
    
    # CBR and WPI plots
    plot_types.extend([
        "CBR vs Depth by Consistency",
        "WPI vs Chainage"
    ])
    
    # Use cached plots from session state (already generated in other tabs)
    try:
        from .batch_plot_generator import create_plots_zip
        
        # Collect cached plots from session state (if they exist)
        cached_plots = {}
        
        # Check for cached plots from various tabs
        plot_storage_keys = [
            'psd_plot_data', 'atterberg_plot_data', 'spt_cohesive_plot_data', 
            'spt_granular_plot_data', 'ucs_depth_plot_data', 'ucs_is50_plot_data',
            'emerson_plot_data', 'spatial_plots', 'material_plots', 'rock_plots'
        ]
        
        # If we have cached plots, use them
        has_cached_plots = any(key in st.session_state for key in plot_storage_keys)
        
        if has_cached_plots:
            # Generate fresh plots quickly (they should be cached)
            from .batch_plot_generator import generate_all_plots_batch
            
            generated_plots = generate_all_plots_batch(
                filtered_data, 
                plot_format.lower(), 
                plot_dpi,
                include_titles,
                include_legends
            )
            
            if generated_plots:
                # Create ZIP file from existing plots
                zip_buffer = create_plots_zip(generated_plots, plot_format.lower())
                
                if zip_buffer:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Direct download button - ONE CLICK like individual plots
                    st.download_button(
                        label=f"Download All {len(generated_plots)} Plots (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"all_plots_{timestamp}.zip",
                        mime="application/zip",
                        key="cached_download_zip",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to create ZIP file.")
            else:
                st.warning("No plots available. Please visit other tabs first to generate plots.")
        else:
            # No cached plots available
            st.info("💡 Please visit other analysis tabs first to generate plots, then return here to download them all at once.")
            
            # Optional: Generate fresh plots
            if st.button("Generate All Plots Now", key="generate_fresh", use_container_width=True):
                from .batch_plot_generator import generate_all_plots_batch
                
                with st.spinner("Generating all plots..."):
                    generated_plots = generate_all_plots_batch(
                        filtered_data, 
                        plot_format.lower(), 
                        plot_dpi,
                        include_titles,
                        include_legends
                    )
                    
                    if generated_plots:
                        zip_buffer = create_plots_zip(generated_plots, plot_format.lower())
                        if zip_buffer:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label=f"Download All {len(generated_plots)} Plots (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"all_plots_{timestamp}.zip",
                                mime="application/zip",
                                key="fresh_download_zip",
                                use_container_width=True
                            )
                        
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")
    
    st.divider()
    
    # Report Templates Section  
    st.subheader("Report Templates")
    
    st.info("Work in progress ...")
    
    report_templates = [
        {
            "name": "Geotechnical Investigation Report",
            "description": "Complete GIR with all test results and interpretations",
            "sections": ["Executive Summary", "PSD Analysis", "Atterberg Limits", "SPT Analysis", "Rock Strength", "Recommendations"],
            "pages": "15-25 pages"
        },
        {
            "name": "Material Characterization Summary",
            "description": "Detailed material properties and classification",
            "sections": ["PSD Analysis", "Atterberg Classification", "Emerson Dispersivity", "Engineering Properties"],
            "pages": "10-15 pages"
        }
    ]
    
    for template in report_templates:
        with st.expander(f"{template['name']}"):
            st.write(f"**Description:** {template['description']}")
            st.write(f"**Estimated Length:** {template['pages']}")
            st.write("**Sections Included:**")
            for section in template['sections']:
                st.write(f"  • {section}")
            
            col1, col2 = st.columns(2)
            with col1:
                button_key = f"gen_{template['name'].replace(' ', '_').replace('/', '_')}"
                if st.button(f"Generate {template['name']}", key=button_key):
                    st.info(f"Report generation for '{template['name']}' would be implemented here.")
                    st.info("This would generate all required plots and compile them into a professional PDF report.")