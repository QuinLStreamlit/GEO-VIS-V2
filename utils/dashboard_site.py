"""
Site Characterization Dashboard Module

This module provides a dashboard view of site characterization analysis,
displaying user-configured plots from individual analysis tabs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import io

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    from .data_processing import get_test_availability
    HAS_DATA_PROCESSING = True
except ImportError:
    HAS_DATA_PROCESSING = False


def render_site_characterization_dashboard(filtered_data: pd.DataFrame):
    """
    Render the Site Characterization Dashboard showing user-configured spatial and geological analysis.
    
    Args:
        filtered_data: Filtered laboratory data DataFrame
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render dashboard.")
        return
    
    st.header("Site Characterization Dashboard")
    st.caption("Summary of your spatial analysis and geological characterization")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload laboratory data first.")
        return
    
    # Check for available plots in session state
    spatial_plots = st.session_state.get('spatial_plots', {})
    
    # Dashboard layout: 2x2 + 1 full width
    # Top row: Geological Distribution + Geological Thickness
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property vs Chainage")
        if 'property_vs_chainage' in spatial_plots:
            st.image(spatial_plots['property_vs_chainage'], use_container_width=True)
        else:
            st.info("📊 Configure chainage analysis to see spatial variation")
            st.caption("Visit **Spatial Analysis** tab to generate property vs chainage plots")
    
    with col2:
        st.subheader("Property vs Depth")
        if 'property_vs_depth' in spatial_plots:
            st.image(spatial_plots['property_vs_depth'], use_container_width=True)
        else:
            st.info("📈 Configure depth analysis to see property variation with depth")
            st.caption("Visit **Spatial Analysis** tab to generate property vs depth plots")
    
    # Middle row: Thickness Distribution (full width)
    st.subheader("Thickness Distribution Analysis")
    if 'thickness_distribution' in spatial_plots:
        st.image(spatial_plots['thickness_distribution'], use_container_width=True)
    else:
        st.info("📍 Configure thickness analysis to see formation thickness distribution")
        st.caption("Visit **Spatial Analysis** tab and configure thickness analysis")
    
    # Bottom row: Borehole Summary + Data Summary
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("SPT Analysis")
        # Check for any SPT analysis plots
        spt_plots = [k for k in spatial_plots.keys() if k.startswith('spt_analysis_')]
        if spt_plots:
            # Show the first available SPT plot
            st.image(spatial_plots[spt_plots[0]], use_container_width=True)
            if len(spt_plots) > 1:
                st.caption(f"Showing {spt_plots[0].replace('_', ' ').title()}")
        else:
            # Generate simple depth summary if data available
            if 'From_mbgl' in filtered_data.columns:
                depth_data = filtered_data['From_mbgl'].dropna()
                if not depth_data.empty:
                    st.metric("Max Depth", f"{depth_data.max():.1f} m")
                    st.metric("Min Depth", f"{depth_data.min():.1f} m")
                    st.metric("Total Samples", len(depth_data))
                else:
                    st.info("📊 Configure SPT analysis to see penetration test results")
                    st.caption("Visit **SPT Analysis** tab to generate SPT plots")
            else:
                st.info("📊 Configure SPT analysis to see penetration test results")
                st.caption("Visit **SPT Analysis** tab to generate SPT plots")
    
    with col4:
        st.subheader("Available Test Data")
        if HAS_DATA_PROCESSING:
            test_availability = get_test_availability(filtered_data)
            
            # Show test counts in a clean format
            for test_type, count in test_availability.items():
                if count > 0:
                    st.metric(test_type, f"{count} tests")
        else:
            st.info("Test availability data not available")
    
    # Dashboard status
    total_plots = 4  # geological_distribution, thickness_analysis, chainage_plot, depth_summary
    configured_plots = len([k for k in ['geological_distribution', 'thickness_analysis', 'chainage_plot', 'depth_summary'] 
                           if k in spatial_plots])
    
    if configured_plots == 0:
        st.info("🚀 **Get Started:** Visit the **Spatial Analysis** tab to configure your site characterization plots")
    elif configured_plots < total_plots:
        st.info(f"📊 **Progress:** {configured_plots}/{total_plots} plots configured. Visit **Spatial Analysis** for more options")
    else:
        st.success("✅ **Complete:** All site characterization plots configured")


def store_spatial_plot(plot_name: str, figure_buffer):
    """
    Store a spatial analysis plot in session state for dashboard display.
    
    Args:
        plot_name: Name identifier for the plot
        figure_buffer: Plot figure buffer (io.BytesIO) or matplotlib figure object
    """
    if 'spatial_plots' not in st.session_state:
        st.session_state.spatial_plots = {}
    
    st.session_state.spatial_plots[plot_name] = figure_buffer


def get_dashboard_status() -> Dict[str, Any]:
    """
    Get the current status of the site characterization dashboard.
    
    Returns:
        Dict containing dashboard status information
    """
    spatial_plots = st.session_state.get('spatial_plots', {})
    
    expected_plots = ['geological_distribution', 'thickness_analysis', 'chainage_plot', 'depth_summary']
    configured_plots = [plot for plot in expected_plots if plot in spatial_plots]
    
    return {
        'total_expected': len(expected_plots),
        'configured_count': len(configured_plots),
        'configured_plots': configured_plots,
        'missing_plots': [plot for plot in expected_plots if plot not in spatial_plots],
        'completion_percentage': (len(configured_plots) / len(expected_plots)) * 100
    }