"""
Material Properties Dashboard Module

This module provides a dashboard view of material properties analysis,
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


def render_material_properties_dashboard(filtered_data: pd.DataFrame):
    """
    Render the Material Properties Dashboard showing user-configured soil classification analysis.
    
    Args:
        filtered_data: Filtered laboratory data DataFrame
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render dashboard.")
        return
    
    st.header("Material Properties Dashboard")
    st.caption("Summary of your soil classification and material characterization analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload laboratory data first.")
        return
    
    # Check for available plots in session state
    material_plots = st.session_state.get('material_plots', {})
    
    # Dashboard layout: 2x2 grid
    # Top row: PSD Curves + Atterberg Classification
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PSD Curves by Geology")
        
        # Show what's available in storage
        if material_plots:
            st.info(f"🔍 Available plots: {list(material_plots.keys())}")
        
        # Method 1: Try matplotlib figure directly (user's suggestion!)
        if 'psd_analysis_fig' in material_plots:
            st.info("📊 Displaying matplotlib figure")
            try:
                st.pyplot(material_plots['psd_analysis_fig'], use_container_width=True)
                st.success("✅ Matplotlib figure displayed successfully!")
            except Exception as e:
                st.error(f"Matplotlib display failed: {e}")
        
        # Method 2: Try BytesIO buffer with st.image()
        elif 'psd_analysis' in material_plots:
            st.info("🖼️ Displaying image buffer")
            try:
                st.image(material_plots['psd_analysis'], use_container_width=True)
                st.success("✅ Image buffer displayed successfully!")
            except Exception as e:
                st.error(f"Image buffer display failed: {e}")
        
        # If no plots available
        else:
            st.info("📊 Configure PSD analysis to see particle size distribution")
            st.caption("Visit **PSD Analysis** tab to generate curves by geological origin")
    
    with col2:
        st.subheader("Atterberg Classification Chart")
        if 'atterberg_chart' in material_plots:
            st.image(material_plots['atterberg_chart'], use_container_width=True)
        else:
            st.info("📈 Configure Atterberg analysis to see plasticity classification")
            st.caption("Visit **Atterberg Limits** tab to generate classification chart")
    
    # Bottom row: Emerson Analysis + Consistency Distribution
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Emerson Class Distribution")
        if 'emerson_analysis' in material_plots:
            st.image(material_plots['emerson_analysis'], use_container_width=True)
        else:
            st.info("📍 Configure Emerson analysis to see dispersivity classification")
            st.caption("Visit **Emerson Analysis** tab to generate dispersivity plots")
    
    with col4:
        st.subheader("Consistency Distribution")
        if 'consistency_distribution' in material_plots:
            st.pyplot(material_plots['consistency_distribution'])
        else:
            # Generate simple consistency summary if data available
            if 'Consistency' in filtered_data.columns:
                consistency_data = filtered_data['Consistency'].dropna()
                if not consistency_data.empty:
                    consistency_counts = consistency_data.value_counts()
                    st.write("**Available Consistency Data:**")
                    for consistency, count in consistency_counts.items():
                        st.metric(consistency, f"{count} samples")
                else:
                    st.info("No consistency data available")
            else:
                st.info("📏 Configure consistency analysis for distribution")
                st.caption("Generate consistency plots in analysis tabs")
    
    # Material classification summary
    st.markdown("---")
    st.subheader("Material Classification Summary")
    
    # Show available material data
    if HAS_DATA_PROCESSING:
        test_availability = get_test_availability(filtered_data)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            psd_count = test_availability.get('PSD', 0)
            st.metric("PSD Tests", psd_count)
        
        with col6:
            atterberg_count = test_availability.get('Atterberg', 0)
            st.metric("Atterberg Tests", atterberg_count)
        
        with col7:
            emerson_count = test_availability.get('Emerson', 0)
            st.metric("Emerson Tests", emerson_count)
        
        with col8:
            # Calculate total classification potential
            total_classification = psd_count + atterberg_count + emerson_count
            st.metric("Total Classification", total_classification)
    
    # Dashboard status
    total_plots = 4  # psd_analysis, atterberg_chart, emerson_analysis, consistency_distribution
    configured_plots = len([k for k in ['psd_analysis', 'atterberg_chart', 'emerson_analysis', 'consistency_distribution'] 
                           if k in material_plots])
    
    if configured_plots == 0:
        st.info("🚀 **Get Started:** Visit **PSD Analysis**, **Atterberg Limits**, and **Emerson Analysis** tabs to configure your material property plots")
    elif configured_plots < total_plots:
        st.info(f"📊 **Progress:** {configured_plots}/{total_plots} plots configured. Continue with individual analysis tabs")
    else:
        st.success("✅ **Complete:** All material properties plots configured")


def store_material_plot(plot_name: str, figure_buffer):
    """
    Store a material properties plot in session state for dashboard display.
    
    Args:
        plot_name: Name identifier for the plot
        figure_buffer: Plot figure buffer (io.BytesIO) or matplotlib figure object
    """
    if 'material_plots' not in st.session_state:
        st.session_state.material_plots = {}
    
    st.session_state.material_plots[plot_name] = figure_buffer


def get_material_dashboard_status() -> Dict[str, Any]:
    """
    Get the current status of the material properties dashboard.
    
    Returns:
        Dict containing dashboard status information
    """
    material_plots = st.session_state.get('material_plots', {})
    
    expected_plots = ['psd_analysis', 'atterberg_chart', 'emerson_analysis', 'consistency_distribution']
    configured_plots = [plot for plot in expected_plots if plot in material_plots]
    
    return {
        'total_expected': len(expected_plots),
        'configured_count': len(configured_plots),
        'configured_plots': configured_plots,
        'missing_plots': [plot for plot in expected_plots if plot not in material_plots],
        'completion_percentage': (len(configured_plots) / len(expected_plots)) * 100
    }


def get_material_classification_summary(filtered_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of material classification data availability.
    
    Args:
        filtered_data: Filtered laboratory data DataFrame
        
    Returns:
        Dict containing classification summary
    """
    summary = {
        'total_samples': len(filtered_data),
        'psd_available': False,
        'atterberg_available': False,
        'emerson_available': False,
        'consistency_available': False
    }
    
    if HAS_DATA_PROCESSING:
        try:
            test_availability = get_test_availability(filtered_data)
            summary['psd_available'] = test_availability.get('PSD', 0) > 0
            summary['atterberg_available'] = test_availability.get('Atterberg', 0) > 0
            summary['emerson_available'] = test_availability.get('Emerson', 0) > 0
        except:
            pass
    
    # Check for consistency data
    if 'Consistency' in filtered_data.columns:
        consistency_data = filtered_data['Consistency'].dropna()
        summary['consistency_available'] = not consistency_data.empty
    
    return summary