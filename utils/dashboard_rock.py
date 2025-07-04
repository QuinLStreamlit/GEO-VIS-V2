"""
Rock Properties Dashboard Module

This module provides a dashboard view of rock properties analysis,
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


def render_rock_properties_dashboard(filtered_data: pd.DataFrame):
    """
    Render the Rock Properties Dashboard showing user-configured rock strength analysis.
    
    Args:
        filtered_data: Filtered laboratory data DataFrame
    """
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. Cannot render dashboard.")
        return
    
    st.header("Rock Properties Dashboard")
    st.caption("Summary of your rock strength and engineering properties analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload laboratory data first.")
        return
    
    # Check for available plots in session state
    rock_plots = st.session_state.get('rock_plots', {})
    
    # Dashboard layout: 2x2 grid
    # Top row: UCS vs Depth + UCS vs Is50 Correlation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("UCS vs Depth by Formation")
        if 'ucs_depth' in rock_plots:
            st.image(rock_plots['ucs_depth'], use_container_width=True)
        else:
            st.info("📊 Configure UCS analysis to see strength vs depth relationship")
            st.caption("Visit **Rock Strength** tab to generate UCS vs depth plots")
    
    with col2:
        st.subheader("UCS vs Is50 Correlation")
        if 'ucs_is50_correlation' in rock_plots:
            st.image(rock_plots['ucs_is50_correlation'], use_container_width=True)
        else:
            st.info("📈 Configure UCS correlation to see strength relationships")
            st.caption("Visit **Rock Strength** tab to generate UCS vs Is50 correlation")
    
    # Bottom row: Rock Strength Distribution + Point Load Distribution
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Rock Strength Distribution")
        if 'strength_distribution' in rock_plots:
            st.image(rock_plots['strength_distribution'], use_container_width=True)
        else:
            # Generate simple UCS summary if data available
            if HAS_DATA_PROCESSING:
                test_availability = get_test_availability(filtered_data)
                ucs_count = test_availability.get('UCS', 0)
                
                if ucs_count > 0:
                    # Find numeric UCS column for plotting
                    ucs_columns = [col for col in filtered_data.columns if 'UCS' in col.upper()]
                    numeric_ucs_col = None
                    
                    # Look for the column with actual numeric UCS values (typically "UCS (MPa)" or similar)
                    for col in ucs_columns:
                        if any(term in col.upper() for term in ['MPA', '(MPA)', 'VALUE', 'STRENGTH']):
                            try:
                                test_data = pd.to_numeric(filtered_data[col].dropna(), errors='coerce')
                                if not test_data.dropna().empty:
                                    numeric_ucs_col = col
                                    break
                            except:
                                continue
                    
                    if numeric_ucs_col:
                        ucs_data = pd.to_numeric(filtered_data[numeric_ucs_col].dropna(), errors='coerce').dropna()
                        if not ucs_data.empty:
                            st.metric("UCS Tests", len(ucs_data))
                            st.metric("Max UCS", f"{ucs_data.max():.1f} MPa")
                            st.metric("Mean UCS", f"{ucs_data.mean():.1f} MPa")
                        else:
                            st.info("UCS data found but no numeric values")
                    else:
                        st.info(f"{ucs_count} UCS tests available")
                else:
                    st.info("📍 Configure rock strength analysis for distribution plots")
                    st.caption("Visit **Rock Strength** tab to generate strength distribution")
            else:
                st.info("📍 Configure rock strength analysis for distribution plots")
                st.caption("Visit **Rock Strength** tab to generate strength distribution")
    
    with col4:
        st.subheader("Point Load Distribution")
        if 'point_load_distribution' in rock_plots:
            st.image(rock_plots['point_load_distribution'], use_container_width=True)
        else:
            # Find numeric Is50 column for plotting
            is50_columns = [col for col in filtered_data.columns if 'Is50' in col or 'IS50' in col]
            numeric_is50_col = None
            
            # Look for the column with actual numeric Is50 values
            for col in is50_columns:
                if any(term in col.upper() for term in ['AXIAL', 'MPA', '(MPA)', 'VALUE']):
                    try:
                        test_data = pd.to_numeric(filtered_data[col].dropna(), errors='coerce')
                        if not test_data.dropna().empty:
                            numeric_is50_col = col
                            break
                    except:
                        continue
            
            if numeric_is50_col:
                is50_data = pd.to_numeric(filtered_data[numeric_is50_col].dropna(), errors='coerce').dropna()
                if not is50_data.empty:
                    st.metric("Is50 Tests", len(is50_data))
                    st.metric("Max Is50", f"{is50_data.max():.2f} MPa")
                    st.metric("Mean Is50", f"{is50_data.mean():.2f} MPa")
                else:
                    st.info("Is50 data found but no numeric values")
            else:
                st.info("📏 Configure point load analysis for distribution")
                st.caption("Visit **Rock Strength** tab to generate Is50 analysis")
    
    # Rock engineering summary
    st.markdown("---")
    st.subheader("Rock Engineering Summary")
    
    # Show available rock data
    if HAS_DATA_PROCESSING:
        test_availability = get_test_availability(filtered_data)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            ucs_count = test_availability.get('UCS', 0)
            st.metric("UCS Tests", ucs_count)
        
        with col6:
            # Count numeric Is50 tests
            is50_columns = [col for col in filtered_data.columns if 'Is50' in col or 'IS50' in col]
            is50_count = 0
            
            # Find the numeric Is50 column
            for col in is50_columns:
                if any(term in col.upper() for term in ['AXIAL', 'MPA', '(MPA)', 'VALUE']):
                    try:
                        is50_numeric = pd.to_numeric(filtered_data[col].dropna(), errors='coerce').dropna()
                        is50_count = len(is50_numeric)
                        break
                    except:
                        continue
            
            st.metric("Is50 Tests", is50_count)
        
        with col7:
            # Check for other rock properties
            rock_columns = [col for col in filtered_data.columns 
                          if any(term in col.upper() for term in ['ROCK', 'STRENGTH', 'MODULUS'])]
            st.metric("Rock Properties", len(rock_columns))
        
        with col8:
            # Calculate engineering suitability
            total_strength_tests = ucs_count + is50_count
            st.metric("Total Strength Tests", total_strength_tests)
    
    # Dashboard status
    total_plots = 4  # ucs_depth, ucs_is50_correlation, strength_distribution, point_load_distribution
    configured_plots = len([k for k in ['ucs_depth', 'ucs_is50_correlation', 'strength_distribution', 'point_load_distribution'] 
                           if k in rock_plots])
    
    if configured_plots == 0:
        st.info("🚀 **Get Started:** Visit **Rock Strength** tab to configure your rock properties analysis")
    elif configured_plots < total_plots:
        st.info(f"📊 **Progress:** {configured_plots}/{total_plots} plots configured. Continue with **Rock Strength** analysis")
    else:
        st.success("✅ **Complete:** All rock properties plots configured")


def store_rock_plot(plot_name: str, figure_buffer):
    """
    Store a rock properties plot in session state for dashboard display.
    
    Args:
        plot_name: Name identifier for the plot
        figure_buffer: Plot figure buffer (io.BytesIO) or matplotlib figure object
    """
    if 'rock_plots' not in st.session_state:
        st.session_state.rock_plots = {}
    
    st.session_state.rock_plots[plot_name] = figure_buffer


def get_rock_dashboard_status() -> Dict[str, Any]:
    """
    Get the current status of the rock properties dashboard.
    
    Returns:
        Dict containing dashboard status information
    """
    rock_plots = st.session_state.get('rock_plots', {})
    
    expected_plots = ['ucs_depth', 'ucs_is50_correlation', 'strength_distribution', 'point_load_distribution']
    configured_plots = [plot for plot in expected_plots if plot in rock_plots]
    
    return {
        'total_expected': len(expected_plots),
        'configured_count': len(configured_plots),
        'configured_plots': configured_plots,
        'missing_plots': [plot for plot in expected_plots if plot not in rock_plots],
        'completion_percentage': (len(configured_plots) / len(expected_plots)) * 100
    }


def get_rock_engineering_summary(filtered_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of rock engineering data availability.
    
    Args:
        filtered_data: Filtered laboratory data DataFrame
        
    Returns:
        Dict containing rock engineering summary
    """
    summary = {
        'total_samples': len(filtered_data),
        'ucs_available': False,
        'is50_available': False,
        'ucs_count': 0,
        'is50_count': 0,
        'strength_range': None
    }
    
    # Check UCS data
    ucs_columns = [col for col in filtered_data.columns if 'UCS' in col.upper()]
    if ucs_columns:
        ucs_data = filtered_data[ucs_columns[0]].dropna()
        if not ucs_data.empty:
            summary['ucs_available'] = True
            summary['ucs_count'] = len(ucs_data)
            summary['strength_range'] = f"{ucs_data.min():.1f} - {ucs_data.max():.1f} MPa"
    
    # Check Is50 data
    is50_columns = [col for col in filtered_data.columns if 'Is50' in col or 'IS50' in col]
    if is50_columns:
        is50_data = filtered_data[is50_columns[0]].dropna()
        if not is50_data.empty:
            summary['is50_available'] = True
            summary['is50_count'] = len(is50_data)
    
    return summary