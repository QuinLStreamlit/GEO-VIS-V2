"""
Simple Plot Download Utility

Provides a consistent single-button download experience across all analysis tabs.
"""

import io
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional


def create_simple_download_button(plot_name: str, key_suffix: str = "", fig=None) -> None:
    """
    Create a simple single-button download for the current matplotlib figure.
    
    Args:
        plot_name: Name for the plot file (without extension)
        key_suffix: Optional suffix for unique button keys
        fig: Optional matplotlib figure object. If not provided, uses plt.gcf()
    """
    try:
        # Use provided figure or get current matplotlib figure
        current_fig = fig if fig is not None else plt.gcf()
        if current_fig and current_fig.get_axes():
            # Prepare download data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{plot_name}_{timestamp}.png"
            
            # Save to buffer
            buffer = io.BytesIO()
            current_fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
            buffer.seek(0)
            
            # Small left-aligned download button with consistent font size
            col1, col2 = st.columns([1, 4])
            with col1:
                # Add CSS for smaller button with left-aligned text
                st.markdown("""
                <style>
                button[data-testid="stDownloadButton"] {
                    font-size: 14px !important;
                    padding: 0.25rem 0.5rem !important;
                    height: auto !important;
                    text-align: left !important;
                    justify-content: flex-start !important;
                    width: auto !important;
                    min-width: fit-content !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Plot",
                    data=buffer.getvalue(),
                    file_name=filename,
                    mime="image/png",
                    key=f"download_{plot_name}_{key_suffix}",
                    use_container_width=True
                )
        else:
            st.info("Generate a plot first to enable download")
    except Exception as e:
        st.info("Generate a plot first to enable download")


def create_download_section(plot_configs: list, section_title: str = "") -> None:
    """
    Create a download section with multiple plot options.
    
    Args:
        plot_configs: List of dictionaries with 'name' and 'key' for each plot
        section_title: Optional section title
    """
    if section_title:
        st.subheader(section_title)
    
    # Create columns for multiple downloads
    num_plots = len(plot_configs)
    if num_plots == 1:
        create_simple_download_button(plot_configs[0]['name'], plot_configs[0]['key'])
    elif num_plots == 2:
        col1, col2 = st.columns(2)
        with col1:
            create_simple_download_button(plot_configs[0]['name'], plot_configs[0]['key'])
        with col2:
            create_simple_download_button(plot_configs[1]['name'], plot_configs[1]['key'])
    else:
        # For more than 2 plots, create rows
        for i in range(0, num_plots, 2):
            if i + 1 < num_plots:
                col1, col2 = st.columns(2)
                with col1:
                    create_simple_download_button(plot_configs[i]['name'], plot_configs[i]['key'])
                with col2:
                    create_simple_download_button(plot_configs[i+1]['name'], plot_configs[i+1]['key'])
            else:
                create_simple_download_button(plot_configs[i]['name'], plot_configs[i]['key'])