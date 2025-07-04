"""
Plot Download Utilities

This module provides functionality to capture and download plots from all analysis modules.
Integrates with the original plotting functions from Functions folder.
"""

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import io
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import base64

# Add Functions folder to path
def setup_functions_path():
    """Add Functions folder to Python path"""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    functions_path = os.path.join(current_dir, 'Functions')
    
    if functions_path not in sys.path:
        sys.path.insert(0, functions_path)
    
    return functions_path

setup_functions_path()


def generate_timestamp():
    """Generate timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_plot_to_buffer(plot_function: Callable, *args, **kwargs) -> Optional[io.BytesIO]:
    """
    Execute a plotting function and capture the result to a buffer for download.
    
    Args:
        plot_function: The plotting function from Functions folder
        *args: Positional arguments for the plotting function
        **kwargs: Keyword arguments for the plotting function
        
    Returns:
        io.BytesIO: Buffer containing the plot image, or None if failed
    """
    # Set non-interactive backend
    matplotlib.use('Agg')
    
    # Clear any existing plots
    plt.close('all')
    
    # Capture the figure
    original_close = plt.close
    original_show = plt.show
    captured_fig = None
    
    def capture_close(fig=None):
        nonlocal captured_fig
        if fig is None:
            fig = plt.gcf()
        captured_fig = fig
        
    def no_show():
        pass
    
    # Temporarily replace close and show functions
    plt.close = capture_close
    plt.show = no_show
    
    try:
        # Ensure show_plot=False and no output_filepath to capture in memory
        plot_kwargs = kwargs.copy()
        plot_kwargs['show_plot'] = False
        plot_kwargs.pop('output_filepath', None)
        
        # Call the plotting function
        plot_function(*args, **plot_kwargs)
        
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        return None
        
    finally:
        # Restore original functions
        plt.close = original_close
        plt.show = original_show
    
    # Save captured figure to buffer
    if captured_fig and captured_fig.get_axes():
        try:
            buffer = io.BytesIO()
            captured_fig.savefig(
                buffer, 
                format='png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            buffer.seek(0)
            
            # Close the figure
            original_close(captured_fig)
            
            return buffer
            
        except Exception as e:
            st.error(f"Error saving plot to buffer: {str(e)}")
            original_close(captured_fig)
            return None
    else:
        st.warning("No plot was generated")
        return None


def create_download_button(plot_function: Callable, filename_prefix: str, 
                         button_label: str = "📥 Download Plot",
                         *args, **kwargs):
    """
    Create a Streamlit download button for a plot.
    
    Args:
        plot_function: The plotting function from Functions folder
        filename_prefix: Prefix for the downloaded filename
        button_label: Label for the download button
        *args: Arguments to pass to the plotting function
        **kwargs: Keyword arguments to pass to the plotting function
    """
    # Generate plot immediately and create download button
    with st.spinner("Preparing download..."):
        plot_buffer = save_plot_to_buffer(plot_function, *args, **kwargs)
    
    if plot_buffer:
        timestamp = generate_timestamp()
        filename = f"{filename_prefix}_{timestamp}.png"
        
        # Single download button - click to download immediately
        st.download_button(
            label=f"📥 Download Plot",
            data=plot_buffer.getvalue(),
            file_name=filename,
            mime="image/png",
            key=f"download_{filename_prefix}",
            use_container_width=True
        )
    else:
        st.error("❌ Failed to prepare plot for download")


def create_plot_download_section(plot_configs: Dict[str, Dict[str, Any]], 
                                analysis_type: str):
    """
    Create a standardized plot download section for any analysis module.
    
    Args:
        plot_configs: Dictionary of plot configurations
            Format: {
                "plot_name": {
                    "function": plotting_function,
                    "args": [],
                    "kwargs": {},
                    "filename": "filename_prefix",
                    "label": "Plot Description"
                }
            }
        analysis_type: Type of analysis (e.g., "PSD", "UCS", "SPT")
    """
    st.subheader(f"{analysis_type} Plot Downloads")
    
    if not plot_configs:
        st.info("No plots available for download")
        return
    
    # Create columns for multiple plots
    num_plots = len(plot_configs)
    if num_plots == 1:
        cols = [st.container()]
    elif num_plots == 2:
        cols = st.columns(2)
    elif num_plots <= 4:
        cols = st.columns(2)
    else:
        cols = st.columns(3)
    
    for i, (plot_name, config) in enumerate(plot_configs.items()):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            st.write(f"**{config['label']}**")
            
            create_download_button(
                config['function'],
                config['filename'],
                f"Download {plot_name}",
                *config.get('args', []),
                **config.get('kwargs', {})
            )


# Specific plotting function imports for download functionality
def get_plot_functions():
    """Import and return all plotting functions"""
    try:
        from Functions.plot_atterberg_chart import plot_atterberg_chart
        from Functions.plot_CBR_swell_WPI_histogram import plot_CBR_swell_WPI_histogram
        from Functions.plot_cbr_vs_consistency import plot_cbr_vs_consistency
        from Functions.plot_UCS_Is50 import plot_UCS_Is50
        from Functions.plot_emerson_by_origin import plot_emerson_by_origin
        from Functions.plot_psd import plot_psd
        from Functions.plot_histogram import plot_histogram
        from Functions.plot_UCS_vs_depth import plot_UCS_vs_depth
        from Functions.plot_SPT_vs_depth_granular import plot_SPT_vs_depth_granular
        from Functions.plot_SPT_vs_depth_cohesive import plot_SPT_vs_depth_cohesive
        from Functions.plot_engineering_property_vs_depth import plot_engineering_property_vs_depth
        from Functions.plot_category_by_thickness import plot_category_by_thickness
        from Functions.plot_by_chainage import plot_by_chainage
        
        return {
            'plot_atterberg_chart': plot_atterberg_chart,
            'plot_CBR_swell_WPI_histogram': plot_CBR_swell_WPI_histogram,
            'plot_cbr_vs_consistency': plot_cbr_vs_consistency,
            'plot_UCS_Is50': plot_UCS_Is50,
            'plot_emerson_by_origin': plot_emerson_by_origin,
            'plot_psd': plot_psd,
            'plot_histogram': plot_histogram,
            'plot_UCS_vs_depth': plot_UCS_vs_depth,
            'plot_SPT_vs_depth_granular': plot_SPT_vs_depth_granular,
            'plot_SPT_vs_depth_cohesive': plot_SPT_vs_depth_cohesive,
            'plot_engineering_property_vs_depth': plot_engineering_property_vs_depth,
            'plot_category_by_thickness': plot_category_by_thickness,
            'plot_by_chainage': plot_by_chainage,
        }
        
    except ImportError as e:
        st.error(f"Failed to import plotting functions: {e}")
        return {}