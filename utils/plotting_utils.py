"""
Universal Plotting Utilities

This module provides a consistent interface for using original plotting functions
from the Functions folder in Streamlit, exactly as they are used in the Jupyter notebook.
"""

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

# Add Functions folder to path (exactly as in Jupyter notebook)
def setup_functions_path():
    """Add Functions folder to Python path, same as Jupyter notebook setup"""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up from utils
    functions_path = os.path.join(current_dir, 'Functions')
    
    if functions_path not in sys.path:
        sys.path.insert(0, functions_path)
    
    return functions_path

# Setup the path when module is imported
setup_functions_path()

def streamlit_plot_wrapper(plot_function, *args, **kwargs):
    """
    Universal wrapper for Functions folder plotting functions to work with Streamlit.
    
    This function handles the matplotlib figure capture that's needed because
    the original plotting functions call plt.close() when show_plot=False.
    
    Args:
        plot_function: The plotting function from Functions folder
        *args: Positional arguments to pass to the plotting function
        **kwargs: Keyword arguments to pass to the plotting function
        
    Returns:
        bool: True if plot was successfully created and displayed, False otherwise
    """
    
    # Set non-interactive backend for Streamlit
    matplotlib.use('Agg')
    
    # Clear any existing plots
    plt.close('all')
    
    # Strategy: Monkey patch plt.close and plt.show to capture the figure
    original_close = plt.close
    original_show = plt.show
    captured_fig = None
    
    def capture_close(fig=None):
        nonlocal captured_fig
        if fig is None:
            fig = plt.gcf()
        # IMPORTANT: Make a copy of the figure before it gets closed
        captured_fig = fig
        # Don't actually close it yet - we need to keep it available for dashboard storage
        
    def capture_show():
        nonlocal captured_fig
        # When show is called, capture the current figure
        captured_fig = plt.gcf()
        # Don't actually show in GUI (which would fail in Streamlit environment)
    
    # Temporarily replace close and show functions
    plt.close = capture_close
    plt.show = capture_show
    
    try:
        # Let the function control show_plot behavior naturally
        # Call the original plotting function with exact same parameters as Jupyter notebook
        plot_function(*args, **kwargs)
        
        # If no figure was captured yet, try to get current figure
        if captured_fig is None:
            current_fig = plt.gcf()
            if current_fig and current_fig.get_axes():
                captured_fig = current_fig
        
    except Exception as e:
        st.error(f"Error calling plotting function {plot_function.__name__}: {str(e)}")
        return False
        
    finally:
        # Always restore original functions
        plt.close = original_close
        plt.show = original_show
    
    # Now display the captured figure in Streamlit
    if captured_fig and captured_fig.get_axes():
        try:
            # Preserve the original figure size from the Functions folder plotting function
            # Don't override the figsize - let the original function control it
            
            # Control actual Streamlit display space using columns
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'plot_display_settings'):
                settings = st.session_state.plot_display_settings
                width_percentage = settings.get('width_percentage', 70)
                
                if width_percentage < 100:
                    # Calculate column ratios for left alignment
                    # For example, 70% means: 70% plot, 30% empty (right side)
                    right_space = 100 - width_percentage
                    
                    # Create columns with left alignment
                    col, _ = st.columns([width_percentage, right_space])
                    with col:
                        st.pyplot(captured_fig, use_container_width=True)
                else:
                    # 100% width - use full container
                    st.pyplot(captured_fig, use_container_width=True)
            else:
                # Default to 70% width, left aligned
                col, _ = st.columns([70, 30])
                with col:
                    st.pyplot(captured_fig, use_container_width=True)
            
            # IMPORTANT: Don't close the figure yet - leave it available for dashboard storage
            # The calling code will handle closing it after storage
            
            return True
            
        except Exception as e:
            st.error(f"Error displaying plot in Streamlit: {str(e)}")
            return False
    else:
        st.warning("No plot was generated by the plotting function")
        return False


def display_plot_with_size_control(fig):
    """
    Display a matplotlib figure with sidebar plot size control.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        bool: True if plot was displayed successfully
    """
    try:
        # Check if we have plot size settings from sidebar
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'plot_display_settings'):
            settings = st.session_state.plot_display_settings
            width_percentage = settings.get('width_percentage', 70)
            
            if width_percentage < 100:
                # Calculate column ratios for left alignment
                # For example, 70% means: 70% plot, 30% empty (right side)
                right_space = 100 - width_percentage
                
                # Create columns with left alignment
                col, _ = st.columns([width_percentage, right_space])
                with col:
                    st.pyplot(fig, use_container_width=True)
            else:
                # 100% width - use full container
                st.pyplot(fig, use_container_width=True)
        else:
            # Default to 70% width, left aligned
            col, _ = st.columns([70, 30])
            with col:
                st.pyplot(fig, use_container_width=True)
        
        return True
        
    except Exception as e:
        st.error(f"Error displaying plot: {str(e)}")
        return False

def import_plotting_functions():
    """
    Import all plotting functions from Functions folder exactly as in Jupyter notebook.
    
    Returns:
        dict: Dictionary of imported plotting functions
    """
    plotting_functions = {}
    
    try:
        # Import exactly as done in Jupyter notebook
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
        
        plotting_functions = {
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
        
        st.success(f"✅ Successfully imported {len(plotting_functions)} plotting functions from Functions folder")
        
    except ImportError as e:
        st.error(f"❌ Failed to import plotting functions: {e}")
        st.info("Make sure Functions folder exists and scipy is installed")
        
    except Exception as e:
        st.error(f"❌ Unexpected error importing plotting functions: {e}")
    
    return plotting_functions

# For convenience, provide direct access to commonly used functions
def plot_spt_cohesive(data, **kwargs):
    """Wrapper for SPT cohesive plotting exactly as used in Jupyter notebook"""
    from Functions.plot_SPT_vs_depth_cohesive import plot_SPT_vs_depth_cohesive
    return streamlit_plot_wrapper(plot_SPT_vs_depth_cohesive, data, **kwargs)

def plot_spt_granular(data, **kwargs):
    """Wrapper for SPT granular plotting exactly as used in Jupyter notebook"""
    from Functions.plot_SPT_vs_depth_granular import plot_SPT_vs_depth_granular
    return streamlit_plot_wrapper(plot_SPT_vs_depth_granular, data, **kwargs)

def plot_atterberg(data, **kwargs):
    """Wrapper for Atterberg chart plotting exactly as used in Jupyter notebook"""
    from Functions.plot_atterberg_chart import plot_atterberg_chart
    
    # Handle save_plot parameter - we don't want to save in Streamlit, just display
    if 'save_plot' not in kwargs:
        kwargs['save_plot'] = False
    
    # If save_plot is True but no output_filepath provided, disable saving
    if kwargs.get('save_plot', False) and not kwargs.get('output_filepath'):
        kwargs['save_plot'] = False
    
    return streamlit_plot_wrapper(plot_atterberg_chart, data, **kwargs)

def plot_psd_analysis(data, **kwargs):
    """Wrapper for PSD plotting exactly as used in Jupyter notebook"""
    from Functions.plot_psd import plot_psd
    
    # plot_psd doesn't have save_plot parameter, just remove output_filepath for Streamlit
    kwargs.pop('save_plot', None)  # Remove save_plot if it exists
    if 'output_filepath' not in kwargs:
        kwargs['output_filepath'] = None
    
    return streamlit_plot_wrapper(plot_psd, data, **kwargs)

def plot_ucs_vs_depth(data, **kwargs):
    """Wrapper for UCS vs depth plotting exactly as used in Jupyter notebook"""
    from Functions.plot_UCS_vs_depth import plot_UCS_vs_depth
    
    # plot_UCS_vs_depth doesn't have save_plot or output_filepath parameters
    # Remove them if they exist in kwargs
    kwargs.pop('save_plot', None)
    kwargs.pop('output_filepath', None)
    
    return streamlit_plot_wrapper(plot_UCS_vs_depth, data, **kwargs)

def plot_ucs_is50(datasets, **kwargs):
    """Wrapper for UCS vs Is50 plotting exactly as used in Jupyter notebook"""
    from Functions.plot_UCS_Is50 import plot_UCS_Is50
    
    # plot_UCS_Is50 has output_filepath but no save_plot parameter
    # Remove save_plot if it exists, set output_filepath to None for Streamlit
    kwargs.pop('save_plot', None)
    if 'output_filepath' not in kwargs:
        kwargs['output_filepath'] = None
    
    return streamlit_plot_wrapper(plot_UCS_Is50, datasets, **kwargs)

def plot_by_chainage(data, **kwargs):
    """Wrapper for chainage plotting exactly as used in Jupyter notebook"""
    from Functions.plot_by_chainage import plot_by_chainage
    
    # plot_by_chainage doesn't have save_plot parameter, just remove output_filepath for Streamlit
    if 'output_filepath' not in kwargs:
        kwargs['output_filepath'] = None
    
    return streamlit_plot_wrapper(plot_by_chainage, data, **kwargs)

def plot_category_by_thickness(data, **kwargs):
    """Wrapper for thickness category plotting exactly as used in Jupyter notebook"""
    from Functions.plot_category_by_thickness import plot_category_by_thickness
    
    # Handle save_plot parameter - we don't want to save in Streamlit, just display
    if 'save_plot' not in kwargs:
        kwargs['save_plot'] = False
    
    # Remove output_filepath since this function doesn't have it
    kwargs.pop('output_filepath', None)
    
    return streamlit_plot_wrapper(plot_category_by_thickness, data, **kwargs)

def plot_engineering_property_vs_depth(data, **kwargs):
    """Wrapper for engineering property vs depth plotting exactly as used in Jupyter notebook"""
    from Functions.plot_engineering_property_vs_depth import plot_engineering_property_vs_depth
    
    # This function doesn't have save_plot or output_filepath parameters
    # Remove them if they exist in kwargs
    kwargs.pop('save_plot', None)
    kwargs.pop('output_filepath', None)
    
    return streamlit_plot_wrapper(plot_engineering_property_vs_depth, data, **kwargs)

def plot_emerson_by_origin(data, **kwargs):
    """Wrapper for Emerson by origin plotting exactly as used in Jupyter notebook"""
    from Functions.plot_emerson_by_origin import plot_emerson_by_origin
    
    # Handle save_plot parameter - we don't want to save in Streamlit, just display
    if 'save_plot' not in kwargs:
        kwargs['save_plot'] = False
    
    # If save_plot is True but no output_filepath provided, disable saving
    if kwargs.get('save_plot', False) and not kwargs.get('output_filepath'):
        kwargs['save_plot'] = False
    
    return streamlit_plot_wrapper(plot_emerson_by_origin, data, **kwargs)