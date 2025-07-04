# Utils package for Geotechnical Data Analysis Tool
"""
Utilities package for geotechnical data analysis and visualization.

This package provides:
- Analysis modules for each plot type following analysis_plot_[name] pattern
- Common utility functions for shared operations
- Data processing and filtering capabilities
- Export and reporting functionality
- Dashboard components for data visualization
"""

# Core data processing
from .data_processing import (
    get_id_columns_from_data,
    extract_test_columns,
    create_test_dataframe,
    get_standard_id_columns,
    apply_global_filters
)

# Common utilities and shared functions
from .common_utility_tool import (
    get_default_parameters,
    get_color_schemes,
    extract_ucs_data,
    extract_is50_data,
    get_available_geologies,
    filter_ucs_by_geology,
    calculate_map_zoom_and_center,
    get_numerical_properties,
    get_categorical_properties,
    parse_tuple
)

# Analysis modules - main render functions
from .analysis_plot_psd import render_psd_analysis_tab
from .analysis_plot_atterberg_chart import render_atterberg_analysis_tab
from .analysis_plot_SPT_vs_depth import render_spt_analysis_tab
from .analysis_plot_UCS_vs_depth import render_ucs_depth_tab
from .analysis_plot_UCS_Is50 import render_ucs_is50_tab
from .analysis_plot_emerson_by_origin import render_emerson_analysis_tab
from .analysis_plot_engineering_property_vs_depth import render_property_depth_tab
from .analysis_plot_by_chainage import render_property_chainage_tab
from .analysis_plot_category_by_thickness import render_thickness_analysis_tab
from .analysis_plot_histogram import render_comprehensive_histograms_tab
from .analysis_plot_CBR_swell_WPI_histogram import render_cbr_wpi_analysis_tab

# Export and batch processing
from .batch_export import render_batch_export_tab
from .batch_plot_generator import generate_all_plots_batch

# Dashboard components
from .dashboard_site import render_site_characterization_dashboard
from .dashboard_materials import render_material_properties_dashboard
from .dashboard_rock import render_rock_properties_dashboard

# Plotting utilities
from .plotting_utils import display_plot_with_size_control
from .plot_download_simple import create_simple_download_button

__version__ = "1.0.0"
__author__ = "Geotechnical Data Analysis Team"

# Define what gets imported with "from utils import *"
__all__ = [
    # Core data processing
    'get_id_columns_from_data',
    'extract_test_columns', 
    'create_test_dataframe',
    'get_standard_id_columns',
    'apply_global_filters',
    
    # Common utilities
    'get_default_parameters',
    'get_color_schemes',
    'extract_ucs_data',
    'extract_is50_data',
    'get_available_geologies',
    'filter_ucs_by_geology',
    'calculate_map_zoom_and_center',
    'get_numerical_properties',
    'get_categorical_properties',
    'parse_tuple',
    
    # Analysis render functions
    'render_psd_analysis_tab',
    'render_atterberg_analysis_tab',
    'render_spt_analysis_tab',
    'render_ucs_depth_tab',
    'render_ucs_is50_tab',
    'render_emerson_analysis_tab',
    'render_property_depth_tab',
    'render_property_chainage_tab',
    'render_thickness_analysis_tab',
    'render_comprehensive_histograms_tab',
    'render_cbr_wpi_analysis_tab',
    
    # Export and batch
    'render_batch_export_tab',
    'generate_all_plots_batch',
    
    # Dashboards
    'render_site_characterization_dashboard',
    'render_material_properties_dashboard',
    'render_rock_properties_dashboard',
    
    # Plotting utilities
    'display_plot_with_size_control',
    'create_simple_download_button'
]