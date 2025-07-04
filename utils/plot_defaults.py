"""
Plot Default Parameters for Geotechnical Analysis Tool

This module provides smart default parameters for different plot types to minimize
user input while maintaining professional visualization standards.
"""

from typing import Dict, Any, List


def get_default_parameters(analysis_type: str) -> Dict[str, Any]:
    """
    Get default parameters for specific analysis types.
    
    Args:
        analysis_type: Type of analysis ('PSD', 'Atterberg', 'SPT', etc.)
        
    Returns:
        Dict[str, Any]: Default parameters for the analysis
    """
    defaults = {
        'PSD': get_psd_defaults(),
        'Atterberg': get_atterberg_defaults(),
        'SPT': get_spt_defaults(),
        'UCS': get_ucs_defaults(),
        'Spatial': get_spatial_defaults(),
        'Rock_Strength': get_rock_strength_defaults()
    }
    
    return defaults.get(analysis_type, {})


def get_psd_defaults() -> Dict[str, Any]:
    """Default parameters for Particle Size Distribution analysis."""
    return {
        'plot_type': 'distribution_curve',
        'x_axis': 'log_scale',
        'y_axis': 'linear_scale',
        'grid': True,
        'legend': True,
        'color_scheme': 'geology_based',
        'line_width': 1.5,
        'marker_size': 4,
        'figure_size': (10, 6),
        'dpi': 300,
        'title': 'Particle Size Distribution',
        'x_label': 'Particle Size (mm)',
        'y_label': 'Percentage Passing (%)',
        'sieve_sizes': [75, 63, 37.5, 26.5, 19, 13.2, 9.5, 6.7, 4.75, 2.36, 1.18, 0.6, 0.425, 0.3, 0.15, 0.075],
        'show_classification': True,
        'show_statistics': True,
        'group_by': 'geology'
    }


def get_atterberg_defaults() -> Dict[str, Any]:
    """Default parameters for Atterberg Limits analysis."""
    return {
        'plot_type': 'plasticity_chart',
        'show_a_line': True,
        'show_u_line': True,
        'show_classification_zones': True,
        'marker_size': 6,
        'alpha': 0.7,
        'grid': True,
        'legend': True,
        'color_scheme': 'geology_based',
        'figure_size': (10, 8),
        'dpi': 300,
        'title': 'Plasticity Chart (BS 1377)',
        'x_label': 'Liquid Limit (%)',
        'y_label': 'Plasticity Index (%)',
        'x_range': (0, 100),
        'y_range': (0, 60),
        'show_statistics': True,
        'group_by': 'geology'
    }


def get_spt_defaults() -> Dict[str, Any]:
    """Default parameters for SPT analysis."""
    return {
        'plot_type': 'n_value_profile',
        'show_corrected_values': True,
        'correction_method': 'N60',
        'marker_size': 4,
        'line_width': 1,
        'grid': True,
        'legend': True,
        'color_scheme': 'depth_based',
        'figure_size': (8, 12),
        'dpi': 300,
        'title': 'SPT N-Value Profile',
        'x_label': 'SPT N-Value',
        'y_label': 'Depth (m bgl)',
        'invert_y': True,
        'show_statistics': True,
        'group_by': 'borehole',
        'max_n_value': 50
    }


def get_ucs_defaults() -> Dict[str, Any]:
    """Default parameters for UCS analysis."""
    return {
        'plot_type': 'strength_profile',
        'marker_size': 6,
        'line_width': 1,
        'grid': True,
        'legend': True,
        'color_scheme': 'strength_based',
        'figure_size': (8, 10),
        'dpi': 300,
        'title': 'Unconfined Compressive Strength Profile',
        'x_label': 'UCS (MPa)',
        'y_label': 'Depth (m bgl)',
        'invert_y': True,
        'log_scale': False,
        'show_statistics': True,
        'group_by': 'geology'
    }


def get_spatial_defaults() -> Dict[str, Any]:
    """Default parameters for spatial analysis."""
    return {
        'plot_type': 'plan_view',
        'marker_size': 8,
        'grid': True,
        'legend': True,
        'color_scheme': 'parameter_based',
        'figure_size': (12, 8),
        'dpi': 300,
        'title': 'Spatial Distribution',
        'x_label': 'Chainage (m)',
        'y_label': 'Parameter Value',
        'show_statistics': False,
        'interpolation': False
    }


def get_rock_strength_defaults() -> Dict[str, Any]:
    """Default parameters for rock strength analysis."""
    return {
        'plot_type': 'strength_distribution',
        'marker_size': 6,
        'line_width': 1,
        'grid': True,
        'legend': True,
        'color_scheme': 'rock_type_based',
        'figure_size': (10, 6),
        'dpi': 300,
        'title': 'Rock Strength Distribution',
        'x_label': 'Strength Parameter',
        'y_label': 'Frequency',
        'show_statistics': True,
        'group_by': 'rock_type'
    }


def get_color_schemes() -> Dict[str, List[str]]:
    """Get available color schemes for different grouping methods."""
    return {
        'geology_based': [
            '#1f77b4',  # Blue - Fill
            '#ff7f0e',  # Orange - Clay
            '#2ca02c',  # Green - Sand
            '#d62728',  # Red - Rock
            '#9467bd',  # Purple - Silt
            '#8c564b',  # Brown - Organic
            '#e377c2',  # Pink - Mixed
            '#7f7f7f',  # Gray - Other
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ],
        'depth_based': [
            '#ffffcc',  # Light yellow - Shallow
            '#c7e9b4',  # Light green
            '#7fcdbb',  # Medium green
            '#41b6c4',  # Light blue
            '#2c7fb8',  # Medium blue
            '#253494'   # Dark blue - Deep
        ],
        'strength_based': [
            '#fee5d9',  # Very light - Very soft/weak
            '#fcbba1',  # Light - Soft/weak
            '#fc9272',  # Medium light - Medium
            '#fb6a4a',  # Medium - Firm/medium
            '#de2d26',  # Dark - Stiff/strong
            '#a50f15'   # Very dark - Very stiff/very strong
        ],
        'parameter_based': [
            '#d7191c',  # Red - High values
            '#fdae61',  # Orange - Medium-high
            '#ffffbf',  # Yellow - Medium
            '#abd9e9',  # Light blue - Medium-low
            '#2c7bb6'   # Blue - Low values
        ],
        'rock_type_based': [
            '#8c510a',  # Brown - Sedimentary
            '#d8b365',  # Light brown - Sandstone
            '#5ab4ac',  # Teal - Metamorphic
            '#01665e',  # Dark teal - Igneous
            '#762a83',  # Purple - Volcanic
            '#c2a5cf'   # Light purple - Other
        ]
    }


def get_classification_standards() -> Dict[str, Dict[str, Any]]:
    """Get classification standards for different test types."""
    return {
        'Atterberg': {
            'A_line': {
                'equation': 'PI = 0.73 * (LL - 20)',
                'description': 'A-Line separating clays from silts'
            },
            'U_line': {
                'equation': 'PI = 0.9 * (LL - 8)',
                'description': 'U-Line (upper limit for natural soils)'
            },
            'zones': {
                'CH': 'High plasticity clay',
                'CL': 'Low plasticity clay',
                'MH': 'High plasticity silt',
                'ML': 'Low plasticity silt'
            }
        },
        'PSD': {
            'gravel': '>4.75mm',
            'sand': '0.075-4.75mm',
            'fines': '<0.075mm',
            'classification_system': 'Unified Soil Classification System (USCS)'
        },
        'SPT': {
            'consistency': {
                'very_loose': '0-4',
                'loose': '4-10',
                'medium_dense': '10-30',
                'dense': '30-50',
                'very_dense': '>50'
            },
            'bearing_capacity': 'Allowable bearing capacity estimation available'
        }
    }


def get_figure_templates() -> Dict[str, Dict[str, Any]]:
    """Get figure template configurations."""
    return {
        'standard': {
            'font_family': 'Arial',
            'font_size': 10,
            'title_size': 12,
            'label_size': 10,
            'legend_size': 9,
            'background_color': 'white',
            'grid_alpha': 0.3,
            'spine_width': 0.8
        },
        'presentation': {
            'font_family': 'Arial',
            'font_size': 12,
            'title_size': 14,
            'label_size': 12,
            'legend_size': 11,
            'background_color': 'white',
            'grid_alpha': 0.4,
            'spine_width': 1.0
        },
        'report': {
            'font_family': 'Times New Roman',
            'font_size': 9,
            'title_size': 11,
            'label_size': 9,
            'legend_size': 8,
            'background_color': 'white',
            'grid_alpha': 0.2,
            'spine_width': 0.6
        }
    }