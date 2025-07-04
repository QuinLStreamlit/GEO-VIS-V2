"""
Batch Plot Generation Module

Generates all plots from the application and packages them into a downloadable ZIP file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import zipfile
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st

# Set matplotlib backend
matplotlib.use('Agg')


def generate_test_distribution_plot(test_type: str, filtered_data: pd.DataFrame, bins) -> Optional[io.BytesIO]:
    """Generate a single test distribution plot."""
    try:
        # Filter data for this test type
        test_col = f"{test_type}?"
        if test_col not in filtered_data.columns:
            return None
            
        # Smart filtering for positive values
        mask = pd.Series(False, index=filtered_data.index)
        col_data = filtered_data[test_col]
        
        for value in col_data.dropna().unique():
            value_str = str(value).strip().lower()
            if value_str in ['yes', 'y', 'true', '1', 'x']:
                mask |= (col_data.astype(str).str.strip().str.lower() == value_str)
            elif value_str == '1.0':
                mask |= (col_data.astype(str).str.strip() == '1.0')
        
        test_data = filtered_data[mask]
        
        if test_data.empty or 'Chainage' not in test_data.columns:
            return None
            
        # Create histogram data
        test_chainage = test_data['Chainage'].dropna()
        if test_chainage.empty:
            return None
            
        hist, bin_edges = np.histogram(test_chainage, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers.astype(int), hist.astype(int), marker='o', markersize=4, linestyle='-', linewidth=2)
        plt.xlabel('Chainage (m)', fontsize=12)
        plt.ylabel('Test Count', fontsize=12)
        plt.title(f'{test_type} Distribution Along Chainage', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Force integer y-axis
        ax = plt.gca()
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        plt.close()
        
        return buffer
        
    except Exception as e:
        print(f"Error generating test distribution plot for {test_type}: {str(e)}")
        plt.close('all')
        return None


def generate_all_plots_batch(filtered_data: pd.DataFrame, plot_format: str = 'png', 
                            plot_dpi: int = 300, include_titles: bool = True, 
                            include_legends: bool = True) -> Dict[str, io.BytesIO]:
    """
    Generate all plots from the application.
    Returns a dictionary of plot names and their buffers.
    """
    generated_plots = {}
    
    # Import necessary modules and functions
    try:
        from .analysis_plot_psd import extract_psd_data, convert_psd_to_long_format
        from .analysis_plot_atterberg_chart import extract_atterberg_data
        from .analysis_plot_SPT_vs_depth import extract_spt_data
        from .common_utility_tool import extract_ucs_data, extract_is50_data
        from .analysis_plot_emerson_by_origin import extract_emerson_data
        from .common_utility_tool import get_numerical_properties, filter_valid_chainage_data
        
        # Import plotting functions from Functions folder
        import sys
        import os
        functions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Functions')
        print(f"DEBUG: Functions path: {functions_path}")
        print(f"DEBUG: Functions path exists: {os.path.exists(functions_path)}")
        if functions_path not in sys.path:
            sys.path.append(functions_path)
            print(f"DEBUG: Added Functions path to sys.path")
        
        try:
            from plot_psd import plot_psd
            from plot_atterberg_chart import plot_atterberg_chart
            from plot_SPT_vs_depth_cohesive import plot_SPT_vs_depth_cohesive
            from plot_SPT_vs_depth_granular import plot_SPT_vs_depth_granular
            from plot_UCS_vs_depth import plot_UCS_vs_depth
            from plot_UCS_Is50 import plot_UCS_Is50
            from plot_emerson_by_origin import plot_emerson_by_origin
            HAS_PLOT_FUNCTIONS = True
            print("DEBUG: Successfully imported all Functions plotting scripts")
        except ImportError as e:
            print(f"DEBUG: Import error for Functions plotting scripts: {str(e)}")
            if st:
                st.warning(f"Some plot functions not available: {str(e)}")
            HAS_PLOT_FUNCTIONS = False
            
    except ImportError as e:
        st.error(f"Error importing modules: {str(e)}")
        return generated_plots
    
    # 1. Generate test distribution plots
    if 'Chainage' in filtered_data.columns:
        chainage_data = filtered_data['Chainage'].dropna()
        if not chainage_data.empty:
            min_chainage = chainage_data.min()
            max_chainage = chainage_data.max()
            bin_interval = 200
            bin_start = int(min_chainage // bin_interval) * bin_interval
            bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
            bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
            
            # Get all test types
            test_columns = [col.replace('?', '') for col in filtered_data.columns if '?' in col]
            
            for test_type in test_columns:
                buffer = generate_test_distribution_plot(test_type, filtered_data, bins)
                if buffer:
                    generated_plots[f"Test_Distribution_{test_type}"] = buffer
    
    print(f"DEBUG: HAS_PLOT_FUNCTIONS = {HAS_PLOT_FUNCTIONS}")
    if HAS_PLOT_FUNCTIONS:
        print("DEBUG: Starting main plotting functions generation")
        # 2. PSD Analysis Plot
        try:
            psd_data = extract_psd_data(filtered_data)
            if not psd_data.empty:
                psd_long_format = convert_psd_to_long_format(psd_data)
                if not psd_long_format.empty:
                    # Clear any existing plots first
                    plt.close('all')
                    matplotlib.use('Agg')
                    
                    # Call plotting function with close_plot=False to keep figure alive
                    print(f"DEBUG: Generating PSD plot with {len(psd_long_format)} data points")
                    plot_psd(psd_long_format, output_filepath=None, show_plot=False,
                            title="Particle Size Distribution" if include_titles else "",
                            xlim=(0.001, 100), ylim=(0, 100), show_grid=True,
                            close_plot=False)
                    
                    # Get the current figure
                    current_fig = plt.gcf()
                    print(f"DEBUG: Figure captured: {current_fig}, has axes: {bool(current_fig.get_axes())}")
                    if current_fig and current_fig.get_axes():
                        buffer = io.BytesIO()
                        current_fig.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                        buffer.seek(0)
                        generated_plots["PSD_Analysis_by_Geology"] = buffer
                        print(f"DEBUG: PSD plot saved to buffer, size: {len(buffer.getvalue())} bytes")
                    else:
                        print("DEBUG: No figure or axes found for PSD plot")
                    
                    plt.close('all')
        except Exception as e:
            print(f"Error generating PSD plot: {str(e)}")
            plt.close('all')
        
        # 3. Atterberg Classification Chart
        try:
            print("DEBUG: Attempting Atterberg plot generation")
            atterberg_data = extract_atterberg_data(filtered_data)
            print(f"DEBUG: Atterberg data shape: {atterberg_data.shape}")
            if not atterberg_data.empty:
                plt.close('all')
                matplotlib.use('Agg')
                
                plot_atterberg_chart(atterberg_data, save_plot=False, show_plot=False,
                                   title="Atterberg Classification Chart" if include_titles else "",
                                   close_plot=False)
                
                current_fig = plt.gcf()
                if current_fig and current_fig.get_axes():
                    buffer = io.BytesIO()
                    current_fig.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                    buffer.seek(0)
                    generated_plots["Atterberg_Classification_Chart"] = buffer
                
                plt.close('all')
        except Exception as e:
            print(f"Error generating Atterberg plot: {str(e)}")
            plt.close('all')
        
        # 4. SPT vs Depth Plots
        try:
            spt_data = extract_spt_data(filtered_data)
            if not spt_data.empty:
                # Cohesive soils
                if 'Type' in spt_data.columns:
                    cohesive_data = spt_data[spt_data['Type'] == 'Cohesive']
                    if not cohesive_data.empty:
                        plt.close('all')
                        matplotlib.use('Agg')
                        
                        plot_SPT_vs_depth_cohesive(cohesive_data, 
                                                 title="SPT N-Value vs Depth (Cohesive)" if include_titles else "",
                                                 xlim=(0, 80), ylim=(0, 30), show_plot=False, close_plot=False)
                        
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            buffer = io.BytesIO()
                            current_fig.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                            buffer.seek(0)
                            generated_plots["SPT_vs_Depth_Cohesive"] = buffer
                        
                        plt.close('all')
                
                    # Granular soils
                    granular_data = spt_data[spt_data['Type'] == 'Granular']
                    if not granular_data.empty:
                        plt.close('all')
                        matplotlib.use('Agg')
                        
                        plot_SPT_vs_depth_granular(granular_data,
                                                 title="SPT N-Value vs Depth (Granular)" if include_titles else "",
                                                 xlim=(0, 80), ylim=(0, 30), show_plot=False, close_plot=False)
                        
                        current_fig = plt.gcf()
                        if current_fig and current_fig.get_axes():
                            buffer = io.BytesIO()
                            current_fig.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                            buffer.seek(0)
                            generated_plots["SPT_vs_Depth_Granular"] = buffer
                        
                        plt.close('all')
        except Exception as e:
            print(f"Error generating SPT plots: {str(e)}")
            plt.close('all')
        
        # 5. UCS vs Depth Plot
        try:
            ucs_data = extract_ucs_data(filtered_data)
            if not ucs_data.empty:
                plt.close('all')
                matplotlib.use('Agg')
                
                plot_UCS_vs_depth(ucs_data, xlim=(0.6, 400.0), ylim=(0.0, 29.0),
                                title_suffix=" All Formations" if include_titles else "",
                                show_strength_indicators=True, show_plot=False, close_plot=False)
                
                current_fig = plt.gcf()
                if current_fig and current_fig.get_axes():
                    buffer = io.BytesIO()
                    current_fig.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                    buffer.seek(0)
                    generated_plots["UCS_vs_Depth_by_Formation"] = buffer
                
                plt.close('all')
        except Exception as e:
            print(f"Error generating UCS vs Depth plot: {str(e)}")
            plt.close('all')
        
        # 6. UCS vs Is50 Correlation
        try:
            ucs_data = extract_ucs_data(filtered_data)
            is50_data = extract_is50_data(filtered_data)
            if not ucs_data.empty and not is50_data.empty:
                # Merge data using dynamic ID columns
                from .data_processing import get_standard_id_columns
                id_columns = get_standard_id_columns(filtered_data)
                merged_data = ucs_data.merge(is50_data, on=id_columns, how='inner')
                
                if not merged_data.empty:
                    plt.close('all')
                    matplotlib.use('Agg')
                    
                    datasets = [{
                        'data_df': merged_data,
                        'x_col': 'Is50a (MPa)',
                        'y_col': 'UCS (MPa)'
                    }]
                    plot_UCS_Is50(datasets, title="UCS vs Is50 Correlation" if include_titles else "",
                                show_trendlines=False, category_by="Geology_Orgin",
                                xlim=(0, 10), ylim=(0, 140), show_plot=False, close_plot=False)
                    
                    current_fig = plt.gcf()
                    if current_fig and current_fig.get_axes():
                        buffer = io.BytesIO()
                        current_fig.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                        buffer.seek(0)
                        generated_plots["UCS_vs_Is50_Correlation"] = buffer
                    
                    plt.close('all')
        except Exception as e:
            print(f"Error generating UCS vs Is50 plot: {str(e)}")
            plt.close('all')
        
        # 7. Emerson Plot
        try:
            emerson_data = extract_emerson_data(filtered_data)
            if not emerson_data.empty:
                if 'Emerson' in emerson_data.columns:
                    emerson_data = emerson_data.rename(columns={'Emerson': 'Emerson class'})
                
                plt.close('all')
                matplotlib.use('Agg')
                
                plot_emerson_by_origin(emerson_data, origin_col='Geology_Orgin', save_plot=False, 
                                     show_plot=False, close_plot=False)
                
                current_fig = plt.gcf()
                if current_fig and current_fig.get_axes():
                    buffer = io.BytesIO()
                    current_fig.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                    buffer.seek(0)
                    generated_plots["Emerson_by_Geological_Origin"] = buffer
                
                plt.close('all')
        except Exception as e:
            print(f"Error generating Emerson plot: {str(e)}")
            plt.close('all')
        
        # 8. Property vs Depth Plots
        try:
            from plot_engineering_property_vs_depth import plot_engineering_property_vs_depth
            
            # Key numerical properties
            numerical_props = ['LL (%)', 'PI (%)', 'MC_%', 'UCS (MPa)', 'Is50a (MPa)', 
                              'Is50d (MPa)', 'SPT N Value', 'CBR (%)', 'CBR Swell (%)', 'WPI']
            
            for prop in numerical_props:
                if prop in filtered_data.columns and filtered_data[prop].notna().sum() > 5:  # At least 5 data points
                    try:
                        plt.figure(figsize=(8, 10))
                        plot_engineering_property_vs_depth(
                            filtered_data, 
                            property_col=prop,
                            depth_col='From_mbgl',
                            category_by_col='Geology_Orgin',
                            title=f"{prop} vs Depth" if include_titles else "",
                            show_plot=False
                        )
                        
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                        buffer.seek(0)
                        plt.close()
                        generated_plots[f"Property_vs_Depth_{prop.replace(' ', '_').replace('(%)', '').replace('%', '')}"] = buffer
                    except Exception as e:
                        print(f"Error generating {prop} vs depth plot: {str(e)}")
                        plt.close('all')
        except ImportError:
            print("Property vs depth plotting function not available")
        except Exception as e:
            print(f"Error in property vs depth plots: {str(e)}")
        
        # 9. Property vs Chainage Plots
        if 'Chainage' in filtered_data.columns:
            try:
                from plot_by_chainage import plot_by_chainage
                
                for prop in numerical_props:
                    if prop in filtered_data.columns and filtered_data[prop].notna().sum() > 5:
                        try:
                            chainage_data = filtered_data[filtered_data[prop].notna() & filtered_data['Chainage'].notna()]
                            if not chainage_data.empty:
                                plt.figure(figsize=(14, 7))
                                plot_by_chainage(
                                    chainage_data,
                                    chainage_col='Chainage',
                                    property_col=prop,
                                    category_by_col='Geology_Orgin',
                                    color_by_col='From_mbgl',
                                    title=f"{prop} along chainage" if include_titles else "",
                                    show_plot=False,
                                    close_plot=False
                                )
                                
                                buffer = io.BytesIO()
                                plt.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                                buffer.seek(0)
                                plt.close()
                                generated_plots[f"Property_vs_Chainage_{prop.replace(' ', '_').replace('(%)', '').replace('%', '')}"] = buffer
                        except Exception as e:
                            print(f"Error generating {prop} vs chainage plot: {str(e)}")
                            plt.close('all')
            except ImportError:
                print("Chainage plotting function not available")
            except Exception as e:
                print(f"Error in chainage plots: {str(e)}")
        
        # 10. Histogram Plots
        try:
            from plot_histogram import plot_histogram
            
            histogram_props = ['LL (%)', 'PI (%)', 'SPT N Value', 'UCS (MPa)']
            for prop in histogram_props:
                if prop in filtered_data.columns and filtered_data[prop].notna().sum() > 5:
                    try:
                        plt.figure(figsize=(10, 6))
                        plot_histogram(
                            filtered_data,
                            property_col=prop,
                            title=f"{prop} Distribution" if include_titles else "",
                            show_plot=False
                        )
                        
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                        buffer.seek(0)
                        plt.close()
                        generated_plots[f"Histogram_{prop.replace(' ', '_').replace('(%)', '').replace('%', '')}"] = buffer
                    except Exception as e:
                        print(f"Error generating {prop} histogram: {str(e)}")
                        plt.close('all')
        except ImportError:
            print("Histogram plotting function not available")
        except Exception as e:
            print(f"Error in histogram plots: {str(e)}")
        
        # 11. CBR and WPI Plots
        try:
            from plot_cbr_vs_consistency import plot_cbr_vs_consistency
            from plot_CBR_swell_WPI_histogram import plot_CBR_swell_WPI_histogram
            
            # CBR vs Consistency
            if 'CBR (%)' in filtered_data.columns and 'Consistency' in filtered_data.columns:
                cbr_data = filtered_data[filtered_data['CBR (%)'].notna() & filtered_data['Consistency'].notna()]
                if not cbr_data.empty:
                    try:
                        plt.figure(figsize=(10, 6))
                        plot_cbr_vs_consistency(
                            cbr_data,
                            title="CBR vs Consistency" if include_titles else "",
                            show_plot=False
                        )
                        
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                        buffer.seek(0)
                        plt.close()
                        generated_plots["CBR_vs_Consistency"] = buffer
                    except Exception as e:
                        print(f"Error generating CBR vs consistency plot: {str(e)}")
                        plt.close('all')
            
            # CBR Swell and WPI Histograms
            cbr_wpi_props = ['CBR (%)', 'CBR Swell (%)', 'WPI']
            for prop in cbr_wpi_props:
                if prop in filtered_data.columns and filtered_data[prop].notna().sum() > 5:
                    try:
                        plt.figure(figsize=(10, 6))
                        plot_CBR_swell_WPI_histogram(
                            data_df=filtered_data,
                            facet_col=prop,
                            title=f"{prop} Distribution" if include_titles else "",
                            show_plot=False,
                            close_plot=False
                        )
                        
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format=plot_format, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
                        buffer.seek(0)
                        plt.close()
                        generated_plots[f"CBR_WPI_{prop.replace(' ', '_').replace('(%)', '').replace('%', '')}"] = buffer
                    except Exception as e:
                        print(f"Error generating {prop} CBR/WPI plot: {str(e)}")
                        plt.close('all')
        except ImportError:
            print("CBR/WPI plotting functions not available")
        except Exception as e:
            print(f"Error in CBR/WPI plots: {str(e)}")
    
    print(f"DEBUG: Total plots generated: {len(generated_plots)}")
    print(f"DEBUG: Plot names: {list(generated_plots.keys())}")
    return generated_plots


def create_plots_zip(generated_plots: Dict[str, io.BytesIO], plot_format: str = 'png') -> Optional[io.BytesIO]:
    """Create a ZIP file containing all generated plots."""
    try:
        zip_buffer = io.BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for plot_name, plot_buffer in generated_plots.items():
                filename = f"{plot_name}_{timestamp}.{plot_format}"
                plot_buffer.seek(0)
                zip_file.writestr(filename, plot_buffer.getvalue())
        
        zip_buffer.seek(0)
        return zip_buffer
        
    except Exception as e:
        st.error(f"Error creating ZIP file: {str(e)}")
        return None