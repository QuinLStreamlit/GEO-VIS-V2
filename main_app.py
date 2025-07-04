"""
Geotechnical Data Analysis Tool - Main Streamlit Application

A professional web application for processing laboratory testing data and generating
standardized engineering visualizations for geotechnical reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import io
from auth import check_password

# Configure page
st.set_page_config(
    page_title="Geotechnical Data Visualisation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        font-size: 14px;
    }
    h1 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
    }
    h3 {
        font-size: 1.4rem;
        margin-bottom: 0.6rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        padding: 8px 16px;
    }
    .metric-container {
        font-size: 12px;
    }
    .stForm {
        border: none;
        box-shadow: none;
    }
    .stMarkdown {
        margin-bottom: 0.2rem;
    }
    .stWarning, .stSuccess, .stInfo, .stError {
        font-size: 11px;
        padding: 0.3rem;
        margin: 0.2rem 0;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.2rem;
    }
    .streamlit-container {
        padding: 0;
    }
    /* Hide number input spinners */
    .stNumberInput button {
        display: none;
    }
    div[data-testid="stNumberInput"] button {
        display: none;
    }
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type="number"] {
        -moz-appearance: textfield;
    }
    /* Hide file size limit text and drag/drop area */
    .uploadedFile {
        font-size: 12px;
    }
    div[data-testid="stFileUploader"] small {
        display: none;
    }
    div[data-testid="stFileUploader"] > div > small {
        display: none;
    }
    /* Hide drag and drop text more aggressively */
    section[data-testid="stFileUploadDropzone"] {
        display: none !important;
    }
    /* Hide the "Drag and drop files here" text */
    div[data-testid="stFileUploader"] div[data-baseweb="file-uploader"] > div:last-child {
        display: none !important;
    }
    /* Alternative selector for drag zone */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    /* Hide any div containing the drag text */
    div:has(> div:contains("Drag and drop")) {
        display: none !important;
    }
    div[data-testid="stFileUploader"] button {
        width: 100%;
    }
    
    /* Make sidebar text smaller */
    .css-1d391kg, [data-testid="stSidebar"] {
        font-size: 11px !important;
    }
    
    /* Make sidebar headers smaller */
    [data-testid="stSidebar"] h1 {
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1.0rem !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
    }
    [data-testid="stSidebar"] h4 {
        font-size: 0.9rem !important;
    }
    
    /* Make sidebar subheaders and labels smaller */
    [data-testid="stSidebar"] .stSubheader {
        font-size: 0.95rem !important;
    }
    [data-testid="stSidebar"] label {
        font-size: 11px !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        font-size: 11px !important;
    }
    
    /* Make input labels and help text smaller */
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        font-size: 11px !important;
    }
    [data-testid="stSidebar"] small {
        font-size: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Import utility functions (will create these)
try:
    # Import core data processing functions
    from utils.data_processing import (
        load_and_validate_data,
        extract_test_columns,
        get_test_availability
    )
    
    # Import all other functions from the utils package interface
    from utils import (
        # Data processing
        apply_global_filters,
        get_id_columns_from_data,
        
        # Analysis render functions
        render_psd_analysis_tab,
        render_atterberg_analysis_tab,
        render_spt_analysis_tab,
        render_ucs_depth_tab,
        render_ucs_is50_tab,
        render_emerson_analysis_tab,
        render_property_depth_tab,
        render_property_chainage_tab,
        render_thickness_analysis_tab,
        render_comprehensive_histograms_tab,
        render_cbr_wpi_analysis_tab,
        
        # Export and batch processing
        render_batch_export_tab,
        
        # Dashboard components
        render_site_characterization_dashboard,
        render_material_properties_dashboard,
        render_rock_properties_dashboard
    )
except ImportError:
    st.error("Required utility modules not found. Please ensure all files are present.")
    st.stop()


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'lab_data' not in st.session_state:
        st.session_state.lab_data = None
    if 'bh_data' not in st.session_state:
        st.session_state.bh_data = None
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'test_availability' not in st.session_state:
        st.session_state.test_availability = {}
    
    # Initialize dashboard plot storage
    if 'spatial_plots' not in st.session_state:
        st.session_state.spatial_plots = {}
    if 'material_plots' not in st.session_state:
        st.session_state.material_plots = {}
    if 'rock_plots' not in st.session_state:
        st.session_state.rock_plots = {}


# Removed complex dashboard progress tracking


# Removed verbose guidance system


# Removed complex workflow progress display




def render_header():
    """Render conditional application header - show on welcome, hide after data upload."""
    if not st.session_state.data_loaded:
        # Show large centered header on welcome page
        st.markdown(
            "<h1 style='text-align: center; font-size: 3.5rem; margin: 2rem 0; color: #1f77b4;'>Geotechnical Data Visualisation</h1>", 
            unsafe_allow_html=True
        )
        


def render_file_upload():
    """Render multi-file upload interface in sidebar."""
    with st.sidebar.expander("📂 Data Upload", expanded=False):
        render_upload_controls()


def render_upload_controls():
    """Render the actual upload controls."""
    # Primary lab data file (required)
    lab_file = st.file_uploader(
        "Lab Summary Data",
        type=['xlsx', 'xls'],
        key="lab_file"
    )
    
    # Optional additional files
    bh_file = st.file_uploader(
        "BH Interpretation Data",
        type=['xlsx', 'xls'],
        key="bh_file",
        help="Required for Thickness Analysis tab. Enables geological formation thickness analysis and interpretation."
    )
    
    # Process uploaded files
    if lab_file is not None:
        try:
            with st.spinner("Loading primary lab data..."):
                lab_data = load_and_validate_data(lab_file)
                st.session_state.lab_data = lab_data
                st.session_state.data_loaded = True
                
                # Get test availability quietly
                test_availability = get_test_availability(lab_data)
                st.session_state.test_availability = test_availability
                
            # Load BH Interpretation data if provided
            if bh_file is not None:
                try:
                    with st.spinner("Loading BH interpretation data..."):
                        import pandas as pd
                        bh_data = pd.read_excel(bh_file)
                        st.session_state.bh_data = bh_data
                except Exception as e:
                    st.error(f"Error loading BH data: {str(e)}")
                    st.session_state.bh_data = None
            else:
                # No BH file uploaded - set to None
                st.session_state.bh_data = None
            
            # Data loaded successfully
                        
        except Exception as e:
            st.error(f"Error loading lab data: {str(e)}")
            st.session_state.data_loaded = False


def render_simple_filters():
    """Render minimal, essential data filters only."""
    if not st.session_state.data_loaded:
        return
    
    st.sidebar.markdown("<p style='font-size: 13px; font-weight: bold; margin-top: 1rem; margin-bottom: 1rem;'>Filters</p>", unsafe_allow_html=True)
    
    lab_data = st.session_state.lab_data
    
    # Reset button
    if st.sidebar.button("Reset"):
        st.session_state.filtered_data = None
        st.rerun()
    
    # Essential global filters
    depth_range = None
    chainage_range = None
    
    # Depth filter
    if 'From_mbgl' in lab_data.columns:
        max_depth = float(lab_data['From_mbgl'].max())
        st.sidebar.markdown("<p style='font-size: 13px; font-weight: bold; margin-top: 1rem; margin-bottom: 0.8rem;'>Depth (mbgl)</p>", unsafe_allow_html=True)
        depth_min = st.sidebar.number_input("From", min_value=0.0, max_value=max_depth, value=0.0, step=0.1, format="%.1f", key="depth_min")
        depth_max = st.sidebar.number_input("To", min_value=0.0, max_value=max_depth, value=max_depth, step=0.1, format="%.1f", key="depth_max")
        depth_range = (depth_min, depth_max)
    
    # Chainage filter
    if 'Chainage' in lab_data.columns:
        chainage_data = lab_data['Chainage'].dropna()
        if not chainage_data.empty:
            data_min_chainage = int(chainage_data.min())
            data_max_chainage = int(chainage_data.max())
            st.sidebar.markdown("<p style='font-size: 13px; font-weight: bold; margin-top: 1rem; margin-bottom: 0.8rem;'>Chainage</p>", unsafe_allow_html=True)
            chainage_min = st.sidebar.number_input("From", value=data_min_chainage, step=1, format="%d", key="chainage_min")
            chainage_max = st.sidebar.number_input("To", value=data_max_chainage, step=1, format="%d", key="chainage_max")
            chainage_range = (int(chainage_min), int(chainage_max))
    
    # Apply filters
    filters = {'depth_range': depth_range, 'chainage_range': chainage_range}
    filtered_data = apply_global_filters(lab_data, filters)
    st.session_state.filtered_data = filtered_data
    


def render_plot_settings():
    """Render plot display size controls in sidebar for app UI optimization."""
    if not st.session_state.data_loaded:
        return
    
    st.sidebar.markdown("<p style='font-size: 13px; font-weight: bold; margin-top: 1rem; margin-bottom: 1rem;'>Plot Size</p>", unsafe_allow_html=True)
    
    # Initialize plot settings in session state if not exists
    if 'plot_display_settings' not in st.session_state:
        st.session_state.plot_display_settings = {
            'width_percentage': 70
        }
    
    # Percentage number input for plot width
    col1, col2 = st.sidebar.columns([1, 2])
    with col1:
        width_percentage = st.number_input(
            "",
            min_value=30,
            max_value=100,
            value=st.session_state.plot_display_settings.get('width_percentage', 60),
            step=5,
            format="%d",
            key="plot_width_percentage",
            help="Control how much horizontal space plots occupy",
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("<div style='padding-top: 5px; font-size: 11px;'>(% of screen)</div>", unsafe_allow_html=True)
    
    
    # Update session state
    st.session_state.plot_display_settings = {
        'width_percentage': width_percentage
    }


def get_global_plot_parameters():
    """Convert display settings to parameters for plotting functions."""
    if 'plot_display_settings' not in st.session_state:
        return {'figsize': (10, 6)}
    
    settings = st.session_state.plot_display_settings
    figure_size = settings.get('figure_size', 'Standard')
    
    # Map display size to matplotlib figsize for app display
    size_map = {
        'Compact': (8, 5),
        'Standard': (10, 6), 
        'Large': (12, 8)
    }
    
    return {
        'figsize': size_map.get(figure_size, (10, 6))
    }


def get_streamlit_plot_config():
    """Get Streamlit plot display configuration."""
    try:
        if hasattr(st.session_state, 'plot_display_settings'):
            settings = st.session_state.plot_display_settings
            figure_size = settings.get('figure_size', 'Standard')
            
            # Map to Streamlit display settings
            if figure_size == 'Compact':
                return {
                    'use_container_width': False,
                    'width': 400,
                    'figsize': (8, 5)
                }
            elif figure_size == 'Large':
                return {
                    'use_container_width': True,
                    'width': None,
                    'figsize': (14, 8)
                }
            else:  # Standard
                return {
                    'use_container_width': False,
                    'width': 600,
                    'figsize': (10, 6)
                }
        else:
            return {
                'use_container_width': False,
                'width': 600,
                'figsize': (10, 6)
            }
    except:
        return {
            'use_container_width': False,
            'width': 600,
            'figsize': (10, 6)
        }


# Removed complex data quality scoring


# Removed verbose recommendations system


@st.cache_data
def calculate_test_availability(data):
    """Calculate test availability with caching for performance."""
    return get_test_availability(data)

def filter_positive_values(data, column):
    """
    Smart filtering for positive test indicators.
    Handles various formats: YES, yes, Yes, Y, y, true, True, 1, etc.
    """
    import pandas as pd
    
    if column not in data.columns:
        return pd.DataFrame()
    
    # Get the column data
    col_data = data[column]
    
    # Create mask for "positive" values
    mask = pd.Series(False, index=data.index)
    
    for value in col_data.dropna().unique():
        value_str = str(value).strip().lower()
        
        # Check for various positive indicators
        if value_str in ['yes', 'y', 'true', '1', 'x']:
            mask |= (col_data.astype(str).str.strip().str.lower() == value_str)
        elif value_str == '1.0':  # Handle float 1.0
            mask |= (col_data.astype(str).str.strip() == '1.0')
    
    return data[mask]

def get_test_types_from_columns(data):
    """
    Extract test types from column names that have '?' in them.
    """
    test_columns = [col for col in data.columns if '?' in col]
    test_types = []
    
    for col in test_columns:
        # Remove the '?' to get the test type name
        test_type = col.replace('?', '').strip()
        if test_type:  # Only add non-empty names
            test_types.append(test_type)
    
    return test_types, test_columns

def render_single_test_chart(test_type, filtered_data, bins):
    """Render a single test distribution chart."""
    st.write(f"**{test_type} Distribution:**")
    
    # Filter data for this test type - look for column with "?" marker
    test_data = None
    found_col = None
    
    # Find the test type column (has "?" in the name)
    test_col = f"{test_type}?"
    
    if test_col in filtered_data.columns:
        # Smart filtering for "positive" values
        test_data_all = filter_positive_values(filtered_data, test_col)
        # Count unique tests by deduplicating on Hole_ID and From_mbgl
        test_data = test_data_all.drop_duplicates(subset=['Hole_ID', 'From_mbgl'])
        yes_count = len(test_data)
        found_col = test_col
    else:
        # Column not found
        test_data = None
        st.caption(f"Column '{test_col}' not found")
    
    if test_data is not None and not test_data.empty and 'Chainage' in test_data.columns:
        # Create histogram data
        test_chainage = test_data['Chainage'].dropna()
        if not test_chainage.empty:
            hist, bin_edges = np.histogram(test_chainage, bins=bins)
            
            # Use bin centers as chainage values for x-axis
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Create chart data with all bins (including zeros for smooth area chart)
            chainage_values = bin_centers.astype(int)
            count_values = hist.astype(int)
            
            # Create chart data
            chart_data = pd.DataFrame({
                'Chainage': chainage_values,
                'Test Count': count_values
            })
            
            if chart_data['Test Count'].sum() > 0:
                # Add padding to force better y-axis scaling for low values
                max_count = chart_data['Test Count'].max()
                if max_count <= 5:
                    # For small datasets, add some padding rows to help with scaling
                    padding_rows = []
                    min_chainage = chart_data['Chainage'].min()
                    max_chainage = chart_data['Chainage'].max()
                    
                    # Add invisible padding points
                    for i in range(max_count + 1, min(max_count + 3, 6)):
                        padding_rows.append({
                            'Chainage': min_chainage - 1000,  # Outside visible range
                            'Test Count': i
                        })
                    
                    if padding_rows:
                        padding_df = pd.DataFrame(padding_rows)
                        chart_data_padded = pd.concat([chart_data, padding_df], ignore_index=True)
                    else:
                        chart_data_padded = chart_data
                else:
                    chart_data_padded = chart_data
                
                # Use Streamlit's native line chart
                st.line_chart(chart_data_padded.set_index('Chainage'))
            else:
                st.info(f"No {test_type} tests found in chainage bins")
        else:
            st.info(f"No {test_type} tests with chainage data")
    else:
        st.info(f"No {test_type} test data available")

@st.cache_data
def calculate_statistics_data(data, numerical_cols):
    """Calculate statistics data with caching for performance."""
    stats_data = []
    for col in numerical_cols:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            stats_data.append({
                'Property': col,
                'Count': len(col_data),
                'Mean': f"{col_data.mean():.2f}",
                'Std Dev': f"{col_data.std():.2f}",
                'Min': f"{col_data.min():.2f}",
                'Max': f"{col_data.max():.2f}",
                'Missing': f"{data[col].isna().sum()} ({data[col].isna().sum()/len(data)*100:.1f}%)"
            })
    return stats_data

def render_data_overview():
    """Render comprehensive data overview with statistics and analysis."""
    if not st.session_state.data_loaded:
        st.warning("Please upload data first.")
        return
    
    filtered_data = st.session_state.filtered_data
    if filtered_data is None:
        filtered_data = st.session_state.lab_data
    
    # Test availability overview (cached for performance)
    test_availability = calculate_test_availability(filtered_data)
    
    st.subheader("Available Tests")
    
    # Create columns for compact display
    col1, col2 = st.columns(2)
    
    test_items = [(test_type, count) for test_type, count in test_availability.items() if count > 0]
    mid_point = len(test_items) // 2
    
    with col1:
        for test_type, count in test_items[:mid_point]:
            st.text(f"{test_type}: {count:,} tests")
    
    with col2:
        for test_type, count in test_items[mid_point:]:
            st.text(f"{test_type}: {count:,} tests")
    
    st.text(f"Total records: {filtered_data.shape[0]:,}")
    
    # Add visual separator
    st.divider()
    
    # Statistical Analysis Section
    st.subheader("Statistical Analysis")
    
    # Key numerical columns for analysis
    numerical_cols = []
    key_properties = [
        'From_mbgl', 'To_mbgl', 'Chainage', 'LL (%)', 'PI (%)', 'MC_%', 
        'UCS (MPa)', 'Is50a (MPa)', 'Is50d (MPa)', 'SPT N Value', 
        'CBR (%)', 'CBR Swell (%)', 'WPI', 'Emerson'
    ]
    
    for col in key_properties:
        if col in filtered_data.columns and filtered_data[col].dtype in ['int64', 'float64']:
            if filtered_data[col].notna().sum() > 0:  # Has actual data
                numerical_cols.append(col)
    
    if numerical_cols:
        # Create statistics table (cached for performance)
        stats_data = calculate_statistics_data(filtered_data, numerical_cols)
        
        if stats_data:
            import pandas as pd
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    # Add visual separator
    st.divider()
    
    # Geological Distribution
    if 'Geology_Orgin' in filtered_data.columns:
        st.subheader("Geological Distribution")
        geo_counts = filtered_data['Geology_Orgin'].value_counts()
        
        # Create table for geological distribution
        geo_data = []
        for geo, count in geo_counts.items():
            percentage = (count / len(filtered_data)) * 100
            geo_data.append({
                'Geological Unit': geo,
                'Count': f"{count:,}",
                'Percentage': f"{percentage:.1f}%"
            })
        
        if geo_data:
            import pandas as pd
            geo_df = pd.DataFrame(geo_data)
            st.dataframe(geo_df, use_container_width=True, hide_index=True)
    
    # Add visual separator  
    st.divider()
    
    # Spatial Coverage
    st.subheader("Spatial Coverage")
    
    # Create table for spatial coverage
    spatial_data = []
    
    if 'From_mbgl' in filtered_data.columns:
        depth_data = filtered_data['From_mbgl'].dropna()
        if not depth_data.empty:
            spatial_data.append({
                'Parameter': 'Depth Range (m)',
                'Value': f"{depth_data.min():.1f} - {depth_data.max():.1f}"
            })
    
    if 'Chainage' in filtered_data.columns:
        chainage_data = filtered_data['Chainage'].dropna()
        if not chainage_data.empty:
            spatial_data.append({
                'Parameter': 'Chainage Range (m)',
                'Value': f"{chainage_data.min():.0f} - {chainage_data.max():.0f}"
            })
    
    if spatial_data:
        import pandas as pd
        spatial_df = pd.DataFrame(spatial_data)
        st.dataframe(spatial_df, use_container_width=True, hide_index=True)
    
    # Add visual separator
    st.divider()
    
    # Test Distribution Chart
    st.subheader("Test Distribution")
    
    try:
        if 'Chainage' in filtered_data.columns:
            # Get test types directly from column names with "?"
            available_test_types, test_columns = get_test_types_from_columns(filtered_data)
            
            if len(available_test_types) > 0:
                # Create bins for chainage to show frequency
                chainage_data = filtered_data['Chainage'].dropna()
                if not chainage_data.empty:
                    min_chainage = chainage_data.min()
                    max_chainage = chainage_data.max()
                    
                    # Create fixed interval bins (200m intervals)
                    bin_interval = 200  # 200m intervals
                    
                    # Round down min_chainage and up max_chainage to nearest intervals
                    bin_start = int(min_chainage // bin_interval) * bin_interval
                    bin_end = int((max_chainage // bin_interval) + 1) * bin_interval
                    
                    # Create bins at fixed intervals
                    bins = np.arange(bin_start, bin_end + bin_interval, bin_interval)
                    
                    
                    # Create charts for all available test types in proper grid layout
                    test_types_to_plot = available_test_types
                    
                    # Process charts in pairs for proper grid alignment
                    for i in range(0, len(test_types_to_plot), 2):
                        # Add spacing between rows
                        if i > 0:
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                        
                        # Create two columns for this row with space between them
                        col1, spacer, col2 = st.columns([5, 1, 5])
                        
                        # Process left chart
                        with col1:
                            test_type = test_types_to_plot[i]
                            render_single_test_chart(test_type, filtered_data, bins)
                        
                        # Process right chart (if exists)
                        with col2:
                            if i + 1 < len(test_types_to_plot):
                                test_type = test_types_to_plot[i + 1]
                                render_single_test_chart(test_type, filtered_data, bins)
                            else:
                                # Empty column to maintain grid alignment
                                st.write("")
                    
                else:
                    st.info("No chainage data available for distribution analysis")
            else:
                st.info("No test data available for distribution analysis")
        else:
            st.warning("Chainage column not found - cannot create spatial distribution")
    
    except Exception as e:
        st.warning(f"Could not generate test distribution chart: {str(e)}")
        st.exception(e)
    
    # Add visual separator
    st.divider()
    
    # Data Preview Options
    st.subheader("Data Preview Options")
    
    # Column selector for data preview
    available_cols = list(filtered_data.columns)
    default_cols = ['Hole_ID', 'Type', 'From_mbgl', 'To_mbgl', 'Chainage', 'Geology_Orgin', 'Consistency']
    available_default_cols = [col for col in default_cols if col in available_cols]
    
    selected_cols = st.multiselect(
        "Select columns to preview:",
        available_cols,
        default=available_default_cols,
        help="Choose which columns to display in the data preview table"
    )
    
    # Data preview (always visible, no checkbox)
    if selected_cols:
        st.dataframe(filtered_data[selected_cols].head(20), use_container_width=True)
        st.caption(f"Showing first 20 rows of {len(filtered_data)} total records")
    else:
        st.warning("Please select at least one column to preview")



def main():
    """Main application function."""
    # Check password first
    if not check_password():
        return
    
    initialize_session_state()
    
    # Sidebar for upload and filters
    render_file_upload()
    if st.session_state.data_loaded:
        render_simple_filters()
        render_plot_settings()
    
    # Main content
    render_header()
    
    # Only show analysis tabs if data is loaded
    if st.session_state.data_loaded:
        
        # Build tab list dynamically based on available data
        tab_names = [
            "Data",
            "PSD", 
            "Atterberg",
            "SPT",
            "Emerson",
            "UCS vs Depth",
            "UCS vs Is50",
            "Property vs Depth",
            "Property vs Chainage"
        ]
        
        # Only add Thickness Analysis tab if BH_Interpretation data is available
        has_bh_data = st.session_state.bh_data is not None
        if has_bh_data:
            tab_names.append("Thickness Analysis")
        
        # Add remaining tabs
        tab_names.extend([
            "Histograms",
            "CBR Swell / WPI",
            "Export"
        ])
        
        # Create tabs
        tabs = st.tabs(tab_names)
        
        data = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.lab_data
        
        # Render tab contents
        tab_index = 0
        
        with tabs[tab_index]:
            render_data_overview()
        tab_index += 1
        
        with tabs[tab_index]:
            render_psd_analysis_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_atterberg_analysis_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_spt_analysis_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_emerson_analysis_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_ucs_depth_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_ucs_is50_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_property_depth_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_property_chainage_tab(data)
        tab_index += 1
        
        # Only render thickness analysis if BH data is available
        if has_bh_data:
            with tabs[tab_index]:
                render_thickness_analysis_tab(data)
            tab_index += 1
        
        with tabs[tab_index]:
            render_comprehensive_histograms_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_cbr_wpi_analysis_tab(data)
        tab_index += 1
        
        with tabs[tab_index]:
            render_batch_export_tab(data)


if __name__ == "__main__":
    main()