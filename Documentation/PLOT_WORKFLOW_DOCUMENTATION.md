# Geotechnical Data Analysis: From Lab Data to Professional Plots

## Overview

This document explains how the Geotechnical Data Analysis Tool transforms raw laboratory testing data into professional engineering plots. The system processes a comprehensive Excel file containing multiple test types and generates standardized visualizations following industry best practices.

## Table of Contents

1. [Data Input & Structure](#1-data-input--structure)
2. [Data Loading & Validation](#2-data-loading--validation)
3. [Dynamic Column Extraction](#3-dynamic-column-extraction)
4. [Test-Specific Processing](#4-test-specific-processing)
5. [Parameter Standardization](#5-parameter-standardization)
6. [Plot Generation Pipeline](#6-plot-generation-pipeline)
7. [Architecture Components](#7-architecture-components)
8. [Technical Implementation Details](#8-technical-implementation-details)

---

## 1. Data Input & Structure

### Human Perspective
*"I have a large Excel file with all my geotechnical lab test results from different boreholes and test types."*

### Technical Reality
- **Primary File**: `Lab_summary_final.xlsx`
- **Size**: 2,459 rows × 167 columns
- **Secondary File**: `Input/BH_Interpretation.xlsx` (for thickness analysis)

### Data Structure
```
Lab Summary Data:
├── Location Data
│   ├── Hole_ID (borehole identifier)
│   ├── From_mbgl (start depth)
│   ├── To_mbgl (end depth)
│   └── Chainage (distance along alignment)
│
├── Classification Data
│   ├── Geology_Orgin (geological formation)
│   ├── Consistency (rock class/soil consistency)
│   └── Material (material type)
│
├── Test Type Flags
│   ├── PSD? (Y/N for particle size distribution)
│   ├── Atterberg? (Y/N for Atterberg limits)
│   ├── UCS? (Y/N for unconfined compressive strength)
│   ├── SPT? (Y/N for standard penetration test)
│   └── [Additional test flags...]
│
└── Test-Specific Data
    ├── UCS (MPa) (strength values)
    ├── LL (%), PI (%) (plasticity indices)
    ├── 0.075mm, 0.15mm... (sieve sizes)
    └── [Test-specific measurements...]
```

---

## 2. Data Loading & Validation

### Implementation
**File**: `utils/data_processing.py`  
**Function**: `load_and_validate_data()`

### Process Steps
1. **Excel File Loading**
   ```python
   lab_data = pd.read_excel(uploaded_file)
   ```

2. **Structure Validation**
   ```python
   required_cols = ['Hole_ID', 'From_mbgl', 'To_mbgl']
   missing_cols = [col for col in required_cols if col not in df.columns]
   ```

3. **Data Type Detection**
   - Automatic identification of numerical vs categorical columns
   - Smart detection of engineering properties using regex patterns

4. **Test Availability Mapping**
   ```python
   test_availability = {
       'PSD': df['PSD?'].eq('Y').sum() if 'PSD?' in df else 0,
       'Atterberg': df['Atterberg?'].eq('Y').sum() if 'Atterberg?' in df else 0,
       'UCS': df['UCS?'].eq('Y').sum() if 'UCS?' in df else 0
   }
   ```

### Output
- Validated DataFrame ready for analysis
- Test availability summary for UI tab activation
- Error messages for any structural issues

---

## 3. Dynamic Column Extraction

### Purpose
*"The system needs to automatically find which columns belong to each test type since lab data formats can vary."*

### Implementation
**File**: `utils/data_processing.py`  
**Function**: `extract_test_columns(df, test_name)`

### Algorithm
```python
def extract_test_columns(df, test_name):
    """
    Intelligently extract columns related to a specific test type
    by finding columns between test identifier flags.
    """
    identifier_col = f"{test_name}?"
    
    # Find start position of this test's columns
    start_idx = df.columns.get_loc(identifier_col)
    
    # Find next test identifier to determine end position
    end_idx = len(df.columns)
    for i, col in enumerate(df.columns[start_idx + 1:], start_idx + 1):
        if col.endswith('?'):
            end_idx = i
            break
    
    # Return columns between start and end
    return df.columns[start_idx + 1:end_idx].tolist()
```

### Examples
- **PSD Input**: `test_name = "PSD"`
- **PSD Output**: `['0.075mm', '0.15mm', '0.3mm', '0.6mm', '1.18mm', '2.36mm', ...]`
- **Atterberg Output**: `['LL (%)', 'PL (%)', 'PI (%)', 'LS (%)']`

---

## 4. Test-Specific Processing

Each test type requires specialized data processing to transform raw measurements into plot-ready data.

### 4.1 Particle Size Distribution (PSD)

**File**: `utils/psd_analysis.py`

#### Process Steps
1. **Data Extraction**
   ```python
   psd_data = df[df['PSD?'] == 'Y']
   ```

2. **Wide-to-Long Transformation**
   ```python
   PSD_long = psd_data.melt(
       id_vars=['Hole_ID', 'Geology_Orgin', 'Consistency'],
       value_vars=['0.075mm', '0.15mm', '0.3mm', ...],
       var_name='Sieve Size',
       value_name='Percentage passing (%)'
   )
   ```

3. **Sieve Size Parsing**
   ```python
   PSD_long['Sieve_Size_mm'] = PSD_long['Sieve Size'].apply(parse_to_mm)
   # Converts "0.075mm" → 0.075, "No.4" → 4.75, etc.
   ```

4. **Geological Grouping**
   ```python
   ALLUVIUM_PSD = PSD_long.groupby('Geology_Orgin').get_group('ALLUVIUM')
   ```

#### Output
Ready-to-plot DataFrame with:
- Numerical sieve sizes for x-axis
- Percentage passing for y-axis
- Geological grouping for color coding

### 4.2 Atterberg Limits Analysis

**File**: `utils/atterberg_analysis.py`

#### Process Steps
1. **Column Identification**
   ```python
   ll_patterns = ['LL (%)', 'LL(%)', 'Liquid_Limit', 'LL']
   pi_patterns = ['PI (%)', 'PI(%)', 'Plasticity_Index', 'PI']
   ```

2. **Data Validation**
   - Remove invalid values (negative plasticity indices)
   - Check for reasonable ranges (LL: 0-200%, PI: 0-100%)

3. **Classification Logic**
   - Apply A-line equation for soil classification
   - Determine soil types (CL, CH, ML, MH, etc.)

#### Output
DataFrame ready for plasticity chart plotting with validated LL and PI values.

### 4.3 Unconfined Compressive Strength (UCS)

**File**: `utils/ucs_analysis.py`

#### Process Steps
1. **Data Extraction & Filtering**
   ```python
   ucs_data = df[df['UCS?'] == 'Y']
   ucs_data = ucs_data.dropna(subset=['UCS (MPa)'])
   ```

2. **Depth Relationship Processing**
   - Calculate midpoint depths for plotting
   - Group by geological formations
   - Apply strength classification thresholds

3. **Statistical Analysis**
   - Calculate mean, median, standard deviation
   - Identify outliers using IQR method

#### Output
Structured data for depth vs strength plots and strength distribution analysis.

### 4.4 Spatial Analysis (Property vs Chainage)

**File**: `utils/spatial_analysis.py`

#### Process Steps
1. **Property Selection**
   ```python
   numerical_props = get_numerical_properties_smart(df)
   selected_property = user_selection  # From UI
   ```

2. **Spatial Filtering**
   ```python
   if chainage_range:
       filtered_data = df[
           (df['Chainage'] >= chainage_range[0]) & 
           (df['Chainage'] <= chainage_range[1])
       ]
   ```

3. **Zone Classification**
   ```python
   zonage = {
       "Zone 1": (21300, 26300),
       "Zone 2": (26300, 32770),
       "Zone 3": (32770, 37100),
       "Zone 4": (37100, 41120)
   }
   ```

#### Output
Data structured for property vs distance plots with zone boundaries and geological coloring.

### 4.5 Thickness Analysis

**File**: `utils/spatial_analysis.py`

#### Process Steps
1. **Separate Data Source**
   ```python
   BH_Interpretation = pd.read_excel("Input/BH_Interpretation.xlsx")
   ```

2. **Formation-Specific Processing**
   ```python
   Tos_data = BH_Interpretation.groupby('Geology_Orgin').get_group('Tos')
   ```

3. **Thickness Proportion Calculation**
   ```python
   thickness_summary = formation_data.pivot_table(
       values='Thickness', 
       index="Consistency", 
       aggfunc='sum'
   ).reset_index()
   
   thickness_summary['thickness_proportion_%'] = (
       thickness_summary['Thickness'] / thickness_summary['Thickness'].sum()
   ) * 100
   ```

#### Output
Percentage distribution of rock classes by thickness for each geological formation.

---

## 5. Parameter Standardization

### Purpose
*"Ensure all plots follow professional engineering standards and maintain visual consistency."*

### Implementation
**File**: `utils/plot_defaults.py`

### Default Parameter System
```python
def get_default_parameters(plot_type):
    """Return standardized parameters for each plot type"""
    base_params = {
        'figsize': (12, 8),
        'title_fontsize': 14,
        'label_fontsize': 12,
        'tick_fontsize': 10,
        'legend_fontsize': 11,
        'dpi': 300
    }
    
    plot_specific = {
        'psd': {'xmin': 0.001, 'xmax': 100, 'log_scale': True},
        'atterberg': {'xlim': (0, 100), 'ylim': (0, 60)},
        'ucs_depth': {'ylim_auto': True, 'strength_lines': True}
    }
    
    return {**base_params, **plot_specific.get(plot_type, {})}
```

### Color Scheme Management
```python
def get_color_schemes():
    """Geological formation color mapping"""
    return {
        'ALLUVIUM': '#8B4513',      # Brown
        'FILL': '#696969',          # Gray
        'RS_XW': '#FF6347',         # Orange-red
        'Dcf': '#4169E1',           # Royal blue
        'Tos': '#32CD32',           # Lime green
        'Rjbw': '#9932CC',          # Dark orchid
        'Rin': '#FF1493'            # Deep pink
    }
```

---

## 6. Plot Generation Pipeline

### Architecture Overview
The system uses a three-layer approach:
1. **Streamlit UI Layer**: Parameter collection and display
2. **Bridge Layer**: Integration utilities
3. **Plotting Engine**: Original Functions folder

### 6.1 UI Parameter Collection

**Location**: Analysis tabs (e.g., `render_psd_analysis_tab()`)

#### Standardized 5×5 Parameter Grid
```python
with st.expander("Plot Parameters", expanded=True):
    # Row 1: Data Display Options
    col1, col2, col3, col4, col5 = st.columns(5)
    # Properties, Faceting, Stacking, Color mapping, Data filters
    
    # Row 2: Plot Configuration  
    col1, col2, col3, col4, col5 = st.columns(5)
    # Figure size, Axis limits, Titles, Axis labels
    
    # Row 3: Visual Style
    col1, col2, col3, col4, col5 = st.columns(5)
    # Colors, Alpha, Grid, Legend, Edge styling
    
    # Row 4: Font Styling
    col1, col2, col3, col4, col5 = st.columns(5)
    # Title size, Label size, Tick size, Legend size, Line widths
    
    # Row 5: Advanced Options
    col1, col2, col3, col4, col5 = st.columns(5)
    # Filters, Output options, Advanced styling
```

### 6.2 Function Call Bridge

**File**: `utils/plotting_utils.py`  
**Function**: `streamlit_plot_wrapper()`

#### Process Flow
```python
def streamlit_plot_wrapper(plot_function, *args, **kwargs):
    """Universal wrapper for Functions folder plotting functions"""
    
    # 1. Set non-interactive backend for Streamlit
    matplotlib.use('Agg')
    
    # 2. Clear any existing plots
    plt.close('all')
    
    # 3. Capture figure before Functions close it
    original_close = plt.close
    original_show = plt.show
    captured_fig = None
    
    def capture_close(fig=None):
        nonlocal captured_fig
        captured_fig = plt.gcf()  # Capture before closing
        
    def capture_show():
        nonlocal captured_fig
        captured_fig = plt.gcf()  # Capture when shown
    
    # 4. Temporarily replace matplotlib functions
    plt.close = capture_close
    plt.show = capture_show
    
    try:
        # 5. Call original plotting function
        plot_function(*args, **kwargs)
        
        # 6. Ensure we have a figure
        if captured_fig is None:
            captured_fig = plt.gcf()
            
    finally:
        # 7. Restore original functions
        plt.close = original_close
        plt.show = original_show
    
    # 8. Display in Streamlit
    if captured_fig and captured_fig.get_axes():
        st.pyplot(captured_fig, use_container_width=True)
        return True
    
    return False
```

### 6.3 Original Plotting Functions

**Location**: `Functions/` folder (16 specialized functions)

#### Key Functions
- `plot_psd()`: Particle size distribution curves
- `plot_atterberg_chart()`: Plasticity classification charts  
- `plot_UCS_vs_depth()`: Strength vs depth profiles
- `plot_by_chainage()`: Property vs distance plots
- `plot_histogram()`: Statistical distributions
- `plot_CBR_swell_WPI_histogram()`: Expansive soil classification
- `plot_category_by_thickness()`: Thickness distribution analysis

#### Function Characteristics
- **Unchanged from Jupyter notebook**: Exact same code and logic
- **Professional output**: High-quality matplotlib figures
- **Extensive parameterization**: 20-50 parameters per function
- **Industry standards**: Following geotechnical engineering conventions

---

## 7. Architecture Components

### 7.1 Directory Structure
```
Data Analysis App/
├── main_app.py                 # Main Streamlit application
├── Functions/                  # Original Jupyter plotting functions
│   ├── plot_psd.py
│   ├── plot_atterberg_chart.py
│   ├── plot_UCS_vs_depth.py
│   └── [14 other plotting functions]
├── utils/                      # Streamlit integration utilities
│   ├── data_processing.py      # Data loading and validation
│   ├── plotting_utils.py       # Plot display integration
│   ├── plot_defaults.py        # Parameter standardization
│   ├── psd_analysis.py         # PSD-specific processing
│   ├── atterberg_analysis.py   # Atterberg-specific processing
│   ├── ucs_analysis.py         # UCS-specific processing
│   ├── spt_analysis.py         # SPT-specific processing
│   ├── spatial_analysis.py     # Spatial/chainage analysis
│   └── comprehensive_analysis.py # Histogram and CBR/WPI analysis
├── Input/                      # Data input directory
│   └── BH_Interpretation.xlsx  # Thickness analysis data
└── Output/                     # Generated outputs
```

### 7.2 Data Flow Between Components

```
User Upload (main_app.py)
        ↓
Data Loading (data_processing.py)
        ↓
Test-Specific Processing (analysis modules)
        ↓
Parameter Collection (Streamlit UI)
        ↓
Function Call Bridge (plotting_utils.py)
        ↓
Original Plotting Functions (Functions/)
        ↓
Figure Capture & Display (Streamlit)
        ↓
Download & Dashboard Storage
```

### 7.3 Key Design Principles

1. **Separation of Concerns**
   - Data processing separate from plotting
   - UI logic separate from analysis logic
   - Original functions preserved unchanged

2. **Golden Standard Fidelity**
   - Functions folder code identical to Jupyter notebook
   - Same parameters, same outputs, same logic
   - Only display mechanism adapted for Streamlit

3. **Extensibility**
   - Easy to add new test types
   - Modular analysis components
   - Standardized parameter interface

4. **Professional Output**
   - Engineering industry standards
   - Consistent visual formatting
   - High-quality figure generation

---

## 8. Technical Implementation Details

### 8.1 Dynamic Property Detection

**Algorithm**: Smart pattern matching for engineering properties
```python
def get_numerical_properties_smart(df):
    """Intelligently detect engineering properties using regex patterns"""
    
    property_patterns = {
        'strength': [
            r'UCS.*MPA',              # UCS (MPa)
            r'IS50[AD]?.*MPA',        # Is50a (MPa), Is50d (MPa)
            r'CBR.*%',                # CBR (%)
            r'SPT.*N.*VALUE',         # SPT N Value
        ],
        'index': [
            r'LL.*%',                 # LL (%)
            r'PI.*%',                 # PI (%)
            r'WPI',                   # WPI
        ],
        'spatial': [
            r'CHAINAGE',              # Chainage
            r'FROM.*MBGL',            # From_mbgl
        ]
    }
    
    # Apply patterns to detect relevant columns
    detected_props = []
    for category, patterns in property_patterns.items():
        for pattern in patterns:
            matches = [col for col in df.columns 
                      if re.search(pattern, col, re.IGNORECASE)]
            detected_props.extend(matches)
    
    return detected_props
```

### 8.2 Parameter Parsing System

**Tuple Parsing**: Converting UI strings to Python objects
```python
def parse_tuple(tuple_str, default):
    """Parse tuple string with error handling"""
    try:
        if tuple_str.strip().startswith('(') and tuple_str.strip().endswith(')'):
            tuple_content = tuple_str.strip()[1:-1]
            values = [float(x.strip()) for x in tuple_content.split(',')]
            return tuple(values)
        else:
            values = [float(x.strip()) for x in tuple_str.split(',')]
            return tuple(values)
    except:
        return default
```

### 8.3 Error Handling & Validation

**Comprehensive Error Management**:
```python
def safe_plot_generation(plot_function, data, parameters):
    """Wrapper for safe plot generation with error handling"""
    try:
        # Validate data
        if data.empty:
            raise ValueError("No data available for plotting")
        
        # Validate parameters
        validated_params = validate_plot_parameters(parameters)
        
        # Generate plot
        success = plot_function(data, **validated_params)
        
        if not success:
            raise RuntimeError("Plot generation failed")
            
        return True
        
    except Exception as e:
        st.error(f"Plot generation error: {str(e)}")
        # Fallback: Show data table instead
        st.dataframe(data.head(20))
        return False
```

### 8.4 Performance Optimization

**Caching Strategy**:
```python
@st.cache_data
def load_and_process_data(file_content):
    """Cache expensive data processing operations"""
    # Data loading and processing only runs when file changes
    pass

@st.cache_data
def extract_test_data(data_hash, test_name):
    """Cache test-specific data extraction"""
    # Avoid re-extracting same test data
    pass
```

### 8.5 Memory Management

**Figure Cleanup**:
```python
def clean_matplotlib_memory():
    """Proper cleanup of matplotlib figures"""
    plt.close('all')  # Close all figures
    plt.clf()         # Clear current figure
    plt.cla()         # Clear current axes
    
# Called after each plot generation
```

---

## Conclusion

This system successfully transforms complex, multi-test geotechnical laboratory data into professional engineering visualizations through a carefully designed pipeline that:

1. **Preserves Data Integrity**: No loss of information during processing
2. **Maintains Professional Standards**: Engineering industry conventions
3. **Provides Flexibility**: Extensive customization options
4. **Ensures Reliability**: Comprehensive error handling and validation
5. **Scales Efficiently**: Handles large datasets (2,400+ records)

The architecture enables engineers to generate publication-quality plots from raw lab data with minimal effort while maintaining full control over visualization parameters and following geotechnical engineering best practices.