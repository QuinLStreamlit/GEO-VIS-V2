# LGCFR Geotechnical Plotting Functions - Style Guide

This document provides detailed coding patterns and style conventions for the LGCFR geotechnical plotting functions. Follow these patterns when creating new functions or enhancing existing ones.

## Function Structure and Organization

### Standard Import Pattern
All plotting functions follow this import structure:
```python
import itertools
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline  # Keep import for optional use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns  # For KDE plots (optional)
import os
import warnings
from pandas.api.types import is_numeric_dtype, is_object_dtype
from matplotlib.container import BarContainer  # NEW IMPORT
from typing import List, Optional, Union, Sequence, Dict, Tuple, Any
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import re
```

### Parameter Organization Pattern
Function parameters should be organized in this hierarchical order with clear section comments:

1. **Essential Data Parameters** (required, most frequently used)
   - DataFrame and core column specifications
   - Primary configuration options

2. **Plot Appearance** (common customization)  
   - `title`, `title_suffix`, `figsize`
   - Core visual settings

3. **Category/Grouping Options** (data organization)
   - `category_col`, `facet_col`, `hue_col`, `stack_col`

4. **Axis Configuration** (layout control)
   - `xlim`, `ylim`, `xticks`, `yticks`, tick intervals

5. **Display Options** (control visibility)
   - `show_plot`, `show_legend`, `show_*` flags

6. **Output Control** (saving and export)
   - `output_filepath`, `save_plot`, `dpi`

7. **Visual Customization** (styling details)
   - Colors, markers, sizes, alpha values
   - Font sizes and weights

8. **Advanced Styling Options** (fine-tuning)
   - Style dictionaries, positioning options
   - Professional styling controls

Example parameter organization:
```python
def plot_function(
    # === Essential Data Parameters ===
    df: pd.DataFrame,
    value_col: str = 'default_col',
    
    # === Plot Appearance ===  
    title: Optional[str] = None,
    title_suffix: Optional[str] = None,
    figsize: tuple = (8, 6),
    
    # === Category Options ===
    category_col: Optional[str] = None,
    
    # === Axis Configuration ===
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    
    # === Display Options ===
    show_plot: bool = True,
    show_legend: bool = True,
    
    # === Output Control ===
    output_filepath: Optional[str] = None,
    save_plot: bool = True,
    dpi: int = 300,
    
    # === Visual Customization ===
    marker_size: int = 40,
    marker_alpha: float = 0.8,
    
    # === Advanced Styling Options ===
    legend_style: Optional[Dict[str, Any]] = None,
    axis_style: Optional[Dict[str, Any]] = None
):
```

## Documentation Standards

### Docstring Structure
Follow this comprehensive documentation pattern:

```python
def function_name():
    """
    Single-line summary of function purpose.
    
    Detailed description explaining the function's purpose, typical use cases,
    and any specialized functionality (2-3 sentences).
    
    Special features or enhanced capabilities should be highlighted here,
    including intelligent defaults, flexible data handling, etc.
    
    Parameters
    ----------
    === Essential Data Parameters ===
    param_name : type, default value
        Description of parameter with typical usage examples.
        Include expected data ranges and formats.
        
    === Plot Appearance ===
    title : str, optional
        Custom plot title. If None, uses intelligent default.
        Example: "Project XYZ Analysis"
        
    title_suffix : str, optional  
        Text to append to default title.
        Example: ": Q3 Results" → "Default Title: Q3 Results"
        
    [Continue for all parameter sections...]
    
    Returns
    -------
    None or specific return type
        Description of what is returned (if anything).
    
    Examples
    --------
    **Basic usage:**
    >>> function_name(data)
    
    **With customization:**
    >>> function_name(data, title="Custom Title", marker_size=60)
    
    **Advanced styling:**
    >>> function_name(data, 
    ...               legend_style={'frameon': True, 'shadow': True},
    ...               axis_style={'title_fontweight': 'normal'})
    """
```

## Input Validation Pattern

### Standard Validation Sequence
All functions should follow this validation pattern:

```python
# === Input Validation ===
# 1. Type validation
if not isinstance(data_df, pd.DataFrame): 
    raise TypeError("'data_df' must be DataFrame.")
if not value_cols: 
    raise ValueError("'value_cols' cannot be empty.")

# 2. Parameter compatibility checks
if hue_col and stack_col: 
    raise ValueError("`hue_col` and `stack_col` cannot be used simultaneously.")

# 3. Column existence validation  
required_cols = [col for col in [value_col, category_col, facet_col] if col is not None]
missing_cols = [col for col in required_cols if col not in data_df.columns]
if missing_cols: 
    raise ValueError(f"DataFrame missing columns: {missing_cols}")

# 4. Parameter value validation with warnings
if facet_orientation not in ['vertical', 'horizontal']:
    warnings.warn(f"Invalid 'facet_orientation': {facet_orientation}. Defaulting to 'vertical'.")
    facet_orientation = 'vertical'

# 5. Backward compatibility handling
if xlim is None and force_xlim is not None:
    xlim = force_xlim  # Maintain backward compatibility
```

## Data Processing Patterns

### Flexible Column Matching
Implement intelligent column pattern matching for geological data:

```python
def _find_column_flexible(df: pd.DataFrame, target_col: Optional[str]) -> Optional[str]:
    """Find column using flexible pattern matching for geological terms."""
    if target_col is None:
        return None
    
    # Exact match first
    if target_col in df.columns:
        return target_col
    
    # Case-insensitive matching
    for col in df.columns:
        if col.lower() == target_col.lower():
            return col
    
    # Pattern matching for common geological terms
    common_patterns = ['geology', 'material', 'formation', 'lithology']
    target_clean = target_col.lower().replace('_', '').replace(' ', '')
    
    for col in df.columns:
        col_clean = col.lower().replace('_', '').replace(' ', '')
        if target_clean in col_clean or col_clean in target_clean:
            return col
    
    return None
```

### Geological Category Recognition
Implement intelligent geological category normalization:

```python
def _normalize_geological_category(category_value):
    """Normalize geological categories using flexible pattern matching."""
    if pd.isna(category_value):
        return category_value
    
    cat_str = str(category_value).upper().strip()
    
    # ALLUVIAL/ALLUVIUM patterns
    alluvial_patterns = ['ALLUVIAL', 'ALLUVIUM', 'QA', 'QUAT', 'QUATERNARY']
    if any(pattern in cat_str for pattern in alluvial_patterns):
        return 'ALLUVIAL'
    
    # RESIDUAL/WEATHERED patterns  
    residual_patterns = ['RESIDUAL', 'RS', 'XW', 'WEATHERED', 'EXTREMELY WEATHERED']
    if any(pattern in cat_str for pattern in residual_patterns):
        return 'RESIDUAL'
    
    # Additional geological units...
    return category_value  # Return original if no match
```

## Styling Implementation Patterns

### Default Style System
Implement consistent default styling with override capabilities:

```python
# Default styling dictionaries
default_grid_style = {'linestyle': '--', 'color': 'grey', 'alpha': 0.35}
default_axis_style = {
    'xlabel_fontsize': 12, 'xlabel_fontweight': 'bold',
    'ylabel_fontsize': 12, 'ylabel_fontweight': 'bold', 
    'title_fontsize': 14, 'title_fontweight': 'bold'
}

# Apply custom styling with defaults as fallback
grid_params = {**default_grid_style, **(grid_style or {})}
axis_params = {**default_axis_style, **(axis_style or {})}

# Use in plotting
ax.grid(True, **grid_params)
ax.set_xlabel(xlabel, fontsize=axis_params['xlabel_fontsize'], 
              fontweight=axis_params['xlabel_fontweight'])
```

### Geological Color Schemes
Use intelligent color assignment for geological data:

```python
geological_colors = {
    'ALLUVIAL': 'darkorange',
    'RESIDUAL': 'green', 
    'FILL': 'lightblue',
    'DCF': 'brown',
    'RIN': 'purple',
    'RJBW': 'red',
    'TOS': 'blue'
}

# Override with user palette if provided
final_colors = {**geological_colors, **(user_palette or {})}
```

## Plot Implementation Structure

### Standard Plotting Flow
Follow this implementation sequence:

```python
def plot_function():
    # 1. Input validation (see validation pattern above)
    
    # 2. Data preparation and processing
    data = df.copy()
    # Apply normalization, filtering, etc.
    
    # 3. Style application  
    if plot_style:
        try: 
            plt.style.use(plot_style)
        except: 
            print(f"Warning: matplotlib style '{plot_style}' not found.")
    
    # 4. Figure creation
    fig, ax = plt.subplots(figsize=figsize)
    
    # 5. Main plotting logic
    # Create plots with intelligent defaults
    
    # 6. Axis customization
    # Apply styling, limits, labels
    
    # 7. Legend handling
    if show_legend:
        legend_params = {'loc': legend_loc, 'fontsize': legend_fontsize}
        if legend_style:
            legend_params.update(legend_style)
        ax.legend(**legend_params)
    
    # 8. Layout optimization
    plt.tight_layout()
    
    # 9. Save and display
    if output_filepath:
        try:
            plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {output_filepath}")
        except Exception as e:
            print(f"Error saving: {e}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # 10. Completion message
    print(f"--- Plotting finished ---")
```

## Error Handling and User Communication

### Graceful Error Management
```python
# Use warnings for non-critical issues
warnings.warn("Non-critical issue with fallback behavior.")

# Use exceptions for critical failures
raise ValueError("Critical validation failure.")

# Provide helpful error messages
try:
    # Plotting operation
except Exception as e:
    print(f"Warning: Operation failed with error: {e}. Using fallback.")
```

### User Feedback Pattern
```python
# Start message
print(f"--- Starting {function_name} ---")

# Progress indicators for complex operations
print(f"Processing {len(categories)} categories...")

# Success confirmations  
print(f"Plot saved successfully to: {filepath}")

# Completion message
print(f"--- Plotting finished ---")
```

## Backward Compatibility Guidelines

1. **Never change existing parameter names** - Add new parameters only
2. **Provide default values** for all new parameters that preserve existing behavior  
3. **Handle deprecated parameters** with warnings:
   ```python
   if old_param is not None:
       warnings.warn("'old_param' is deprecated. Use 'new_param' instead.")
       new_param = old_param if new_param is None else new_param
   ```
4. **Maintain function signatures** - Only add optional parameters
5. **Test existing functionality** after any modifications

## Advanced Styling Pattern Examples

### Dictionary-Based Styling
All styling should use dictionary parameters for maximum flexibility:

```python
# Grid styling
grid_style = {'linestyle': ':', 'color': 'blue', 'alpha': 0.3, 'linewidth': 0.8}

# Legend styling  
legend_style = {'frameon': True, 'shadow': True, 'fancybox': True, 'framealpha': 0.9}

# Scatter styling
scatter_style = {'edgecolors': 'black', 'linewidths': 0.8, 'zorder': 10}

# Trendline styling
trendline_style = {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.9}

# Equation text styling
equation_style = {'fontsize': 10, 'bbox': dict(boxstyle='square', facecolor='yellow')}
```

### Position and Layout Control
Provide flexible positioning options:

```python
# Equation positioning in axes coordinates (0-1)
equation_position = (0.02, 0.98)  # Top-left
equation_position = (0.98, 0.02)  # Bottom-right (default)

# Legend positioning
legend_loc = 'upper left'  # Standard matplotlib locations
legend_bbox_to_anchor = (1.05, 1)  # Outside plot area
```

This style guide ensures consistency across all plotting functions while maintaining the flexibility and professional quality that characterizes the LGCFR geotechnical analysis suite.