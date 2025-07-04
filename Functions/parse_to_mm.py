import itertools
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline # Keep import for optional use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns # For KDE plots (optional)
import os
import warnings
from pandas.api.types import is_numeric_dtype, is_object_dtype
from matplotlib.container import BarContainer # NEW IMPORT
from typing import List, Optional, Union, Sequence, Dict, Tuple, Any
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import re



# ------------------------------------------------------------
# parse_to_mm
# ------------------------------------------------------------
def parse_to_mm(val):
    """Converts a string size value with units (mm, cm, um) to numeric mm."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Normalize micro characters and make lowercase
    s = s.replace('μ', 'u').replace('µ', 'u').lower()
    # Match number and optional unit
    m = re.match(r'([\d\.\-]+)\s*([a-z]+)?', s)
    if not m:
        return np.nan # Could not parse
    try:
        num = float(m.group(1))
        unit = m.group(2) or 'mm' # Default unit is mm
    except (ValueError, TypeError):
         return np.nan # Could not convert number

    # Unit conversion
    if unit == 'mm':
        return num
    elif unit == 'cm':
        return num * 10.0
    elif unit == 'um':
        return num / 1000.0
    else:
        # Unrecognized unit, assume mm (or handle as error)
        # print(f"Unrecognized unit '{unit}' for value {val}. Assuming mm.") # Optional warning
        return num # Assume mm if unit unknown

