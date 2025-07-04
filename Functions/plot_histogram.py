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

def plot_histogram(
        data_df: pd.DataFrame,
        value_cols: Union[str, List[str]],
        facet_col: Optional[str] = None,
        facet_orientation: str = 'vertical',
        hue_col: Optional[str] = None,
        stack_col: Optional[str] = None,
        stack_group_order: Optional[List[str]] = None,  # MODIFIED: New parameter
        bins: Optional[Union[int, Sequence[float], str, Dict[Any, Optional[Union[int, Sequence[float], str]]]]] = None,
        density: bool = False,
        kde: bool = False,
        log_scale: Union[bool, str] = False,
        show_stats: bool = False,
        stats_location: str = 'upper right',
        integer_xticks: bool = False,
        xtick_interval: Optional[float] = None,
        tick_direction: str = 'out',
        show_minor_ticks: bool = False,
        sharex: bool = False,
        sharey: bool = False,
        xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        force_xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,  # Kept for backward compatibility
        force_ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,  # Kept for backward compatibility
        output_filepath: Optional[str] = None,
        save_plot: bool = False,
        show_plot: bool = True,
        dpi: int = 300,
        style: Optional[str] = 'seaborn-v0_8-colorblind',
        figsize: Optional[Tuple[float, float]] = None,
        color: Optional[str] = '#0072B2',
        cmap_name: Optional[str] = 'tab10',
        palette: Optional[Union[List[str], Dict[Any, str]]] = None,
        alpha: float = 0.9,
        bar_width: float = 0.8,
        hist_edgecolor: Optional[str] = 'black',
        hist_linewidth: float = 0.6,
        kde_linewidth: float = 2.0,
        title: Optional[str] = None,
        title_suffix: Optional[str] = None,
        show_legend: bool = True,
        fig_title_fontsize: int = 15,
        fig_title_fontweight: str = 'bold',
        subplot_title_fontsize: int = 14,
        subplot_title_fontweight: str = 'normal',
        label_fontsize: int = 13,
        label_fontweight: str = 'normal',
        tick_fontsize: int = 12,
        legend_fontsize: int = 11,
        legend_title_fontsize: int = 12,
        stats_fontsize: int = 10,
        show_grid: bool = True,
        grid_axis: str = 'y',
        grid_linestyle: str = '--',
        grid_alpha: float = 0.4,
        legend_loc: str = 'best',
        legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
        custom_xlabels: Optional[Dict[str, str]] = None,
        custom_ylabel: Optional[str] = None,
        debug_mode: bool = False,
        
        # === Plot Cleanup Control ===
        close_plot: bool = True
) -> None:
    # --- Input Validation & Setup ---
    if not isinstance(data_df, pd.DataFrame): raise TypeError("'data_df' must be DataFrame.")
    if not value_cols: raise ValueError("'value_cols' cannot be empty.")
    if hue_col and stack_col: raise ValueError("`hue_col` and `stack_col` cannot be used simultaneously.")
    if isinstance(value_cols, str):
        value_cols_list = [value_cols]
    else:
        value_cols_list = list(value_cols)
    required_cols = value_cols_list[:]
    if facet_col:
        if not isinstance(facet_col, str): raise TypeError("`facet_col` must be a string.")
        required_cols.append(facet_col)
        if len(value_cols_list) > 1: raise ValueError("If `facet_col` is used, `value_cols` must be a single string.")
        if facet_orientation not in ['vertical', 'horizontal']:
            warnings.warn(f"Invalid 'facet_orientation': {facet_orientation}. Defaulting to 'vertical'.");
            facet_orientation = 'vertical'
    grouping_col = None;
    is_stacking = False;
    is_hueing = False
    if hue_col:
        if not isinstance(hue_col, str): raise TypeError("`hue_col` must be a string.")
        required_cols.append(hue_col);
        grouping_col = hue_col;
        is_hueing = True
    if stack_col:
        if not isinstance(stack_col, str): raise TypeError("`stack_col` must be a string.")
        required_cols.append(stack_col);
        grouping_col = stack_col;
        is_stacking = True
        if kde: warnings.warn("KDE plot ignored when `stack_col` is used."); kde = False
        if show_stats: warnings.warn(
            "When 'stack_col' is used, 'show_stats' will display stats for the entire facet, ignoring stack components.")
    # Handle axis limits with backward compatibility
    if xlim is None and force_xlim is not None:
        xlim = force_xlim  # Use force_xlim for backward compatibility
    if ylim is None and force_ylim is not None:
        ylim = force_ylim  # Use force_ylim for backward compatibility
        
    if xlim is not None and not (isinstance(xlim, tuple) and len(xlim) == 2):
        warnings.warn(f"`xlim` must be a tuple (min, max). Received: {xlim}. Ignoring.");
        xlim = None
    if ylim is not None and not (isinstance(ylim, tuple) and len(ylim) == 2):
        warnings.warn(f"`ylim` must be a tuple (min, max). Received: {ylim}. Ignoring.");
        ylim = None
    missing_cols = [col for col in required_cols if col is not None and col not in data_df.columns]
    if missing_cols: raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if kde and not is_stacking:
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError("Seaborn required for KDE plots.")
    if log_scale not in [True, False, 'x', 'y', 'both']: raise ValueError("`log_scale` invalid.")
    if log_scale is True: log_scale = 'y'
    if save_plot and not output_filepath: raise ValueError("If `save_plot`, `output_filepath` required.")

    # --- Filepath Handling ---
    if save_plot and output_filepath:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                warnings.warn(f"Cannot create dir '{output_dir}'. Save disabled. {e}"); save_plot = False

    # --- Data Preprocessing ---
    df_processed = data_df[list(set(col for col in required_cols if col is not None))].copy();
    initial_rows = len(df_processed)
    numeric_value_cols = [];
    categorical_value_cols = []
    for col in value_cols_list:
        if col not in df_processed.columns: continue
        if is_object_dtype(df_processed[col]) or isinstance(df_processed[col].dtype, pd.CategoricalDtype):  # MODIFIED
            categorical_value_cols.append(col);
            df_processed[col] = df_processed[col].astype(str).fillna('NaN')
        elif is_numeric_dtype(df_processed[col]):
            numeric_value_cols.append(col)
        else:
            original_nan_count = df_processed[col].isna().sum()
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            if df_processed[col].isna().sum() > original_nan_count: warnings.warn(f"Coerced NaNs in '{col}'.")
            if is_numeric_dtype(df_processed[col]):
                numeric_value_cols.append(col)
            else:
                categorical_value_cols.append(col); df_processed[col] = df_processed[col].astype(str).fillna(
                    'NaN'); warnings.warn(f"'{col}' treated as categorical.")
    if grouping_col and grouping_col in df_processed.columns: df_processed[grouping_col] = df_processed[
        grouping_col].fillna('NaN').astype(str)
    if facet_col and facet_col in df_processed.columns: df_processed[facet_col] = df_processed[facet_col].fillna(
        'NaN').astype(str)
    df_processed.dropna(subset=value_cols_list, inplace=True)  # Drop rows where value_col is NaN for plotting
    if len(df_processed) < initial_rows and not df_processed.empty: print(
        f"INFO: Removed {initial_rows - len(df_processed)} rows with NaNs in 'value_cols'.")
    if df_processed.empty: print("Warning: DataFrame empty. No plot."); return

    # --- Determine Grouping Order and Colors ---
    ordered_groups = [None];
    color_map = {}
    default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if grouping_col and grouping_col in df_processed.columns:
        unique_grps = sorted(df_processed[grouping_col].unique())
        num_grps = len(unique_grps)
        if num_grps == 0:
            grouping_col = None;
            is_stacking = False;
            is_hueing = False
            warnings.warn(f"No unique groups in '{grouping_col}'. Grouping ignored.")
        else:
            # MODIFIED: Logic for ordered_groups based on stack_group_order
            if is_stacking:
                if stack_group_order:
                    # Validate that stack_group_order contains all and only the unique groups from the data
                    if sorted(list(set(stack_group_order))) == sorted(
                            list(set(unique_grps))):  # Ensure sets for comparison
                        ordered_groups = [g for g in stack_group_order if
                                          g in unique_grps]  # Maintain order, filter by actual groups
                        if len(ordered_groups) != len(unique_grps):
                            warnings.warn(
                                f"Provided 'stack_group_order' ({stack_group_order}) "
                                f"contains items not present in actual data groups for '{grouping_col}' ({unique_grps}) "
                                "or is missing some. Using intersection in specified order and then remaining groups by frequency."
                            )
                            # Fallback for partial match: use provided order for common elements, then append others by frequency
                            remaining_groups = [g for g in df_processed[grouping_col].value_counts().index.tolist() if
                                                g not in ordered_groups]
                            ordered_groups.extend(remaining_groups)

                    else:
                        warnings.warn(
                            f"Provided 'stack_group_order' ({stack_group_order}) "
                            f"does not perfectly match unique groups in '{grouping_col}' ({unique_grps}). "
                            "Using default frequency-based order."
                        )
                        ordered_groups = df_processed[grouping_col].value_counts().index.tolist()
                else:
                    ordered_groups = df_processed[grouping_col].value_counts().index.tolist()
            elif is_hueing:  # For hueing, typically sorted unique groups are fine
                ordered_groups = unique_grps
            else:  # Should not be reached if grouping_col is set and num_grps > 0
                ordered_groups = unique_grps

            if palette:
                if isinstance(palette, list):
                    if len(palette) < num_grps: warnings.warn("Palette list shorter than groups. Colors will cycle.")
                    color_map = {g: palette[i % len(palette)] for i, g in enumerate(ordered_groups)}
                elif isinstance(palette, dict):
                    color_map = palette.copy()
                    for i, g in enumerate(ordered_groups):
                        if g not in color_map: color_map[g] = default_color_list[
                            i % len(default_color_list)]; warnings.warn(f"Palette missing group '{g}'. Using default.")
                else:
                    warnings.warn("Invalid 'palette'. Using 'cmap_name'."); palette = None
            if not palette:  # palette could have been set to None if invalid
                try:
                    cmap_obj = plt.get_cmap(cmap_name);
                    colors_from_cmap = cmap_obj(np.linspace(0.1, 0.9, num_grps) if num_grps > 0 else [])
                    color_map = {g: colors_from_cmap[i] for i, g in enumerate(ordered_groups)}
                except ValueError:
                    warnings.warn(f"Invalid 'cmap_name'. Using defaults."); color_map = {
                        g: default_color_list[i % len(default_color_list)] for i, g in enumerate(ordered_groups)}
    if not grouping_col and color: color_map = {None: color}

    # --- Determine Plot Layout & Size ---
    facets_values = sorted(df_processed[facet_col].unique()) if facet_col and facet_col in df_processed.columns else [
        None]
    is_true_faceting = bool(facet_col and facet_col in df_processed.columns and len(facets_values) > 0 and not (
                len(facets_values) == 1 and facets_values[0] == 'NaN' and df_processed[facet_col].nunique() == 1))
    if not is_true_faceting: facets_values = [None]; facet_col = None
    n_actual_facets = len(facets_values)
    if is_true_faceting:
        n_rows, n_cols = (1, n_actual_facets) if facet_orientation == 'horizontal' else (n_actual_facets, 1)
    else:
        n_rows, n_cols = len(value_cols_list), 1
    if figsize is None:
        rh = 4.5 if (show_stats or (kde and not is_stacking)) else 4.0;
        cw = 7.0
        figsize = (max(cw, cw * n_cols), max(rh, rh * n_rows))

    # --- Plotting ---
    context_manager = plt.style.context(style) if style else warnings.catch_warnings()
    with context_manager:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=False)
        axes = axes.flatten();
        plot_idx = 0

        for facet_val_current_loop in facets_values:
            subplot_df = df_processed
            if is_true_faceting and facet_val_current_loop is not None:
                subplot_df = df_processed[df_processed[facet_col] == facet_val_current_loop]
            if subplot_df.empty and is_true_faceting:
                if plot_idx < len(axes): ax = axes[plot_idx]; ax.set_title(
                    f"{str(facet_col).replace('_', ' ').title()} = {str(facet_val_current_loop)}"); ax.text(0.5, 0.5,
                                                                                                            "No data",
                                                                                                            ha='center',
                                                                                                            va='center',
                                                                                                            transform=ax.transAxes); ax.set_yticks(
                    []); ax.set_xticks([]); plot_idx += 1
                continue
            cols_for_this_ax = value_cols_list if not is_true_faceting else [value_cols_list[0]]
            for val_col_name in cols_for_this_ax:
                if plot_idx >= len(axes): break
                ax = axes[plot_idx];
                data_series_to_plot = subplot_df[val_col_name]
                if data_series_to_plot.empty:
                    title_empty_subplot = "";
                    if is_true_faceting and facet_val_current_loop is not None:
                        title_empty_subplot = f"{str(facet_col).replace('_', ' ').title()} = {str(facet_val_current_loop)}"
                    elif not is_true_faceting and len(value_cols_list) > 1:
                        title_empty_subplot = val_col_name.replace('_', ' ').title()
                    ax.set_title(f"{title_empty_subplot}\n({val_col_name}: No data)");
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes);
                    ax.set_yticks([]);
                    ax.set_xticks([]);
                    plot_idx += 1
                    continue

                is_cat_plot = val_col_name in categorical_value_cols
                plot_handles, plot_labels = [], []

                if is_cat_plot:
                    series_cat = data_series_to_plot;
                    color_cat = color_map.get(None, default_color_list[0])
                    if is_stacking and stack_col in subplot_df.columns:
                        try:
                            ct = pd.crosstab(series_cat, subplot_df[stack_col])
                            # Use ordered_groups (which respects stack_group_order if provided)
                            grps_cat_stk = ordered_groups if ordered_groups != [None] and all(
                                g in ct.columns for g in ordered_groups if g is not None) else sorted(
                                subplot_df[stack_col].unique())
                            # Ensure ct is reindexed with groups present in the actual crosstab data
                            actual_ct_cols = [g for g in grps_cat_stk if g in ct.columns]
                            ct = ct.reindex(columns=actual_ct_cols, fill_value=0)

                            colors_cat_stk = [color_map.get(str(g), default_color_list[i % len(default_color_list)]) for
                                              i, g in enumerate(ct.columns)]
                            ct.plot(kind='bar', stacked=True, ax=ax, color=colors_cat_stk, alpha=alpha, width=bar_width,
                                    edgecolor=hist_edgecolor, linewidth=hist_linewidth, legend=False, zorder=2)
                            plot_handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors_cat_stk];
                            plot_labels = [str(g) for g in ct.columns]
                        except Exception as e:
                            warnings.warn(
                                f"Stacked bar for '{val_col_name}' failed: {e}"); series_cat.value_counts().sort_index().plot(
                                kind='bar', ax=ax, color=color_cat, alpha=alpha, width=bar_width,
                                edgecolor=hist_edgecolor, linewidth=hist_linewidth, zorder=2)
                    else:
                        if is_hueing: warnings.warn(f"hue_col ignored for categorical '{val_col_name}'.")
                        series_cat.value_counts().sort_index().plot(kind='bar', ax=ax, color=color_cat, alpha=alpha,
                                                                    width=bar_width, edgecolor=hist_edgecolor,
                                                                    linewidth=hist_linewidth, zorder=2)
                    ax.tick_params(axis='x', rotation=0)
                else:  # Numerical Plot
                    bins_hist = bins
                    if is_true_faceting and isinstance(bins, dict):
                        key_b = str(facet_val_current_loop)
                        if key_b in bins:
                            bins_hist = bins[key_b]
                        elif '__default__' in bins:
                            bins_hist = bins['__default__']
                        else:
                            bins_hist = 'auto'; warnings.warn(
                                f"Bin config for facet '{key_b}' not found. Using 'auto'.")

                    if show_stats and not data_series_to_plot.empty:
                        c, m, md, s = len(
                            data_series_to_plot), data_series_to_plot.mean(), data_series_to_plot.median(), data_series_to_plot.std()
                        sc, sm, smd, ss = f"{c}", f"{m:.2f}" if pd.notna(m) else "nan", f"{md:.2f}" if pd.notna(
                            md) else "nan", f"{s:.2f}" if pd.notna(s) else "nan"
                        base_txt_s = f"N={sc}\nMean={sm}\nMed={smd}\nStd={ss}"
                        if is_stacking or not is_hueing:
                            pfx = "Statistics\n" if is_stacking else "";
                            txt = f"{pfx}{base_txt_s}"
                            loc_stat = 'upper left' if is_stacking and stats_location == 'upper right' else stats_location
                            xp, yp, hap, vap = _get_stats_position(loc_stat, 0, 1, stats_fontsize)
                            ax.text(xp, yp, txt, transform=ax.transAxes, fontsize=stats_fontsize, ha=hap, va=vap,
                                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, edgecolor='grey'))

                    active_kde_plt = kde and not is_stacking
                    # current_ordered_groups_for_plot is derived from ordered_groups which now respects stack_group_order
                    current_ordered_groups_for_plot = [g for g in ordered_groups if
                                                       g is not None] if ordered_groups != [None] else []

                    if is_stacking and stack_col in subplot_df.columns:
                        d_stk, c_stk, l_stk = [], [], []
                        for i_s, s_val in enumerate(
                                current_ordered_groups_for_plot):  # This uses the potentially re-ordered groups
                            s_data = data_series_to_plot[subplot_df[stack_col] == s_val]
                            if not s_data.empty: d_stk.append(s_data); c_stk.append(
                                color_map.get(s_val, default_color_list[i_s % len(default_color_list)])); l_stk.append(
                                str(s_val))
                        if d_stk: _, _, patches = ax.hist(d_stk, bins=bins_hist, density=density, color=c_stk,
                                                          alpha=alpha, edgecolor=hist_edgecolor,
                                                          linewidth=hist_linewidth, label=l_stk, stacked=True, zorder=2)
                        if patches: plot_handles = patches  # patches is list of BarContainers
                        plot_labels = l_stk if d_stk else []
                    elif is_hueing and hue_col in subplot_df.columns:
                        # For hueing, current_ordered_groups_for_plot is sorted unique_grps
                        h_grps_data = [h for h in current_ordered_groups_for_plot if
                                       not data_series_to_plot[subplot_df[hue_col] == h].empty()]
                        n_h = len(h_grps_data);
                        h_idx = 0
                        temp_handles_hue, temp_labels_hue = [], []
                        for i_h, h_val in enumerate(
                                current_ordered_groups_for_plot):  # Iterates by sorted unique group names
                            h_data = data_series_to_plot[subplot_df[hue_col] == h_val]
                            if h_data.empty: continue
                            h_color = color_map.get(h_val, default_color_list[i_h % len(default_color_list)])
                            _, _, patches_h = ax.hist(h_data, bins=bins_hist, density=density, color=h_color,
                                                      alpha=alpha, edgecolor=hist_edgecolor, linewidth=hist_linewidth,
                                                      label=str(h_val), zorder=2)
                            if patches_h: temp_handles_hue.append(patches_h[0]); temp_labels_hue.append(str(h_val))
                            if active_kde_plt:
                                try:
                                    sns.kdeplot(h_data, ax=ax, color=h_color, linewidth=kde_linewidth, label=None,
                                                zorder=3)
                                except Exception as e:
                                    warnings.warn(f"KDE for hue '{h_val}' failed: {e}")
                            if show_stats:
                                ch, mh, mdh, sdh = len(h_data), h_data.mean(), h_data.median(), h_data.std()
                                smh, smdh, ssdh = f"{mh:.2f}" if pd.notna(mh) else "nan", f"{mdh:.2f}" if pd.notna(
                                    mdh) else "nan", f"{sdh:.2f}" if pd.notna(sdh) else "nan"
                                txth = f"[{h_val}]\nN={ch}\nMean={smh}\nMed={smdh}\nStd={ssdh}"
                                xh, yh, hah, vah = _get_stats_position(stats_location, h_idx, n_h, stats_fontsize)
                                ax.text(xh, yh, txth, transform=ax.transAxes, fontsize=stats_fontsize, ha=hah, va=vah,
                                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, edgecolor='grey'));
                                h_idx += 1
                        plot_handles = temp_handles_hue;
                        plot_labels = temp_labels_hue
                    else:  # Single histogram
                        s_data = data_series_to_plot;
                        s_color = color_map.get(None, default_color_list[0])
                        if not s_data.empty:
                            ax.hist(s_data, bins=bins_hist, density=density, color=s_color, alpha=alpha,
                                    edgecolor=hist_edgecolor, linewidth=hist_linewidth, zorder=2)
                            if active_kde_plt:
                                try:
                                    sns.kdeplot(s_data, ax=ax, color=s_color, linewidth=kde_linewidth, zorder=3)
                                except Exception as e:
                                    warnings.warn(f"KDE for single histogram failed: {e}")
                    if integer_xticks: ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
                    # Custom X-tick interval support
                    elif xtick_interval is not None and xtick_interval > 0:
                        ax.xaxis.set_major_locator(mticker.MultipleLocator(xtick_interval))
                    if log_scale in ['x', 'both']: ax.set_xscale('log')
                    if log_scale in ['y', 'both']: ax.set_yscale('log')
                    if ax.get_xscale() != 'log' and not integer_xticks: xlims = ax.get_xlim();ax.set_xlim(
                        max(0, xlims[0]), xlims[1])

                # --- Common Axis & Label Styling & LEGEND ---
                title_subplot = "";
                if is_true_faceting and facet_val_current_loop is not None:
                    title_subplot = f"{str(facet_col).replace('_', ' ').title()} = {str(facet_val_current_loop)}"
                elif not is_true_faceting and len(value_cols_list) > 1:
                    title_subplot = val_col_name.replace('_', ' ').title()
                ax.set_title(title_subplot, fontsize=subplot_title_fontsize, fontweight=subplot_title_fontweight,
                             pad=10)
                xlabel_current = custom_xlabels.get(val_col_name, val_col_name.replace('_',
                                                                                       ' ').title()) if custom_xlabels else val_col_name.replace(
                    '_', ' ')
                ax.set_xlabel(xlabel_current, fontsize=label_fontsize, fontweight=label_fontweight)
                if (plot_idx % n_cols == 0) or not sharey:
                    ylabel_current = custom_ylabel if custom_ylabel else (
                        'Frequency Count' if is_cat_plot else ('Probability Density' if density else 'Frequency Count'))
                    ax.set_ylabel(ylabel_current, fontsize=label_fontsize, fontweight=label_fontweight)
                elif sharey:
                    ax.set_ylabel("")
                ax.tick_params(axis='both', labelsize=tick_fontsize, direction=tick_direction, which='major')
                if show_minor_ticks: ax.minorticks_on(); ax.tick_params(axis='both', which='minor',
                                                                        direction=tick_direction)
                ax.yaxis.set_major_locator(
                    mticker.MaxNLocator(integer=(not density or is_cat_plot), nbins=6 if n_rows * n_cols > 2 else 8))
                if ax.get_yscale() != 'log': ax.set_ylim(bottom=0)
                
                # Apply individual axis limits if specified
                if xlim is not None:
                    xmin_new = xlim[0] if xlim[0] is not None else ax.get_xlim()[0]
                    xmax_new = xlim[1] if xlim[1] is not None else ax.get_xlim()[1]
                    ax.set_xlim(xmin_new, xmax_new)
                if ylim is not None:
                    ymin_new = ylim[0] if ylim[0] is not None else ax.get_ylim()[0]
                    ymax_new = ylim[1] if ylim[1] is not None else ax.get_ylim()[1]
                    ax.set_ylim(ymin_new, ymax_new)
                
                ax.spines[['top', 'right']].set_visible(False);
                ax.spines[['left', 'bottom']].set_linewidth(0.8)
                if show_grid:
                    ax.grid(axis=grid_axis, linestyle=grid_linestyle, alpha=grid_alpha,
                            which='major'); ax.set_axisbelow(True)
                else:
                    ax.grid(False)

                # --- LEGEND LOGIC (REVISED BASED ON BarContainer KNOWLEDGE) ---
                if debug_mode:
                    print(f"\n[DEBUG LEGEND] Subplot: Facet='{facet_val_current_loop}', ValueCol='{val_col_name}'")
                    print(
                        f"[DEBUG LEGEND] grouping_col: '{grouping_col}', is_stacking: {is_stacking}, is_hueing: {is_hueing}, is_cat_plot: {is_cat_plot}")
                    print(
                        f"[DEBUG LEGEND] Raw plot_handles type: {type(plot_handles)}, Has content: {bool(plot_handles)}")
                    if isinstance(plot_handles, list) and plot_handles: print(
                        f"[DEBUG LEGEND] First handle type: {type(plot_handles[0])}")
                    print(
                        f"[DEBUG LEGEND] Raw plot_labels: {plot_labels} (Length: {len(plot_labels) if plot_labels else 0})")
                    print(f"[DEBUG LEGEND] ordered_groups for legend consideration: {ordered_groups}")

                if grouping_col and plot_handles and plot_labels:
                    final_legend_handles = []
                    final_legend_labels = plot_labels

                    if is_stacking:
                        if not is_cat_plot:  # Numerical Stacked
                            if isinstance(plot_handles, list) and all(
                                    isinstance(item, BarContainer) for item in plot_handles):
                                final_legend_handles = plot_handles
                            elif debug_mode:
                                print(
                                    f"[DEBUG LEGEND] Numerical Stacked: plot_handles was NOT list of BarContainers. Type: {type(plot_handles)}")
                        else:  # Categorical Stacked
                            if isinstance(plot_handles, list) and all(
                                    isinstance(item, plt.Rectangle) for item in plot_handles):
                                final_legend_handles = plot_handles
                            elif debug_mode:
                                print(
                                    f"[DEBUG LEGEND] Categorical Stacked: plot_handles was NOT list of Rectangles. Type: {type(plot_handles)}")
                    elif is_hueing:  # Numerical Hued
                        if isinstance(plot_handles, list) and all(
                                hasattr(item, 'get_facecolor') for item in plot_handles):  # Check for patch-like
                            final_legend_handles = plot_handles
                        elif debug_mode:
                            print(
                                f"[DEBUG LEGEND] Numerical Hued: plot_handles was NOT list of Patches. Type: {type(plot_handles)}")

                    if debug_mode:
                        print(f"[DEBUG LEGEND] Processed final_legend_handles count: {len(final_legend_handles)}")
                        print(f"[DEBUG LEGEND] Processed final_legend_labels count: {len(final_legend_labels)}")

                    if show_legend and final_legend_handles and final_legend_labels and len(final_legend_handles) == len(
                            final_legend_labels):
                        if debug_mode: print(f"[DEBUG LEGEND] ---> Conditions MET. Drawing legend.")
                        lgd_title = str(grouping_col).replace('_', ' ').title()

                        # Ensure legend order matches desired ordered_groups if possible
                        # This part is tricky if handles/labels from plot don't directly match ordered_groups
                        # For now, we assume plot_labels and plot_handles are already in the correct order
                        # as determined by how they were generated (which should follow ordered_groups)

                        # If using stack_group_order or hueing with sorted unique_grps,
                        # plot_labels should already respect this order.

                        ax.legend(handles=final_legend_handles, labels=final_legend_labels, title=lgd_title,
                                  fontsize=legend_fontsize, title_fontsize=legend_title_fontsize,
                                  loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, frameon=False)
                    elif debug_mode:
                        print(f"[DEBUG LEGEND] ---> Legend SKIPPED. Reason(s):")
                        if not final_legend_handles: print(
                            f"[DEBUG LEGEND]     final_legend_handles is empty (original plot_handles type: {type(plot_handles)}).")
                        if not final_legend_labels: print(f"[DEBUG LEGEND]     final_legend_labels is empty.")
                        if final_legend_handles and final_legend_labels: print(
                            f"[DEBUG LEGEND]     Length mismatch: handles ({len(final_legend_handles)}), labels ({len(final_legend_labels)}).")
                elif grouping_col and debug_mode:
                    print(
                        f"[DEBUG LEGEND] ---> SKIPPING legend. plot_handles or plot_labels were None/empty despite grouping_col='{grouping_col}'.")
                # --- End Legend Logic ---
                plot_idx += 1
                if plot_idx >= n_rows * n_cols: break
            if plot_idx >= n_rows * n_cols: break
        for k_clean in range(plot_idx, n_rows * n_cols):
            if k_clean < len(axes): fig.delaxes(axes[k_clean])

    # --- Apply Forced Limits ---
    if fig and axes.size > 0 and plot_idx > 0:
        first_ax = axes[0]
        if sharex and xlim:
            xmin, xmax = first_ax.get_xlim();
            new_xmin = xlim[0] if xlim[0] is not None else xmin;
            new_xmax = xlim[1] if xlim[1] is not None else xmax
            if (new_xmin, new_xmax) != (xmin, xmax): first_ax.set_xlim(new_xmin, new_xmax)
        if sharey and ylim:
            ymin, ymax = first_ax.get_ylim();
            new_ymin = ylim[0] if ylim[0] is not None else ymin;
            new_ymax = ylim[1] if ylim[1] is not None else ymax
            if (new_ymin, new_ymax) != (ymin, ymax): first_ax.set_ylim(new_ymin, new_ymax)

    # --- Global Figure Title & Layout ---
    final_title = None
    if title is not None:
        final_title = title
    elif title_suffix is not None:
        # Create default title with suffix
        if len(value_cols_list) == 1:
            final_title = f"Histogram of {value_cols_list[0]}: {title_suffix}"
        else:
            final_title = f"Histogram: {title_suffix}"
    
    if final_title:
        title_y = 0.98
        if legend_bbox_to_anchor and legend_bbox_to_anchor[1] > 1.0 and n_rows == 1:
            title_y = min(0.98, fig.subplotpars.top + 0.08 if hasattr(fig,
                                                                      'subplotpars') and fig.subplotpars.top is not None else 0.98)
        elif n_rows == 1:
            title_y = 0.95
        fig.suptitle(final_title, fontsize=fig_title_fontsize, fontweight=fig_title_fontweight, y=title_y)
    try:
        rect_top = 0.93 if final_title else 0.96
        has_xlabel = False
        if plot_idx > 0:
            for ax_idx_check in range(min(plot_idx, len(fig.axes))):
                if fig.axes[ax_idx_check].axison and fig.axes[ax_idx_check].get_xlabel():
                    has_xlabel = True
                    break
        rect_bottom = 0.12 if has_xlabel else 0.05
        fig.tight_layout(rect=[0.04, rect_bottom, 0.96, rect_top], h_pad=1.8 if n_rows > 1 else 0.5,
                         w_pad=1.8 if n_cols > 1 else 0.5)
    except Exception as e:
        warnings.warn(f"Tight layout failed: {e}. Consider `plt.subplots_adjust()`.", UserWarning)
        if title: plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9)

    # --- Saving & Showing ---
    if save_plot and output_filepath:
        try:
            plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight'); print(f"Plot saved to: {output_filepath}")
        except Exception as e:
            print(f"Error saving: {e}")
    if show_plot: plt.show()
    if close_plot:
        plt.close(fig)
    print(f"--- Plotting finished ---")

def _get_stats_position(location_str: str, item_index: int = 0, total_items: int = 1, font_size: int = 10) -> Tuple[float, float, str, str]:
    if not isinstance(location_str, str): location_str = 'upper right'
    if 'left' in location_str: x_pos, ha = 0.02, 'left'
    elif 'center' in location_str: x_pos, ha = 0.5, 'center'
    else: x_pos, ha = 0.98, 'right'
    vertical_offset_factor = 0.05 + (font_size / 10.0) * 0.13
    if 'lower' in location_str:
        y_pos_base = 0.02; y_pos = y_pos_base + (item_index * vertical_offset_factor); va = 'bottom'
        if total_items > 1 and (y_pos_base + ((total_items - 1) * vertical_offset_factor) + 0.1) > 0.95: y_pos = 0.02 + (item_index * 0.05)
    elif 'center' in location_str:
        y_pos_base = 0.5; total_height_approx = (total_items -1) * vertical_offset_factor
        y_pos = y_pos_base + total_height_approx / 2 - (item_index * vertical_offset_factor); va = 'top'
        if total_items == 1: va = 'center'
    else:
        y_pos_base = 0.95; y_pos = y_pos_base - (item_index * vertical_offset_factor); va = 'top'
        if total_items > 1 and (y_pos_base - ((total_items - 1) * vertical_offset_factor) - 0.1) < 0.05: y_pos = 0.95 - (item_index * 0.05)
    return x_pos, y_pos, ha, va


