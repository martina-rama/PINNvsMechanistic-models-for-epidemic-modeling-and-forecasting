"""
Comparison of the probabilistic forecasts of the PINN and of the Mechanistic model (and, if available, of a
baseline), for the synthetic scenarios and for the real case. All the forecasts are read from files in the
same format (`forecast_results.pickle`, see `pinnvsmech.forecasting.forecast` and `pinnvsmech.mechanistic.convert`):
a dict indexed by the label of the week, whose entries have `raw`, `quantiles` (columns t, q01..q99), `t0`, `date`,
`y_obs`.

Figures:

    plot_probabilistic_forecasts   grid with one panel per rolling window (data, 90% and 50% bands, median)
    plot_fixed_horizon_grid        one panel per horizon: the forecast that was made h weeks earlier for every week
    plot_single_window             fit of one window (synthetic scenarios: retrospective fit at the last week T_LAST)
    plot_age_forecast_compare      synthetic scenarios: fit by age class, PINN vs Mechanistic, at the last week T_LAST
    fit_coverage                   empirical coverage of the intervals of the fit of one window

The same functions are used for the synthetic scenarios (x axis: week number = t + 1, observations = new infections)
and for the real case (x axis: dates MM-DD, observations = new deaths): the axis is chosen from the type of `dates`.
Positions of the plots = index t of the week; the first week is t = 0.

`T_LAST` (last week used in the plots) depends on the scenario: 33 for R0 = 1.5 and 23 for R0 = 2.0
(`pinnvsmech.inference.cases.SYNTHETIC_T0_LAST`). `trim_forecasts` keeps the windows whose forecast ends before T_LAST.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.transforms import offset_copy

from pinnvsmech.forecasting.windows import entry_at, trim_forecasts   # noqa: F401  (also available from here)

# ---------------------------------------------------------------------------
# Colours and styles
# ---------------------------------------------------------------------------

COLOR_PINN = mcolors.to_rgb("darkviolet")
COLOR_MECH = mcolors.to_rgb("darkcyan")
COLOR_BASELINE = mcolors.to_rgb("#E0A100")   # amber, dashed median

STYLE_BY_MODEL = {
    "pinn": dict(c90=(*COLOR_PINN, 0.18), c50=(*COLOR_PINN, 0.32), median=COLOR_PINN, median_ls="-"),
    "mech": dict(c90=(*COLOR_MECH, 0.18), c50=(*COLOR_MECH, 0.32), median=COLOR_MECH, median_ls="-"),
    "baseline": dict(c90=(*COLOR_BASELINE, 0.20), c50=(*COLOR_BASELINE, 0.35), median=COLOR_BASELINE, median_ls="--"),
}

AGE_LABELS = ("0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
              "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75+")


def _over(fg, bg=(1, 1, 1)):
    """Colour seen when `fg` (with its alpha) is on top of `bg`."""
    r, g, b, a = to_rgba(fg)
    return tuple(a * c + (1 - a) * d for c, d in zip((r, g, b), bg))


# ---------------------------------------------------------------------------
# Helpers: forecasts, windows, x axis
# ---------------------------------------------------------------------------

def describe_forecasts(models):
    """Print the number of windows and the range of t0 of every model ({name: forecasts dict}) and the common ones."""
    for name, d in models.items():
        t = sorted(int(v["t0"]) for v in d.values())
        print(f"{name:10s} {len(t):2d} windows | t0 {t[0]} -> {t[-1]}")
    common = set.intersection(*[set(int(v["t0"]) for v in d.values()) for d in models.values()])
    print("common:", len(common))


def _is_date_axis(dates):
    d = np.asarray(dates)
    return np.issubdtype(d.dtype, np.datetime64) or isinstance(d.flat[0], (pd.Timestamp, np.datetime64))


def _week_formatter(dates, date_format="%m-%d"):
    """Tick label of the position t: the date MM-DD (real case) or the week number t + 1 (synthetic scenarios)."""
    if _is_date_axis(dates):
        labels = [pd.Timestamp(d).strftime(date_format) for d in dates]

        def fmt(x, pos):
            i = int(round(x))
            return labels[i] if 0 <= i < len(labels) else ""
    else:
        def fmt(x, pos):
            return f"{int(round(x)) + 1}"
    return FuncFormatter(fmt)


def _common_keys(active):
    keys = None
    for d in active.values():
        keys = set(d.keys()) if keys is None else keys & set(d.keys())
    return keys


def _active(**forecasts):
    active = {k: v for k, v in forecasts.items() if v is not None}
    if not active:
        raise ValueError("Provide at least one of the forecasts")
    return active


# ---------------------------------------------------------------------------
# Grid: one panel per rolling window
# ---------------------------------------------------------------------------

def plot_probabilistic_forecasts(
    dates,
    obs,
    dt_train,
    forecasts_pinn=None,
    forecasts_mech=None,
    forecasts_baseline=None,
    grid_size="all",
    n_cols=4,
    error_width=90,
    error_width_1=50,
    ylim=(0, 1800),
    ylabel="Weekly new infections",
    title=None,
    show_title=True,
    figsize=None,
    dpi=300,
    label_pinn="PINN",
    label_mech="Mechanistic",
    label_baseline="Baseline",
    label_past="All data",
    label_forecast_obs="Look-back window",
    show_fit=True,
    legend_y=None,
    t0_min=None,
    t_last=None,
    drop_last=0,
    xtick_step=None,
):
    """
    Grid of panels, one for every rolling window t0: observations, 90% and 50% bands and median of every model.

    Parameters
    ----------
    dates : np.ndarray
        Dates of the weeks (real case: x axis with the dates MM-DD) or labels "week 1", ... (synthetic scenarios:
        x axis with the week number).
    obs : np.ndarray
        Observed weekly new deaths (real) or infections (synthetic).
    dt_train : int
        Length of the look-back window of the forecasts (weeks); the window t0 - dt_train + 1 .. t0 is drawn filled.
    forecasts_pinn, forecasts_mech, forecasts_baseline : dict or None
        At least one, in the `forecast_results` format. Only the windows that all of them have are shown.
    grid_size : int or "all"
        "all": `n_cols` columns and as many rows as needed for all the windows.
        int: square grid grid_size x grid_size; if there are more windows than panels, equispaced windows are chosen.
    ylim : tuple, or list with one tuple for every ROW of the grid
        y limits (of all the panels, or row by row to give room to the wide bands of the first windows).
    ylabel, title : str
        y label and title of the figure (`show_title=False` hides the title).
    show_fit : bool
        True: bands and median from week 0; False: only from t0 on (the future is shown, the fit is not).
    t0_min : int, optional
        First window shown (index).
    t_last : int, optional
        Last week shown (index): observations, bands and medians stop there.
    drop_last : int
        Number of windows removed at the end of the selection.
    xtick_step : int, optional
        Step of the ticks of the x axis. Default: 2 for dates; for week numbers 2, 3 or 4 depending on the row.
    """
    active = _active(pinn=forecasts_pinn, mech=forecasts_mech, baseline=forecasts_baseline)
    multi = len(active) > 1
    is_date = _is_date_axis(dates)
    obs = np.asarray(obs)

    n_weeks = len(obs)
    if t_last is not None:
        n_weeks = min(n_weeks, t_last + 1)

    # windows shared by all the models
    keys_common = _common_keys(active)
    ref_name = next(iter(active))
    ref = active[ref_name]
    cand = sorted(keys_common, key=lambda k: int(ref[k]["t0"]))
    if t0_min is not None:
        cand = [k for k in cand if int(ref[k]["t0"]) >= t0_min]
    if grid_size == "all":
        t0_keys = cand
    else:
        if not isinstance(grid_size, int) or grid_size < 1:
            raise ValueError("grid_size must be a positive integer or 'all'")
        n_cols = grid_size
        if drop_last > 0:
            cand = cand[:-drop_last]
        if len(cand) > grid_size ** 2:
            pick = np.linspace(0, len(cand) - 1, grid_size ** 2, dtype=int)
            cand = [cand[i] for i in pick]
        t0_keys = cand
        drop_last = 0
    if drop_last > 0:
        t0_keys = t0_keys[:-drop_last]

    n_plots = len(t0_keys)
    if n_plots == 0:
        raise ValueError("No forecast available for the chosen windows")
    n_rows = int(np.ceil(n_plots / n_cols))

    if (isinstance(ylim, (list, tuple)) and len(ylim) == n_rows
            and all(isinstance(x, (list, tuple)) for x in ylim)):
        row_ylims = list(ylim)
    else:
        row_ylims = [ylim] * n_rows

    if figsize is None:
        figsize = (13, 2.5 * n_rows) if grid_size == "all" else (13, 10)
    if legend_y is None:
        legend_y = 0.93 if is_date else 0.96

    font_scale = {2: 1.0, 3: 0.95, 4: 0.85, 5: 0.7}
    marker_scale = {2: 1.6, 3: 1.0, 4: 0.75, 5: 0.6}
    fs = font_scale.get(n_cols, 0.85)
    ms = marker_scale.get(n_cols, 1.0)
    label_fontsize = int((16 if is_date else 17) * fs)
    tick_fontsize = int(11 * fs)
    legend_fontsize = int(10 * fs)
    suptitle_fontsize = int(20 * fs)
    markersize = 7 * ms
    linewidth_data = 1.2 * ms

    label_by_model = {"pinn": label_pinn, "mech": label_mech, "baseline": label_baseline}
    styles = {name: STYLE_BY_MODEL[name] for name in active}
    formatter = _week_formatter(dates)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if show_title and title:
        fig.suptitle(title, fontsize=suptitle_fontsize, fontweight="bold")

    legend_elements = None
    for idx, key in enumerate(t0_keys):
        t0_train = int(ref[key]["t0"])
        for name, d in active.items():
            if name != ref_name and int(d[key]["t0"]) != t0_train:
                print(f"  (warning: t0 mismatch at {key}: {ref_name}={t0_train} {name}={d[key]['t0']} - "
                      f"using {ref_name}'s)")

        # last week of this panel: the last one with a forecast for at least one model
        t_max_plot = max(int(d[key]["quantiles"]["t"].max()) for d in active.values())
        t_max_plot = min(t_max_plot, n_weeks - 1)
        t_epi = np.arange(0, t_max_plot + 1)

        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        row = idx // n_cols
        window_train = np.arange(max(0, t0_train - dt_train + 1), t0_train + 1)

        # observations: all the data (open circles), look-back window (filled), current week (filled, same style)
        ax.plot(t_epi, obs[t_epi], ".-", color="black", markerfacecolor="white", markeredgecolor="black",
                markersize=markersize, markeredgewidth=0.8, linewidth=linewidth_data)
        ax.plot(t_epi[window_train], obs[window_train], ".-", color="black", markerfacecolor="black",
                markeredgecolor="black", markersize=markersize, markeredgewidth=0.8, linewidth=linewidth_data)
        ax.plot(t_epi[t0_train], obs[t0_train], "o", color="black", markerfacecolor="black",
                markeredgecolor="black", markersize=markersize, markeredgewidth=1.0)

        # bands and median of every model
        for name, d in active.items():
            style = styles[name]
            suf = f" ({label_by_model[name]})" if multi else ""
            data_q = d[key]["quantiles"]
            if not show_fit:
                data_q = data_q[data_q["t"] >= t0_train]
            if t_last is not None:
                data_q = data_q[data_q["t"] <= t_last]
            ax.fill_between(data_q["t"], data_q["q05"], data_q["q95"], color=style["c90"],
                            label=f"{error_width}% PI{suf}")
            ax.fill_between(data_q["t"], data_q["q25"], data_q["q75"], color=style["c50"],
                            label=f"{error_width_1}% PI{suf}")
            ax.plot(data_q["t"], data_q["q50"], color=style["median"], linestyle=style["median_ls"],
                    linewidth=linewidth_data, label=f"Median{suf}")

        # x axis
        if xtick_step is not None:
            step = xtick_step
        elif is_date:
            step = 2
        elif row >= n_rows - 2:
            step = 4
        elif row >= n_rows - 4:
            step = 3
        else:
            step = 2
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(formatter)
        if is_date:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_fontsize * 0.8)
        else:
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=tick_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if idx == 0:
            legend_elements = [
                plt.Line2D([0], [0], color="black", linestyle="-", marker=".", markersize=legend_fontsize,
                           markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.8, label=label_past),
                plt.Line2D([0], [0], color="black", linestyle="-", marker=".", markersize=legend_fontsize,
                           markerfacecolor="black", markeredgecolor="black", markeredgewidth=0.8,
                           label=label_forecast_obs),
            ]
            if multi:
                # empty entry: the observations take 3 rows so that every model has its own column
                legend_elements.append(plt.Line2D([0], [0], color="none", label=" "))
            for name in active:
                suf = f" ({label_by_model[name]})" if multi else ""
                style = styles[name]
                c90_leg = _over(style["c90"])
                c50_leg = _over(style["c50"], c90_leg)
                legend_elements += [
                    Patch(facecolor=c90_leg, edgecolor="none", label=f"{error_width}% PI{suf}"),
                    Patch(facecolor=c50_leg, edgecolor="none", label=f"{error_width_1}% PI{suf}"),
                    plt.Line2D([0], [0], color=style["median"], linestyle=style["median_ls"],
                               linewidth=linewidth_data * 1.8, label=f"Median{suf}"),
                ]
            if not multi:
                ax.legend(handles=legend_elements, loc="upper left", fontsize=legend_fontsize, ncol=1)

        if row == n_rows - 1 or idx + 1 > n_plots - n_cols:
            ax.set_xlabel("Weeks", fontsize=label_fontsize, labelpad=8)
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=8)
        ax.set_ylim(*row_ylims[row])

    if multi:
        fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, legend_y),
                   ncol=len(active) + 1, fontsize=legend_fontsize * 1.8, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, legend_y - (0.08 if is_date else 0.05)))
    else:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fixed horizon: one panel per horizon
# ---------------------------------------------------------------------------

def _draw_horizon(ax, horizon, active, keys_common, styles, label_by_model, multi, style, bar_offset, t0_min, t_last):
    """Draw on `ax` the forecasts at horizon `horizon`: for every window, the quantiles at the week t0 + horizon."""
    n_active = len(active)
    offsets = {name: (i - (n_active - 1) / 2) * bar_offset for i, name in enumerate(active)}

    for name, forecasts in active.items():
        mstyle = styles[name]
        rows = []
        for key, entry in forecasts.items():
            if key not in keys_common:
                continue
            t0 = int(entry["t0"])
            if t0_min is not None and t0 < t0_min:
                continue
            target_t = t0 + horizon
            if t_last is not None and target_t > t_last:
                continue
            q = entry["quantiles"]
            match = q[q["t"] == target_t]
            if len(match) == 0:
                continue
            row = match.iloc[0]
            rows.append((target_t, row["q05"], row["q25"], row["q50"], row["q75"], row["q95"]))
        if not rows:
            print(f"  (no window has horizon {horizon} available for '{name}')")
            continue
        rows.sort(key=lambda r: r[0])
        ts, q05, q25, q50, q75, q95 = (np.array(c) for c in zip(*rows))

        suf = f" ({label_by_model[name]})" if multi else ""
        solid = mstyle["median"]

        if style == "ribbon":
            ax.fill_between(ts, q05, q95, color=mstyle["c90"], label=f"90% PI{suf}")
            ax.fill_between(ts, q25, q75, color=mstyle["c50"], label=f"50% PI{suf}")
            ax.plot(ts, q50, color=mstyle["median"], linestyle=mstyle["median_ls"], linewidth=1.8,
                    label=f"Median{suf}")

        elif style == "bars":
            x = ts + offsets[name]
            for xi, lo90, hi90, lo50, hi50 in zip(x, q05, q95, q25, q75):
                ax.plot([xi, xi], [lo90, hi90], color=solid, alpha=0.55, linewidth=2.2,
                        solid_capstyle="round", zorder=3)
                ax.plot([xi, xi], [lo50, hi50], color=solid, alpha=0.9, linewidth=5.5,
                        solid_capstyle="round", zorder=4)
            ax.plot(x, q50, "o", color="white", markersize=5.5, markeredgecolor=solid, markeredgewidth=1.6,
                    zorder=5)
            c90_leg = _over(to_rgba(solid, 0.55))
            c50_leg = _over(to_rgba(solid, 0.90), c90_leg)
            ax.plot([], [], color=c90_leg, linewidth=2.2, label=f"90% PI{suf}")
            ax.plot([], [], color=c50_leg, linewidth=5.5, label=f"50% PI{suf}")
            ax.plot([], [], "o", color="white", markersize=5.5, markeredgecolor=solid, markeredgewidth=1.6,
                    label=f"Median{suf}")

        elif style == "bars_overlap":
            rank = list(active).index(name)
            lw90 = 5.0 - rank * 1.2
            lw50 = 9.0 - rank * 2.2
            for xi, lo90, hi90, lo50, hi50 in zip(ts, q05, q95, q25, q75):
                ax.plot([xi, xi], [lo90, hi90], color=solid, alpha=0.45, linewidth=lw90,
                        solid_capstyle="round", zorder=3 + rank)
                ax.plot([xi, xi], [lo50, hi50], color=solid, alpha=0.8, linewidth=lw50,
                        solid_capstyle="round", zorder=4 + rank)
            ax.plot(ts, q50, "o", color="white", markersize=5.5, markeredgecolor=solid, markeredgewidth=1.6,
                    zorder=10 + rank)
            c90_leg = _over(to_rgba(solid, 0.45))
            c50_leg = _over(to_rgba(solid, 0.80), c90_leg)
            ax.plot([], [], color=c90_leg, linewidth=lw90, label=f"90% PI{suf}")
            ax.plot([], [], color=c50_leg, linewidth=lw50, label=f"50% PI{suf}")
            ax.plot([], [], "o", color="white", markersize=5.5, markeredgecolor=solid, markeredgewidth=1.6,
                    label=f"Median{suf}")

        elif style == "box":
            x = ts + offsets[name]
            box_w = bar_offset * 0.85
            for xi, lo90, hi90, lo50, hi50, med in zip(x, q05, q95, q25, q75, q50):
                ax.add_patch(Rectangle((xi - box_w / 2, lo90), box_w, hi90 - lo90, facecolor=solid, alpha=0.15,
                                       edgecolor=solid, linewidth=0.8, zorder=2))
                ax.add_patch(Rectangle((xi - box_w / 2, lo50), box_w, hi50 - lo50, facecolor=solid, alpha=0.35,
                                       edgecolor=solid, linewidth=1.3, zorder=4))
                ax.plot([xi - box_w / 2, xi + box_w / 2], [med, med], color=solid, linewidth=2.0, zorder=5)
            c90_leg = _over(to_rgba(solid, 0.15))
            c50_leg = _over(to_rgba(solid, 0.35), c90_leg)
            ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=c90_leg, edgecolor=c90_leg, linewidth=0.8,
                                   label=f"90% PI{suf}"))
            ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=c50_leg, edgecolor=c50_leg, linewidth=1.3,
                                   label=f"50% PI{suf}"))
            ax.plot([], [], color=solid, linewidth=2.0, label=f"Median{suf}")

        else:
            raise ValueError("style must be 'bars', 'bars_overlap', 'box' or 'ribbon'")


# Defaults of the sizes, for the two kinds of x axis
_FIXED_HORIZON_DEFAULTS = {
    "weeks": dict(xtick_step=2, xtick_rotation=0, xtick_fontsize=16, ytick_fontsize=16, ylabel_fontsize=20,
                  legend_fontsize=16.5, ylabel="New infections", bottom=0.05, hspace=0.18),
    "dates": dict(xtick_step=1, xtick_rotation=45, xtick_fontsize=13, ytick_fontsize=15, ylabel_fontsize=18,
                  legend_fontsize=14, ylabel="Weekly Deaths", bottom=0.08, hspace=0.35),
}


def plot_fixed_horizon_grid(
    dates,
    obs,
    horizons=(1, 2, 3, 4),
    forecasts_pinn=None,
    forecasts_mech=None,
    forecasts_baseline=None,
    label_pinn="PINN",
    label_mech="Mechanistic",
    label_baseline="Baseline",
    style="box",
    ytick_step=None,
    y_nbins=None,
    y_sci=True,
    bar_offset=0.28,
    obs_markersize=8,
    ylim=None,
    t0_min=None,
    t_last=None,
    ylabel=None,
    xlabel="Weeks",
    show_title=False,
    title=None,
    title_fontsize=22,
    show_xlabel=True,
    show_ylabel=True,
    show_side_label=True,
    side_label_fontsize=22,
    side_label_offset=100,
    ylabel_offset=58,
    xlabel_offset=None,
    show_legend=True,
    legend_loc="upper right",
    legend_fontsize=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    ylabel_fontsize=None,
    xtick_step=None,
    xtick_rotation=None,
    figsize=(12, 16),
    margins=None,
    dpi=100,
):
    """
    One panel per horizon, one below the other, with the same x axis: the observations and, for every week, the
    forecast that was made `h` weeks before (the quantiles of the window t0 = week - h at the week t0 + h).

    Parameters
    ----------
    dates, obs : as in `plot_probabilistic_forecasts` (the x axis shows the dates for the real case, the week
        numbers for the synthetic scenarios).
    horizons : weeks ahead, one panel each.
    style : "box" (boxes: 90% and 50% intervals and median), "bars", "bars_overlap" or "ribbon".
    ytick_step : float or list (one per horizon), step of the major ticks of the y axis.
    ylim : tuple, or list with one tuple per horizon.
    t0_min, t_last : first window and last week (indices) shown.
    margins : dict(left, right, bottom, top, hspace). To align two figures side by side use the same figsize,
        bottom, top and hspace; `left` can be smaller in the figure without labels.
    xtick_step, xtick_rotation, font sizes, ylabel : if None, defaults of the kind of axis (weeks / dates).

    Returns
    -------
    fig, axes
    """
    active = _active(pinn=forecasts_pinn, mech=forecasts_mech, baseline=forecasts_baseline)
    keys_common = _common_keys(active)
    label_by_model = {"pinn": label_pinn, "mech": label_mech, "baseline": label_baseline}
    styles = {name: STYLE_BY_MODEL[name] for name in active}
    multi = len(active) > 1

    is_date = _is_date_axis(dates)
    dflt = _FIXED_HORIZON_DEFAULTS["dates" if is_date else "weeks"]
    xtick_step = dflt["xtick_step"] if xtick_step is None else xtick_step
    xtick_rotation = dflt["xtick_rotation"] if xtick_rotation is None else xtick_rotation
    xtick_fontsize = dflt["xtick_fontsize"] if xtick_fontsize is None else xtick_fontsize
    ytick_fontsize = dflt["ytick_fontsize"] if ytick_fontsize is None else ytick_fontsize
    ylabel_fontsize = dflt["ylabel_fontsize"] if ylabel_fontsize is None else ylabel_fontsize
    legend_fontsize = dflt["legend_fontsize"] if legend_fontsize is None else legend_fontsize
    ylabel = dflt["ylabel"] if ylabel is None else ylabel

    obs = np.asarray(obs)
    n_weeks = len(obs)
    if t_last is not None:
        n_weeks = min(n_weeks, t_last + 1)
    t_epi = np.arange(n_weeks)
    formatter = _week_formatter(dates)

    n = len(horizons)
    if ylim is not None and isinstance(ylim[0], (list, tuple)):
        ylims = list(ylim)
    else:
        ylims = [ylim] * n

    if margins is None:
        left = 0.15 if (show_ylabel or show_side_label) else 0.07
        margins = dict(left=left, right=0.98, bottom=dflt["bottom"], top=0.95 if show_title else 0.985,
                       hspace=dflt["hspace"])

    fig, axes = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)
    axes = np.atleast_1d(axes)

    for i, (ax, h) in enumerate(zip(axes, horizons)):
        ax.plot(t_epi, obs[:n_weeks], ".-", color="black", markerfacecolor="black", markeredgecolor="black",
                markersize=obs_markersize, markeredgewidth=1.0, linewidth=1.4, label="Data", zorder=5)

        _draw_horizon(ax, h, active, keys_common, styles, label_by_model, multi, style, bar_offset, t0_min, t_last)

        # axes
        ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim(-0.6, n_weeks - 0.4)
        ax.tick_params(axis="x", which="major", labelsize=xtick_fontsize, labelbottom=True)
        for lab in ax.get_xticklabels():
            lab.set_rotation(xtick_rotation)
            lab.set_ha("right" if xtick_rotation else "center")
            lab.set_rotation_mode("anchor")
        if y_sci:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(axis="y", which="major", labelsize=ytick_fontsize)
        ax.yaxis.get_offset_text().set_fontsize(ytick_fontsize)
        if ylims[i] is not None:
            ax.set_ylim(*ylims[i])
        step = ytick_step[i] if isinstance(ytick_step, (list, tuple)) else ytick_step
        if step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(step))
            ax.yaxis.set_minor_locator(MultipleLocator(step / 2))   # intermediate tick without number
        elif y_nbins is not None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
            ax.yaxis.set_label_coords(0, 0.5, transform=offset_copy(ax.transAxes, fig=fig, x=-ylabel_offset,
                                                                    units="points"))
        if show_side_label:
            ax.annotate(f"Horizon {h}", xy=(0, 0.5), xycoords="axes fraction", xytext=(-side_label_offset, 0),
                        textcoords="offset points", rotation=90, ha="center", va="center", annotation_clip=False,
                        fontsize=side_label_fontsize, fontweight="bold")

        # legend only in the first panel: Data, then PINN, then Mechanistic (one column)
        if i == 0 and show_legend:
            hdl, lbl = ax.get_legend_handles_labels()
            by_label = dict(zip(lbl, hdl))
            order = ["Data"]
            for name in active:
                suf = f" ({label_by_model[name]})" if multi else ""
                order += [f"90% PI{suf}", f"50% PI{suf}", f"Median{suf}"]
            order = [lab for lab in order if lab in by_label]
            ax.legend([by_label[lab] for lab in order], order, ncol=1, loc=legend_loc, fontsize=legend_fontsize,
                      frameon=False, handlelength=1.8, labelspacing=0.35)

    if show_xlabel:
        axes[-1].set_xlabel(xlabel, fontsize=ylabel_fontsize)
        off = ylabel_offset if xlabel_offset is None else xlabel_offset
        axes[-1].xaxis.set_label_coords(0.5, 0, transform=offset_copy(axes[-1].transAxes, fig=fig, y=-off,
                                                                      units="points"))

    if show_title:
        fig.text(0.5, 0.995, title or "Horizon-specific probabilistic forecast across all weeks", ha="center",
                 va="top", fontsize=title_fontsize, fontweight="bold")

    fig.subplots_adjust(**margins)
    return fig, axes


# ---------------------------------------------------------------------------
# Fit of one window (synthetic scenarios: retrospective fit at the last week)
# ---------------------------------------------------------------------------

def samples_to_quantiles(samples, qlevels=(5, 25, 50, 75, 95)):
    """
    Quantiles of the samples of one series.

    samples : np.ndarray (n_samples, T)
    Returns a dict {'q05': ..., 'q25': ..., 'q50': ..., 'q75': ..., 'q95': ...}, every one with shape (T,).
    """
    qs = np.percentile(samples, qlevels, axis=0)
    return {f"q{int(q):02d}": qs[i] for i, q in enumerate(qlevels)}


def plot_single_window(
    dates,
    obs,
    t0,
    forecasts_pinn=None,
    mech_samples=None,
    dt_train=None,
    t_end=None,
    error_width=90,
    error_width_1=50,
    ylim=None,
    figsize=(12, 4.5),
    dpi=300,
    ax=None,
    label_pinn="PINN",
    label_mech="Mechanistic",
    label_past="All data",
    label_forecast_obs="Look-back window",
    ylabel="Weekly new infections",
    show_lookback=False,
    show_title=False,
    title=None,
    title_fontsize=20,
    margins=None,
    legend="figure",
    legend_y=None,
    label_fontsize=19,
    tick_fontsize=14,
    legend_fontsize=15,
    ytick_step=None,
    y_nbins=None,
    xtick_step=2,
    linewidth_data=1.2,
    markersize=7,
    data_markersize=10,
    y_sci=True,
    labelpad=14,
):
    """
    One panel for the window `t0`, in the same style as the grid of `plot_probabilistic_forecasts`.

    - PINN: the entry of `forecasts_pinn` with t0 (fit and forecast of the window).
    - Mechanistic: quantiles computed from `mech_samples`, cut at t_end.
    - Data and bands are shown from t = 0 to t_end (default t0: only the fit).
    - Legend (`legend="figure"`): above the panel, "All data" in the first block and one column per model
      (Median, 90% PI, 50% PI). `legend="axes"`: inside the panel; None: no legend.

    Parameters
    ----------
    dates, obs : as in `plot_probabilistic_forecasts`.
    t0 : int
        Window to show, e.g. T_LAST for the retrospective fit of the synthetic scenarios.
    forecasts_pinn : dict, optional
        Forecasts of the PINN in the `forecast_results` format (not trimmed).
    mech_samples : np.ndarray (n_samples, n_weeks), optional
        Simulated trajectories of the Mechanistic model for that window (`last_week[t0]["trajectories"]`).
    dt_train : int, optional
        Look-back, used only with `show_lookback`. None: all the history up to t0.
    t_end : int, optional
        Last week shown. None: t0 (fit only); t0 + 4: fit and forecast.

    Returns
    -------
    fig, ax
    """
    obs = np.asarray(obs)
    n_weeks = len(obs)
    if t_end is None:
        t_end = t0
    t_end = min(t_end, n_weeks - 1)
    if dt_train is None:
        dt_train = n_weeks
    t_epi = np.arange(0, t_end + 1)

    lo, hi = (100 - error_width) / 2, 100 - (100 - error_width) / 2
    lo1, hi1 = (100 - error_width_1) / 2, 100 - (100 - error_width_1) / 2
    q_lo, q_hi = f"q{int(lo):02d}", f"q{int(hi):02d}"
    q_lo1, q_hi1 = f"q{int(lo1):02d}", f"q{int(hi1):02d}"

    # active models, all as a DataFrame of quantiles
    models = {}
    if forecasts_pinn is not None:
        dq = entry_at(forecasts_pinn, t0)["quantiles"]
        models["pinn"] = dq[dq["t"] <= t_end].reset_index(drop=True)
    if mech_samples is not None:
        s = np.asarray(mech_samples)[:, :t_end + 1]
        models["mech"] = pd.DataFrame({
            "t": np.arange(s.shape[1]),
            q_lo: np.percentile(s, lo, axis=0),
            q_lo1: np.percentile(s, lo1, axis=0),
            "q50": np.percentile(s, 50, axis=0),
            q_hi1: np.percentile(s, hi1, axis=0),
            q_hi: np.percentile(s, hi, axis=0),
        })
    if not models:
        raise ValueError("Provide forecasts_pinn and/or mech_samples")
    multi = len(models) > 1
    label_by_model = {"pinn": label_pinn, "mech": label_mech}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    # observations
    ax.plot(t_epi, obs[t_epi], ".-", color="black", markerfacecolor="white", markeredgecolor="black",
            markersize=data_markersize, markeredgewidth=0.9, linewidth=linewidth_data, zorder=5)
    if show_lookback:
        window_train = np.arange(max(0, t0 - dt_train + 1), t0 + 1)
        ax.plot(window_train, obs[window_train], ".-", color="black", markerfacecolor="black",
                markeredgecolor="black", markersize=markersize, markeredgewidth=0.8, linewidth=linewidth_data,
                zorder=6)
        ax.plot(t0, obs[t0], "o", color="black", markerfacecolor="black", markeredgecolor="black",
                markersize=markersize, markeredgewidth=1.0, zorder=6)

    # bands and median
    for name, dq in models.items():
        style = STYLE_BY_MODEL[name]
        ax.fill_between(dq["t"], dq[q_lo], dq[q_hi], color=style["c90"], linewidth=0)
        ax.fill_between(dq["t"], dq[q_lo1], dq[q_hi1], color=style["c50"], linewidth=0)
        ax.plot(dq["t"], dq["q50"], color=style["median"], linestyle=style["median_ls"],
                linewidth=linewidth_data * 1.3)

    # axes
    ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(_week_formatter(dates))
    ax.set_xlim(-0.5, t_end + 0.5)
    if y_sci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Weeks", fontsize=label_fontsize, labelpad=labelpad)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=labelpad)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ytick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
        ax.yaxis.set_minor_locator(MultipleLocator(ytick_step / 2))
    elif y_nbins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))

    # legend: data block + models block (column-major: medians | PI of the first model | PI of the second)
    data_handles = [plt.Line2D([0], [0], color="black", linestyle="-", marker=".", markersize=data_markersize,
                               markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.9,
                               label=label_past)]
    if show_lookback:
        data_handles.append(plt.Line2D([0], [0], color="black", linestyle="-", marker=".", markersize=markersize,
                                       markerfacecolor="black", markeredgecolor="black", markeredgewidth=0.8,
                                       label=label_forecast_obs))
    med_handles, pi_handles = [], []
    for name in models:
        style = STYLE_BY_MODEL[name]
        suf = f" ({label_by_model[name]})" if multi else ""
        c90_leg = _over(style["c90"])
        c50_leg = _over(style["c50"], c90_leg)
        med_handles.append(plt.Line2D([0], [0], color=style["median"], linestyle=style["median_ls"],
                                      linewidth=linewidth_data * 1.8, label=f"Median{suf}"))
        pi_handles += [Patch(facecolor=c90_leg, edgecolor="none", label=f"{error_width}% PI{suf}"),
                       Patch(facecolor=c50_leg, edgecolor="none", label=f"{error_width_1}% PI{suf}")]
    if multi:
        model_handles = med_handles + pi_handles
        model_ncol = 1 + len(models)
    else:
        model_handles = pi_handles + med_handles
        model_ncol = 1

    # title, legend, layout
    fig_h_pt = fig.get_figheight() * 72
    top = 1.0
    if show_title and title:
        # fig.text and not suptitle: tight_layout must not reserve more room for it
        fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=title_fontsize, fontweight="bold")
        top -= (title_fontsize * 1.6) / fig_h_pt
    if legend_y is None:
        legend_y = top

    if legend == "figure":
        inv = fig.transFigure.inverted()
        kw = dict(fontsize=legend_fontsize, frameon=False)
        # 1) width of the data block, to centre the whole group
        leg_d = fig.legend(handles=data_handles, loc="upper left", bbox_to_anchor=(0, 0), **kw)
        fig.canvas.draw()
        w_d = leg_d.get_window_extent().transformed(inv).width
        leg_d.remove()
        # 2) models block, moved to the right by half of the data block
        leg_m = fig.legend(handles=model_handles, loc="upper center", ncol=model_ncol,
                           bbox_to_anchor=(0.5 + w_d / 2, legend_y), columnspacing=2.5, **kw)
        fig.canvas.draw()
        bb = leg_m.get_window_extent().transformed(inv)
        # 3) data block on the left, centred vertically on the models block
        fig.legend(handles=data_handles, loc="center right", bbox_to_anchor=(bb.x0, (bb.y0 + bb.y1) / 2), **kw)
        fig.tight_layout(rect=(0, 0, 1, bb.y0 - 0.01))
    elif legend == "axes":
        # one column, upper right: data, then all the entries of the first model, then of the second
        by_model = []
        for name in models:
            style = STYLE_BY_MODEL[name]
            suf = f" ({label_by_model[name]})" if multi else ""
            c90_leg = _over(style["c90"])
            c50_leg = _over(style["c50"], c90_leg)
            by_model += [
                Patch(facecolor=c90_leg, edgecolor="none", label=f"{error_width}% PI{suf}"),
                Patch(facecolor=c50_leg, edgecolor="none", label=f"{error_width_1}% PI{suf}"),
                plt.Line2D([0], [0], color=style["median"], linestyle=style["median_ls"],
                           linewidth=linewidth_data * 1.8, label=f"Median{suf}"),
            ]
        ax.legend(handles=data_handles + by_model, loc="upper right", ncol=1, fontsize=legend_fontsize,
                  frameon=False)
        fig.tight_layout(rect=(0, 0, 1, top))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.94 if show_title else 1))
    if margins is not None:
        fig.subplots_adjust(**margins)
    return fig, ax


def fit_coverage(obs, t0, forecasts_pinn, mech_samples, level=90, t_start=0):
    """
    Empirical coverage of the `level`% intervals of the fit of the window t0, on the weeks t_start..t0: fraction of
    observations inside the interval, for the PINN (quantiles of its forecast entry) and for the Mechanistic model
    (quantiles of its samples).

    Returns
    -------
    cov_pinn, cov_mech, n_weeks
    """
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    y = np.asarray(obs)[t_start:t0 + 1]

    q = entry_at(forecasts_pinn, t0)["quantiles"].set_index("t")
    p_lo = q.loc[t_start:t0, f"q{int(lo):02d}"].to_numpy()
    p_hi = q.loc[t_start:t0, f"q{int(hi):02d}"].to_numpy()

    s = np.asarray(mech_samples)[:, t_start:t0 + 1]
    m_lo, m_hi = np.percentile(s, [lo, hi], axis=0)

    cov_pinn = np.mean((y >= p_lo) & (y <= p_hi))
    cov_mech = np.mean((y >= m_lo) & (y <= m_hi))
    return cov_pinn, cov_mech, len(y)


# ---------------------------------------------------------------------------
# Sanity check by age class (synthetic scenarios)
# ---------------------------------------------------------------------------

def plot_age_forecast_compare(
    forecasts_age,
    mech_samples_age,
    dates,
    true_age,
    t0=None,
    age_rows=((1, 2, 3, 4, 5), (7, 9, 11, 13, 16)),
    age_labels=AGE_LABELS,
    col_titles=("PINN", "Mechanistic"),
    show_band=True,
    ci=90,
    figsize=(14, 10),
    dpi=300,
    show_title=True,
    title="Weekly incidence by age class - synthetic scenario",
    ylabel="Weekly Age Incidence",
    xtick_step=3,
    true_lw=2.0,
    labelpad=12,
):
    """
    Grid (n_rows x 2): every row is a group of age classes (`age_rows`, numbered from 1), left column = fit of the
    PINN (window t0), right column = fit of the Mechanistic model. Every panel shows, for every class, the true
    incidence without noise (dashed), the median (continuous) and the `ci`% band of the same colour. Panels of the
    same row share the y axis.

    Parameters
    ----------
    forecasts_age : dict {a (0-based): forecasts dict}
        Forecast files of every age class (notebook 04 with `AGE_FORECASTS = True`).
    mech_samples_age : np.ndarray (n_samples, k_age, n_weeks), or None
        Incidence of every age class simulated by the Mechanistic model (`last_week[t0]["weekly_age"]`).
    dates : np.ndarray, only for the type of the x axis (week numbers or dates).
    true_age : np.ndarray (n_weeks, k_age)
        True incidence of the age classes without noise, up to the last week.
    t0 : int, optional
        Window of the fit. Default: the last week of `true_age` (T_LAST).
    """
    n_weeks, k_age = true_age.shape
    t_epi = np.arange(n_weeks)
    if t0 is None:
        t0 = n_weeks - 1

    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    q_lo, q_hi = f"q{int(lo):02d}", f"q{int(hi):02d}"
    cmap = plt.get_cmap("viridis")
    formatter = _week_formatter(dates)

    n_rows = len(age_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, dpi=dpi, sharex=True, sharey="row", squeeze=False)
    if show_title:
        fig.suptitle(title, fontsize=20, fontweight="bold")

    # Mechanistic: quantiles of the samples, cut at the length of the series
    if mech_samples_age is not None:
        n_y = min(mech_samples_age.shape[2], n_weeks)
        t_y = np.arange(n_y)
        y_lo = np.percentile(mech_samples_age[:, :, :n_y], lo, axis=0)    # (k_age, n_y)
        y_md = np.percentile(mech_samples_age[:, :, :n_y], 50, axis=0)
        y_hi = np.percentile(mech_samples_age[:, :, :n_y], hi, axis=0)

    for r, ages in enumerate(age_rows):
        idx = [a - 1 for a in ages]                    # 1-based -> 0-based
        for c in range(2):
            ax = axes[r, c]
            for i, a in enumerate(idx):
                color = cmap(0.05 + 0.85 * i / max(len(idx) - 1, 1))    # distinct colours inside the row
                ax.plot(t_epi, true_age[:, a], "--", color=color, alpha=0.7, linewidth=true_lw)

                if c == 0:      # PINN
                    fa = forecasts_age.get(a)
                    if fa is None:
                        continue
                    dq = entry_at(fa, t0)["quantiles"]
                    dq = dq[dq["t"] <= n_weeks - 1]
                    if show_band:
                        ax.fill_between(dq["t"], dq[q_lo], dq[q_hi], color=color, alpha=0.30, linewidth=0)
                    ax.plot(dq["t"], dq["q50"], "-", color=color, linewidth=1.4, label=f"{age_labels[a]} years")

                elif mech_samples_age is not None:     # Mechanistic
                    if show_band:
                        ax.fill_between(t_y, y_lo[a], y_hi[a], color=color, alpha=0.30, linewidth=0)
                    ax.plot(t_y, y_md[a], "-", color=color, linewidth=1.4)

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis="both", labelsize=13, pad=6)

            if r == 0:
                ax.set_title(col_titles[c], fontsize=17, fontweight="bold")
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=15, labelpad=labelpad)
                ax.legend(fontsize=15, loc="upper left", frameon=False)
            if r == n_rows - 1:
                ax.set_xlabel("Weeks", fontsize=15, labelpad=labelpad)

    # one legend for the styles, at the top
    style_handles = [
        plt.Line2D([0], [0], color="grey", linestyle="--", alpha=0.7, linewidth=true_lw,
                   label="True incidence (no noise)"),
        plt.Line2D([0], [0], color="grey", linestyle="-", linewidth=1.4, label="Median"),
        Patch(facecolor="grey", alpha=0.30, label=f"{ci}% PI"),
    ]
    fig.legend(handles=style_handles, loc="upper center", ncol=3, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 0.95 if show_title else 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92 if show_title else 0.96))
    return fig
