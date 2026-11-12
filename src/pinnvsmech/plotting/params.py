"""
Comparison of the parameter estimates of the PINN (MCMC) and of the Mechanistic model (ABC-SMC), on the weeks that
both models have. Both models are read from files in the same format (`mcmc_distributions.pkl` of the PINN and
`abc_distributions.pkl` of the Mechanistic model, see `pinnvsmech.mechanistic.convert`).

The main figure is `plot_params_comparison_violin`: one split violin per week, the left half is the PINN and the right
half the Mechanistic model. For the synthetic scenarios `plot_R0_density` shows the density of R0 at the last week. It works for the synthetic scenarios (x axis: week number) and for the real case (x axis:
dates). For the synthetic scenarios both R0 and the effective reproduction number Rt can be shown; for the real case
only Rt (R0 is not identifiable there and has no reference value).

Positions of the plots = index of the week t0; labels of the x axis = week number t0 + 1 (synthetic) or date MM-DD (real).
"""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from scipy.stats import gaussian_kde

COLOR_A = "#9400D3"      # PINN (darkviolet)
COLOR_B = "#008B8B"      # Mechanistic (darkcyan)
COLOR_REF = "#333333"    # reference lines
COLOR_TRUE = "#2E7D32"   # true value


# ---------------------------------------------------------------------------
# Defaults of the cases
# ---------------------------------------------------------------------------

# Parameter shown, as (key in the file, title, ylim, prior bounds or None, horizontal threshold or None,
# show the peak week on this panel?).
PARAMS = {
    "R0": ("R0s_samples", r"$\hat{R}_0$", (1.0, 3.5), (1.3, 3.0), None, False),
    "Rt": ("Rts_samples", r"$\hat{R}_t$", (0.0, 3.5), None, 1.0, True),
}

# y limits of Rt that depend on the case
RT_YLIM = {("synthetic", 1.5): (0.5, 2.5), ("synthetic", 2.0): (0.2, 3.2), ("real", None): (0.0, 3.5)}


def parameter_spec(which, kind, R0=None, ylim=None):
    """
    Specification of the parameter to plot.

    Parameters
    ----------
    which : "R0" or "Rt"
    kind : "synthetic" or "real"
    R0 : float
        True R0 of the synthetic scenario (1.5 or 2.0): sets the y limits of Rt.
    ylim : tuple, optional
        Replaces the default y limits.

    Returns
    -------
    tuple, an element of the `params` argument of the plot functions.
    """
    if which not in PARAMS:
        raise ValueError(f"which must be 'R0' or 'Rt', got '{which}'")
    if which == "R0" and kind == "real":
        raise ValueError("R0 is shown only for the synthetic scenarios: in the real case only Rt can be plotted")

    key, title, default_ylim, bounds, hline, show_peak = PARAMS[which]
    if which == "Rt":
        default_ylim = RT_YLIM.get((kind, float(R0) if kind == "synthetic" else None), default_ylim)
    return key, title, ylim or default_ylim, bounds, hline, show_peak


def default_t0_range(kind, dates, R0=None):
    """
    Weeks (t0 indices) shown by default: for the real case from 2020-03-15 to 2020-07-19; for the synthetic scenarios
    from t0 = 3 to 29 (R0 = 1.5, weeks 4-30) or to 19 (R0 = 2.0, weeks 4-20), the rolling windows also of the forecasts.

    Returns
    -------
    (t0_min, t0_max), None meaning no limit.
    """
    if kind == "real":
        d = np.asarray(dates, dtype="datetime64[ns]")
        return (int(np.where(d == np.datetime64("2020-03-15"))[0][0]),
                int(np.where(d == np.datetime64("2020-07-19"))[0][0]))
    return 3, {1.5: 29, 2.0: 19}[float(R0)]


def peak_band(obs, half_width=0.45):
    """Band around the week of the peak of the observations, to draw it as a reference."""
    p = int(np.argmax(obs))
    return p - half_width, p + half_width


def default_peak_week(kind, obs):
    """
    Peak week drawn by default: for the synthetic scenarios the band around the peak of the (noisy) observations; for
    the real case the band (6.55, 7.45), i.e. around t0 = 7 (2020-04-05), set by hand.
    """
    return (6.55, 7.45) if kind == "real" else peak_band(obs)


# ---------------------------------------------------------------------------
# Loading and helpers
# ---------------------------------------------------------------------------

def load_distributions(path):
    """Load a file of distributions (`mcmc_distributions.pkl` or `abc_distributions.pkl`)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def common_weeks(data_a, data_b, t0_min=None, t0_max=None):
    """Weeks (t0) present in both files, in increasing order, optionally within [t0_min, t0_max]."""
    wa = np.asarray(data_a["weeks"]).astype(int)
    wb = np.asarray(data_b["weeks"]).astype(int)
    w = sorted(set(wa.tolist()) & set(wb.tolist()))
    if t0_min is not None:
        w = [x for x in w if x >= t0_min]
    if t0_max is not None:
        w = [x for x in w if x <= t0_max]
    if not w:
        raise ValueError("No week in common between the two models in the chosen interval")
    return np.array(w)


def _aligned_samples(data, key, weeks_sel):
    """List of arrays of samples of `key`, ordered as weeks_sel; None if the key is not in the file."""
    if key not in data:
        return None
    weeks = np.asarray(data["weeks"]).astype(int).tolist()
    return [data[key][weeks.index(w)] for w in weeks_sel]


def _over(fg, bg=(1, 1, 1)):
    """Colour seen when `fg` (with its alpha) is on top of `bg`."""
    r, g, b, a = to_rgba(fg)
    return tuple(a * c + (1 - a) * d for c, d in zip((r, g, b), bg))


def _is_date_axis(dates):
    return np.issubdtype(np.asarray(dates).dtype, np.datetime64)


def _style_axis(ax, dates, weeks_sel, ylim, title, xlabel, ylabel, title_pad=12, xtick_step=None):
    """
    Common axes. With dates (real case): labels MM-DD every week, rotated. With week names (synthetic scenarios):
    labels = week number t0 + 1, every 2 weeks.
    """
    date_axis = _is_date_axis(dates)
    xtick_step = xtick_step or (1 if date_axis else 2)

    if title:
        ax.set_title(title, fontsize=17, fontweight="bold", pad=title_pad)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, labelpad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13, labelpad=12)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.xaxis.set_major_locator(FixedLocator(np.arange(weeks_sel[0], weeks_sel[-1] + 1, xtick_step)))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    if date_axis:
        labels = [pd.Timestamp(d).strftime("%m-%d") for d in dates]
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, p: labels[int(round(x))] if 0 <= int(round(x)) < len(labels) else ""))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(round(x)) + 1}"))
    ax.tick_params(axis="x", which="major", labelbottom=True)
    if date_axis:
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")
    ax.set_xlim(weeks_sel[0] - 1, weeks_sel[-1] + 1)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)


def _draw_threshold_and_peak(ax, hline_value, peak_week):
    """
    Draw the horizontal threshold (e.g. Rt = 1) and the reference of the peak week. `peak_week` can be None (nothing),
    a number (dashed vertical line) or a pair (a, b) (shaded band between the two weeks). Returns the legend handles.
    """
    handles = []
    if hline_value is not None:
        ax.axhline(hline_value, color="black", linewidth=1.3, alpha=0.8, zorder=1)
        handles.append(plt.Line2D([0], [0], color="black", lw=1.3, alpha=0.8, label=r"Threshold: $R_t = 1$"))
    if peak_week is not None:
        if np.isscalar(peak_week):
            ax.axvline(peak_week, color="0.35", linestyle=(0, (4, 4)), linewidth=1.3, alpha=0.8, zorder=1)
            handles.append(plt.Line2D([0], [0], color="0.35", lw=1.3, linestyle=(0, (4, 4)), alpha=0.8,
                                      label="Peak week"))
        else:
            ax.axvspan(min(peak_week), max(peak_week), color="0.35", alpha=0.12, zorder=0)
            handles.append(Patch(facecolor="0.35", alpha=0.12, label="Peak week"))
    return handles


def _select_params(params, which):
    """Keep only the parameters in `which`: keys of the file or the aliases "R0" and "Rt"."""
    aliases = {"R0": "R0s_samples", "Rt": "Rts_samples"}
    which = [which] if isinstance(which, str) else list(which)
    which = [aliases.get(k, k) for k in which]
    available = [p[0] for p in params]
    missing = [k for k in which if k not in available]
    if missing:
        raise ValueError(f"Keys not found: {missing}. Available: {available}")
    return [p for p in params if p[0] in which]


def _default_params():
    return [PARAMS["R0"], PARAMS["Rt"]]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_params_comparison_violin(
    data_a, data_b,
    dates,
    label_a="PINN",
    label_b="Mechanistic",
    params=None,
    show=None,                  # "R0", "Rt" or ("R0", "Rt")
    show_prior_bounds=True,
    peak_week=None,
    width=0.85,
    t0_min=None,                # first week shown (index)
    t0_max=None,                # last week shown (index)
    face_alpha=0.30,
    edge_alpha=0.60,
    median_color=None,          # None: colour of the model; e.g. "black"
    show_title=True,
    show_legend=True,
    side_label=None,            # e.g. r"$\mathbf{R_0=2.0}$" or "COVID-19\nLondon"
    side_label_fontsize=20,
    side_label_offset=85,       # points, on the left of the y axis
    true_values=None,           # e.g. {"R0s_samples": 1.5}
    figsize=None,
    legend_loc="upper right",
    legend_ncol=1,
    legend_fontsize=14,
    dpi=300,
):
    """
    Split violin: for every week one body, the left half is model A (PINN) and the right half model B (Mechanistic).
    It shows the whole shape of the posterior (bimodality, accumulation at the bounds), with the median as a short line.

    Parameters
    ----------
    data_a, data_b : dict
        Distributions of the two models (`load_distributions`).
    dates : np.ndarray
        Weekly dates (real case: x axis with dates) or names of the weeks (synthetic: x axis with week numbers).
    params : list of tuples
        Parameters in the format of `parameter_spec`. Default: R0 and Rt.
    show : str or tuple
        Which of the parameters to draw.
    peak_week : None, number or pair
        Reference of the peak week, drawn only where the parameter asks for it (Rt).
    """
    params = params if params is not None else _default_params()
    if show is not None:
        params = _select_params(params, show)

    weeks_sel = common_weeks(data_a, data_b, t0_min, t0_max)
    date_axis = _is_date_axis(dates)

    n = len(params)
    figsize = figsize or (10, 4.5 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)
    axes = np.atleast_1d(axes)

    for i, (ax, (key, title, ylim, bounds, hline, show_peak)) in enumerate(zip(axes, params)):
        for data, color, side in ((data_a, COLOR_A, -1), (data_b, COLOR_B, +1)):
            samples = _aligned_samples(data, key, weeks_sel)
            if samples is None:
                continue
            parts = ax.violinplot(samples, positions=weeks_sel, widths=width, showextrema=False, showmedians=False)
            for body, w in zip(parts["bodies"], weeks_sel):
                verts = body.get_paths()[0].vertices
                # keep only the half of the violin on the side of this model
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, w) if side < 0 else np.clip(verts[:, 0], w, np.inf)
                body.set_alpha(None)                          # the RGBA colours decide
                body.set_facecolor(to_rgba(color, face_alpha))
                body.set_edgecolor(to_rgba(color, edge_alpha))
                body.set_linewidth(0.8)
            # median as a short line on the side of the model
            for w, m in zip(weeks_sel, [np.median(s) for s in samples]):
                ax.plot([w, w + side * width * 0.42], [m, m], color=median_color or color, lw=2.0, zorder=5)

        if show_prior_bounds and bounds is not None:
            for b in bounds:
                ax.axhline(b, linestyle=(0, (4, 4)), lw=1.2, color=COLOR_REF, alpha=0.55, zorder=1)

        true_handle = []
        if true_values and key in true_values:
            ax.axhline(true_values[key], color=COLOR_TRUE, lw=1.6, linestyle="-.", zorder=2)
            true_handle = [plt.Line2D([0], [0], color=COLOR_TRUE, lw=1.6, linestyle="-.",
                                      label=f"True value ({true_values[key]:g})")]

        ref_handles = _draw_threshold_and_peak(ax, hline, peak_week if show_peak else None)

        _style_axis(ax, dates, weeks_sel, ylim, title if show_title else None,
                    xlabel=("Date" if date_axis else "Weeks") if i == n - 1 else None,
                    ylabel="Parameter value", title_pad=12)

        if side_label:
            ax.annotate(side_label, xy=(0, 0.5), xycoords="axes fraction",
                        xytext=(-side_label_offset, 0), textcoords="offset points",
                        rotation=90, ha="center", va="center", annotation_clip=False,
                        fontsize=side_label_fontsize, fontweight="bold")

        if show_legend:
            handles = [
                Patch(facecolor=_over(to_rgba(COLOR_A, face_alpha)), edgecolor=to_rgba(COLOR_A, edge_alpha),
                      label=label_a),
                Patch(facecolor=_over(to_rgba(COLOR_B, face_alpha)), edgecolor=to_rgba(COLOR_B, edge_alpha),
                      label=label_b),
            ]
            if median_color is not None:
                handles.append(plt.Line2D([0], [0], color=median_color, lw=2.0, label="Median"))
            if show_prior_bounds and bounds is not None:
                handles.append(plt.Line2D([0], [0], color=COLOR_REF, lw=1.2, linestyle=(0, (4, 4)), alpha=0.55,
                                          label="Prior bounds"))
            handles += true_handle
            handles += ref_handles            # threshold Rt = 1 and "Peak week"
            ax.legend(handles=handles, fontsize=legend_fontsize, ncol=legend_ncol, frameon=False, loc=legend_loc)

    plt.tight_layout(pad=2.0, h_pad=2.0)
    return fig


# ---------------------------------------------------------------------------
# Density of R0 at the last week of a synthetic scenario
# ---------------------------------------------------------------------------

# x limits of the density of R0 of every scenario
DENSITY_XLIM = {1.5: (1.39, 1.64), 2.0: (1.78, 2.25)}


def load_pinn_R0(data_pinn, t0):
    """R0 samples of the PINN for the window t0 (index: t0 = 33 -> week 34)."""
    weeks = np.asarray(data_pinn["weeks"]).astype(int).tolist()
    if t0 not in weeks:
        raise KeyError(f"t0={t0} is not in the weeks of the PINN ({weeks[0]}..{weeks[-1]})")
    return np.asarray(data_pinn["R0s_samples"][weeks.index(t0)])


def load_mech_R0(posterior_csv, column="R0"):
    """R0 samples of the Mechanistic model from a raw posterior.csv (already equally weighted)."""
    return pd.read_csv(posterior_csv)[column].to_numpy()


def load_mech_last_week_R0(data_mech, t0):
    """R0 samples of the Mechanistic model for the extra fit at the last week t0 (key `last_week` of the file)."""
    last = data_mech.get("last_week", {})
    if t0 not in last:
        raise KeyError(f"No fit at t0={t0} in the Mechanistic results (available: {sorted(last)}): "
                       f"convert the folder 'retro_w{t0 + 1}' with notebook 05")
    return np.asarray(last[t0]["R0"])


def plot_R0_density(
    samples_by_model,            # {"PINN": array, "Mechanistic": array}
    colors=None,                 # {"PINN": "#9400D3", "Mechanistic": "#008B8B"}
    true_value=None,
    xlim=None,                   # None: range of the samples with a margin
    normalize=True,              # True: every density is scaled to maximum 1
    bw_method=None,              # bandwidth of the KDE (None = Scott); e.g. 0.3 for smoother
    face_alpha=0.30,
    show_medians=True,           # dotted vertical line at the median of every model
    show_legend=True,
    legend_loc="top",            # "top" (above the panel, 2 rows) or a matplotlib loc
    show_yticks=True,
    show_title=False,
    title=r"$\hat{R}_0$",
    xlabel=r"$R_0$",
    ylabel=None,                 # None: "Density (scaled to max = 1)" or "Density"
    side_label=None,
    side_label_fontsize=18,
    side_label_offset=62,
    label_fontsize=19,
    tick_fontsize=17,
    legend_fontsize=14.5,
    n_grid=800,
    figsize=(6, 4),
    dpi=100,
    ax=None,
    margins=None,
):
    """Overlapping densities (KDE) of the posterior of R0, one per model."""
    colors = colors or {"PINN": COLOR_A, "Mechanistic": COLOR_B}
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    allv = np.concatenate([np.asarray(v) for v in samples_by_model.values()])
    if xlim is None:
        lo, hi = np.percentile(allv, [0.5, 99.5])
        pad = 0.15 * (hi - lo)
        xlim = (lo - pad, hi + pad)
    xs = np.linspace(*xlim, n_grid)

    for name, s in samples_by_model.items():
        s = np.asarray(s)
        c = colors[name]
        d = gaussian_kde(s, bw_method=bw_method)(xs)
        if normalize:
            d = d / d.max()
        ax.fill_between(xs, d, color=to_rgba(c, face_alpha), lw=0)
        ax.plot(xs, d, color=c, lw=1.6)
        if show_medians:
            ax.axvline(np.median(s), color=c, lw=1.6, ls=":", zorder=3)

    if true_value is not None:
        ax.axvline(true_value, color="black", lw=1.4, ls="--", zorder=4)

    # --- legend handles ---
    true_h = [plt.Line2D([0], [0], color="black", lw=1.4, ls="--", label="True value")] if true_value is not None else []
    model_h = []                                   # per model: patch + median
    for name in samples_by_model:
        c = colors[name]
        model_h.append(Patch(facecolor=to_rgba(c, face_alpha), edgecolor=c, lw=1.2, label=name))
        if show_medians:
            short = "Mech" if name == "Mechanistic" else name
            model_h.append(plt.Line2D([0], [0], color=c, lw=1.6, ls=":", label=f"Median ({short})"))

    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0, top=1.3 if normalize else None)
    if not show_yticks:
        ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel(ylabel or ("Density (scaled to max = 1)" if normalize else "Density"),
                  fontsize=label_fontsize - 2, labelpad=10)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if side_label:
        ax.annotate(side_label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-side_label_offset, 0),
                    textcoords="offset points", rotation=90, ha="center", va="center", annotation_clip=False,
                    fontsize=side_label_fontsize, fontweight="bold")

    # --- title, legend and layout ---
    top = 1.0
    fig_h_pt = fig.get_figheight() * 72
    if show_title and title:
        fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=label_fontsize + 2, fontweight="bold")
        top -= (label_fontsize + 2) * 1.7 / fig_h_pt

    if show_legend and legend_loc == "top":
        inv = fig.transFigure.inverted()
        kw = dict(fontsize=legend_fontsize, frameon=False, handlelength=1.6, handletextpad=0.5)
        ncol_m = len(samples_by_model)
        # width of the "True value" block, to centre the whole group
        w_t = 0.0
        if true_h:
            tmp = fig.legend(handles=true_h, loc="upper left", bbox_to_anchor=(0, 0), **kw)
            fig.canvas.draw()
            w_t = tmp.get_window_extent().transformed(inv).width
            tmp.remove()
        # block of the models: one column per model (patch above, median below)
        leg_m = fig.legend(handles=model_h, loc="upper center", ncol=ncol_m,
                           bbox_to_anchor=(0.5 + w_t / 2, top), columnspacing=1.2, **kw)
        fig.canvas.draw()
        bb = leg_m.get_window_extent().transformed(inv)
        # "True value" on the left, centred between the two rows
        if true_h:
            fig.legend(handles=true_h, loc="center right", bbox_to_anchor=(bb.x0, (bb.y0 + bb.y1) / 2), **kw)
        fig.tight_layout(rect=(0, 0, 1, bb.y0 - 0.01))
    elif show_legend:
        ax.legend(handles=true_h + model_h, fontsize=legend_fontsize, frameon=False, loc=legend_loc)
        fig.tight_layout(rect=(0, 0, 1, top))
    else:
        fig.tight_layout(rect=(0, 0, 1, top))
    if margins is not None:
        fig.subplots_adjust(**margins)
    return fig, ax
