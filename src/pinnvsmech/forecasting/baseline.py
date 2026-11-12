"""
Baseline forecasting model, reproduced from:

    Gozzi N, Perra N, Vespignani A (2025). "Comparative evaluation of behavioral epidemic models using COVID-19
    data." PNAS 122(24):e2421993122. Materials and Methods, "Baseline forecasting model" (p. 8).

Definition from the paper:

    The baseline predicts as median the last data point of the calibration period. The predictive intervals come from
    the previous 1-step increments: delta = (d_2, d_3, ..., d_t) are the 1-step differences up to time t and, to keep
    the median on the last calibration point, they are symmetrised, delta' = (delta, -delta). If the maximum horizon is
    H, H differences are sampled from delta'; the prediction at horizon h is f_h = v_t + sum_{i=1}^{h} d_hat_i, where
    v_t is the last observed value and d_hat_i are the sampled 1-step differences. 10,000 trajectories are generated
    and their quantiles give the predictive intervals.

It is the neutral reference (a symmetrised random walk from the last observed value) against which the models are
scored: relative WIS = WIS of the model / WIS of the baseline.

The output has the same format as the forecasts of the PINN and of the Mechanistic model (`forecast_results.pickle`):
one entry per window, with `raw`, `summary_90`, `summary_50`, `quantiles`, `t0`, `date`, `y_obs`, so that it can be used
in the same plots and scores. The baseline has no fit over the calibration period (it only resamples past increments):
its tables cover t = t0 (the last known point, with no uncertainty, for continuity with the observed point in the
plots) up to t0 + horizon.
"""

import numpy as np
import pandas as pd

from pinnvsmech.constants import quantiles as QUANTILES, rename_map as RENAME_MAP
from pinnvsmech.inference.cases import week_label


def baseline_label(case):
    """Name of the folder of the baseline of a case: the label of the case without the look-back (it does not use it)."""
    return case.label.split("_lookback")[0]


def compute_baseline_window(y_obs, t0, horizon=4, n_trajectories=10000, seed=None, clip_negative=True,
                            quantiles=None, rename_map=None):
    """
    Baseline forecast for one rolling window.

    Parameters
    ----------
    y_obs : array-like, length t0 + 1
        Observed weekly values of the weeks 0..t0 (the calibration period).
    t0 : int
        Index of the current week (last week of the calibration period).
    horizon : int
        Number of weeks to forecast (H in the paper).
    n_trajectories : int
        Number of bootstrap trajectories (10,000 in the paper).
    seed : int or None
        Seed of the random numbers (None: not reproducible).
    clip_negative : bool
        NOT in the paper: the symmetrised bootstrap can give negative values (the sum of the increments can go below
        zero near the peak of the epidemic), and weekly counts cannot be negative: True clips the trajectories at 0
        before the summaries. False: literal reproduction of the algorithm of the paper.

    Returns
    -------
    dict with `raw`, `summary_90`, `summary_50`, `quantiles`, `t0`, `y_obs` (one entry of `forecast_results.pickle`,
    without `date`).
    """
    quantiles = QUANTILES if quantiles is None else quantiles
    rename_map = RENAME_MAP if rename_map is None else rename_map

    y_obs = np.asarray(y_obs, dtype=float)
    if len(y_obs) != t0 + 1:
        raise ValueError(f"y_obs has length {len(y_obs)}, expected t0 + 1 = {t0 + 1}")

    v_t = y_obs[-1]

    # 1-step differences of the calibration period: delta = (d_2, ..., d_t)
    diffs = np.diff(y_obs)
    if len(diffs) == 0:
        # a single point: no increments to resample, flat forecast with no uncertainty
        diffs = np.array([0.0])

    # symmetrise: delta' = (delta, -delta)
    diffs_sym = np.concatenate([diffs, -diffs])

    rng = np.random.default_rng(seed)

    # H differences per trajectory, sampled with replacement, then cumulative sum
    sampled = rng.choice(diffs_sym, size=(n_trajectories, horizon), replace=True)
    forecasts = v_t + np.cumsum(sampled, axis=1)             # f_h = v_t + sum_{i=1}^{h} d_hat_i

    # t0 itself as a degenerate point (all the trajectories at v_t)
    t_range = np.arange(t0, t0 + horizon + 1)
    part = np.concatenate([np.full((n_trajectories, 1), v_t), forecasts], axis=1)

    if clip_negative:
        part = np.clip(part, 0, None)

    raw = pd.DataFrame({"t": np.tile(t_range, n_trajectories), "v": part.flatten()})

    mean = part.mean(axis=0)
    p05, p25, p75, p95 = np.percentile(part, [5, 25, 75, 95], axis=0)
    summary_90 = pd.DataFrame({"t": t_range, "mean": mean, "lower": p05, "upper": p95})
    summary_50 = pd.DataFrame({"t": t_range, "mean": mean, "lower": p25, "upper": p75})

    qvals = np.percentile(part, [100 * q for q in quantiles], axis=0)    # (n_quantiles, horizon + 1)
    quantiles_df = pd.DataFrame({"t": t_range})
    for q, vals in zip(quantiles, qvals):
        quantiles_df[rename_map[q]] = vals

    return {
        "raw": raw,
        "summary_90": summary_90,
        "summary_50": summary_50,
        "quantiles": quantiles_df,
        "t0": t0,
        "y_obs": y_obs,
    }


def compute_baseline_forecasts_from_series(cases, t0_list, dates=None, horizon=4, n_trajectories=10000, seed=None,
                                           clip_negative=True):
    """
    Baseline forecasts for a list of windows, built directly from the observed series (no file of the models needed).

    Parameters
    ----------
    cases : array-like, length n_weeks
        Observed weekly series: new deaths (real case) or noisy new infections (synthetic scenarios), the ones the
        models are calibrated on.
    t0_list : iterable of int
        Index of the current week of every window.
    dates : array-like, length n_weeks, or None
        Dates or labels ("week 12") of the weeks, for the keys of the dict: 2020-03-15 (real) or week_13 (synthetic).
        None: week_<t0>.
    horizon, n_trajectories, seed, clip_negative : see `compute_baseline_window`.

    Returns
    -------
    dict in the format of `forecast_results.pickle`, one entry per window.
    """
    cases = np.asarray(cases, dtype=float)
    out = {}
    for t0 in t0_list:
        t0 = int(t0)
        entry = compute_baseline_window(cases[:t0 + 1], t0, horizon=horizon, n_trajectories=n_trajectories,
                                        seed=seed, clip_negative=clip_negative)
        key = week_label(dates[t0]) if dates is not None else f"week_{t0}"
        entry["date"] = key
        out[key] = entry
    return out
