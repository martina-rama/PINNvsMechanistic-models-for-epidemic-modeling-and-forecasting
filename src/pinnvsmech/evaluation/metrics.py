"""
Quantitative comparison of the forecasts of the models (PINN, Mechanistic and baseline) with
the observations: scores per horizon, pooled over the horizons and normalised by the baseline.

All the models are scored on the windows that they have in common. The observations (`true_incidence`) are the series
of the case, indexed by the absolute week (new deaths for the real case, noisy new infections for the synthetic
scenarios), not the `y_obs` field of the forecast files. The horizon is h = t - t0 (h >= 1: only the future is scored,
not the fit).

Scores (see `forecast_validation`), not weighted: MAE and RMSE of the median, WIS, IS of the 80% interval, wMAPE and
coverage of the 50% and 90% intervals. They are computed once on the pooled array of all the windows (the scores
already average over the points).
"""

import numpy as np
import pandas as pd

from pinnvsmech.evaluation.forecast_validation import coverage_error, is_error, wis_error, wmape_error

ERROR_METRICS = ["MAE", "WIS", "RMSE", "IS_80", "wMAPE"]       # normalised by the baseline
COVERAGE_METRICS = ["Coverage_50", "Coverage_90"]               # not normalised
TABLE_COLUMNS = ["MAE", "wMAPE", "RMSE", "WIS", "IS_80", "Coverage_50", "Coverage_90"]


def _scores(y_arr, q50_arr, quantiles_rows):
    """Scores of the pooled forecasts (arrays in the same order)."""
    cov50, cov90 = coverage_error(y_arr, quantiles_rows)
    return {
        "MAE": np.mean(np.abs(q50_arr - y_arr)),
        "RMSE": np.sqrt(np.mean((q50_arr - y_arr) ** 2)),
        "WIS": wis_error(y_arr, quantiles_rows, weighted=False),
        "IS_80": is_error(y_arr, quantiles_rows, alpha=0.20),
        "wMAPE": wmape_error(y_arr, q50_arr, weighted=False),
        "Coverage_50": cov50,
        "Coverage_90": cov90,
    }


def _common_keys(forecasts_by_model):
    keys = None
    for forecasts in forecasts_by_model.values():
        keys = set(forecasts.keys()) if keys is None else keys & set(forecasts.keys())
    if not keys:
        raise ValueError("No window in common between the models")
    return keys


def compute_metrics_per_horizon(forecasts_by_model, true_incidence, horizons=(1, 2, 3, 4), t0_min=None, t_last=None):
    """
    Scores of every model at every horizon, on the windows common to all the models.

    Parameters
    ----------
    forecasts_by_model : dict[str, dict]
        {model name: forecasts dict}, e.g. {"PINN": ..., "Mechanistic": ..., "Baseline": ...}.
    true_incidence : array-like
        Observed series of the whole period, indexed by the absolute week.
    horizons : tuple of int
        Horizons (t - t0) to score.
    t0_min : int, optional
        First window (index).
    t_last : int, optional
        Last week with observations: the targets t0 + h must be <= t_last.

    Returns
    -------
    DataFrame with a row for every (Model, Horizon): n_rounds, MAE, RMSE, WIS, IS_80, wMAPE, Coverage_50, Coverage_90.
    """
    true_incidence = np.asarray(true_incidence)
    if t_last is not None:
        true_incidence = true_incidence[:t_last + 1]

    keys_common = _common_keys(forecasts_by_model)

    records = []
    for model_name, forecasts in forecasts_by_model.items():
        for h in horizons:
            q50_list, y_list, rows = [], [], []
            for key in keys_common:
                entry = forecasts[key]
                t0 = entry["t0"]
                if t0_min is not None and t0 < t0_min:
                    continue
                t_target = t0 + h
                q_df = entry["quantiles"]
                row_match = q_df[q_df["t"] == t_target]
                if row_match.empty:
                    continue                       # this window does not cover the horizon h
                if t_target >= len(true_incidence):
                    continue                       # beyond the observations
                y_list.append(true_incidence[t_target])
                q50_list.append(row_match["q50"].iloc[0])
                rows.append(row_match)

            n = len(y_list)
            if n == 0:
                continue
            quantiles_rows = pd.concat(rows, ignore_index=True)      # one row per window, same order as y and q50
            records.append({"Model": model_name, "Horizon": h, "n_rounds": n,
                            **_scores(np.array(y_list), np.array(q50_list), quantiles_rows)})

    return pd.DataFrame(records)


def _pool_points(forecasts, true_incidence, horizons, t0_min=None):
    """(y, q50, quantile row) of every (t0, h) available in `forecasts`."""
    q50_list, y_list, rows = [], [], []
    for entry in forecasts.values():
        t0 = entry["t0"]
        if t0_min is not None and t0 < t0_min:
            continue
        q_df = entry["quantiles"]
        for h in horizons:
            t_target = t0 + h
            row_match = q_df[q_df["t"] == t_target]
            if row_match.empty:
                continue
            if t_target >= len(true_incidence):
                continue
            y_list.append(true_incidence[t_target])
            q50_list.append(row_match["q50"].iloc[0])
            rows.append(row_match)
    return q50_list, y_list, rows


def compute_metrics_pooled(forecasts_by_model, true_incidence, horizons=(1, 2, 3, 4), common_only=True, t0_min=None,
                           t_last=None):
    """
    Scores of every model pooled over the horizons: every (t0, h) is an independent point.

    Parameters
    ----------
    common_only : bool
        True: only the windows common to all the models (same number of points for all). False: all the windows
        available for every model (models with fewer windows have fewer points).
    other parameters : see `compute_metrics_per_horizon`.

    Returns
    -------
    DataFrame with a row for every model: n_points, MAE, RMSE, WIS, IS_80, wMAPE, Coverage_50, Coverage_90.
    """
    true_incidence = np.asarray(true_incidence)
    if t_last is not None:
        true_incidence = true_incidence[:t_last + 1]

    keys_common = _common_keys(forecasts_by_model) if common_only else None

    records = []
    for model_name, forecasts in forecasts_by_model.items():
        use = {k: forecasts[k] for k in forecasts if k in keys_common} if common_only else forecasts
        q50_list, y_list, rows = _pool_points(use, true_incidence, horizons, t0_min=t0_min)
        n = len(y_list)
        if n == 0:
            continue
        quantiles_rows = pd.concat(rows, ignore_index=True)
        records.append({"Model": model_name, "n_points": n,
                        **_scores(np.array(y_list), np.array(q50_list), quantiles_rows)})

    return pd.DataFrame(records)


def normalize_by_baseline(results_df, baseline_model="Baseline"):
    """
    Divide the errors (MAE, WIS, RMSE, IS_80, wMAPE) by those of the baseline, horizon by horizon (table of
    `compute_metrics_per_horizon`). The coverage is not changed.
    """
    df_norm = results_df.copy()
    for h in df_norm["Horizon"].unique():
        mask_baseline = (df_norm["Horizon"] == h) & (df_norm["Model"] == baseline_model)
        if mask_baseline.sum() == 0:
            continue
        baseline_vals = df_norm[mask_baseline][ERROR_METRICS].iloc[0]
        mask_h = df_norm["Horizon"] == h
        df_norm.loc[mask_h, ERROR_METRICS] = df_norm.loc[mask_h, ERROR_METRICS].values / baseline_vals.values
    return df_norm


def normalize_pooled_by_baseline(results_df, baseline_model="Baseline"):
    """Divide the errors by those of the baseline (table of `compute_metrics_pooled`). The coverage is not changed."""
    df_norm = results_df.copy()
    mask_baseline = df_norm["Model"] == baseline_model
    if mask_baseline.sum() == 0:
        print("Warning: baseline model not found in the table")
        return df_norm
    baseline_vals = df_norm[mask_baseline][ERROR_METRICS].iloc[0]
    df_norm[ERROR_METRICS] = df_norm[ERROR_METRICS].values / baseline_vals.values     # also the baseline row (= 1)
    return df_norm
