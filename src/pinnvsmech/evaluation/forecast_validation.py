"""
Scores of probabilistic forecasts from quantiles: coverage of the 50% and 90% intervals, interval score (IS),
weighted interval score (WIS) and wMAPE.

Inputs: `data_inc`, the observed values (one per forecast), and `forecast_df`, a DataFrame with one row per forecast
(same order as `data_inc`) and the columns `q01 ... q99` of the quantiles (see `pinnvsmech.constants`). Every function
returns the mean over the forecasts. With `weighted=True` the mean is replaced by a weighted mean with harmonic weights
that increase towards the most recent observations (the same weights as in the likelihood of the MCMC inference).
"""

import warnings

import numpy as np


def coverage_error(data_inc, forecast_df):
    """
    Compute coverage metrics for prediction intervals.

    Parameters
    ----------
    data_inc : array-like
        Observed incidence values.
    forecast_df : dict or DataFrame
        Forecast quantiles with keys/columns: 'q05', 'q25', 'q75', 'q95'.

    Returns
    -------
    cov_50 : float
        Coverage of 50% prediction interval (q25-q75).
    cov_90 : float
        Coverage of 90% prediction interval (q05-q95).
    """
    cov_50 = 0
    cov_90 = 0

    for idx_inc in range(len(data_inc)):
        # Check if observation falls within 50% PI
        if (data_inc[idx_inc] >= forecast_df['q25'].iloc[idx_inc]) & (data_inc[idx_inc] <= forecast_df['q75'].iloc[idx_inc]):
            cov_50 = cov_50 + 1

        # Check if observation falls within 90% PI
        if (data_inc[idx_inc] >= forecast_df['q05'].iloc[idx_inc]) & (data_inc[idx_inc] <= forecast_df['q95'].iloc[idx_inc]):
            cov_90 = cov_90 + 1

    cov_50 = cov_50 / len(data_inc)
    cov_90 = cov_90 / len(data_inc)

    return cov_50, cov_90


def is_error(data_inc, forecast_df, alpha=0.20):
    """
    Compute Interval Score (IS) for a single prediction interval at level (1-alpha).

    IS_alpha(l, u, y) = (u - l) + (2/alpha) * max(l - y, 0) + (2/alpha) * max(y - u, 0)

    Parameters
    ----------
    data_inc    : array-like - observed values
    forecast_df : DataFrame with the quantile columns
    alpha       : float - nominal level (default 0.20 -> 80% PI)
                  alpha=0.20 -> PI 80% (q10, q90)
                  alpha=0.10 -> PI 90% (q05, q95)
                  alpha=0.50 -> PI 50% (q25, q75)

    Returns
    -------
    is_score : float - mean IS over all the points
    """

    quantile_bounds = {
        0.02: ('q01',   'q99'),   # PI 98%: [0.01, 0.99]
        0.05: ('q025',  'q975'),  # PI 95%: [0.025, 0.975]
        0.10: ('q05',   'q95'),   # PI 90%: [0.05, 0.95]
        0.20: ('q10',   'q90'),   # PI 80%: [0.10, 0.90]
        0.30: ('q15',   'q85'),   # PI 70%: [0.15, 0.85]
        0.40: ('q20',   'q80'),   # PI 60%: [0.20, 0.80]
        0.50: ('q25',   'q75'),   # PI 50%: [0.25, 0.75]
        0.60: ('q30',   'q70'),   # PI 40%: [0.30, 0.70]
        0.70: ('q35',   'q65'),   # PI 30%: [0.35, 0.65]
        0.80: ('q40',   'q60'),   # PI 20%: [0.40, 0.60]
        0.90: ('q45',   'q55'),   # PI 10%: [0.45, 0.55]
    }

    if alpha not in quantile_bounds:
        raise ValueError(f"alpha={alpha} not supported. Choose between: {list(quantile_bounds.keys())}")

    lower_col, upper_col = quantile_bounds[alpha]
    is_score = 0.0

    for idx_inc in range(len(data_inc)):
        l = forecast_df[lower_col].iloc[idx_inc]
        u = forecast_df[upper_col].iloc[idx_inc]
        y = data_inc[idx_inc]

        IS = (
            (u - l) +
            (2 / alpha) * np.clip(l - y, a_min=0, a_max=None) +
            (2 / alpha) * np.clip(y - u, a_min=0, a_max=None)
        )
        is_score += IS

    return is_score / len(data_inc)


def wis_error(data_inc, forecast_df, weighted=False):
    """
    Compute Weighted Interval Score (WIS) as weighted sum of IS across multiple intervals.
    It uses `is_error` for every alpha.
    """
    alpha_list = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    K = len(alpha_list)
    w_0 = 0.5

    n = len(data_inc)
    wis_per_point = np.empty(n)

    for idx_inc in range(n):
        wis_components = []
        for alpha in alpha_list:
            w = alpha / 2
            IS = is_error([data_inc[idx_inc]], forecast_df.iloc[[idx_inc]], alpha=alpha)
            wis_components.append(w * IS)
        abs_error_median = abs(forecast_df['q50'].iloc[idx_inc] - data_inc[idx_inc])
        wis_components.append(w_0 * abs_error_median)
        wis_per_point[idx_inc] = sum(wis_components) / (K + 0.5)

    if weighted:
        weights = np.array([1.0 / (1 + n - i) for i in range(1, n + 1)])
        weights = weights / np.sum(weights)
        return weights @ wis_per_point
    else:
        return wis_per_point.mean()


def wmape_error_old(data_inc, forecast_median, weighted=False):
    """
    Compute weighted Mean Absolute Percentage Error (wMAPE).

    wMAPE = sum|pred - true| / sum|true|

    Parameters
    ----------
    data_inc        : array-like - observed values
    forecast_median : array-like - forecast median

    Returns
    -------
    wmape : float
    """
    data_inc = np.asarray(data_inc)
    forecast_median = np.asarray(forecast_median)

    if weighted:
        n = len(data_inc)
        weights = np.array([1.0 / (1 + n - i) for i in range(1, n + 1)])
        weights = weights / np.sum(weights)
        wmape_value = np.abs(forecast_median - data_inc) / np.abs(data_inc)
        return weights @ wmape_value
    else:
        denom = np.sum(np.abs(data_inc))
        if denom == 0:
            return np.nan
        return np.sum(np.abs(forecast_median - data_inc)) / denom


def wmape_error(data_inc, forecast_median, weighted=False):
    """
    Compute (weighted) Mean Absolute Percentage Error (wMAPE).

    Unweighted: wMAPE = sum|pred - true| / sum|true|
    Weighted:   weighted average of per-point relative errors, using harmonic
                weights that increase towards the most recent observations
                (same weighting scheme as in wis_error / MCMC inference).

    Parameters
    ----------
    data_inc        : array-like - observed values
    forecast_median : array-like - forecast median
    weighted        : bool - if True, apply harmonic recency weighting

    Returns
    -------
    wmape : float
    """
    data_inc = np.asarray(data_inc, dtype=float)
    forecast_median = np.asarray(forecast_median, dtype=float)
    n = len(data_inc)

    if weighted:
        weights = np.array([1.0 / (1 + n - i) for i in range(1, n + 1)])
        weights = weights / np.sum(weights)

        # count trailing zeros = most recent observations = highest weights
        trailing_zeros = 0
        for v in data_inc[::-1]:
            if v == 0:
                trailing_zeros += 1
            else:
                break

        if trailing_zeros == n:
            mae = np.mean(np.abs(forecast_median - data_inc))
            warnings.warn(
                "wmape_error: the entire observation window is zero, weighted wMAPE is undefined. "
                "Falling back to MAE for this window.",
                RuntimeWarning,
            )
            return mae

        if trailing_zeros > 0:
            excluded_weight = np.sum(weights[n - trailing_zeros:])
            warnings.warn(
                f"wmape_error: {trailing_zeros} zero value(s) at the end of the window "
                f"(excluded weight ~ {excluded_weight:.1%}). Weighted wMAPE computed on the "
                f"remaining {n - trailing_zeros} points only: result may be unreliable for this round.",
                RuntimeWarning,
            )

        mask = np.abs(data_inc) > 0   # remove the zero data and reweight the remaining points
        wmape_value = np.abs(forecast_median[mask] - data_inc[mask]) / np.abs(data_inc[mask])
        valid_weights = weights[mask] / np.sum(weights[mask])
        return valid_weights @ wmape_value
    else:
        denom = np.sum(np.abs(data_inc))
        if denom == 0:
            mae = np.mean(np.abs(forecast_median - data_inc))
            warnings.warn(
                "wmape_error: the entire observation window is zero, wMAPE is undefined. "
                "Falling back to MAE for this window.",
                RuntimeWarning,
            )
            return mae
        else:
            return np.sum(np.abs(forecast_median - data_inc)) / denom
