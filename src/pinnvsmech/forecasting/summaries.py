"""
Summaries of the forecast trajectories: quantiles and prediction intervals for every week.
"""

import numpy as np


def truncate_forecast_blocks(T, V, window_forecast, window_train):
    """
    Retain only true forecast horizons.

    Parameters
    ----------
    T, V : ndarray
        Stacked times and forecast values: one block of len(window_forecast) entries for every posterior sample.
    window_forecast : ndarray
        Full forecast window.
    window_train : ndarray
        Observation window indices, used to determine how many entries to discard from each forecast block.

    Returns
    -------
    T_trunc, V_trunc : ndarray
        From each block only the last `len(window_forecast) - len(window_train) + 1` entries are kept: the
        cutoff week and the weeks of the forecast horizon.
    """
    block_dim = len(window_forecast)
    last_n = block_dim - len(window_train) + 1

    T_trunc, V_trunc = [], []

    for i in range(0, len(T), block_dim):
        end = i + block_dim
        if end <= len(T):
            T_trunc.extend(T[end - last_n:end])
            V_trunc.extend(V[end - last_n:end])

    return np.array(T_trunc), np.array(V_trunc)


def summarize_forecast(df, error_width=90):
    """
    Compute mean forecast and prediction interval.

    Parameters
    ----------
    df : DataFrame
        Forecast samples with columns ['t', 'v'].
    error_width : int
        Width of the prediction interval (e.g. 90).

    Returns
    -------
    summary : DataFrame
        Mean, lower and upper bounds per time step (columns `t`, `mean`, `lower`, `upper`).
    """
    alpha = (100 - error_width) / 2

    summary = (
        df.groupby("t")["v"]
        .agg(
            mean="mean",
            lower=lambda x: np.percentile(x, alpha),
            upper=lambda x: np.percentile(x, 100 - alpha),
        )
        .reset_index()
    )

    return summary


def compute_quantiles(df, quantiles, rename_map):
    """
    Compute quantile-based probabilistic forecasts.

    Parameters
    ----------
    df : DataFrame
        Forecast samples with columns ['t', 'v'].
    quantiles : list
        Quantile levels to compute.
    rename_map : dict
        Mapping from quantile to column name.

    Returns
    -------
    DataFrame
        Quantile forecasts per time step (columns `t` and the names of `rename_map`).
    """
    return (
        df.groupby("t")["v"]
        .quantile(quantiles)
        .unstack()
        .rename(columns=rename_map)
        .reset_index()
    )
