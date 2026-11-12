"""
Probabilistic forecasts: the posterior samples of the parameters, estimated in the rolling-window
MCMC step (`pinnvsmech.inference.mcmc`), are propagated forward through the PINN.

For each week t0 the forecast covers the weeks of the window of observations (the "fit") and the
`horizon` weeks after t0. Every posterior sample gives one trajectory; the trajectories are
summarised by quantiles (`pinnvsmech.forecasting.summaries`).
"""

import os
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pinnvsmech.constants import quantiles as QUANTILES, rename_map as RENAME_MAP
from pinnvsmech.forecasting.summaries import compute_quantiles, summarize_forecast, truncate_forecast_blocks
from pinnvsmech.inference.cases import week_label
from pinnvsmech.inference.mcmc import cut_times, load_mcmc_chain, observable_cumulative
from pinnvsmech.pinn.approximation import D_incidence, evaluation_pinn

# Time grid on which the PINN is evaluated for the forecasts (days). It goes beyond the training domain
# of the PINN, [0, 365], to have enough points after any onset tau0: for late onsets the forecast is an
# extrapolation of the PINN.
FORECAST_TIMES = np.linspace(0, 600, 601)


def forecast_pinn(x, model, N_age, dt, window, delta_fixed=None, observable="deaths", times=FORECAST_TIMES):
    """
    Incidence forecast of the PINN for a single set of parameters.

    Parameters
    ----------
    x : array-like
        Parameters [R0, Delta, tau0], or [R0, tau0] if `delta_fixed` is given.
    model : torch.nn.Module
        Pretrained PINN.
    N_age : np.ndarray (k_age,)
        Population of the age classes.
    dt : int
        Time step of the incidence (days).
    window : int
        Number of weeks of the forecast, counted from the first week after the onset tau0.
    delta_fixed : float or None
        Value of Delta when it is not estimated.
    observable : "deaths" or "infections"
        See `pinnvsmech.inference.mcmc.model_fun_pinn`.
    times : array-like
        Time grid on which the PINN is evaluated.

    Returns
    -------
    ndarray, shape (window,)
        Weekly incidence of the observable.
    """
    if delta_fixed is None:
        R0, Delta, tau0 = x
    else:
        R0, tau0 = x
        Delta = delta_fixed

    model_eval = evaluation_pinn([R0, Delta], model, times)
    t_inc = times[::dt].astype(int)

    # Align with the onset tau0
    t_inc, _ = cut_times(t_inc, window, tau0, forecast=True)

    cumulative = observable_cumulative(model_eval, N_age, observable)
    return D_incidence(cumulative, dt=dt, times=t_inc)


def forecast_TV(time_plot_x, chain_samples, N_age, dt, window_forecast, pinn_model=None,
                delta_fixed=None, observable="deaths", times=FORECAST_TIMES):
    """
    Forecast trajectories of all the posterior samples, stacked.

    Returns
    -------
    T, V : ndarray, shape (n_samples * len(window_forecast),)
        Week index and forecast value; the first len(window_forecast) entries belong to the first sample, etc.
    """
    if pinn_model is None:
        raise ValueError("pinn_model must be provided.")

    T = []
    V = []
    for params in chain_samples:
        T.append(np.asarray(time_plot_x))
        V.append(forecast_pinn(params, pinn_model, N_age, dt, len(window_forecast),
                               delta_fixed, observable, times))

    return np.concatenate(T), np.concatenate(V)


def generate_forecast_samples(t0_train, dt_train, horizon, chain_samples, N_age, dt, dates,
                              pinn_model=None, delta_fixed=None, observable="deaths", times=FORECAST_TIMES):
    """
    Forecast samples for a single cutoff week.

    Parameters
    ----------
    t0_train : int
        Index of the last observed week.
    dt_train : int
        Length of the window of observations (weeks).
    horizon : int
        Forecast horizon (weeks after t0_train).
    chain_samples : ndarray
        Posterior samples.
    dates : np.ndarray
        Weekly dates; the forecast does not go beyond the last one.

    Returns
    -------
    T, V : ndarray
        Weeks and values, stacked (see `forecast_TV`).
    window_forecast : ndarray
        Indices of the weeks of the forecast window: from the first week of the window of observations to
        t0_train + horizon.
    """
    start = max(0, t0_train - dt_train + 1)
    end = min(len(dates), t0_train + horizon + 1)
    window_forecast = np.arange(start, end)

    T, V = forecast_TV(window_forecast, chain_samples, N_age, dt, window_forecast,
                       pinn_model=pinn_model, delta_fixed=delta_fixed, observable=observable, times=times)

    return T, V, window_forecast


def run_probabilistic_forecast(
    chains_dir,
    dates,
    obs,
    dt_train,
    horizon,
    N_age,
    pinn_model,
    t0_range,
    delta_fixed=None,
    observable="deaths",
    dt=7,
    times=FORECAST_TIMES,
    fit=True,
    quantiles=None,
    rename_map=None,
    sample_strategy="tail",
    nsample=1000,
    nsim=10000,
    tag=None,
    output_dir=None,
):
    """
    Probabilistic forecast at every week of `t0_range`.

    For each week t0 it loads the posterior samples saved by `run_rolling_mcmc_inference`
    (`chains_dir/obs=<week>.pickle`), propagates them through the PINN and saves everything in
    `output_dir/forecast_results.pickle`.

    Parameters
    ----------
    chains_dir : str or Path
        Folder with the chains of one run (one seed).
    dates : np.ndarray
        Weekly dates (or labels) of the observations.
    obs : np.ndarray
        Weekly observations (saved with the forecast, used to evaluate it).
    dt_train : int
        Length of the window of observations (weeks).
    horizon : int
        Forecast horizon in weeks (4 for short-term, 10 for long-term).
    N_age : np.ndarray (k_age,)
        Population of the age classes. To forecast one age class, put to zero all the others and give
        the observations of that class.
    pinn_model : torch.nn.Module
        Pretrained PINN.
    t0_range : iterable of int
        Indices of the cutoff weeks.
    delta_fixed, observable :
        See `forecast_pinn`.
    dt : int
        Time step of the incidence (days).
    times : np.ndarray
        Time grid on which the PINN is evaluated.
    fit : bool
        True: the forecast includes also the weeks of the window of observations (the fit);
        False: only t0 and the weeks after it.
    quantiles, rename_map : list, dict
        Quantiles of the forecast and the names of their columns. Default: those of `pinnvsmech.constants`.
    sample_strategy, nsample, nsim :
        Posterior samples, see `load_mcmc_chain`.
    tag : str, optional
        Suffix of the output folder, e.g. "age0".
    output_dir : str or Path, optional
        Default: `chains_dir/forecast_h<horizon>[_<tag>]`.

    Returns
    -------
    forecasts : dict
        Indexed by the week of the cutoff. For every week: `raw` (DataFrame t, v with all the samples),
        `summary_90`, `summary_50`, `quantiles`, `t0`, `date`, `y_obs` (observations of the window).
    """
    if pinn_model is None:
        raise ValueError("pinn_model must be provided.")
    quantiles = QUANTILES if quantiles is None else quantiles
    rename_map = RENAME_MAP if rename_map is None else rename_map

    suffix = f"_{tag}" if tag else ""
    if output_dir is None:
        if tag:
            output_dir = f"{chains_dir}/forecast{suffix}"      # e.g. forecast_age0
        else:
            output_dir = f"{chains_dir}/forecast_h{horizon}"    
    os.makedirs(output_dir, exist_ok=True)

    forecasts = {}

    for t0_train in tqdm(t0_range, desc=f"Forecast h={horizon}{suffix}", leave=False):

        label = week_label(dates[t0_train])
        chain = load_mcmc_chain(f"{chains_dir}/obs={label}.pickle",
                                sample_strategy=sample_strategy, nsample=nsample, nsim=nsim)

        T, V, window_forecast = generate_forecast_samples(
            t0_train, dt_train, horizon, chain, N_age, dt, dates,
            pinn_model=pinn_model, delta_fixed=delta_fixed, observable=observable, times=times,
        )

        window_train = np.arange(max(0, t0_train - dt_train + 1), t0_train + 1)

        if fit:
            T_tr, V_tr = T, V
        else:
            T_tr, V_tr = truncate_forecast_blocks(T, V, window_forecast, window_train)

        df = pd.DataFrame({"t": T_tr, "v": V_tr})

        forecasts[label] = {
            "raw": df,
            "summary_90": summarize_forecast(df, error_width=90),
            "summary_50": summarize_forecast(df, error_width=50),
            "quantiles": compute_quantiles(df, quantiles, rename_map),
            "t0": t0_train,
            "date": label,
            "y_obs": obs[max(0, t0_train - dt_train + 1):t0_train + 1],
        }

    with open(f"{output_dir}/forecast_results.pickle", "wb") as f:
        pickle.dump(forecasts, f)

    return forecasts
