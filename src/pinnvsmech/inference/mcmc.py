"""
Rolling-window MCMC estimation of the epidemic parameters with the pretrained PINN.

At every week t0 the parameters are estimated on the observations of a window that ends at t0:

- R0                : basic reproduction number;
- Delta             : mean delay in days (estimated only if `delta_fixed` is None);
- t_0 (tau0)        : epidemic onset, in days from the start of the PINN time axis.

The PINN gives the cumulative trajectory of the observable on the daily grid `times`; the weekly
incidence is computed on the weekly grid that starts at the onset tau0 and compared with the
observations through a Poisson log-likelihood (or wMAPE / MSE). The posterior is sampled with
pymcmcstat (adaptive Metropolis).

Two steps:

1. `run_rolling_mcmc_inference`   one chain per week t0, saved as obs=<week>.pickle
2. `save_mcmc_chain_tails`        samples of the last iterations of every chain, with the
                                  derived Rt and I(t_start), saved as mcmc_distributions.pkl
"""

import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.special import gammaln

# pymcmcstat (through mcmcplot) imports pi, sin and cos from scipy, which recent versions of scipy no
# longer provide. They are the same as the numpy ones.
for _name in ("pi", "sin", "cos"):
    if not hasattr(scipy, _name):
        setattr(scipy, _name, getattr(np, _name))
from pymcmcstat.MCMC import MCMC  # noqa: E402
from pymcmcstat.propagation import define_sample_points  # noqa: E402

from pinnvsmech.inference.cases import load_week, week_label  # noqa: E402
from pinnvsmech.pinn.approximation import D_incidence, evaluation_pinn  # noqa: E402

# Continuous time grid on which the PINN is evaluated (days)
DEFAULT_TIMES = np.linspace(0, 365, 366)


def default_bounds(delta_fixed=None):
    """
    Initial value and bounds of the estimated parameters: (R0, Delta, tau0), or (R0, tau0) if Delta is fixed.
    The bounds of R0 and Delta are the ranges of the training set of the PINN.

    Returns
    -------
    theta0, lb, ub : tuples
    """
    if delta_fixed is None:
        return (2.15, 21, 182), (1.3, 7, 0), (3, 35, 365)
    return (2.15, 182), (1.3, 0), (3, 365)


# ---------------------------------------------------------------------------
# Forward model and likelihood
# ---------------------------------------------------------------------------

def cut_times(t_inc, y_inc, tau0_guess, forecast=False):
    """
    Align incidence times and observations after the inferred epidemic onset.

    Parameters
    ----------
    t_inc : array-like
        Discrete incidence time grid.
    y_inc : array-like or int
        Observed incidence values (or their number, in forecast mode).
    tau0_guess : float
        Estimated epidemic start time.
    forecast : bool, optional
        If True, only the time grid is truncated (used for the forecasts).

    Returns
    -------
    t_inc : array
        Truncated incidence time grid.
    y_inc : array
        Truncated incidence observations (in inference mode).
    """

    # Keep only incidence times after the inferred epidemic onset
    t_inc = t_inc[t_inc >= tau0_guess]

    if forecast:
        # Forecasting: y_inc is only the number of weeks
        min_len = np.min([len(t_inc), y_inc])

    else:
        # Inference: time grid and observations have equal length
        min_len = np.min([len(t_inc), len(y_inc)])
        y_inc = y_inc[:min_len]

    t_inc = t_inc[:min_len]

    return t_inc, y_inc


def observable_cumulative(model_eval, N_age, observable):
    """
    Cumulative trajectory of the observable, aggregated over the age classes.

    Parameters
    ----------
    model_eval : ndarray, shape (T, 6*k_age)
        Output of the PINN (see `evaluation_pinn`).
    N_age : array-like
        Population of the age classes (the PINN works with fractions of N_age).
    observable : "deaths" or "infections"
        "deaths": observed deaths (compartment Dobs); "infections": minus the susceptibles (-S), whose
        increments are the new infections.

    Returns
    -------
    ndarray, shape (T,)
    """
    k_age = len(N_age)
    if observable == "deaths":
        return (model_eval[:, 5 * k_age:6 * k_age] * N_age).sum(axis=1)
    if observable == "infections":
        return -(model_eval[:, 0:k_age] * N_age).sum(axis=1)
    raise ValueError(f"observable must be 'deaths' or 'infections', got '{observable}'")


def model_fun_pinn(x, param, t, dt, N_age, y_inc, model, delta_fixed=None, observable="deaths"):
    """
    Forward model: parameters -> weekly incidence of the observable, aligned with the observations.

    Parameters
    ----------
    x : array-like
        Dummy input required by pymcmcstat (not used).
    param : array-like
        Parameters [R0, Delta, tau0], or [R0, tau0] if `delta_fixed` is given.
    t : array-like
        Continuous time grid.
    dt : int
        Time step of the incidence (days).
    N_age : array-like
        Population of the age classes.
    y_inc : array-like
        Observed incidence.
    model : torch.nn.Module
        Pretrained PINN.
    delta_fixed : float or None
        Value of Delta when it is not estimated.
    observable : "deaths" or "infections"
        "deaths": new observed deaths (compartment Dobs);
        "infections": new infections (decrease of the susceptibles S).

    Returns
    -------
    y_inc : array
        Observations after the onset tau0.
    val : array
        Model incidence on the same weeks.
    """

    if delta_fixed is None:
        R0, Delta, tau0 = param
    else:
        R0, tau0 = param
        Delta = delta_fixed

    model_eval = evaluation_pinn([R0, Delta], model, t)

    # The incidence is computed on a weekly grid
    t_inc = t[::dt].astype("i")

    # Align the incidence with the inferred epidemic onset
    t_inc, y_inc = cut_times(t_inc, y_inc, tau0)

    cumulative = observable_cumulative(model_eval, N_age, observable)

    val = D_incidence(D=cumulative, dt=dt, times=t_inc)

    return y_inc, val


def make_ssfun_pinn(t, dt, N_age, y_inc, model, loss, weight, delta_fixed=None, observable="deaths"):
    """
    Build the function that pymcmcstat minimises.

    With loss='poisson' it returns the negative Poisson log-likelihood of the observations given the
    model incidence (appropriate for counts); 'wmape' and 'mse' return the corresponding errors.

    Parameters
    ----------
    t : array-like
        Continuous time grid.
    dt : int
        Time step of the incidence (days).
    N_age : array-like
        Population of the age classes.
    y_inc : array-like
        Observed incidence of the window.
    model : torch.nn.Module
        Pretrained PINN.
    loss : 'poisson', 'wmape' or 'mse'
    weight : bool
        Harmonic weights on the log-likelihood (last weeks count more). Only for the Poisson loss.
    delta_fixed, observable :
        See `model_fun_pinn`.

    Returns
    -------
    ssfun : callable
        Function `ssfun(theta, data, custom=None)` for pymcmcstat.
    """

    def ssfun(theta, data, custom=None):
        y_data, fun_val = model_fun_pinn(
            data.xdata[0],
            theta,
            t,
            dt,
            N_age,
            y_inc,
            model,
            delta_fixed,
            observable,
        )

        # Penalise if after cut_times there are fewer points than observations (onset too late)
        n = len(y_data)
        if n < len(y_inc):
            return 1e6

        if loss == "poisson":
            # Clip the predictions to avoid log(0)
            epsilon = 1e-10
            fun_val = np.clip(fun_val, epsilon, None)

            # Poisson log-likelihood: sum_i [ y_i*log(mu_i) - mu_i - log(y_i!) ]
            log_lik = -fun_val + y_data * np.log(fun_val) - gammaln(y_data + 1)
            if weight:
                weights = np.array([1.0 / (1 + n - i) for i in range(1, n + 1)])  # harmonic weights
                weights = weights / np.sum(weights)
                log_lik = weights * log_lik

            # pymcmcstat minimises, so the negative log-likelihood is returned
            return -np.sum(log_lik)

        elif loss == "wmape":
            # wMAPE = sum|pred - true| / sum|true|
            denom = np.sum(np.abs(y_data))
            if denom == 0:
                return np.nan
            return np.sum(np.abs(fun_val - y_data)) / denom

        elif loss == "mse":
            res = y_data - fun_val
            return (res ** 2).sum(axis=0) / len(res)

        else:
            raise ValueError(f"loss must be 'poisson', 'wmape' or 'mse', got '{loss}'.")

    return ssfun


# ---------------------------------------------------------------------------
# Step 1: one chain per week
# ---------------------------------------------------------------------------

def run_rolling_mcmc_inference(
    N_age,
    dt_train,
    output_dir,
    dates,
    obs,
    pinn_model,
    t0_range,
    times=DEFAULT_TIMES,
    delta_fixed=None,
    observable="deaths",
    dt=7,
    weight=True,
    nsim=1e4,
    theta0=None,
    lb=None,
    ub=None,
    priors="uniform",
    loss="poisson",
    seed=None,
    n_early=4,
    early_t0=(75, 150),
):
    """
    Run the rolling-window MCMC estimation.

    For each week t0 of `t0_range` the parameters are estimated on the window of observations that
    ends at t0 (length `dt_train` weeks) and the chain is saved in `output_dir/obs=<week>.pickle`
    (dict with `results` of pymcmcstat, `runtime_seconds`, `seed`, `week`, `y_obs`).

    Parameters
    ----------
    N_age : array-like
        Population of the age classes.
    dt_train : int
        Length of the window of observations (weeks).
    output_dir : str or Path
        Folder where the chains are saved.
    dates : array-like
        Weekly dates (or labels) of the observations.
    obs : array-like
        Weekly observations.
    pinn_model : torch.nn.Module
        Pretrained PINN.
    t0_range : iterable of int
        Indices of the weeks at which the estimation is run.
    times : array-like
        Continuous time grid on which the PINN is evaluated (days).
    delta_fixed : float or None
        None: Delta is estimated (parameters R0, Delta, tau0). Number: Delta fixed to this value
        (parameters R0, tau0). It must be in the range of the training set of the PINN.
    observable : "deaths" or "infections"
        See `model_fun_pinn`.
    dt : int, optional
        Time step of the incidence (default: 7 days).
    weight : bool, optional
        Harmonic weights on the Poisson log-likelihood.
    nsim : int, optional
        Number of MCMC iterations (default: 1e4).
    theta0, lb, ub : tuple, optional
        Initial value, lower and upper bounds of the estimated parameters. By default those of
        `default_bounds(delta_fixed)`: the ranges of the training set of the PINN, tau0 in [0, 365].
    priors : str, optional
        - 'uniform': flat prior within [lb, ub].
        - 'normal': Gaussian prior on R0 and Delta, centred at theta0 with sigma = (ub - lb) / 3.
    loss : 'poisson', 'wmape' or 'mse'
    seed : int, optional
        Seed of the chain.
    n_early, early_t0 : int, (float, float)
        In the first `n_early` weeks of `t0_range` there are few observations: the onset tau0 starts
        from early_t0[0] and is bounded above by early_t0[1].

    Returns
    -------
    output_dir : str or Path
    """

    if priors not in ("uniform", "normal"):
        raise ValueError("Priors distribution has to be correctly specified. Please select either 'uniform' or 'normal'.")
    if pinn_model is None:
        raise ValueError("pinn_model must be provided")

    default_theta0, default_lb, default_ub = default_bounds(delta_fixed)
    theta0 = default_theta0 if theta0 is None else theta0
    lb = default_lb if lb is None else lb
    ub = default_ub if ub is None else ub

    names = ["R0"] + (["Delta"] if delta_fixed is None else []) + ["t_0"]
    if not (len(theta0) == len(lb) == len(ub) == len(names)):
        raise ValueError(f"theta0, lb and ub need {len(names)} values (parameters: {names})")

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    for t0 in t0_range:
        t_start = time.perf_counter()

        # Observations of the window that ends at week t0
        y = load_week(dates, obs, week_id=t0, dt_train=dt_train)
        y = np.asarray(y, dtype=float)

        label = week_label(dates[t0])

        window = np.arange(t0 - dt_train + 1, t0 + 1)
        window = window[window >= 0]

        # Dummy x-data required by pymcmcstat
        x = window.reshape(len(window), 1)

        # Initialize MCMC
        mcstat = MCMC(rngseed=seed)
        mcstat.data.add_data_set(x, y)

        mcstat.simulation_options.define_simulation_options(
            nsimu=int(nsim),
            updatesigma=True
        )

        # Disable pymcmcstat internal output (progress bars, tables)
        mcstat.simulation_options.waitbar = False
        mcstat.simulation_options.verbosity = 0

        for i, name in enumerate(names):
            th0, hi = theta0[i], ub[i]
            if name == "t_0":
                if t0 in t0_range[:n_early]:
                    th0, hi = early_t0
                mcstat.parameters.add_model_parameter(name=name, theta0=th0, minimum=lb[i], maximum=hi)
            elif priors == "normal":
                mcstat.parameters.add_model_parameter(
                    name=name, theta0=th0, minimum=lb[i], maximum=hi,
                    prior_mu=theta0[i], prior_sigma=(ub[i] - lb[i]) / 3
                )
            else:
                mcstat.parameters.add_model_parameter(name=name, theta0=th0, minimum=lb[i], maximum=hi)

        # Function to minimise (negative log-likelihood)
        ssfun_local = make_ssfun_pinn(times, dt, N_age, y, pinn_model, loss, weight, delta_fixed, observable)

        mcstat.model_settings.define_model_settings(sos_function=ssfun_local)

        # Run MCMC
        mcstat.run_simulation()

        t_elapsed = time.perf_counter() - t_start

        results = mcstat.simulation_results.results

        # Save posterior samples
        with open(f"{output_dir}/obs={label}.pickle", "wb") as f:
            pickle.dump({"results": results, "runtime_seconds": t_elapsed, "seed": seed, "week": label, "y_obs": y}, f)

    return output_dir


# ---------------------------------------------------------------------------
# Step 2: posterior samples and derived quantities
# ---------------------------------------------------------------------------

def load_mcmc_chain(path, sample_strategy="tail", nsample=1000, nsim=10000):
    """
    Load the chain saved by `run_rolling_mcmc_inference` and select the posterior samples.

    Parameters
    ----------
    path : str or Path
        File obs=<week>.pickle.
    sample_strategy : str
        'tail'   : the last `nsample` iterations of the chain (after the burn-in);
        'random' : `nsample` iterations drawn at random from the whole chain (pymcmcstat `define_sample_points`,
                   random numbers of numpy);
        'None'   : the whole chain.
    nsample : int
        Number of samples.
    nsim : int
        Number of iterations of the chain (used by 'random').

    Returns
    -------
    chain : ndarray, shape (n_samples, n_parameters)
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    chain = np.asarray(d["results"]["chain"])

    if sample_strategy == "tail":
        return chain[-nsample:]
    if sample_strategy == "random":
        idx_sample, _ = define_sample_points(nsample=nsample, nsimu=nsim)
        return chain[idx_sample]
    if sample_strategy in ("None", None):
        return chain
    raise ValueError(f"sample_strategy must be 'tail', 'random' or 'None', got '{sample_strategy}'")


def save_mcmc_chain_tails(
    path,
    N_age,
    dt_train,
    dates,
    t0_range,
    pinn_model,
    times=DEFAULT_TIMES,
    delta_fixed=None,
    dt=7,
    sample_strategy="tail",
    nsample=1000,
    nsim=10000,
):
    """
    Collect the posterior samples of all the weeks and compute the derived quantities.

    For each week t0 it loads the saved chain, takes the samples of the parameters and, with the
    PINN, computes for every sample
    - the effective reproduction number Rt = R0 * S(t) / N_tot at the current week;
    - the number of infected people I at the first week of the observations.
    Everything is saved in a single file `mcmc_distributions.pkl`, used by the forecasts.

    Parameters
    ----------
    path : str or Path
        Folder with the chains obs=<week>.pickle; the output is saved here too.
    N_age : array-like
        Population of the age classes.
    dt_train : int
        Length of the window of observations (weeks).
    dates : array-like
        Weekly dates (or labels) of the observations.
    t0_range : iterable of int
        Weeks at which the estimation was run.
    pinn_model : torch.nn.Module
        Pretrained PINN.
    times : array-like
        Continuous time grid on which the PINN is evaluated (days).
    delta_fixed : float or None
        As in `run_rolling_mcmc_inference`.
    dt : int
        Time step of the incidence (days).
    sample_strategy, nsample, nsim :
        See `load_mcmc_chain`.

    Returns
    -------
    output_path : str
        File mcmc_distributions.pkl, a dict with:
        - 'weeks'          : array of t0 indices
        - 'R0s_samples'    : list of R0 posterior arrays
        - 'Deltas_samples' : list of Delta posterior arrays (only if Delta is estimated)
        - 'delta_fixed'    : the fixed value of Delta (only if Delta is fixed)
        - 'tau0s_samples'  : list of tau0 posterior arrays
        - 'Rts_samples'    : list of Rt = R0 * S(t) / N_tot posterior arrays
        - 'Is_samples'     : list of arrays with the infected people at the first week of the window
        Samples whose onset is beyond the time grid are discarded from Rts and Is.
    """

    n_skipped = 0

    R0s_samples = []
    Deltas_samples = []
    tau0s_samples = []
    Rts_samples = []
    Is_samples = []
    weeks = []

    t_inc = times[0:365:dt].astype("i")
    k_age = len(N_age)
    N_tot = N_age.sum()

    for t0_train in t0_range:
        n_skipped_week = 0

        # Load MCMC posterior samples
        chain_path = f"{path}/obs={week_label(dates[t0_train])}.pickle"
        chain = load_mcmc_chain(chain_path, sample_strategy=sample_strategy, nsample=nsample, nsim=nsim)

        R0s = chain[:, 0]
        if delta_fixed is None:
            Deltas = chain[:, 1]
            tau0s = chain[:, 2]
        else:
            Deltas = np.full(len(R0s), float(delta_fixed))
            tau0s = chain[:, 1]

        # Store ALL samples for each parameter
        R0s_samples.append(R0s)
        if delta_fixed is None:
            Deltas_samples.append(Deltas)
        tau0s_samples.append(tau0s)
        weeks.append(t0_train)

        Rt_week_samples = []
        I_week_samples = []

        # Window of observations
        window_train = np.arange(t0_train - dt_train + 1, t0_train + 1)
        window_train = window_train[window_train >= 0]

        for R0_j, Delta_j, tau0_j in zip(R0s, Deltas, tau0s):

            # Cut time axis based on the epidemic onset
            t0_j_inc, _ = cut_times(t_inc, len(window_train), tau0_j, forecast=True)

            if len(t0_j_inc) == 0:
                n_skipped_week += 1
                n_skipped += 1
                continue

            t0_j_current = t0_j_inc[-1]  # t0_j_inc has one point per week of the window, up to the current week
            t0_j_week0 = int(t0_j_inc[0]) - int(window_train[0]) * dt  # day of the first week of the window
            t0_j_week0 = max(0, t0_j_week0)

            model_eval = evaluation_pinn([R0_j, Delta_j], pinn_model, np.array([t0_j_week0, t0_j_current]))

            S_at_t0 = (model_eval[1, :k_age] * N_age).sum() / N_tot      # S at t0_j_current
            I_at_0 = (model_eval[0, 2 * k_age:3 * k_age] * N_age).sum()  # I at t0_j_week0

            Rt_week_samples.append(R0_j * S_at_t0)
            I_week_samples.append(I_at_0)

        if n_skipped_week > 0:
            print(f"week {t0_train}: discarded {n_skipped_week}/{len(R0s)} samples (onset beyond the time grid)")

        Is_samples.append(np.array(I_week_samples))
        Rts_samples.append(np.array(Rt_week_samples))

    data_dict = {
        "weeks": np.array(weeks),
        "R0s_samples": R0s_samples,
        "tau0s_samples": tau0s_samples,
        "Rts_samples": Rts_samples,
        "Is_samples": Is_samples,
    }
    if delta_fixed is None:
        data_dict["Deltas_samples"] = Deltas_samples
    else:
        data_dict["delta_fixed"] = delta_fixed  # a single value, not a list of samples

    output_path = os.path.join(path, "mcmc_distributions.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"File saved in: {output_path}")
    if n_skipped:
        print(f"Total samples discarded over the whole estimation: {n_skipped}")
    return output_path


def export_last_week_samples(path, filename=None):
    """
    Save as CSV the posterior samples of the last week (columns R0, Delta, tau0, Rt).

    Parameters
    ----------
    path : str or Path
        Folder with mcmc_distributions.pkl (see `save_mcmc_chain_tails`).
    filename : str or Path, optional
        Output file. Default: `posterior_samples_<last week>.csv` in `path`.
    """
    with open(f"{path}/mcmc_distributions.pkl", "rb") as f:
        data = pickle.load(f)

    week_last = data["weeks"][-1]
    R0_last = data["R0s_samples"][-1]
    Delta_last = data["Deltas_samples"][-1] if "Deltas_samples" in data else np.full(len(R0_last), data["delta_fixed"])

    df_params = pd.DataFrame({
        "R0": R0_last,
        "Delta": Delta_last,
        "tau0": data["tau0s_samples"][-1],
        "Rt": data["Rts_samples"][-1],
    })

    filename = f"{path}/posterior_samples_{week_last}.csv" if filename is None else filename
    df_params.to_csv(filename, index=False)
    print(f"Saved: {filename} - shape: {df_params.shape}")
    return df_params
