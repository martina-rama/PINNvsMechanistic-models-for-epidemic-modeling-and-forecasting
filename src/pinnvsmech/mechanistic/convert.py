"""
Conversion of the results of the Mechanistic model (ABC-SMC, calibrated in its own repository) into the format
used for the PINN, so that the same code can load, plot and score both models:

    abc_distributions.pkl                      like the PINN `mcmc_distributions.pkl` (parameter posteriors)
    forecast_h<horizon>/forecast_results.pickle   like the PINN forecasts (`pinnvsmech.forecasting.forecast`)

Folder of the raw results of the Mechanistic model, for one case:

    <raw_dir>/
      nall/                 one folder per look-back: "nall" (all the history), "n4", "n5", "n6" (weeks)
        4/ (or w4/, w_4/)   one folder per week t0 (index of the current week)
          posterior.csv     one row per particle, columns: R0, inf_t0 and, only for the real case, Delta and delta_t
          Rt.npz            Rt (n_particles, n_weeks): Rt = R0 * S(t) / N
          trajectories.npz  trajectories (n_particles, n_weeks): simulated weekly incidence of the whole period
          weights.npz       weights (n_particles,): importance weights of the ABC-SMC (optional)
        retro_w34/          synthetic scenarios, optional: extra fit at the last week (the number is the week, t0 = 34 - 1):
                            posterior.csv (density of R0), trajectories.npz (aggregate weekly incidence, retrospective
                            fit) and infections_by_age.npz (key `weekly_age`, (n_particles, 16, n_weeks): incidence of
                            every age class, sanity check by age)

Real and synthetic cases differ in the parameters: in the real case Delta and delta_t are estimated, in the
synthetic ones Delta is fixed (21) and there is neither Delta nor delta_t in posterior.csv.

The parameter `delta_t` is stored under the key `tau0s_samples` only so that the field exists where the code of the
PINN expects it. It is NOT the same quantity as the onset tau0 of the PINN: the Mechanistic model estimates the
initial number of infected (`inf_t0`), while the PINN fixes it to one and lets tau0 absorb the timing. R0 and Delta
are directly comparable.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd

from pinnvsmech.constants import quantiles as QUANTILES, rename_map as RENAME_MAP
from pinnvsmech.forecasting.summaries import compute_quantiles, summarize_forecast
from pinnvsmech.inference.cases import week_label


def variant_dt_train(variant, n_weeks):
    """Look-back in weeks of a variant folder: "nall" -> n_weeks (all the history), "n5" -> 5."""
    m = re.fullmatch(r"n?(all|\d+)", variant)
    if m is None:
        raise ValueError(f"Variant '{variant}' not recognised: expected 'nall' or 'n<weeks>', e.g. 'n5'")
    return n_weeks if m.group(1) == "all" else int(m.group(1))


def find_variants(raw_dir):
    """Names of the variant folders ('nall', 'n4', ...) in raw_dir."""
    return sorted(d for d in os.listdir(raw_dir)
                  if os.path.isdir(os.path.join(raw_dir, d)) and re.fullmatch(r"n?(all|\d+)", d))


def find_weeks(variant_dir):
    """Week folders of a variant ('4', 'w4' or 'w_4'): list of (folder name, week index), sorted by week."""
    entries = []
    for d in os.listdir(variant_dir):
        if os.path.isdir(os.path.join(variant_dir, d)):
            m = re.fullmatch(r"[wW]?_?(\d+)", d)
            if m:
                entries.append((d, int(m.group(1))))
    return sorted(entries, key=lambda e: e[1])


def find_retro_weeks(variant_dir):
    """Folders of the extra fits at the last week ('retro_w34'): list of (folder name, index t0 = week - 1)."""
    entries = []
    for d in os.listdir(variant_dir):
        m = re.fullmatch(r"retro_[wW]_?(\d+)", d)
        if m and os.path.isdir(os.path.join(variant_dir, d)):
            entries.append((d, int(m.group(1)) - 1))
    return sorted(entries, key=lambda e: e[1])


def convert_variant(
    raw_dir,
    variant,
    dates,
    obs,
    horizons=(4,),
    delta_fixed=None,
    resample=False,
    n_resample=1000,
    seed=0,
    quantiles=None,
    rename_map=None,
    verbose=True,
):
    """
    Convert one look-back variant of the Mechanistic results.

    Parameters
    ----------
    raw_dir : str or Path
        Folder with the variants of one case (see the module docstring).
    variant : str
        "nall", "n4", "n5", ...
    dates, obs : arrays
        Weekly dates (or labels) and observations of the case (`pinnvsmech.inference.cases.load_case`). They give the
        names of the weeks and the observations of the window of every forecast.
    horizons : iterable of int
        Forecast horizons in weeks. The trajectories cover the whole period, so every horizon is available.
    delta_fixed : float or None
        Delta of the synthetic cases (fixed). Used only if posterior.csv has no Delta column.
    resample : bool
        True: bootstrap resample of n_resample particles with the ABC-SMC weights (weights.npz is then required);
        False: use the particles as they are.
    seed : int
        Seed of the resampling.

    Returns
    -------
    distributions : dict
        `weeks`, `R0s_samples`, `inf_t0s_samples`, `Rts_samples` (Rt of the current week) and, if present,
        `Deltas_samples` (or `delta_fixed`) and `tau0s_samples`; one array of samples per week. If there are
        extra fits at the last week (`retro_w<week>` folders), `last_week`: {t0: {"R0": samples, "inf_t0": samples}} and,
        if the files are there, "trajectories" (n_particles, n_weeks) and "weekly_age" (n_particles, 16, n_weeks).
    forecasts : dict
        Indexed by horizon; each one is a dict indexed by the week label (like the forecasts of the PINN):
        `raw`, `summary_90`, `summary_50`, `quantiles`, `t0`, `date`, `y_obs`.
    """
    quantiles = QUANTILES if quantiles is None else quantiles
    rename_map = RENAME_MAP if rename_map is None else rename_map

    variant_dir = os.path.join(raw_dir, variant)
    if not os.path.isdir(variant_dir):
        raise FileNotFoundError(f"No folder for variant '{variant}' in {raw_dir}")
    weeks_found = find_weeks(variant_dir)
    if not weeks_found:
        raise FileNotFoundError(f"No week folders in {variant_dir}")

    n_weeks = len(obs)
    dt_train = variant_dt_train(variant, n_weeks)
    if verbose:
        print(f"variant {variant}: {len(weeks_found)} windows, weeks {weeks_found[0][1]}..{weeks_found[-1][1]}, "
              f"look-back {dt_train}")

    weeks = []
    R0s, Deltas, tau0s, inf_t0s, Rts = [], [], [], [], []
    forecasts = {h: {} for h in horizons}

    for name, week in weeks_found:
        wdir = os.path.join(raw_dir, variant, name)
        post = pd.read_csv(os.path.join(wdir, "posterior.csv"))
        rt_all = np.load(os.path.join(wdir, "Rt.npz"))["Rt"]
        traj_all = np.load(os.path.join(wdir, "trajectories.npz"))["trajectories"]

        n_particles = len(post)
        if rt_all.shape[0] != n_particles or traj_all.shape[0] != n_particles:
            raise ValueError(f"week {week}: posterior.csv has {n_particles} rows, Rt.npz {rt_all.shape[0]}, "
                             f"trajectories.npz {traj_all.shape[0]}")

        if resample:
            weights = np.load(os.path.join(wdir, "weights.npz"))["weights"]
            if len(weights) != n_particles:
                raise ValueError(f"week {week}: weights.npz has {len(weights)} entries, posterior.csv {n_particles} rows")
            sel = np.random.default_rng(seed + week).choice(
                n_particles, size=n_resample, replace=True, p=weights / weights.sum())
        else:
            sel = np.arange(n_particles)

        n_domain = traj_all.shape[1]
        weeks.append(week)
        R0s.append(post["R0"].to_numpy()[sel])
        inf_t0s.append(post["inf_t0"].to_numpy()[sel])
        if "Delta" in post.columns:
            Deltas.append(post["Delta"].to_numpy()[sel])
        if "delta_t" in post.columns:
            tau0s.append(post["delta_t"].to_numpy()[sel])
        # Rt of the current week of the window
        Rts.append(rt_all[sel][:, min(week, n_domain - 1)])

        traj = traj_all[sel]
        label = week_label(dates[week])

        # Weeks of the window of observations (as for the PINN, `fit=True`): from the first week of the look-back
        start = max(0, week - dt_train + 1)
        for h in horizons:
            t_range = np.arange(start, min(week + h, n_domain - 1) + 1)
            sub = traj[:, t_range]
            df = pd.DataFrame({"t": np.tile(t_range, sub.shape[0]), "v": sub.reshape(-1)})
            forecasts[h][label] = {
                "raw": df,
                "summary_90": summarize_forecast(df, error_width=90),
                "summary_50": summarize_forecast(df, error_width=50),
                "quantiles": compute_quantiles(df, quantiles, rename_map),
                "t0": week,
                "date": label,
                "y_obs": obs[start:week + 1],
            }

    distributions = {
        "weeks": np.array(weeks),
        "R0s_samples": R0s,
        "inf_t0s_samples": inf_t0s,
        "Rts_samples": Rts,
    }
    if Deltas:
        distributions["Deltas_samples"] = Deltas
    else:
        if delta_fixed is None:
            raise ValueError("posterior.csv has no Delta column: delta_fixed is required")
        distributions["delta_fixed"] = delta_fixed
    if tau0s:
        distributions["tau0s_samples"] = tau0s

    retro = {}
    for name, t0 in find_retro_weeks(variant_dir):
        post = pd.read_csv(os.path.join(variant_dir, name, "posterior.csv"))
        retro[t0] = {"R0": post["R0"].to_numpy(), "inf_t0": post["inf_t0"].to_numpy()}
        traj_path = os.path.join(variant_dir, name, "trajectories.npz")
        if os.path.exists(traj_path):
            retro[t0]["trajectories"] = np.load(traj_path)["trajectories"]
        age_path = os.path.join(variant_dir, name, "infections_by_age.npz")
        if os.path.exists(age_path):
            retro[t0]["weekly_age"] = np.load(age_path)["weekly_age"]
    if retro:
        distributions["last_week"] = retro
        if verbose:
            print(f"  extra fits at the last week: t0 = {sorted(retro)}")

    return distributions, forecasts


def save_converted(output_dir, distributions, forecasts):
    """Write `abc_distributions.pkl` and `forecast_h<horizon>/forecast_results.pickle` in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "abc_distributions.pkl"), "wb") as f:
        pickle.dump(distributions, f)
    for h, fc in forecasts.items():
        os.makedirs(os.path.join(output_dir, f"forecast_h{h}"), exist_ok=True)
        with open(os.path.join(output_dir, f"forecast_h{h}", "forecast_results.pickle"), "wb") as f:
            pickle.dump(fc, f)
