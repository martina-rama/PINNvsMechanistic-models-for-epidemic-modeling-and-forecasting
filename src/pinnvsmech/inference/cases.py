"""
Observations and settings for the parameter estimation.

One function, `load_case`, returns everything the rolling-window estimation needs for
one of the five cases of the paper: the four synthetic scenarios (R0 = 1.5 / 2.0, noise
"light" / "mild") and the real COVID-19 deaths in London.

The two kinds of case differ in what is observed and in how the model is fitted:

- synthetic: the observable is the weekly number of new infections (decrease of S), the
  delay Delta is fixed (the data were generated with it, so only R0 and the onset tau0 are
  estimated), the whole history up to the current week is used.
- real: the observable is the weekly number of observed deaths (Dobs), R0, Delta and tau0 are
  estimated, a rolling window of 5 weeks is used.

The look-back (`dt_train`, length of the window of observations in weeks) can be changed. The harmonic
weights on the likelihood are used by default when the window covers all the history (look-back = number
of weeks: all the synthetic scenarios, or the real case with `dt_train=None`) and not otherwise.
"""

import os
import pickle
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from pinnvsmech.paths import EPI_DIR, SYNTHETIC_DIR

# Real data: weekly COVID-19 deaths in London (columns: date, deaths)
DEATHS_CSV = EPI_DIR / "london" / "deaths.csv"
FIRST_WAVE = ("2020-02-16", "2020-08-02")

# dt_train=None: the window covers all the history. weight=None: True if the window covers all the history.
PRESETS = {
    "real": dict(observable="deaths", delta_fixed=None, dt_train=5, weight=None,
                 t0_start=3, t0_end_offset=2),
    # t0_end_offset=None: the last week of the rolling windows depends on the scenario (SYNTHETIC_T0_STOP)
    "synthetic": dict(observable="infections", delta_fixed=21, dt_train=None, weight=None,
                      t0_start=3, t0_end_offset=None),
}

# Rolling windows of the synthetic scenarios: t0 from 3 up to SYNTHETIC_T0_STOP - 1, i.e. weeks 4-30 (R0 = 1.5) and
# 4-20 (R0 = 2.0). They are used for the estimation, the forecasts and the plots of the parameters.
SYNTHETIC_T0_STOP = {1.5: 30, 2.0: 20}

# Extra window of the synthetic scenarios, estimated too, used for the densities of R0 at the last week:
# t0 = 33 (week 34) for R0 = 1.5 and t0 = 23 (week 24) for R0 = 2.0.
SYNTHETIC_T0_LAST = {1.5: 33, 2.0: 23}


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def load_covid_deaths(path=DEATHS_CSV, period=FIRST_WAVE, verbose=True):
    """
    Weekly COVID-19 deaths of the first wave.

    Parameters
    ----------
    path : str or Path
        CSV with the columns `date` (first column) and `deaths`. A leading index column is also accepted.
    period : (str, str)
        First and last week kept.

    Returns
    -------
    dates  : np.ndarray of datetime64, weekly dates
    deaths : np.ndarray of int, new deaths observed every week
    """
    df = pd.read_csv(path, index_col=0).reset_index()
    df["date"] = pd.to_datetime(df["date"])

    df = df[(df["date"] >= period[0]) & (df["date"] <= period[1])].reset_index(drop=True)

    dates = df["date"].values
    deaths = df["deaths"].values.astype(int)

    if verbose:
        print(f"Loaded {len(deaths)} weekly observations from {df['date'].iloc[0].date()} "
              f"to {df['date'].iloc[-1].date()}")
    return dates, deaths


def load_synthetic_scenario(R0_val, noise_label, data_dir=SYNTHETIC_DIR):
    """
    Load one synthetic scenario (a file synthetic_R0_<R0>_noise_<noise>.pkl).

    Returns
    -------
    dates : np.ndarray of str, ["week 1", "week 2", ...]
    cases : np.ndarray, weekly incidence of the whole population, with noise
    """
    path = os.path.join(data_dir, f"synthetic_R0_{float(R0_val)}_noise_{noise_label}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)

    cases = np.asarray(data["noisy incidence"])
    dates = np.array([f"week {i + 1}" for i in range(len(data["t"]))])
    return dates, cases


def load_week(dates, values, week_id, dt_train):
    """
    Observations of the window that ends at week `week_id`.

    Parameters
    ----------
    dates    : np.ndarray, weekly dates
    values   : np.ndarray, weekly observations
    week_id  : int or date, index (or date) of the current week
    dt_train : int, length of the window in weeks

    Returns
    -------
    np.ndarray, the last `dt_train` observations up to `week_id` included
    (fewer at the beginning of the series).
    """
    if isinstance(week_id, (int, np.integer)):
        idx = week_id
    else:
        # closest date
        idx = np.argmin(np.abs(dates - np.datetime64(week_id)))

    start = max(0, idx - dt_train + 1)
    return values[start:idx + 1]


def week_label(date):
    """Label used in the file names: 2020-03-15 for real data, week_5 for synthetic ones."""
    if isinstance(date, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(date).strftime("%Y-%m-%d")
    return str(date).replace(" ", "_")


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """Observations and settings of one case. `settings()` gives the part that is saved with the results."""
    kind: str                # "real" or "synthetic"
    label: str               # name of the folder of the results
    dates: Any
    obs: Any                 # weekly observations
    observable: str          # "deaths" or "infections"
    delta_fixed: Any         # None: Delta is estimated; number: Delta is fixed to this value
    dt_train: int            # length of the window of observations (weeks)
    weight: bool             # harmonic weights on the Poisson log-likelihood
    t0_range: range          # indices of the current weeks of the rolling windows (estimation, forecasts, plots)
    t0_last: Any = None      # extra window for the density of R0 at the last week (synthetic scenarios only)

    def t0_estimation(self):
        """Weeks at which the parameters are estimated: the rolling windows and, if there is, the extra last window."""
        weeks = list(self.t0_range)
        if self.t0_last is not None and self.t0_last not in weeks:
            weeks.append(self.t0_last)
        return weeks

    def settings(self):
        d = asdict(self)
        d.pop("dates"), d.pop("obs")
        d["t0_range"] = [self.t0_range.start, self.t0_range.stop]
        d["t0_estimation"] = self.t0_estimation()
        d["n_weeks"] = len(self.obs)
        return d


def load_case(kind, R0=None, noise=None, deaths_path=DEATHS_CSV, synthetic_dir=SYNTHETIC_DIR, **overrides):
    """
    Load the observations and the settings of a case.

    Parameters
    ----------
    kind : "real" or "synthetic"
    R0, noise : synthetic scenario (R0 = 1.5 or 2.0; noise = "light" or "mild")
    overrides : replace entries of PRESETS[kind], e.g. dt_train=8, weight=True, delta_fixed=None.
                dt_train=None: look-back on all the history (then the weights are on by default).
                t0_start / t0_end_offset give t0_range = range(t0_start, n_weeks - t0_end_offset). For the synthetic
                scenarios, if t0_end_offset is not given, the last window depends on R0 (SYNTHETIC_T0_STOP).
    """
    if kind not in PRESETS:
        raise ValueError(f"kind must be 'real' or 'synthetic', got '{kind}'")
    unknown = set(overrides) - set(PRESETS[kind])
    if unknown:
        raise ValueError(f"Unknown settings {sorted(unknown)}; valid ones: {sorted(PRESETS[kind])}")
    s = {**PRESETS[kind], **overrides}

    if kind == "real":
        dates, obs = load_covid_deaths(deaths_path)
        label = "real_London"
    else:
        if R0 is None or noise is None:
            raise ValueError("For a synthetic case R0 and noise are required")
        dates, obs = load_synthetic_scenario(R0, noise, synthetic_dir)
        label = f"synthetic_R0_{float(R0)}_noise_{noise}"

    n_weeks = len(obs)
    if kind == "synthetic" and s["t0_end_offset"] is None:
        t0_stop, t0_last = SYNTHETIC_T0_STOP[float(R0)], SYNTHETIC_T0_LAST[float(R0)]
    else:
        t0_stop = n_weeks - (s["t0_end_offset"] or 0)
        t0_last = None
    dt_train = n_weeks if s["dt_train"] is None else s["dt_train"]
    full_history = dt_train >= n_weeks
    weight = full_history if s["weight"] is None else bool(s["weight"])

    # The folder name tells the look-back (always for the real case; for the synthetic ones only if it is not
    # the whole history) and the weights if they are not the default ones
    if kind == "real" or not full_history:
        label += f"_lookback{dt_train}"
    if weight != full_history:
        label += "_weighted" if weight else "_noweight"

    return Case(
        kind=kind, label=label, dates=dates, obs=obs,
        observable=s["observable"], delta_fixed=s["delta_fixed"],
        dt_train=dt_train, weight=weight,
        t0_range=range(s["t0_start"], t0_stop), t0_last=t0_last,
    )
