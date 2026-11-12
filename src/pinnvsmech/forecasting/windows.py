"""
Weeks and windows used to plot and score the forecasts of a case (notebooks 08 and 09).

The last week used (T_LAST) depends on the scenario: for the synthetic scenarios it is the week of the extra window
(t0 = 33 for R0 = 1.5, t0 = 23 for R0 = 2.0, `case.t0_last`) and only the windows whose forecast ends before it are
kept; for the real case the weeks go from 2020-03-15 to 2020-07-19 and the windows keep their shorter horizons near the
end.
"""

import numpy as np


def trim_forecasts(fc, t_last, horizon=4):
    """
    Keep only the windows with t0 + horizon <= t_last and cut every table to t <= t_last. It does not modify the
    original dict.
    """
    out = {}
    for k, e in fc.items():
        if int(e["t0"]) + horizon > t_last:
            continue
        e2 = dict(e)
        for name in ("quantiles", "raw", "summary_90", "summary_50"):
            if name in e2:
                e2[name] = e2[name][e2[name]["t"] <= t_last].reset_index(drop=True)
        out[k] = e2
    return out


def entry_at(forecasts, t0):
    """Entry of the forecasts dict whose window ends at week t0 (the key is not needed)."""
    for k, e in forecasts.items():
        if int(e["t0"]) == int(t0):
            return e
    raise KeyError(f"No window with t0 = {t0} (available: {sorted(int(e['t0']) for e in forecasts.values())})")


def evaluation_window(case):
    """
    Weeks of a case used for plots and scores.

    Returns
    -------
    t0_min : int or None
        First window (index). None: no limit.
    t_last : int
        Last week (index) of the observations and of the forecasts.
    trim : bool
        True (synthetic scenarios): the windows with t0 + horizon > t_last are removed (`trim_forecasts`), so that
        every window has the whole horizon. False (real case): all the windows with t0 >= t0_min are kept.
    """
    if case.kind == "synthetic":
        return None, int(case.t0_last), True
    d = np.asarray(case.dates, dtype="datetime64[ns]")
    return (int(np.where(d == np.datetime64("2020-03-15"))[0][0]),
            int(np.where(d == np.datetime64("2020-07-19"))[0][0]), False)


def select_windows(case, forecasts_by_model, horizon=4):
    """
    Apply the window of the case to the forecasts of every model ({name: forecasts dict}).

    Returns
    -------
    forecasts : dict, the same models, trimmed for the synthetic scenarios (unchanged for the real case)
    t0_min, t_last : see `evaluation_window`
    """
    t0_min, t_last, trim = evaluation_window(case)
    if trim:
        forecasts_by_model = {name: trim_forecasts(f, t_last, horizon) for name, f in forecasts_by_model.items()}
    return forecasts_by_model, t0_min, t_last
