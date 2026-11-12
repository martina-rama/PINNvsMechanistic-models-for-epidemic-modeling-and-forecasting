"""
Choice of the best PINN run (seed) among the independent runs of notebooks 03 and 04.

Every seed has its own chains and forecasts (`posteriors/pinn/<case>/seed_<seed>/forecast_h<H>/forecast_results.pickle`).
The seeds are ranked by the score of the FIT: for every rolling window t0 the forecast file contains the weeks of the
window of observations (t <= t0), which are compared with the observations; no future week is used. The score of a seed
is the mean over the windows of the score of the fit of the window.

Criteria: "WIS" (weighted interval score of the quantiles) or "wMAPE" (of the median). The score of a window uses the
harmonic weights on its weeks (more weight to the most recent ones) when the likelihood of the case uses them
(`case.weight`), as in the inference: weighted for the synthetic scenarios and for the real case with the look-back on
all the history, not weighted for the real case with look-back 5 weeks. The best seed is the one with the lowest score.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pinnvsmech.evaluation.forecast_validation import wis_error, wmape_error

CRITERIA = ("WIS", "wMAPE")
BEST_SEED_FILE = "best_seed.json"
BEST_DIR = "best_WIS"   # folder of the best run inside `posteriors/pinn/<case>/` (same files as `seed_<seed>/`)


def list_seeds(case_dir):
    """Seeds with a run folder `seed_<seed>` in `case_dir` (sorted)."""
    case_dir = Path(case_dir)
    seeds = []
    for p in case_dir.glob("seed_*"):
        m = re.fullmatch(r"seed_(\d+)", p.name)
        if m and p.is_dir():
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def fit_score_window(entry, obs, criterion="WIS", weighted=False):
    """
    Score of the fit of one window: the weeks t <= t0 of the forecast compared with the observations.

    Parameters
    ----------
    entry : dict
        Entry of `forecast_results` (needs `t0` and `quantiles`, which must include the fit: notebook 04 with `fit=True`).
    obs : array-like
        Observed series of the whole period, indexed by the absolute week.
    """
    q = entry["quantiles"]
    t0 = int(entry["t0"])
    rows = q[(q["t"] <= t0) & (q["t"] < len(obs))].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"The forecast of the window t0={t0} does not include the fit (weeks t <= t0)")
    y = np.asarray(obs, dtype=float)[rows["t"].to_numpy(dtype=int)]
    if criterion == "WIS":
        return float(wis_error(y, rows, weighted=weighted))
    if criterion == "wMAPE":
        return float(wmape_error(y, rows["q50"].to_numpy(), weighted=weighted))
    raise ValueError(f"criterion must be one of {CRITERIA}")


def fit_score_seed(forecasts, obs, windows, criterion="WIS", weighted=False):
    """
    Score of the fit of one seed: mean over the windows `windows` (values of t0) of `fit_score_window`.

    Returns
    -------
    (mean score, DataFrame with the score of every window: t0, score)
    """
    scores = []
    for entry in forecasts.values():
        t0 = int(entry["t0"])
        if t0 in windows:
            scores.append({"t0": t0, "score": fit_score_window(entry, obs, criterion, weighted)})
    if not scores:
        raise ValueError("None of the windows requested is in the forecasts")
    per_window = pd.DataFrame(scores).sort_values("t0").reset_index(drop=True)
    return float(per_window["score"].mean()), per_window


def rank_seeds(case, case_dir, horizon=4, criterion="WIS", weighted=None, seeds=None, verbose=False):
    """
    Score of the fit of every seed of `case_dir` and ranking.

    Parameters
    ----------
    case : Case
        Case of the runs (`obs`, `t0_range`, `weight`).
    case_dir : path
        `.../posteriors/pinn/<case.label>`, with the folders `seed_<seed>`.
    horizon : int
        Horizon of the forecast files to read (the fit is the same for every horizon).
    weighted : bool or None
        Harmonic weights on the weeks of every window. None: those of the likelihood of the case (`case.weight`).
    seeds : list of int, optional
        Seeds to score (default: all those in `case_dir` with the forecast file).

    Returns
    -------
    DataFrame with the columns seed, score, n_windows, sorted by score (best first).
    """
    case_dir = Path(case_dir)
    weighted = bool(case.weight) if weighted is None else bool(weighted)
    windows = set(int(t) for t in case.t0_range)
    seeds = list_seeds(case_dir) if seeds is None else list(seeds)

    records = []
    for seed in seeds:
        path = case_dir / f"seed_{seed:03d}" / f"forecast_h{horizon}" / "forecast_results.pickle"
        if not path.exists():
            if verbose:
                print(f"seed {seed}: no forecast file, skipped")
            continue
        with open(path, "rb") as f:
            forecasts = pickle.load(f)
        score, per_window = fit_score_seed(forecasts, case.obs, windows, criterion, weighted)
        records.append({"seed": seed, "score": score, "n_windows": len(per_window)})

    if not records:
        raise FileNotFoundError(f"No forecast files of horizon {horizon} in {case_dir}: run notebook 04 first")
    ranking = pd.DataFrame(records).sort_values("score").reset_index(drop=True)
    return ranking.rename(columns={"score": f"fit_{criterion}" + ("_weighted" if weighted else "")})


def save_best_seed(case_dir, ranking, criterion, weighted):
    """Write `best_seed.json` in `case_dir` with the best seed of `ranking` (first row)."""
    case_dir = Path(case_dir)
    best = ranking.iloc[0]
    info = {"seed": int(best["seed"]), "criterion": criterion, "weighted": bool(weighted),
            "score": float(best.iloc[1]), "n_seeds": int(len(ranking))}
    with open(case_dir / BEST_SEED_FILE, "w") as f:
        json.dump(info, f, indent=2)
    return info


def resolve_seed(seed, case_dir):
    """`seed` if it is an integer; if it is "best", the seed written in `case_dir/best_seed.json` (notebook 10)."""
    if isinstance(seed, str) and seed == "best":
        path = Path(case_dir) / BEST_SEED_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run notebook 10 to choose the best seed, or set the seed by hand")
        with open(path) as f:
            return int(json.load(f)["seed"])
    return int(seed)


def resolve_run_dir(run, case_dir):
    """
    Folder of the PINN run to use: `case_dir` is `.../posteriors/pinn/<case.label>`.

    - an integer `run`: the folder of that seed, `case_dir/seed_<run>` (3 digits);
    - "best": the folder `case_dir/best_WIS` if it exists (best run shipped in `precomputed/`), otherwise the seed
      written in `case_dir/best_seed.json` by notebook 10.
    """
    case_dir = Path(case_dir)
    if isinstance(run, str) and run == "best":
        best_dir = case_dir / BEST_DIR
        if best_dir.is_dir():
            return best_dir
        return case_dir / f"seed_{resolve_seed('best', case_dir):03d}"
    path = case_dir / f"seed_{int(run):03d}"
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found")
    return path
