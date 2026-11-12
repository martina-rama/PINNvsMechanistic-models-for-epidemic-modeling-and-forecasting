"""
Synthetic data: deterministic SEIRDDobs trajectories for a few (R0, delta) scenarios,
and the weekly incidence (true and with Negative Binomial noise) derived from them.

Pipeline
--------
1. generate_deterministic_data : ODE trajectories of all compartments, per age class
2. compute_incidences          : weekly incidence (aggregated and per age class) + noisy versions
3. save_* / load_*             : read and write files in data/synthetic/
4. plot_*                      : plots of the generated data
"""

import glob
import os
import os.path as osp
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pinnvsmech.ode.Data_SEIRDDobs_age_structured import Data_SEIRDDobs_age
from pinnvsmech.paths import SYNTHETIC_DIR
from pinnvsmech.pinn.approximation import D_incidence, load_train_data

# File names (they encode: 2 R0 scenarios, weekly time grid)
DETERMINISTIC_CSV = "deterministic_syntheticdata_normalized_2R0s_Tweekly.csv"
PARAMETER_PAIRS_PICKLE = "SEIRDDobs_syntheticdata_parameterspairs_2R0s_Tweekly.pickle"


# ---------------------------------------------------------------------------
# Initial conditions and noise
# ---------------------------------------------------------------------------

def initial_conditions(N_age, mode="standard", age_idx=None, n_infected=1,
                       inf_t0=None, xi=None, eps=None):
    """Initial conditions (E0, I0): integer arrays of length n_age, counts per age class.

    Modes
    -----
    "standard"     : 1 infected (I) in every age class, the rest susceptible.
    "age_specific" : `n_infected` infected (I) only in the classes `age_idx`
                     (int or list; n_infected scalar or list of the same
                     length), the rest susceptible.
    "total"        : `inf_t0` total initial infections, split among the age
                     classes in proportion to N_age (with int()) and then
                     split into I and E with
                         inf_t0_age = int(inf_t0 * N_age[age] / N_age.sum())
                         I = int(inf_t0_age * (1/xi) / ((1/xi) + (1/eps)))
                         E = inf_t0_age - I
                     (requires xi = recovery rate and eps = incubation rate,
                     i.e. 1/xi and 1/eps are the mean infectious and latent periods).

    The counts are always integers.
    """
    N_age = np.asarray(N_age)
    n_age = len(N_age)

    E0 = np.zeros(n_age, dtype=int)
    I0 = np.zeros(n_age, dtype=int)

    if mode == "standard":
        I0[:] = 1

    elif mode == "age_specific":
        if age_idx is None:
            raise ValueError("mode='age_specific' requires age_idx")
        idx = np.atleast_1d(age_idx)
        if idx.min() < 0 or idx.max() >= n_age:
            raise ValueError(f"age_idx out of range (0..{n_age - 1})")
        I0[idx] = np.broadcast_to(np.asarray(n_infected), idx.shape)

    elif mode == "total":
        if inf_t0 is None or xi is None or eps is None:
            raise ValueError("mode='total' requires inf_t0, xi and eps")
        for age in range(n_age):
            inf_t0_age = int(inf_t0 * N_age[age] / np.sum(N_age))
            # I
            I0[age] = int(int(inf_t0_age) * (1 / xi) / ((1 / xi) + (1 / eps)))
            # E
            E0[age] = inf_t0_age - I0[age]

    else:
        raise ValueError("mode must be 'standard', 'age_specific' or 'total'")

    return E0, I0


def negbin_noise(incidence, r, rng=None):
    """
    Apply Negative Binomial noise to a series of expected incidence (mean).

    Parametrization (the usual one in the literature, e.g. Flaxman et al. 2020
    for deaths, EpiNow2 for cases):
        Var[y] = incidence + incidence**2 / r

    incidence : array-like, shape (T,) or (T, k_age) — "true" incidence.
    r         : float or array-like broadcastable to incidence — dispersion
                parameter (also called psi/phi/size in the literature).
                Smaller => more overdispersion (stronger noise).
                For r -> infinity the Poisson limit is recovered (Var = mean).
    rng       : optional np.random.Generator (e.g. np.random.default_rng(42))
                for reproducibility; if None the global numpy state is used.

    Returns: array of non-negative integers, same shape as `incidence`.
    """
    incidence = np.asarray(incidence, dtype=float)
    if np.any(incidence < 0):
        raise ValueError("incidence must be >= 0 (it is an expected count)")

    r = np.broadcast_to(np.asarray(r, dtype=float), incidence.shape)

    p = r / (r + incidence)

    generator = rng if rng is not None else np.random
    return generator.negative_binomial(r, p)


# ---------------------------------------------------------------------------
# 1. Deterministic trajectories
# ---------------------------------------------------------------------------

def generate_deterministic_data(p_train, C, N_age, ifr, epsilon, xi,
                                inf_t0=1000, domain=(0, 245), step=7, ic_mode="total"):
    """
    Integrate the age-structured SEIRDDobs system for every parameter pair in p_train.
    Used both for the synthetic scenarios (ic_mode="total") and for the PINN training set
    (ic_mode="standard": one infected per age class).

    p_train : array (n_scenarios, 2), columns [R0, delta]
    C       : contact matrix (k_age x k_age)
    N_age   : population per age class
    inf_t0  : total initial infections (only for ic_mode="total", see initial_conditions)
    domain  : (t0, t_end) in days
    step    : time step in days (7 = weekly)
    ic_mode : mode of initial_conditions ("total" or "standard")

    Returns x_train (n_t*n_scenarios, 3) = [t, R0, delta_param],
            y_train (n_t*n_scenarios, 6*k_age) = [S, E, I, R, D, Dobs] per age class,
            idx_train (n_t*n_scenarios, 1) = scenario index.
    """
    p_train = np.asarray(p_train)
    N_age = np.asarray(N_age)
    zeros = np.zeros(len(N_age))

    E0_age, I0_age = initial_conditions(N_age=N_age, mode=ic_mode, inf_t0=inf_t0, xi=xi, eps=epsilon)
    data = Data_SEIRDDobs_age(N_age=N_age, E0_age=E0_age, I0_age=I0_age, R0_age=zeros,
                              D0_age=zeros, D0obs_age=zeros, C=C, domain=list(domain))

    t_train = data.t_SEIRDDobs[::step]
    x_train, y_train = data.train_data_constrains(t_train, p_train, epsilon, xi, ifr=ifr, noise=False)

    # Indices to recover each single parameter pair
    idx_train = np.tile(np.arange(len(p_train))[:, None], (1, len(t_train))).reshape(-1, 1)
    return x_train, y_train, idx_train


def save_deterministic_data(x_train, y_train, idx_train, p_train, folder=SYNTHETIC_DIR,
                            csv_name=DETERMINISTIC_CSV, pairs_name=PARAMETER_PAIRS_PICKLE):
    """Save the parameter pairs (pickle) and [x_train | y_train | idx_train] (csv, no header)."""
    os.makedirs(folder, exist_ok=True)
    with open(osp.join(folder, pairs_name), "wb") as b:
        pickle.dump(p_train, b)
    pd.DataFrame(np.c_[x_train, y_train, idx_train]).to_csv(osp.join(folder, csv_name), header=None, index=None)


def load_deterministic_data(k_age, folder=SYNTHETIC_DIR, csv_name=DETERMINISTIC_CSV):
    """Load x_train, y_train, idx_train from the csv saved by save_deterministic_data."""
    return load_train_data(osp.join(folder, csv_name), k_age=k_age)


# ---------------------------------------------------------------------------
# 2. Weekly incidence, true and noisy
# ---------------------------------------------------------------------------

def compute_incidences(x_train, y_train, idx_train, N_age, noise_levels=None, seed=42):
    """
    Weekly incidence (new exposed = decrease of S) of every scenario, aggregated and per age
    class, plus versions with Negative Binomial noise (the noise is applied to the aggregated
    incidence).

    noise_levels : dict {label: r}; default {"light": 100, "mild": 50}. Smaller r => more noise.
    seed         : seed of the generator (a single generator for all the scenarios, in the order
                   scenario -> noise levels, so that the same data are always obtained).

    Returns a dict indexed by R0:
        {R0: {"t": (T-1,), "incidence": (T-1,), "age incidence": (T-1, k_age),
              "noisy": {label: (T-1,)}}}
    """
    if noise_levels is None:
        noise_levels = {"light": 100, "mild": 50}
    N_age = np.asarray(N_age)
    k_age = len(N_age)
    idx_train = np.asarray(idx_train).ravel()
    rng = np.random.default_rng(seed)

    scenarios = {}
    for idx in np.unique(idx_train):
        idx_p = np.where(idx_train == idx)[0]
        x_sel = x_train[idx_p]
        y_true = y_train[idx_p]
        times = x_sel[1:, 0]

        S_true_all = y_true[:, :k_age]
        S_true_counts = S_true_all * N_age              # (T, k_age), counts per age class
        S_true_agg = S_true_counts.sum(axis=1)
        S_inc = D_incidence(-S_true_agg, dt=1)

        S_inc_age = np.column_stack([
            D_incidence(-S_true_counts[:, a], dt=1) for a in range(k_age)
        ])  # (T-1, k_age)

        R0_val = float(x_sel[0][1])
        if R0_val in scenarios:
            raise ValueError(f"Two scenarios with the same R0={R0_val}: the files are indexed by R0")
        scenarios[R0_val] = {
            "t": times,
            "incidence": S_inc,
            "age incidence": S_inc_age,
            "noisy": {label: negbin_noise(S_inc, r=r, rng=rng) for label, r in noise_levels.items()},
        }
    return scenarios


def save_scenarios(scenarios, out_dir=SYNTHETIC_DIR):
    """One file for each pair (R0, noise level): synthetic_R0_<R0>_noise_<label>.pkl"""
    os.makedirs(out_dir, exist_ok=True)
    for R0_val, sc in scenarios.items():
        for noise_label, inc_noisy in sc["noisy"].items():
            out_path = os.path.join(out_dir, f"synthetic_R0_{R0_val}_noise_{noise_label}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump({
                    "t": sc["t"],
                    "noisy incidence": inc_noisy,
                    "age incidence": sc["age incidence"],
                    "incidence": sc["incidence"],
                }, f)
            print(f"Saved {out_path}")


_SCENARIO_FILE = re.compile(r"synthetic_R0_(?P<R0>[\d.]+)_noise_(?P<noise>\w+)\.pkl")


def load_scenarios(data_dir=SYNTHETIC_DIR):
    """
    Load all the files synthetic_R0_*_noise_*.pkl in data_dir.
    Returns the same dict as compute_incidences, sorted by R0.
    """
    files = sorted(Path(data_dir).glob("synthetic_R0_*_noise_*.pkl"))
    scenarios = {}
    for path in files:
        m = _SCENARIO_FILE.fullmatch(path.name)
        if m is None:
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        sc = scenarios.setdefault(float(m["R0"]), {
            "t": d["t"],
            "incidence": d["incidence"],
            "age incidence": d["age incidence"],
            "noisy": {},
        })
        sc["noisy"][m["noise"]] = d["noisy incidence"]
    if not scenarios:
        raise FileNotFoundError(f"No file synthetic_R0_*_noise_*.pkl in '{data_dir}'")
    return dict(sorted(scenarios.items()))


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

def plot_noise_comparison(t_train, true_inc, noisy_dict, title):
    n_scenari = len(true_inc)
    fig, axes = plt.subplots(1, n_scenari, figsize=(6 * n_scenari, 4))
    if n_scenari == 1:
        axes = [axes]
    for ax, idx in zip(axes, sorted(true_inc.keys())):
        t_inc = t_train[-len(true_inc[idx]):]
        ax.plot(t_inc, true_inc[idx], color="black", linewidth=2, label="no noise")
        ax.scatter(t_inc, noisy_dict[idx], color="tab:red", s=20, alpha=0.7, label="with noise")
        ax.set_title(f"scenario {idx} (R0={idx})")
        ax.set_xlabel("days")
        ax.set_ylabel("weekly incidence")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_noise_comparison_agg_grid(scenarios):
    """
    Aggregated weekly incidence, without and with noise.
    One row per noise level, one column per R0. `scenarios` as returned by load_scenarios.
    """
    R0_values = list(scenarios)
    labels = list(scenarios[R0_values[0]]["noisy"])
    fig, axes = plt.subplots(len(labels), len(R0_values),
                             figsize=(6 * len(R0_values), 4 * len(labels)), squeeze=False)
    for i, label in enumerate(labels):
        for j, R0_val in enumerate(R0_values):
            sc = scenarios[R0_val]
            ax = axes[i, j]
            ax.plot(sc["t"], sc["incidence"], color="black", linewidth=2, label="no noise")
            ax.scatter(sc["t"], sc["noisy"][label], color="tab:red", s=20, alpha=0.7, label="with noise")
            ax.set_title(f"R0 = {R0_val}, noise: {label}")
            if i == len(labels) - 1:
                ax.set_xlabel("days")
            if j == 0:
                ax.set_ylabel("weekly incidence")
            if i == 0 and j == 0:
                ax.legend()
    fig.tight_layout()
    return fig


def load_age_incidence(R0_val, data_dir=SYNTHETIC_DIR):
    """
    Load 't' and 'age incidence' for a given R0. Takes the first file found for that R0,
    whatever the noise level, because 'age incidence' (no noise) is the same for 'light' and 'mild'.
    """
    pattern = os.path.join(data_dir, f"synthetic_R0_{R0_val}_noise_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for R0={R0_val} in {data_dir}")
    with open(files[0], "rb") as f:
        data = pickle.load(f)
    return data["t"], data["age incidence"]


def plot_age_trajectories(R0_values, data_dir=SYNTHETIC_DIR, figsize=None):
    """
    One panel per R0, with the true incidence trajectories of all the age classes
    overlaid (no noise).
    """
    n_cols = len(R0_values)
    if figsize is None:
        figsize = (6 * n_cols, 4.5)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)
    axes = axes[0]

    cmap = plt.cm.viridis
    k_age = None

    for j, R0_val in enumerate(R0_values):
        t, age_inc = load_age_incidence(R0_val, data_dir=data_dir)  # (T-1, k_age)
        k_age = age_inc.shape[1]
        colors = [cmap(a / max(k_age - 1, 1)) for a in range(k_age)]

        ax = axes[j]
        for a in range(k_age):
            ax.plot(t, age_inc[:, a], color=colors[a], linewidth=1.5)

        ax.set_title(f"R0 = {R0_val}", fontsize=13, fontweight="bold")
        ax.set_xlabel("days")
        if j == 0:
            ax.set_ylabel("age-specific weekly incidence")

    colors = [cmap(a / max(k_age - 1, 1)) for a in range(k_age)]
    handles = [plt.Line2D([0], [0], color=colors[a], lw=2, label=f"age class {a}") for a in range(k_age)]
    fig.legend(handles=handles, loc="upper center", ncol=min(k_age, 8),
              bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
