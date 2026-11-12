"""
Utility functions for epidemiological analysis and visualization, and to validate the approximation ability of the pretrained PINN.
"""

#%% 
import torch
import numpy as np
import pandas as pd
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from pinnvsmech.pinn.NN_free_params import NN
from torch import nn
import csv
from pinnvsmech.ode.Data_SEIRDDobs_age_structured import Data_SEIRDDobs_age
from numpy.linalg import eigvals

torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------------

def evaluation_pinn(x,model,times):
    """
    Evaluate the PINN at the given time points for a single parameter set.

    Parameters
    ----------
    x : array-like
        Parameters [R0, Delta] (Delta = mean delay in days).
    model : torch.nn.Module
        Pretrained PINN.
    times : array-like
        Time grid used for the evaluation.

    Returns
    -------
    model_eval : ndarray, shape (len(times), 6*k_age)
        Forward evaluation of the PINN on the whole time grid.
    """

    # Build PINN input: [time, R0, Delta]
    input_eval = torch.tensor(np.hstack([times[:, None], np.tile([x[0], x[1]], (len(times), 1))]), dtype=torch.float64) 

    model.eval()  
    # Forward evaluation of the PINN
    with torch.no_grad():
        model_eval = model(input_eval).numpy()

    return model_eval


def D_incidence(D, dt=1, times=None):
    """
    Incidence as difference between consecutive time points.
    D   : np.ndarray (T,) — aggregated cumulative trajectory
    dt  : int — number of steps to sum (default 1 = difference between adjacent points)
    times : optional array-like — indices at which the incidence is computed:
            the value at index t is max(0, D[t] - D[t-dt]).
            If None, returns T-1 values, one for each index i = 0..T-2 (the value at i=0 is 0). 
    """

    D_diff = np.diff(D)
    if times is not None:
        incidence = np.array([
            max(0, np.sum(D_diff[max(0, t - dt):t]))
            for t in times
        ])
    else:
        incidence = np.array([
            max(0, np.sum(D_diff[max(0, i - dt):i]))
            for i in range(len(D_diff))
        ])
        
    return incidence

def load_train_data(filepath, k_age=16):
    """
    Load x_train, y_train, idx_train from the CSV saved after train_data_constrains.
    
    Struttura colonne: [x_train (3) | y_train (6*k_age) | idx_train (1)]
    
    x_train colonne: [t, R0, delta_param]
    y_train colonne: [S_1..S_k, E_1..E_k, I_1..I_k, R_1..R_k, D_1..D_k, Dobs_1..Dobs_k]
    """
    df = pd.read_csv(filepath, header=None)
    
    n_x = 3
    n_y = 6 * k_age  # 96
    
    x_train   = df.iloc[:, :n_x].values                    # (n, 3)
    y_train   = df.iloc[:, n_x:n_x + n_y].values           # (n, 96)
    idx_train = df.iloc[:, n_x + n_y:].values.astype(int)  # (n, 1)
    
    print(f"x_train   shape: {x_train.shape}")
    print(f"y_train   shape: {y_train.shape}")
    print(f"idx_train shape: {idx_train.shape}")
    
    return x_train, y_train, idx_train

def find_scenario_by_params(x_train, idx_train, R0_target, Delta_target):
    """
    Find the scenario_idx closest to (R0_target, Delta_target).

    Parameters
    ----------
    x_train      : torch.Tensor (N_pts, 3) = [t, R0, Delta]
    idx_train    : np.ndarray (N_pts,)
    R0_target    : float
    Delta_target : float

    Returns
    -------
    scenario_idx : int   – ID of the scenario found
    R0_found     : float – actual R0 of the scenario
    Delta_found  : float – actual Delta of the scenario
    """
    x_np = x_train.detach().cpu().numpy()

    unique_scenarios = np.unique(idx_train)

    R0_list    = np.array([x_np[np.where(idx_train == s)[0][0], 1] for s in unique_scenarios])
    Delta_list = np.array([x_np[np.where(idx_train == s)[0][0], 2] for s in unique_scenarios])

    R0_range    = R0_list.max()    - R0_list.min()
    Delta_range = Delta_list.max() - Delta_list.min()

    dist = np.sqrt(
        ((R0_list    - R0_target)    / R0_range)    ** 2 +
        ((Delta_list - Delta_target) / Delta_range) ** 2
    )

    best_i       = np.argmin(dist)
    scenario_idx = unique_scenarios[best_i]
    R0_found     = R0_list[best_i]
    Delta_found  = Delta_list[best_i]

    print(f"Target:  R0={R0_target:.4f},  Delta={Delta_target:.4f}")
    print(f"Found:  R0={R0_found:.4f},  Delta={Delta_found:.4f},  scenario_idx={scenario_idx:.0f}")

    return int(scenario_idx), R0_found, Delta_found

def plot_age_classes(age_idx,
                     y_true,
                     y_pred,
                     times,
                     compartments,
                     colors_true,
                     colors_pred,
                     k_age,
                     R0=None):

    n_cols = len(age_idx)

    fig, axes = plt.subplots(
        len(compartments),
        n_cols,
        figsize=(3*n_cols, 14),
        squeeze=False
    )

    title = "S, E, I, R, D, Dobs training set behavior"
    if R0 is not None:
        title += f": R0 = {R0:.2f}"

    fig.suptitle(title, fontsize=15, fontweight='bold')

    for col, j in enumerate(age_idx):

        for c, comp in enumerate(compartments):

            ax = axes[c, col]

            y_true_c = y_true[:, j + c*k_age]
            y_pred_c = y_pred[:, j + c*k_age]

            if col == 0:
                ax.plot(
                    times,
                    y_true_c,
                    lw=2,
                    color=colors_true[c],
                    label='True'
                )

                ax.plot(
                    times,
                    y_pred_c,
                    '--',
                    lw=2,
                    color=colors_pred[c],
                    label='PINN'
                )

                ax.legend(fontsize=8)

            else:
                ax.plot(
                    times,
                    y_true_c,
                    lw=2,
                    color=colors_true[c]
                )

                ax.plot(
                    times,
                    y_pred_c,
                    '--',
                    lw=2,
                    color=colors_pred[c]
                )

            if c == 0:
                ax.set_title(f'Age class {j+1}')

            if col == 0:
                ax.set_ylabel(comp, fontsize=12)

            if c == len(compartments) - 1:
                ax.set_xlabel('Time')

            ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


"""
Approximation error of the PINN with respect to the age-structured SEIRDDobs ODE system.

Model output:
    y_pred shape: (T, 6 * k_age)
    Columns: [S_0..S_{k-1}, E_0..E_{k-1}, I_0..I_{k-1}, R_0..R_{k-1},
              D_0..D_{k-1}, Dobs_0..Dobs_{k-1}]

Model input:
    x shape: (T, 3) = [t, R0, Delta]   with Delta = 1/delta

Aggregation over the age classes:
    X_agg(t) = sum_j(X_j(t) * N_k[j]) / N_tot     (fraction of the total population)
"""
 

def wMAPE(y_true, y_pred):
    """
    Weighted Mean Absolute Percentage Error (wMAPE):
        wMAPE = sum|y_pred - y_true| / sum|y_true|
    Robust to y_true ~ 0 because it normalizes by the total sum, not point by point.
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_pred - y_true)) / denom
 
def pinn_error_full_train(
    model,
    x_train,           # torch.Tensor shape (N_pts, 3): [t, R0, Delta]
    y_train,           # torch.Tensor shape (N_pts, 6*k_age)
    idx_train,
    N_k,
    device="cpu",
    daily=False
):
    """
    For each scenario of the training set computes:

      1. MSE per age class      (mean over S,E,I,R,D,Dobs and over all the k_age classes)
      2. Aggregated MSE         (mean over the 6 aggregated compartments)
      3. MSE on aggregated D    (true deaths, weighted by N_k)
      4. MSE on aggregated Dobs (observed deaths, weighted by N_k)
      5. MAE and wMAPE on the Dobs incidence

    Returns
    -------
    dict with keys (all np.ndarray of shape (n_scenarios,)):
        'errors_per_age'      : mean error per age class
        'errors_agg'          : aggregated error (6 compartments)
        'errors_S_agg', 'errors_E_agg', 'errors_I_agg', 'errors_R_agg',
        'errors_D_agg', 'errors_Dobs_agg' : error on each aggregated compartment
        'errors_Dobs_inc_mae', 'errors_Dobs_inc_wmape' : errors on the Dobs incidence
        'R0_list'             : R0 of each scenario
        'Delta_list'          : Delta of each scenario
        'scenario_ids'        : index of each scenario
    """
 
    model.eval()
    N_k = np.asarray(N_k, dtype=float)
    N_tot = N_k.sum()
    k_age = len(N_k)
    n_comp = 6  # S, E, I, R, D, Dobs
 
    x_np = x_train.detach().cpu().numpy()
    y_np = y_train.detach().cpu().numpy()
 
    unique_scenarios = np.unique(idx_train)
    n_scenarios = len(unique_scenarios)
 
    errors_per_age  = np.zeros(n_scenarios)
    errors_agg      = np.zeros(n_scenarios)
    errors_S_agg    = np.zeros(n_scenarios)
    errors_E_agg    = np.zeros(n_scenarios)
    errors_I_agg    = np.zeros(n_scenarios)
    errors_R_agg    = np.zeros(n_scenarios)
    errors_D_agg    = np.zeros(n_scenarios)
    errors_Dobs_agg    = np.zeros(n_scenarios)
    errors_Dobs_inc_mae     = np.zeros(n_scenarios)
    errors_Dobs_inc_wmape   = np.zeros(n_scenarios)
    R0_list         = np.zeros(n_scenarios)
    Delta_list      = np.zeros(n_scenarios)
 
    with torch.no_grad():
        y_pred_all = model(x_train.to(device)).detach().cpu().numpy()  # (N_pts, 6*k_age)
 
    for s_i, scenario_idx in enumerate(unique_scenarios):
        mask = idx_train == scenario_idx
        y_true = y_np[mask]          # (T, 6*k_age)
        y_pred = y_pred_all[mask]    # (T, 6*k_age)
 
        first_idx = np.where(mask)[0][0]
        R0_list[s_i]    = x_np[first_idx, 1]
        Delta_list[s_i] = x_np[first_idx, 2]
 
        mse_age = 0.0
        for c in range(n_comp):  # S, E, I, R, D, Dobs
            sl = slice(c * k_age, (c + 1) * k_age)
            diff = y_pred[:, sl] - y_true[:, sl]   # (T, k_age)
            mse_age += np.mean(diff ** 2)
        errors_per_age[s_i] = mse_age / n_comp
 
        comp_errors = []
        comp_true_agg = {}
        comp_pred_agg = {}
        comp_names = ['S', 'E', 'I', 'R', 'D', 'Dobs']
        for c, name in enumerate(comp_names):
            sl = slice(c * k_age, (c + 1) * k_age)
            true_agg = (y_true[:, sl] * N_k).sum(axis=1) / N_tot
            pred_agg = (y_pred[:, sl] * N_k).sum(axis=1) / N_tot
            comp_true_agg[name] = true_agg
            comp_pred_agg[name] = pred_agg
            comp_errors.append(np.mean((pred_agg - true_agg) ** 2))
 
        errors_agg[s_i] = np.mean(comp_errors)


        errors_S_agg[s_i]    = comp_errors[0]
        errors_E_agg[s_i]    = comp_errors[1]
        errors_I_agg[s_i]    = comp_errors[2]
        errors_R_agg[s_i]    = comp_errors[3]
        errors_D_agg[s_i]    = comp_errors[4]
        errors_Dobs_agg[s_i] = comp_errors[5]

        Dobs_inc_true = D_incidence(comp_true_agg['Dobs']*N_tot, dt=1)
        Dobs_inc_pred = D_incidence(comp_pred_agg['Dobs']*N_tot, dt=1)

        if daily:
            times=x_train.detach().cpu().numpy()[np.where(mask)[0],0]
            times = times[::7].astype(int)
            Dobs_inc_true = D_incidence(comp_true_agg['Dobs']*N_tot, dt=7, times=times)
            Dobs_inc_pred = D_incidence(comp_pred_agg['Dobs']*N_tot, dt=7, times=times)

        errors_Dobs_inc_mae[s_i]   = np.mean(np.abs(Dobs_inc_pred - Dobs_inc_true))

        errors_Dobs_inc_wmape[s_i] = wMAPE(Dobs_inc_true, Dobs_inc_pred)


  
    return {
        'errors_per_age':  errors_per_age,
        'errors_agg':      errors_agg,
        'errors_S_agg':    errors_S_agg,
        'errors_E_agg':    errors_E_agg,
        'errors_I_agg':    errors_I_agg,
        'errors_R_agg':    errors_R_agg,
        'errors_D_agg':    errors_D_agg,
        'errors_Dobs_agg':    errors_Dobs_agg,
        'errors_Dobs_inc_mae':   errors_Dobs_inc_mae,
        'errors_Dobs_inc_wmape': errors_Dobs_inc_wmape,
        'R0_list':         R0_list,
        'Delta_list':      Delta_list,
        'scenario_ids':    unique_scenarios,
    }

 
def print_error_summary(results, label="training set", n_extreme=5, error_key='errors_Dobs_inc_wmape'):
    """
    Print a summary of the errors computed by pinn_error_full_train,
    including the best and worst scenarios for the chosen metric.

    Parameters
    ----------
    results    : dict returned by pinn_error_full_train
    label      : descriptive string (e.g. "training set", "test set")
    n_extreme  : how many best/worst scenarios to show
    error_key  : metric used for the ranking
                 ('errors_agg', 'errors_per_age',
                  'errors_S_agg', 'errors_E_agg', 'errors_I_agg',
                  'errors_R_agg', 'errors_D_agg', 'errors_Dobs_agg',
                  'errors_Dobs_inc_mae', 'errors_Dobs_inc_wmape')
    """
    print(f"\n{'='*60}")
    print(f"  PINN SEIRDDobs age-structured errors – {label}")
    print(f"{'='*60}")
 
    comp_keys  = ['errors_S_agg', 'errors_E_agg', 'errors_I_agg',
                  'errors_R_agg', 'errors_D_agg', 'errors_Dobs_agg', 
                  'errors_Dobs_inc_wmape']
    comp_names = ['S', 'E', 'I', 'R', 'D', 'Dobs', 'DobsInc_wmape']
    stats_names = ['mean', 'median', 'max', 'min']
    stats_funcs = [np.mean, np.median, np.max, np.min]
 
    col_width = 12
    header = f"  {'':<9}" + "".join(f"{name:>{col_width}}" for name in comp_names)
    print(f"\n  Aggregated MSE per compartment ({len(results['scenario_ids'])} scenarios):")
    print(header)
    print("  " + "─" * (9 + col_width * len(comp_names)))
    for stat_name, stat_func in zip(stats_names, stats_funcs):
        row = f"  {stat_name:<9}"
        for key in comp_keys:
            vals = results[key]
            row += f"{stat_func(vals):>{col_width}.4e}"
        print(row)
 
    for key, name in [
        ('errors_per_age', "MSE per age class (mean over the 6 compartments)"),
        ('errors_agg',     'Aggregated MSE (S+E+I+R+D+Dobs, weighted)'),
    ]:
        vals = results[key]
        if vals is None:
            print(f"\n  {name}: not computed")
            continue
        print(f"\n  {name}:")
        print(f"    mean    = {np.mean(vals):.4e}")
        print(f"    median  = {np.median(vals):.4e}")
        print(f"    max     = {np.max(vals):.4e}")
        print(f"    min     = {np.min(vals):.4e}")
 
    vals    = results[error_key]
    ids     = results['scenario_ids']
    R0_l    = results['R0_list']
    Delta_l = results['Delta_list']
 
    if vals is None:
        print(f"\n  '{error_key}' not available for the ranking.")
    else:
        sorted_asc  = np.argsort(vals)
        sorted_desc = sorted_asc[::-1]
 
        n = min(n_extreme, len(vals))
        print(f"\n  {'─'*60}")
        print(f"Ranking su: {error_key}")
        print(f"  {'─'*60}")
 
        print(f"\nTop-{n} BEST scenarios:")
        print(f"    {'Rank':<6} {'Scenario ID':<14} {'R0':>8} {'Delta':>8} {'Error':>12}")
        for rank, i in enumerate(sorted_asc[:n], 1):
            print(f"    {rank:<6} {ids[i]:<14.1f} {R0_l[i]:>8.3f} {Delta_l[i]:>8.3f} {vals[i]:>12.4e}")
 
        print(f"\nTop-{n} WORST scenarios:")
        print(f"    {'Rank':<6} {'Scenario ID':<14} {'R0':>8} {'Delta':>8} {'Error':>12}")
        for rank, i in enumerate(sorted_desc[:n], 1):
            print(f"    {rank:<6} {ids[i]:<14.1f} {R0_l[i]:>8.3f} {Delta_l[i]:>8.3f} {vals[i]:>12.4e}")
 
    print(f"{'='*60}\n")

def get_best_worst_scenarios(
    results,
    n=6,
    error_key='errors_Dobs_inc_wmape',
    R0_min=None,
    R0_max=None,
):
    """
    Returns the IDs of the best and worst scenarios, with an optional filter on R0.

    Parameters
    ----------
    results   : dict from pinn_error_full_train
    n         : how many scenarios to return for best/worst
    error_key : metric used for the ranking
    R0_min    : float – exclude scenarios with R0 < R0_min
    R0_max    : float – exclude scenarios with R0 > R0_max

    Returns
    -------
    best_ids  : np.ndarray – scenario IDs with the lowest error
    worst_ids : np.ndarray – scenario IDs with the highest error
    """
    vals = results[error_key]
    ids  = results['scenario_ids']
    R0_array = results['R0_list']

    if vals is None:
        raise ValueError(f"'{error_key}' not available in the results.")

    mask = np.ones(len(ids), dtype=bool)
    if R0_min is not None:
        mask &= R0_array >= R0_min
    if R0_max is not None:
        mask &= R0_array <= R0_max

    vals_f = vals[mask]
    ids_f  = ids[mask]

    if len(ids_f) == 0:
        raise ValueError("No scenario survives the R0 filter. Widen the limits.")

    n = min(n, len(ids_f))
    sorted_asc = np.argsort(vals_f)
    best_ids  = ids_f[sorted_asc[:n]]
    worst_ids = ids_f[sorted_asc[::-1][:n]]
    return best_ids, worst_ids

def plot_agg_trajectories(
    model,
    x_train,
    y_train,
    idx_train,
    N_k,
    scenario_ids,
    title="PINN vs ODE – aggregated SEIRDDobs trajectories",
    n_cols=3,
    daily=False,
    figsize=None,
    savepath=None,
):
    """
    Plots the weekly Dobs incidence (aggregated over the age classes, weighted by N_k)
    for a list of scenarios, comparing PINN vs ODE (training data).

    Parameters
    ----------
    model        : PINN (PyTorch)
    x_train      : torch.Tensor (N_pts, 3) = [t, R0, Delta]
    y_train      : torch.Tensor (N_pts, 6*k_age)
    idx_train    : np.ndarray  (N_pts,)
    N_k          : np.ndarray  (k_age,)
    scenario_ids : array-like  – IDs of the scenarios to plot
    title        : title of the figure
    n_cols       : number of columns of the grid
    daily        : if True the data are daily and the incidence is computed over 7-day windows
    figsize      : tuple (width, height); computed automatically if None
    savepath     : if given, the figure is saved at this path
    """
    import torch

    model.eval()
    N_k   = np.asarray(N_k, dtype=float)
    N_tot = N_k.sum()
    k_age = len(N_k)
    n_comp = 6
    comp_names = ['S', 'E', 'I', 'R', 'D', 'Dobs']

    colors_true = {
        'S': '#E69F00', 'E': '#F0E442', 'I': '#56B4E9',
        'R': '#009E73', 'D': '#CC79A7', 'Dobs': '#999999'
    }
    colors_pred = {
        'S': '#D55E00', 'E': '#B8B800', 'I': '#0072B2',
        'R': '#117733', 'D': '#882255', 'Dobs': '#444444'
    }

    x_np = x_train.detach().cpu().numpy()
    y_np = y_train.detach().cpu().numpy()

    with torch.no_grad():
        y_pred_all = model(x_train).detach().cpu().numpy()  # (N_pts, 6*k_age)

    scenario_ids = np.asarray(scenario_ids)
    n_plots = len(scenario_ids)
    n_rows  = int(np.ceil(n_plots / n_cols))

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=15, fontweight='bold')

    for ax_i, scenario_idx in enumerate(scenario_ids):
        ax   = axes[ax_i]
        mask = idx_train == scenario_idx
        rows = np.where(mask)[0]

        y_true = y_np[rows]           # (T, 6*k_age)
        y_pred = y_pred_all[rows]     # (T, 6*k_age)
        times  = x_np[rows, 0]
        R0     = x_np[rows[0], 1]
        Delta  = x_np[rows[0], 2]

        c, comp = 5,'Dobs'
        #for c, comp in [(4,'D')]: #enumerate(comp_names):
        sl = slice(c * k_age, (c + 1) * k_age)
        true_agg = (y_true[:, sl] * N_k).sum(axis=1) 
        pred_agg = (y_pred[:, sl] * N_k).sum(axis=1) 

        Dobs_inc_true = D_incidence(true_agg, dt=1)
        Dobs_inc_pred = D_incidence(pred_agg, dt=1)

        if daily:
            times = times[::7].astype(int)
            Dobs_inc_true = D_incidence(true_agg, dt=7, times=times)
            Dobs_inc_pred = D_incidence(pred_agg, dt=7, times=times)

        n_weeks = len(Dobs_inc_true)
        weeks = np.arange(1, n_weeks + 1)  # Week 1, 2, 3, ...
        
        ax.plot(weeks, Dobs_inc_true, lw=2,
                color=colors_true['D'], label=f"{comp} incidence ODE")
        ax.plot(weeks, Dobs_inc_pred, '--', lw=2,
                color=colors_pred['D'], label=f"{comp} incidence PINN")

        ax.set_xticks(weeks[::8])
        ax.set_xticklabels([f"W{w}" for w in weeks[::8]], fontsize=8)

        ax.set_title(
            f"ID={scenario_idx:.0f}  $R_0$={R0:.2f}  $\\Delta$={Delta:.2f}",
            fontsize=9
        )
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel("Time - weeks", fontsize=8)
        ax.set_ylabel("Weekly Deaths", fontsize=8)

    axes[0].legend(fontsize=7, ncol=2)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved in: {savepath}")

    return fig


def plot_sir_conservation(
    x_train_np,
    idx_train,
    idx_plot,
    S_pred, I_pred, R_pred,
):

    """
    Plot population conservation (S + I + R ≈ 1) for a subset of training scenarios.
    
    For each selected scenario, shows the sum S+I+R predicted by the SIR-INN
    over time, together with a reference line at 1. The subplot title reports
    the MAE and maximum absolute error for that scenario.
    
    Parameters
    ----------
    x_train_np : np.ndarray
        Training input array of shape (N, 3) with columns [time, beta, gamma].
    idx_train : np.ndarray
        Scenario indices associated with each training point.
    idx_plot : array-like
        Subset of scenario indices to visualize (up to 12).
    S_pred, I_pred, R_pred : np.ndarray
        SIR compartments predicted by the SIR-INN.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the (4 x 3) conservation panel.
    """
    
    color_sum = "#8B008B"
    color_ref = "#888888"

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle("Population Conservation: S + I + R ≈ N", fontsize=16)
    
    n_plots = len(idx_plot)
    n_cols = 3
    n_rows = 4


    for i, idx in enumerate(idx_plot):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        idx_p = np.where(idx_train == idx)[0]
        t = x_train_np[idx_p, 0]

        beta  = x_train_np[idx_p[0], 1]
        gamma = x_train_np[idx_p[0], 2]
        R0    = beta / gamma

        N_sum = S_pred[idx_p] + I_pred[idx_p] + R_pred[idx_p]
        mae   = np.mean(np.abs(N_sum - 1.0))
        max_e = np.max(np.abs(N_sum - 1.0))

        # Reference line
        ax.axhline(1.0, color=color_ref, lw=1.5, ls='--', label='N = 1')
        # S+I+R curve
        ax.plot(t, N_sum, color=color_sum, lw=2, label='S+I+R')

        ax.set_title(
            r'$\mathbf{R_0 = %.2f}$' % R0 + '\n' +
            r'MAE=%.2e   max|err|=%.2e' % (mae, max_e),
            fontsize=10,
            loc='center'
        )

        ax.set_ylim([max(0, N_sum.min() - 0.05), N_sum.max() + 0.05])
        ax.grid(True)

        # X-axis label only on last row
        if row == n_rows - 1:
            ax.set_xlabel("Time (days)", fontsize=10)
        else:
            ax.tick_params(labelbottom=False)

        # Y-axis label only on first column
        if col == 0:
            ax.set_ylabel("S + I + R", fontsize=10)
        else:
            ax.tick_params(labelleft=False)
        
        if i == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def compute_conservation_error(
    idx_train,
    S_pred, I_pred, R_pred,
):
    """
    Compute population conservation error (S+I+R ≈ 1) 
    across the entire training set and per scenario.

    Parameters
    ----------
    idx_train : np.ndarray
        Scenario indices associated with each training point.
    S_pred, I_pred, R_pred : np.ndarray
        SIR compartments predicted by the SIR-INN across all training points.
    
    Returns
    -------
    df : pd.DataFrame
        Per-scenario conservation error with columns MAE, max_err, std_err,
        indexed by scenario index.
    """
    
    N_sum = S_pred + I_pred + R_pred
    abs_err = np.abs(N_sum - 1.0)

    scenarios = np.unique(idx_train)
    results = []

    for idx in scenarios:
        idx_p = np.where(idx_train == idx)[0]
        err = abs_err[idx_p] # for all the time domain
        results.append({
            'scenario': idx,
            'MAE':      np.mean(err),
            'max_err':  np.max(err),
            'std_err':  np.std(err),
        })

    df = pd.DataFrame(results).set_index('scenario')

    # Global stats across all training points
    print("=" * 45)
    print("  Conservation Error — Full Training Set")
    print("=" * 45)
    print(f"  Global MAE     : {abs_err.mean():.4e}")
    print(f"  Global max|err|: {abs_err.max():.4e}")
    print(f"  Global std     : {abs_err.std():.4e}")
    print(f"  # scenarios    : {len(scenarios)}")
    print(f"  # total points : {len(N_sum)}")
    print("=" * 45)

    return df

