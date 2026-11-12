"""
Data for the PINN training.

- Training set: deterministic trajectories on a grid of (R0, Delta) pairs, weekly time grid,
  one infected per age class at t=0. It is generated with `generate_deterministic_data`
  (see data/synthetic.py, with ic_mode="standard") and saved with `save_deterministic_data`.
- Collocation set: points (t, R0, Delta) where the ODE residual is evaluated. They are the
  midpoints of the parameter grid, on a daily time grid.
"""

import os
import os.path as osp
import pickle

import numpy as np

from pinnvsmech.paths import TRAINING_DIR

# File names (they encode: 33x33 parameter grid, I0 = 1 per age class, weekly/daily time grid)
TRAIN_CSV = "deterministic_dataset_PINN.csv"
TRAIN_PAIRS_PICKLE = "SEIRDDobs_training_parameter_pairs_33.pickle"
COLLOCATION_PICKLE = "SEIRDDobs_collocation_parameter_pairs_32.pickle"


def parameter_grid(R0_range=(1.3, 3.0), Delta_range=(7, 35), n_1d=33):
    """
    Regular grid of (R0, Delta) pairs, n_1d x n_1d points. Delta is the mean delay in days.
    Returns an array (n_1d**2, 2), columns [R0, Delta].
    """
    xx, yy = np.meshgrid(np.linspace(R0_range[0], R0_range[1], n_1d),
                         np.linspace(Delta_range[0], Delta_range[1], n_1d))
    return np.c_[xx.flatten(), yy.flatten()]


def generate_collocation_set(p_train, domain=(0, 365)):
    """
    Collocation set: midpoints of the (R0, Delta) grid of p_train (n_1d - 1 values per axis),
    each one on a daily time grid over `domain`.
    Returns an array ((n_1d - 1)**2 * n_days, 3), columns [t, R0, Delta].
    """
    R0_unique = np.sort(np.unique(p_train[:, 0]))
    Delta_unique = np.sort(np.unique(p_train[:, 1]))
    R0_mid = (R0_unique[:-1] + R0_unique[1:]) / 2
    Delta_mid = (Delta_unique[:-1] + Delta_unique[1:]) / 2

    xx_mid, yy_mid = np.meshgrid(R0_mid, Delta_mid)
    p_coll = np.c_[xx_mid.flatten(), yy_mid.flatten()]

    t_coll = np.linspace(int(domain[0]), int(domain[1]), int(domain[1]) - int(domain[0]) + 1)
    n_p_c = len(p_coll)
    n_t_c = len(t_coll)
    t_rep_c = np.tile(t_coll, n_p_c)
    p_rep_c = np.repeat(p_coll, n_t_c, axis=0)
    return np.column_stack([t_rep_c, p_rep_c])


def save_collocation_set(x_coll, folder=TRAINING_DIR, name=COLLOCATION_PICKLE):
    os.makedirs(folder, exist_ok=True)
    with open(osp.join(folder, name), "wb") as f:
        pickle.dump(x_coll, f)


def load_collocation_set(folder=TRAINING_DIR, name=COLLOCATION_PICKLE):
    """Load the collocation set, array (n_points, 3), columns [t, R0, Delta]."""
    with open(osp.join(folder, name), "rb") as f:
        return pickle.load(f)
