"""
Training of the PINN: loss = alpha['ode'] * ODE residual (on the collocation points)
                           + alpha['mse'] * MSE on the training data.
"""

import csv
import gc
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from pinnvsmech.utils_age_structured import compute_loss_age

torch.set_default_dtype(torch.float64)


def compute_rho(C, N_age):
    """
    Spectral radius of the structural next-generation matrix C_ij * N_i / N_j
    (the same quantity used in Data_SEIRDDobs_age.beta_from_R0, so that beta = R0 * xi / rho).
    """
    C_np = C.cpu().numpy() if isinstance(C, torch.Tensor) else np.asarray(C)
    N_np = N_age.cpu().numpy() if isinstance(N_age, torch.Tensor) else np.asarray(N_age)
    N_ratio = N_np[:, None] / N_np[None, :]   # (k_age, k_age)
    K_struct = C_np * N_ratio
    return np.max(np.abs(np.linalg.eigvals(K_struct)))


def train_pinn(model, x_train, y_train, x_coll_full, C, N_age, epsilon, xi, ifr,
               alpha=None, n_epochs=5000, lr=1e-3, weight_decay=0, batch_size=100, train_size=0.9,
               model_path="model.pth", log_csv=None, runtime_json=None):
    """
    Train the PINN with Adamax.

    model       : NN (input [t, R0, Delta], output 6*k_age compartments)
    x_train     : tensor (n, 3) = [t, R0, Delta]
    y_train     : tensor (n, 6*k_age)
    x_coll_full : tensor (n_coll, 3), collocation set; batch_size points are sampled at random
                  at every batch, for both training and validation
    C, N_age, epsilon, xi, ifr : ODE parameters (contact matrix, population, rates, ifr per age class)
    alpha       : dict with the loss weights, default {'ode': 1000, 'mse': 1}
    train_size  : fraction of (x_train, y_train) used for training, the rest for validation
    model_path  : where the weights are saved (every time the training loss improves)
    log_csv     : optional csv with the losses of every epoch
    runtime_json: optional json with the time of every epoch

    Returns (history, runtime_log).
    """
    if alpha is None:
        alpha = {'ode': 1000, 'mse': 1}
    C = torch.tensor(C) if not isinstance(C, torch.Tensor) else C
    rho = compute_rho(C, N_age)

    dataset = TensorDataset(x_train, y_train)
    train_size_int = int(train_size * len(dataset))
    val_size_int = len(dataset) - train_size_int
    train_dataset, val_dataset = random_split(dataset, [train_size_int, val_size_int])
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for path in (model_path, log_csv, runtime_json):
        if path is not None and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    if log_csv is not None:
        with open(log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_ode', 'train_data', 'val_loss', 'val_ode', 'val_data'])

    optimizer = torch.optim.Adamax(model.parameters(),
                                   lr=lr,
                                   weight_decay=weight_decay)

    history = {}
    history['train'] = {'total': [],
                        'ode': [],
                        'mse': [],
                        'ic': []}
    history['val'] = {'total': [],
                      'ode': [],
                      'mse': [],
                      'ic': []}

    best_loss = 99999

    train_start = time.time()
    runtime_log = {"epoch_times": [], "total_seconds": None}

    for ep in range(n_epochs):
        t0 = time.time()

        # Train
        train_loss = {'total': 0, 'mse': 0, 'ode': 0}
        model.train()

        for batch_idx, data in enumerate(train_data):
            optimizer.zero_grad()

            idx_coll = torch.randperm(len(x_coll_full))[:batch_size]
            x_coll_batch = x_coll_full[idx_coll]

            loss, ode_loss, mse_loss = compute_loss_age(nn=model,
                                                        x=data[0],
                                                        y=data[1],
                                                        x_coll=x_coll_batch,
                                                        C_matrix=C,
                                                        N_age=N_age,
                                                        epsilon=epsilon,
                                                        xi=xi,
                                                        ifr=ifr,
                                                        rho=rho,
                                                        alpha=alpha
                                                        )
            train_loss['total'] += loss.item()
            train_loss['ode'] += ode_loss.item()
            train_loss['mse'] += mse_loss.item()

            loss.backward()
            optimizer.step()
            del loss, ode_loss, mse_loss
            gc.collect()

        history['train']['total'] += [train_loss['total'] / len(train_data.dataset)]
        history['train']['ode'] += [train_loss['ode'] / len(train_data.dataset)]
        history['train']['mse'] += [train_loss['mse'] / len(train_data.dataset)]
        history['train']['ic'] += [0]

        # Validation
        val_loss = {'total': 0, 'mse': 0, 'ode': 0}
        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(val_data):

                idx_coll_val = torch.randperm(len(x_coll_full))[:batch_size]
                x_coll_val_batch = x_coll_full[idx_coll_val]

                loss, ode_loss, mse_loss = compute_loss_age(nn=model,
                                                            x=data[0],
                                                            y=data[1],
                                                            x_coll=x_coll_val_batch,
                                                            C_matrix=C,
                                                            N_age=N_age,
                                                            epsilon=epsilon,
                                                            xi=xi,
                                                            ifr=ifr,
                                                            rho=rho,
                                                            alpha=alpha
                                                            )
                val_loss['total'] += loss.item()
                val_loss['ode'] += ode_loss.item()
                val_loss['mse'] += mse_loss.item()

                del loss, ode_loss, mse_loss
                gc.collect()

            history['val']['total'] += [val_loss['total'] / len(val_data.dataset)]
            history['val']['ode'] += [val_loss['ode'] / len(val_data.dataset)]
            history['val']['mse'] += [val_loss['mse'] / len(val_data.dataset)]
            history['val']['ic'] += [0]

        if history['train']['total'][-1] < best_loss:
            best_loss = history['train']['total'][-1]
            torch.save(model.state_dict(), model_path)

        if log_csv is not None:
            with open(log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep,
                    history['train']['total'][-1],
                    history['train']['ode'][-1],
                    history['train']['mse'][-1],
                    history['val']['total'][-1],
                    history['val']['ode'][-1],
                    history['val']['mse'][-1]
                ])

        epoch_time = time.time() - t0
        runtime_log["epoch_times"].append(epoch_time)

        if ep % 100 == 0:
            print('Training Completed: %3d%%, Average Train Loss: % 2.2e, Average Val Loss: % 2.2e' % (100 * (ep+1)/n_epochs,
                                                                                                     history['train']['total'][-1],
                                                                                                     history['val']['total'][-1]))

    runtime_log["total_seconds"] = time.time() - train_start
    print(f"Training completed in {runtime_log['total_seconds']:.1f}s "
          f"({runtime_log['total_seconds']/60:.1f} min)")

    if runtime_json is not None:
        with open(runtime_json, "w") as fp:
            json.dump(runtime_log, fp, indent=2)

    return history, runtime_log
