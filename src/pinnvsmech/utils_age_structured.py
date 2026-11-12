"""
Utilities for the age-structured PINN: loss functions, ODE right-hand side, parameter estimation,
forecasting and forecast metrics.
"""

from pinnvsmech.pinn.NN_free_params import NN
try:
    from pinnvsmech.pinn.NN_free_params_age_structured_3 import NN_age
except ImportError:  # the convolutional network is not included in this repository
    class NN_age:
        """Placeholder, so that isinstance(model, NN_age) is always False."""
        pass
import torch
torch.set_default_dtype(torch.float64)
#import torch.func
import functorch
import numpy as np
from matplotlib import cm
import scipy.spatial
import matplotlib.tri as mtri
from scipy.optimize import minimize, Bounds
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigvals

def load_C_ages(basin = "Italy"):
    from epydemix.population import load_epydemix_population
    # Load the population of Italy from GitHub. For offline use, download the population data from
    # https://github.com/epistorm/epydemix_data and provide the local path to 
    # the epydemix_data folder using the argument path_to_data
    population = load_epydemix_population(basin)
    contact_matrices = population.contact_matrices
    return contact_matrices

def load_N_ages(basin = "Italy"):
    from epydemix.population import load_epydemix_population
    # Load the population of Italy from GitHub. For offline use, download the population data from
    # https://github.com/epistorm/epydemix_data and provide the local path to 
    # the epydemix_data folder using the argument path_to_data
    population = load_epydemix_population(basin)
    N_k = population.Nk
    return N_k

#%%
#def df(f: NN, x: torch.Tensor) -> torch.Tensor:
#    df_value = torch.vmap(torch.func.jacrev(f))(x)[:, :, 0]
#    return df_value 

def df(f, x: torch.Tensor) -> torch.Tensor:
    """
    Time derivative of the model output with respect to t (first column of x), row by row.
    Works with the standard NN (output: tensor (batch, 6*k_age)) and with NN_age
    (output: tuple of 12 tensors). Returns a tensor of shape (batch, 6*k_age).
    """
    if isinstance(f, NN_age):
        def f_disagg(x_single):
            out = f(x_single.unsqueeze(0))
            S, E, I, R, D, Dobs, _, _, _, _, _, _ = out
            return torch.cat([S, E, I, R, D, Dobs], dim=1).squeeze(0)  # shape: (6*k_age,)
        
        df_value = torch.vmap(torch.func.jacrev(f_disagg))(x)[:, :, 0]
    else:
        df_value = torch.vmap(torch.func.jacrev(f))(x)[:, :, 0]
    
    return df_value

def compute_loss_age(nn: NN,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 x_coll: torch.Tensor,
                 C_matrix, 
                 N_age,
                 epsilon,
                 xi,
                 ifr,
                 rho,
                 alpha: np.array = [1, 1],
                 ) -> torch.float:
    """Total loss = alpha['ode'] * ODE loss + alpha['mse'] * data MSE loss.
    Returns (total, ode_loss, mse_loss). Everything is built with differentiable tensors,
    so .backward() can be applied to the total loss.
    rho: spectral radius of the structural next-generation matrix (C * N_i / N_j), computed once outside.
    alpha: dict with keys 'ode' and 'mse'.
    """

    # ode_loss = ((df(nn, x_coll) - y_coll) ** 2).sum(axis=0).mean()
    ode_loss = ((df(nn, x_coll) - F_age(nn, x_coll, C_matrix, N_age, epsilon, xi, ifr, rho)) ** 2).sum(axis=0).mean()
    #ode_loss = ((df(nn, x_coll) - F(nn, x_coll)) ** 2).mean()
    
     # ----- MSE loss -----
    if isinstance(nn, NN_age):
        print("Warning: this branch is not maintained, the standard NN is expected.")
        k_age = N_age.shape[0]
        N_tot = N_age.sum()

        if y.shape[1] == 6:
            print("Warning: this branch is not maintained, the standard NN is expected (y.shape[1] == 6).")
            y_agg = y
        elif y.shape[1] == 6 * k_age:
            print("Warning: this branch is not maintained, the standard NN is expected (y.shape[1] == 6*k_age).")
            S_train = (y[:, :k_age]*N_age).sum(dim=1, keepdim=True)
            I_train = (y[:, k_age:2*k_age]*N_age).sum(dim=1, keepdim=True)
            R_train = (y[:, 2*k_age:3*k_age]*N_age).sum(dim=1, keepdim=True)
            y_agg = torch.cat([S_train/N_tot, I_train/N_tot, R_train/N_tot], dim=1)
        else:
            raise ValueError(f"Size of y ({y.shape[1]}) not compatible with k_age={k_age}.")

        S_pred, E_pred, I_pred, R_pred, D_pred, Dobs_pred, S_tot, E_tot, I_tot, R_tot, D_tot, Dobs_tot  = nn(x)
        y_pred = torch.cat([S_tot/N_tot, I_tot/N_tot, R_tot/N_tot], dim=1)

    else:
        y_agg = y
        y_pred = nn(x)

    mse_loss = ((y_pred - y_agg) ** 2).sum(axis=0).mean()
    
    # ----- MSE loss -----
    #if isinstance(nn, NN_age):
    #    k_age = N_age.shape[0]
    #    S_train = y[:, :k_age].sum(dim=1, keepdim=True)
    #    I_train = y[:, k_age:2*k_age].sum(dim=1, keepdim=True)
    #    R_train = y[:, 2*k_age:3*k_age].sum(dim=1, keepdim=True)
    #    y_agg = torch.cat([S_train, I_train, R_train], dim=1)
    #    S_pred, I_pred, R_pred, S_tot, I_tot, R_tot = nn(x)
    #    y_pred = torch.cat([S_tot, I_tot, R_tot], dim=1)
    #else:
    #    y_agg = y
    #    y_pred = nn(x)
   
    #mse_loss = ((y_pred - y_agg)**2).sum(axis=0).mean()

    #mse_loss = ((nn(x) - y) ** 2).mean()
    
    tot_loss = alpha['ode'] * ode_loss + alpha['mse'] * mse_loss
    
    return tot_loss, ode_loss, mse_loss


def compute_loss_age_D(nn: NN,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 x_coll: torch.Tensor,
                 C_matrix,
                 N_age,
                 epsilon,
                 xi,
                 ifr,
                 rho,
                 alpha: dict = {'ode': 1.0, 'mse': 1.0, 'D': 100.0},
                 ) -> torch.float:
    """
    Loss = alpha['ode'] * ode_loss
         + alpha['mse'] * mse_loss      (MSE on all the 6 compartments)
         + alpha['D']   * mse_D_loss    (additional MSE on D only)
    """
    ode_loss = ((df(nn, x_coll) - F_age(nn, x_coll, C_matrix, N_age, epsilon, xi, ifr, rho)) ** 2).sum(axis=0).mean()

    if isinstance(nn, NN_age):
        print("Warning: this branch is not maintained, the standard NN is expected.")
        # ... (invariato)
    else:
        y_agg  = y
        y_pred = nn(x)

    k_age = N_age.shape[0]
    D_sl  = slice(4 * k_age, 5 * k_age)

    mse_loss  = ((y_pred - y_agg) ** 2).sum(axis=0).mean()

    mse_D_loss = ((y_pred[:, D_sl] - y_agg[:, D_sl]) ** 2).sum(axis=0).mean()

    tot_loss = alpha['ode'] * ode_loss + alpha['mse'] * mse_loss + alpha['D'] * mse_D_loss

    return tot_loss, ode_loss, mse_loss, mse_D_loss


def F_age(f, x, C, N_age, epsilon, xi, ifr, rho):
    """
    f     : network (standard NN, or NN_age)
    x     : tensor (batch, 3) = [t, R0, Delta], with Delta the mean delay in days (delta = 1/Delta)
    C     : contact matrix (k_age x k_age)
    N_age : population per age class (k_age,)
    epsilon, xi, ifr : constant rates; ifr is a vector with one value per age class
    rho   : spectral radius of C * N_i / N_j (beta = R0 * xi / rho)

    Return: tensor of shape (batch, 6*k_age) with the derivatives dS, dE, dI, dR, dD, dDobs
    """
    out = f(x)
    if isinstance(out, tuple) and len(out) == 12:  # NN_age: tuple of 12 tensors (per-class compartments, then aggregated totals)
        S, E, I, R, D, Dobs = out[:6]
    else:  # NN
        k_age = N_age.shape[0]
        S = out[:, :k_age]
        E = out[:, k_age:2*k_age]
        I = out[:, 2*k_age:3*k_age]
        R = out[:, 3*k_age:4*k_age]
        D = out[:, 4*k_age:5*k_age]
        Dobs = out[:, 5*k_age:6*k_age]

    if isinstance(C, np.ndarray):
        C = torch.tensor(C, dtype=x.dtype, device=x.device)
    
    lambda_ = torch.matmul(I, C.T)

    #C_np = C.cpu().numpy() if isinstance(C, torch.Tensor) else C
    #N_np = N_age.cpu().numpy() if isinstance(N_age, torch.Tensor) else N_age

    #N_ratio = N_np[:, None] / N_np[None, :]   # (k_age, k_age)
    #K_struct = C_np * N_ratio
    #rho = np.max(np.abs(np.linalg.eigvals(K_struct)))  # scalare numpy

    beta  = (x[:, 1] * xi / rho).unsqueeze(1)   # shape (batch, 1)
    delta = (1.0 / x[:, 2]).unsqueeze(1)         # shape (batch, 1)
    
    if isinstance(ifr, np.ndarray):
        ifr = torch.tensor(ifr, dtype=x.dtype, device=x.device)  # shape (k_age,)

    dS = - beta * S * lambda_
    dE = beta * S * lambda_ - epsilon * E
    dI = epsilon * E - xi * I
    dR = xi*(1-ifr) * I
    dD = xi*ifr * I - delta * D
    dDobs = delta * D

    return torch.cat([dS, dE, dI, dR, dD, dDobs], dim=1) # shape: (batch, 3*k_age)

def F_age_convolution(f: NN_age, x, C, N):
    """
    (legacy SIR version)
    f: network returning S, I, R for each age class (output shape: 3*k_age)
    x: tensor of shape (batch, 3), each row is (t, beta, gamma)
    C: contact matrix (k_age x k_age)
    N: population per age class (k_age,)

    Return: tensor of shape (batch, 3*k_age) with dS, dI, dR
    """
    S_value, I_value, R_value, S_tot_value, I_tot_value, R_tot_value = f(x)  # shape: (batch, 3*k_age)
    k_age = N.shape[0]
    F_value = []

    for p in range(x.shape[0]):
        t, beta, gamma = x[p]  
        S = S_value[p,:]
        I = I_value[p,:]
        R = R_value[p,:]

        dS = torch.zeros(k_age)
        dI = torch.zeros(k_age)
        dR = torch.zeros(k_age)

        for i in range(k_age):
            lambda_i = 0.0
            for j in range(k_age):
                lambda_i += C[i, j] * I[j] / N[j]
            dS[i] = - beta * S[i] * lambda_i
            dI[i] = beta * S[i] * lambda_i - gamma * I[i]
            dR[i] = gamma * I[i]

        F_value.append(torch.cat([dS, dI, dR]))

    return torch.stack(F_value)  # shape: (batch, 3*k_age)


def F_age_no_convolution(f: NN, x, C, N):
    """
    (legacy SIR version)
    f: network returning S, I, R for each age class (output shape: 3*k_age)
    x: tensor of shape (batch, 3), each row is (t, beta, gamma)
    C: contact matrix (k_age x k_age)
    N: population per age class (k_age,)

    Return: tensor of shape (batch, 3*k_age) with dS, dI, dR
    """
    f_value = f(x)  # shape: (batch, 3*k_age)
    k_age = N.shape[0]
    F_value = []

    for p in range(x.shape[0]):
        t, beta, gamma = x[p]  
        S = f_value[p, :k_age]
        I = f_value[p, k_age:2*k_age]
        R = f_value[p, 2*k_age:]

        dS = torch.zeros(k_age)
        dI = torch.zeros(k_age)
        dR = torch.zeros(k_age)

        for i in range(k_age):
            lambda_i = 0.0
            for j in range(k_age):
                lambda_i += C[i, j] * I[j] / N[j]
            dS[i] = - beta * S[i] * lambda_i
            dI[i] = beta * S[i] * lambda_i - gamma * I[i]
            dR[i] = gamma * I[i]

        F_value.append(torch.cat([dS, dI, dR]))

    return torch.stack(F_value)  # shape: (batch, 3*k_age)

#%%

#%%
def compute_loss_age_conv_choice(nn: NN_age,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 x_coll: torch.Tensor,
                 C_matrix, 
                 N_age,
                 alpha: np.array = [1, 1]
                 ) -> torch.float:
    """Compute the full loss function as pde loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    # ode_loss = ((df(nn, x_coll) - y_coll) ** 2).sum(axis=0).mean()

    #ode_loss = ((df(nn, x_coll) - F_age_no_convolution(nn, x_coll, C_matrix, N_age)) ** 2).sum(axis=0).mean() #if no_convolution
    ode_loss = ((df_age(nn, x_coll) - F_age_convolution(nn, x_coll, C_matrix, N_age)) ** 2).sum(axis=0).mean()
    #ode_loss = ((df(nn, x_coll) - F(nn, x_coll)) ** 2).mean()
    
    mse_loss = ((nn(x) - y) ** 2).sum(axis=0).mean()
    #mse_loss = ((nn(x) - y) ** 2).mean()
    
    tot_loss = alpha['ode'] * ode_loss + alpha['mse'] * mse_loss
    
    return tot_loss, ode_loss, mse_loss


def F(f: NN, x) -> torch.Tensor:
    f_value = f(x) 
    F_value = []
    for p in range(x.shape[0]):
        # -beta * S * I / N
        # beta * S * I / N - gamma * I
        # gamma * I
        
        # s = S/N, i=I/N, r=R/N -> dS = N * ds, dI = N * di, dR = N * dr
        
        # -beta * s * N * i * N / N = (-beta * s * i) * N
        # beta * S * I / N - gamma * I = (beta * s * i - gamma * i) * N 
        # gamma * I = gamma * i * N
        
        # No more N dependent
               
        F_value.append([-x[p][1].item() * f_value[p][0].item() * f_value[p][1].item(), 
                        x[p][1].item() * f_value[p][0].item() * f_value[p][1].item() - x[p][2].item() * f_value[p][1].item(), 
                        x[p][2].item() * f_value[p][1].item()])
    val = torch.tensor(F_value) 
    return val



#%%
def compute_incidence(S, I, R, times, dt, N, beta, gamma, incidence_type):
    S_diff = np.diff(S) * 1000 / N
    I_diff = np.diff(I) * 1000 / N
    R_diff = np.diff(R) * 1000 / N

    if incidence_type == 'S':
        incidence = [-S_diff[0]]
        for t in times[1:]:
            incidence += [-np.sum(S_diff[t-dt:t])]

    if incidence_type == 'IR':
        IR_diff = I_diff + R_diff
        incidence = [IR_diff[0]]
        for t in times[1:]:
            incidence += [np.sum(IR_diff[t-dt:t])]

    if incidence_type == 'beta':
        SI = S[:-1] * I[:-1] * 1000 / N / N
        incidence = [beta * SI[0]]
        for t in times[1:]:
            incidence += [beta * np.sum(SI[t-dt:t])]

    if incidence_type == 'gamma':
        I_diff_gamma_I = I_diff + gamma * I[:-1] * 1000 / N
        incidence = [I_diff_gamma_I[0]]
        for t in times[1:]:
            incidence += [np.sum(I_diff_gamma_I[t-dt:t])]
       
    incidence = np.array(incidence)    
    return incidence


def compute_incidence_no_zero(S, I, R, times, dt, N, beta, gamma, incidence_type):
    S_diff = np.diff(S) * 1000 / N
    I_diff = np.diff(I) * 1000 / N
    R_diff = np.diff(R) * 1000 / N

    incidence = []
    
    if incidence_type == 'S':
        for t in times:
            incidence += [-np.sum(S_diff[t-dt:t])]

    if incidence_type == 'IR':
        IR_diff = I_diff + R_diff
        for t in times:
            incidence += [np.sum(IR_diff[t-dt:t])]

    if incidence_type == 'beta':
        SI = S[:-1] * I[:-1] * 1000 / N / N
        for t in times:
            incidence += [beta * np.sum(SI[t-dt:t])]

    if incidence_type == 'gamma':
        I_diff_gamma_I = I_diff + gamma * I[:-1] * 1000 / N
        for t in times:
            incidence += [np.sum(I_diff_gamma_I[t-dt:t])]
       
    incidence = np.array(incidence)    
    return incidence


#%%
def surf(x, y, z, ax):
    pts = np.vstack([x, y]).T
    tess = scipy.spatial.Delaunay(pts) # tessilation
    # Create the matplotlib Triangulation object
    tri = tess.simplices # or tess.simplices depending on scipy version
    triDat = mtri.Triangulation(x=pts[:, 0], y=pts[:, 1], triangles=tri)
    # ax.plot_trisurf(x, y, z,
    # cmap=cm.Spectral)
    ax.plot_trisurf(triDat, z, cmap=cm.Spectral)
    
    
#%%
def inc(S, times, dt):
    S_diff = np.diff(S) * 1000
    incidence = []
    for t in times:
        incidence += [-np.sum(S_diff[t-dt:t])]
    incidence = np.array(incidence)
    return incidence

def cost(param, t, dt, y_inc, model):
    # param = [beta, gamma, t0]
    n_times = len(t)
    input_eval = torch.tensor(np.hstack([t[:, None], np.tile(param[:2], (n_times, 1))]), dtype=torch.float64) 
    with torch.no_grad():
        model_eval = model(input_eval).numpy()
        
    t_inc = t[0:590:dt].astype('i')
        
    t_inc, y_inc = cut_times(t_inc, y_inc, param[2])
    
    incidence = inc(model_eval[:, 0], t_inc, dt)    
    
    val = np.sum((incidence - y_inc) ** 2) / len(t_inc)
    return val


def cost_exact(param, t, dt, y_inc, data_new):
    S, _, _ = data_new.SIR_components(param[0], param[1])
    
    t_inc = t[0:590:dt].astype('i')
    
    t_inc, y_inc = cut_times(t_inc, y_inc, param[2])
    
    incidence = inc(S / 1e6, t_inc, dt)    
        
    val = np.sum((incidence - y_inc) ** 2) / len(t_inc)
    return val


def cut_times(t_inc, y_inc, t0_guess):
    t_inc = t_inc[t_inc>=t0_guess] # t[0:590:dt] = [0,7,14,..] but only the ones > t0_guess. E.g., [77.,  84.,  91.,...] if t0_guess = 75
    min_len = np.min([len(t_inc), len(y_inc)])
    t_inc = t_inc[:min_len]
    y_inc = y_inc[:min_len] # the original one, not cutted, just shortened in the lenght to be the same as t_inc
    return t_inc, y_inc


def cut_times_len(t_inc, w, t0_guess):
    t_inc = t_inc[t_inc>=t0_guess]
    min_len = np.min([len(t_inc), w])
    t_inc = t_inc[:min_len]
    return t_inc


def params_transformation(max_t0, vector, inverse=False):  
    #M_0 = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]]) # old_one
    M_0_new =(2**(1/2)/2)*np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2/(2**(1/2))]]) # new and correct one
    
    #f or model a, b, d and e (check)
    #M = np.diag([1/0.7, 1/0.08, 1/max_t0]) @ M_0 # old one
    M = np.diag([1/0.6, 1/0.11, 1/max_t0]) @ M_0_new # M_new: new and correct one
    #M = np.diag([1/0.7, 1/0.16, 1/max_t0]) @ M_0 # M_new_2: correct D and old one
    
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]]) @ np.diag([0.7, 0.08, max_t0]) # M_inv_old: old wrong one
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.08, max_t0]) # M_inv: old one with t_max = 300
    M_inv = (1/(2**(1/2))) *  np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2**(1/2)]]) @ np.diag([0.6, 0.11, max_t0]) # M_inv_new: inv of M_new
    #M_inv =  0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.16, max_t0]) # M_inv_new_2: inv of M_new_2

    # for model c and f, g (check)
    #M = np.diag([1/0.7, 1/0.06, 1/max_t0]) @ M_0 # old one
    #M = np.diag([1/0.5, 1/0.04, 1/max_t0]) @ M_0_new # M_new: new and correct one
    
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.06, max_t0]) # M_inv: old one with t_max = 300
    #M_inv = (1/(2**(1/2))) *  np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2**(1/2)]]) @ np.diag([0.5, 0.04, max_t0]) # M_inv_new: inv of M_new
    
    if inverse == False:
        
        #f or model a, b, d and e (check)
        #lb=[0.2/0.7, -1, 0]
        #ub=[0.9/0.7, -0.2, 1]
        lb=[0.20, -0.5, 0] # lb_new
        ub=[1.05, -0.1, 1] # ub_new
        #lb=[0.25, -0.5, 0] # lb_new_2
        #ub=[1.25, -0.1, 1] # ub_new_2
        # for model c and f, g (check)
        #lb=[0.2, -1.35, 0]
        #ub=[1.25, -0.2, 1]
        
        return M@vector,lb,ub
    
    else:
        
        lb = [0.12, 1/12, 0]
        ub = [0.45, 1/2.5, max_t0]
        
        return M_inv@vector,lb,ub

#%% 
def estimate_pinn_powell(x0, times, dt, incidence, max_t0, model):
    options = {'xtol': 1e-12}
    method = 'powell'
    
    x0_tranf,lb,ub = params_transformation(max_t0, x0, inverse=False)
    
    bounds = Bounds(lb=lb, ub=ub, keep_feasible=[True, True, True])
    
    fun = lambda x: cost(params_transformation(max_t0, x, inverse=True)[0], times, dt, incidence, model)
    
    res = minimize(fun=fun, 
                   x0=x0_tranf,
                   method=method,
                   bounds=bounds,
                   options=options)
    
    x = params_transformation(max_t0, res.x, inverse=True)[0]
    return x


#%%
def estimate_sir_powell(x0, times, dt, incidence, max_t0, data_new):
    options = {'xtol': 1e-12}
    method = 'powell'
    
    x0_tranf,lb,ub = params_transformation(max_t0, x0, inverse=False)
        
    bounds = Bounds(lb=lb, ub=ub, keep_feasible=[True, True, True])
    
    fun_exact = lambda x: cost_exact(params_transformation(max_t0, x, inverse=True)[0], times, dt, incidence, data_new)    
    
    res_exact = minimize(fun=fun_exact, 
                   x0=x0_tranf,
                   method=method,
                   bounds=bounds,
                   options=options)

    x = params_transformation(max_t0, res_exact.x, inverse=True)[0]
    return x

def estimate_pinn_old(times, dt, incidence, max_t0, model):
    x0 = [0.2, 0.15, int(max_t0/2)]
    #x0 = [0.3, 0.25, int(max_t0/2)]
    options = {'xtol': 1e-12}
    method = 'powell'
    
    M_0 = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]]) # old_one
    M_0_new =(2**(1/2)/2)*np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2/(2**(1/2))]]) # new and correct one
    
    #f or model a, b, d and e (check)
    #M = np.diag([1/0.7, 1/0.08, 1/max_t0]) @ M_0 # old one
    M = np.diag([1/0.6, 1/0.11, 1/max_t0]) @ M_0_new # M_new: new and correct one
    #M = np.diag([1/0.7, 1/0.16, 1/max_t0]) @ M_0 # M_new_2: correct D and old one
    
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]]) @ np.diag([0.7, 0.08, max_t0]) # M_inv_old: old wrong one
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.08, max_t0]) # M_inv: old one with t_max = 300
    M_inv = (1/(2**(1/2))) *  np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2**(1/2)]]) @ np.diag([0.6, 0.11, max_t0]) # M_inv_new: inv of M_new
    #M_inv =  0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.16, max_t0]) # M_inv_new_2: inv of M_new_2

    #lb=[0.2/0.7, -1, 0]
    #ub=[0.9/0.7, -0.2, 1]
    lb=[0.20, -0.5, 0] # lb_new
    ub=[1.05, -0.1, 1] # ub_new
    #lb=[0.25, -0.5, 0] # lb_new_2
    #ub=[1.25, -0.1, 1] # ub_new_2
    
    # for model c and f, g (check)
    #M = np.diag([1/0.7, 1/0.06, 1/max_t0]) @ M_0 # old one
    #M = np.diag([1/0.5, 1/0.04, 1/max_t0]) @ M_0_new # M_new: new and correct one
    
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.06, max_t0]) # M_inv: old one with t_max = 300
    #M_inv = (1/(2**(1/2))) *  np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2**(1/2)]]) @ np.diag([0.5, 0.04, max_t0]) # M_inv_new: inv of M_new
    
    #lb=[0.2, -1.35, 0]
    #ub=[1.25, -0.2, 1]
    
    bounds = Bounds(lb=lb, ub=ub, keep_feasible=[True, True, True])
    
    fun = lambda x: cost(M_inv @ x, times, dt, incidence, model)
    
    res = minimize(fun=fun, 
                   x0=M@x0,
                   method=method,
                   bounds=bounds,
                   options=options)
    
    x = M_inv @ res.x
    return x


#%%
def estimate_sir_old(times, dt, incidence, max_t0, data_new):
    x0 = [0.2, 0.15, int(max_t0/2)]
    #x0 = [0.3, 0.25, int(max_t0/2)]
    options = {'xtol': 1e-12}
    method = 'powell'
    
    M_0 = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]]) # old_one
    M_0_new =(2**(1/2)/2)*np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2/(2**(1/2))]]) # new and correct one
    
    #f or model a, b, d and e (check)
    #M = np.diag([1/0.7, 1/0.08, 1/max_t0]) @ M_0 # old one
    M = np.diag([1/0.6, 1/0.11, 1/max_t0]) @ M_0_new # M_new: new and correct one
    #M = np.diag([1/0.7, 1/0.16, 1/max_t0]) @ M_0 # M_new_2: correct D and old one
    
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]]) @ np.diag([0.7, 0.08, max_t0]) # M_inv_old: old wrong one
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.08, max_t0]) # M_inv: old one with t_max = 300
    M_inv = (1/(2**(1/2))) *  np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2**(1/2)]]) @ np.diag([0.6, 0.11, max_t0]) # M_inv_new: inv of M_new
    #M_inv =  0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.16, max_t0]) # M_inv_new_2: inv of M_new_2

    #lb=[0.2/0.7, -1, 0]
    #ub=[0.9/0.7, -0.2, 1]
    lb=[0.20, -0.5, 0] # lb_new
    ub=[1.05, -0.1, 1] # ub_new
    #lb=[0.25, -0.5, 0] # lb_new_2
    #ub=[1.25, -0.1, 1] # ub_new_2
    
    # for model c and f, g (check)
    #M = np.diag([1/0.7, 1/0.06, 1/max_t0]) @ M_0 # old one
    #M = np.diag([1/0.5, 1/0.04, 1/max_t0]) @ M_0_new # M_new: new and correct one
    
    #M_inv = 0.5 * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) @ np.diag([0.7, 0.06, max_t0]) # M_inv: old one with t_max = 300
    #M_inv = (1/(2**(1/2))) *  np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2**(1/2)]]) @ np.diag([0.5, 0.04, max_t0]) # M_inv_new: inv of M_new
    
    #lb=[0.2, -1.35, 0]
    #ub=[1.25, -0.2, 1]
     
    bounds = Bounds(lb=lb, ub=ub, keep_feasible=[True, True, True])
    
    fun_exact = lambda x: cost_exact(M_inv @ x, times, dt, incidence, data_new)
    
    
    res_exact = minimize(fun=fun_exact, 
                   x0=M@x0,
                   method=method,
                   bounds=bounds,
                   options=options)

    x = M_inv @ res_exact.x
    return x

#%%
def load_influcast_data(data_path, season='2023-2024'):
    data = pd.read_csv(data_path)
    week = data['year_week'].apply(lambda x: int(x.split('-')[1]))
    week[week < 42] = week[week < 42] + 52     
    data['week'] = week
    
    week = data[data['flu_season']==season]['week'].values
    inc = data[data['flu_season']==season]['incidence'].values
    return week, inc

#%%
def forecast_pinn(x, model, N_age, times, dt, window):
    # x = [beta, gamma, t0]
    input_eval = torch.tensor(np.hstack([times[:, None], np.tile([x[0], x[1]], (len(times), 1))]), dtype=torch.float64) 
    with torch.no_grad():
        model_eval = model(input_eval).numpy()
    
    t_inc = times[0:590:dt].astype('i')
    t_inc = cut_times_len(t_inc, window, x[2])

    N_tot = N_age.sum()
    k_age = len(N_age)
    model_eval_age = (model_eval[:, :k_age]* N_age).sum(axis=1) / N_tot
    
    incidence_estimated = inc(model_eval_age, t_inc, dt)
    return incidence_estimated

def forecast_TV(time_plot_x,chain_samples,model,N_age,times,dt,window_forecast):
    forecast = {'pinn': [], 'sir': [], 't': []}
    forecast['t'] = time_plot_x
    forecast['pinn'] = forecast_pinn(chain_samples[0], model, N_age, times, dt, len(window_forecast))

    for params in chain_samples[1:]: # [::10]
        forecast['t'] = np.concatenate((forecast['t'], time_plot_x), axis=0)
        forecast['pinn'] = np.concatenate((forecast['pinn'], forecast_pinn(params, model, N_age, times, dt, len(window_forecast))), axis=0)
        #forecast['sir'] += [forecast_sir(x_sir[idx], sir_model, times, dt, len(window_forecast))]

    T = np.vstack(forecast['t'])[:, 0]
    V = np.vstack(forecast['pinn'])[:, 0]
    #x_pinn = np.stack(x_pinn)
    #x_sir = np.stack(x_sir)
    
    return T,V

def forecast_sir(x, model, times, dt, window):
    S, _, _ = model.SIR_components(x[0], x[1])
    
    t_inc = times[0:590:dt].astype('i')
    t_inc = cut_times_len(t_inc, window, x[2])
       
    incidence_estimated = inc(S / 1e6, t_inc, dt)
    return incidence_estimated


def forecast_gillespie(x, model, sir_model, times):
    import gillespie
    
    S, I, R = sir_model.SIR_components(x[0], x[1])
    
    t0 = int(x[2])
    y0 = [S[t0], I[t0], R[t0]]
    
    T = np.max(times) - t0

    propensities = [lambda s, i, r: -x[0] * s * i / model.N,
                    lambda s, i, r: x[1] * i] 

    stoichiometry = [[-1, 1, 0],  # S -> I, Population change: S-1, I+1, R+0
                      [0, -1, 1]]  # I -> R Population change: S+0, I-1, R+1

    t, SIR = gillespie.simulate(y0, propensities, stoichiometry, T)

    S, I, R = zip(*SIR)


#%%
def downsample(t, y, dt=7):
    min_t = np.floor(np.min(t))
    max_t = np.ceil(np.max(t))
    tt = np.arange(min_t, max_t, dt)
    yy = np.zeros((len(tt), 3))
    for idx in range(3):
        try:
            yy[:, idx] = np.interp(tt, t, y[:, idx])
        except:
            breakpoint()
    return tt, yy


def absolute_error(data_inc,model_inc):
    abs_err=0
    for idx_inc in range(len(data_inc)):
        abs_err = abs_err + abs(model_inc[idx_inc]-data_inc[idx_inc])
    abs_err = abs_err/(len(data_inc))
    return abs_err

def coverage_error(data_inc,model_all):
    cov_50=0
    cov_90=0
    for idx_inc in range(len(data_inc)):
        if (data_inc[idx_inc] >= model_all['q25'][idx_inc]) & (data_inc[idx_inc] <= model_all['q75'][idx_inc]):
            cov_50 = cov_50+1
        if (data_inc[idx_inc] >= model_all['q05'][idx_inc]) & (data_inc[idx_inc] <= model_all['q95'][idx_inc]):
            cov_90 = cov_90+1
    cov_50 = cov_50/(len(data_inc))
    cov_90 = cov_90/(len(data_inc))
    return cov_50,cov_90

def wis_error(data_inc,model_all):                                                              
    alpha_list = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    quantile_bounds = {
        0.02: ('q01',  'q99'),
        0.05: ('q025', 'q975'),
        0.10: ('q05',  'q95'),
        0.20: ('q10',  'q90'),
        0.30: ('q20',  'q80'),
        0.40: ('q25',  'q75'),
        0.50: ('q30',  'q70'),
        0.60: ('q35',  'q65'),
        0.70: ('q15',  'q85'),
        0.80: ('q10',  'q90'),
        0.90: ('q05',  'q95'),
    }
    
    wis=0
    for idx_inc in range(len(data_inc)):                                                     
        wis_components = []

        for alpha in alpha_list:
            lower, upper = quantile_bounds[alpha]
            
            w = alpha / 2

            IS = (
                (model_all[upper][idx_inc] - model_all[lower][idx_inc]) +
                (2 / alpha) * np.clip(model_all[lower][idx_inc] - data_inc[idx_inc], a_min=0, a_max=None) +
                (2 / alpha) * np.clip(data_inc[idx_inc] - model_all[upper][idx_inc], a_min=0, a_max=None)
            )

            wis_components.append(w * IS)

        w_0 = 0.5
        abs_error_median = abs(model_all['q50'][idx_inc] - data_inc[idx_inc])
        wis_components.append(w_0 * abs_error_median)

        K = len(alpha_list)
        wis=wis+sum(wis_components)/(K + 0.5)
    wis = wis/(len(data_inc))
    return wis