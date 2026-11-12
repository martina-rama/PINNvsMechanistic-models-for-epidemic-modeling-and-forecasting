import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from scipy.integrate import odeint
from tqdm.auto import tqdm
from numpy.linalg import eigvals

class Data_SEIRDDobs_age:
    def __init__(self, N_age, E0_age, I0_age, R0_age, D0_age, D0obs_age, C, domain): # no more N
        """
        Age-structured SEIRDDobs model (compartments S, E, I, R, D, Dobs).
        N_age: array-like, population per age class (len = k_age)
        E0_age, I0_age, R0_age, D0_age, D0obs_age: array-like, initial counts per age class
        C: contact matrix (k_age x k_age)
        domain: tuple (t0, t_end), in days
        """
        self.N_age = np.array(N_age)
        self.E0_age = np.array(E0_age)
        self.I0_age = np.array(I0_age)
        self.R0_age = np.array(R0_age)
        self.D0_age = np.array(D0_age)
        self.D0obs_age = np.array(D0obs_age)
        #self.S0_age = self.N_age - self.E0_age - self.I0_age - self.R0_age - self.D0_age - self.D0obs_age
        self.S0_age = 1 - self.E0_age/self.N_age - self.I0_age/self.N_age - self.R0_age/self.N_age - self.D0_age/self.N_age - self.D0obs_age/self.N_age
        self.C = np.array(C)
        self.k_age = len(N_age)
        self.domain = domain
        
        #self.ft0 = torch.tensor([self.S0, self.I0, self.R0])
        #self.y0 = self.S0, self.I0, self.R0

        # time grid
        self.t_SEIRDDobs = np.linspace(int(self.domain[0]),
                                 int(self.domain[1]),
                                 int(self.domain[1]) - int(self.domain[0]) + 1)

        # concatenated initial state [S, E, I, R, D, Dobs], each of length k_age
        self.y0 = np.concatenate([self.S0_age, self.E0_age/self.N_age, self.I0_age/self.N_age, self.R0_age/self.N_age, self.D0_age/self.N_age, self.D0obs_age/self.N_age])

    def deriv_age(self, y, t, beta, epsilon, xi, ifr, delta):
        """
        Right-hand side of the age-structured SEIRDDobs ODE system.
        """
        k = self.k_age
        S = y[:k]
        E = y[k:2*k]
        I = y[2*k:3*k]
        R = y[3*k:4*k]
        D = y[4*k:5*k]
        Dobs = y[5*k:]

        lambda_i = np.zeros(k)
        for i in range(k):
            for j in range(k):
                lambda_i[i] += self.C[i, j] * I[j] # / self.N_age[j]

        dSdt = -beta * S * lambda_i
        dEdt = beta * S * lambda_i - epsilon * E
        dIdt = epsilon * E - xi * I
        dRdt = xi*(1-ifr) * I
        dDdt = xi*ifr * I - delta * D
        dDobsdt = delta * D

        #return dSdt, dIdt, dRdt
        return np.concatenate([dSdt, dEdt, dIdt, dRdt, dDdt, dDobsdt])


    def SEIRDDobs_components(self, beta, epsilon, xi, ifr, delta, vector=False):
        """
        Integrates the system for the given parameters.
        Returns S, E, I, R, D, Dobs, each of shape (T, k_age), as fractions of N_age
        (or the full solution matrix, shape (T, 6*k_age), if vector=True).
        """
        ret = odeint(self.deriv_age, self.y0, self.t_SEIRDDobs, args=(beta, epsilon, xi, ifr, delta))
        k = self.k_age

        S    = ret[:, :k]
        E    = ret[:, k:2*k]
        I    = ret[:, 2*k:3*k]
        R    = ret[:, 3*k:4*k]
        D    = ret[:, 4*k:5*k]
        Dobs = ret[:, 5*k:6*k]
        if vector:
            return ret
        else:
            return S, E, I, R, D, Dobs
        #S_tot = S.sum(axis=1)
        #I_tot = I.sum(axis=1)
        #R_tot = R.sum(axis=1)
        #return S_tot, I_tot, R_tot

    # A function for all the collocation points generation
    def generate_t_points(self,n_points, bounds=True, grad=False):
        if bounds == True:
            collocation_points = torch.linspace(self.domain[0], self.domain[1], 
                                                steps=n_points, 
                                                requires_grad=grad).reshape(-1, 1) 
        else:
            collocation_points = torch.linspace(self.domain[0]+1, self.domain[1]-1, 
                                                steps=n_points, 
                                                requires_grad=grad).reshape(-1, 1)
        return collocation_points #t        
       
    def data_y_no_noise(self, t, beta, epsilon, xi, ifr, delta): 
            S, E, I, R, D, Dobs = self.SEIRDDobs_components(beta, epsilon, xi, ifr, delta, vector=False)
           
            #old code:
            #y = np.concatenate([S[t.astype(int)], I[t.astype(int)], R[t.astype(int)]], axis=1)
            #N_vec = np.tile(self.N_age, 3)  # [N1..Nk, N1..Nk, N1..Nk]
            #y = y / N_vec # y data without noise (len(t), 3*k_age)



            t_idx = t.astype(int)
            S_t, E_t, I_t, R_t, D_t, Dobs_t = S[t_idx, :], E[t_idx, :], I[t_idx, :], R[t_idx, :], D[t_idx, :], Dobs[t_idx, :]

            #N_tot = self.N_age.sum()
            #N_vec = np.tile(self.N_age, 3)
            S_t_norm = S_t #/ self.N_age
            E_t_norm = E_t #/ self.N_age
            I_t_norm = I_t #/ self.N_age
            R_t_norm = R_t #/ self.N_age
            D_t_norm = D_t #/ self.N_age
            Dobs_t_norm = Dobs_t #/ self.N_age


            # Concatenazione: (T, 3*k_age)
            y = np.concatenate([S_t_norm, E_t_norm, I_t_norm, R_t_norm, D_t_norm, Dobs_t_norm], axis=1)
            return y

    
    def train_data_constrains(self, t_train, p_train, epsilon, xi, ifr, noise=True):
        # Input
        n_p = len(p_train)
        n_t = len(t_train)
        t_rep = np.tile(t_train, n_p) # shape (n_t * n_p,)
        p_rep = np.repeat(p_train, n_t, axis=0) # shape (n_t * n_p, 2)
        x_train = np.column_stack([t_rep, p_rep])  # shape (n_t * n_p, 3)
        
        # Output
        y = []
        for p in tqdm(p_train, desc="Generating training data"):
            beta = self.beta_from_R0(p[0], xi, self.C, self.N_age)
            Delta = 1/p[1]
            if noise == True:
                y += [self.data_y_noise(t_train, beta, epsilon, xi, ifr, Delta)]
            else:
                y += [self.data_y_no_noise(t_train, beta, epsilon, xi, ifr, Delta)]
        y_train = np.vstack(y)
        
        return x_train, y_train
    
    
    def R0_from_beta(self, beta, xi, C, N_age):
        """
        Compute R0 for the age-structured SEIRD model (largest eigenvalue of the next-generation matrix).
        beta  : transmission rate (scalar)
        xi    : removal rate (scalar)
        C     : contact matrix (k x k)
        N_age : population per age class (k,)
        """
        k = len(N_age)
        # NGM: K[i,j] = (beta / xi) * (N_i / N_j) * C[i,j]
        N_ratio = N_age[:, None] / N_age[None, :]  # shape (k, k)
        K = (beta / xi) * C * N_ratio
        
        R0 = np.max(np.abs(np.linalg.eigvals(K)))
        return R0

    def beta_from_R0(self, R0_target, xi, C, N_age):
        """
        Compute beta for a given R0 in the age-structured SEIRD model.

        R0_target : desired value of R0 (scalar)
        xi        : removal rate (scalar)
        C         : contact matrix (k x k)
        N_age     : population per age class (k,)
        """
        N_ratio = N_age[:, None] / N_age[None, :]  # D_N C D_N^{-1}
        K_struct = C * N_ratio                     # shape (k, k)
        
        rho = np.max(np.abs(np.linalg.eigvals(K_struct)))
        
        beta = (R0_target * xi) / rho
        return beta  
        
        
    
    
    
    
    
    
    
    
    