import numpy as np
import squidpy as sq
from numba import njit, prange

"""
Modified library to train STHD model for CODEX Spatial Proteomics.
Modifications include:
1. Gaussian Continuous Log-Likelihood (replacing Poisson count model).
2. Distance-Weighted spatial cross-entropy penalty for irregular cell graphs.
"""

# CODEX MODIFICATION: Gaussian Continuous Log-Likelihood
@njit(parallel=True, fastmath=True)
def fill_F(X, Y, Z, N, Mu, Var, F):
    for a in prange(X):
        for t in range(Z):
            for g in range(Y):
                diff = N[a, g] - Mu[t, g]
                # CODEX FIX: Divide by variance to prevent noisy markers from exploding the loss
                F[a, t] = F[a, t] - 0.5 * (diff * diff) / Var[g]


# def prepare_constants(sthd_data):
#     X = sthd_data.adata.obs.shape[0]  # n of segmented cells
#     Y = sthd_data.adata.shape[1]  # n of protein markers
#     Z = sthd_data.lambda_cell_type_by_gene_matrix.shape[0]  # n of cell types

#     # get raw continuous data
#     N = sthd_data.adata.to_df().values  # [X,Y] continuous intensity per cell
#     Mu = sthd_data.lambda_cell_type_by_gene_matrix.astype("float32")  # [Z,Y] mean intensity profile

#     # CODEX MODIFICATION: Build an irregular distance-weighted spatial graph
#     sq.gr.spatial_neighbors(
#         sthd_data.adata, spatial_key="spatial", coord_type="generic", n_neighs=6
#     )
#     A_csr = sthd_data.adata.obsp["spatial_connectivities"]  # contains distance weights
#     print("Currently we only support symmetric adjacency matrix of neighbors")
#     Acsr_row = A_csr.indptr
#     Acsr_col = A_csr.indices
#     Acsr_data = A_csr.data  # Extract edge weights for the spatial penalty

#     # Calculate variance for each protein marker (add 1e-6 to prevent division by zero)
#     Var = np.var(N, axis=0).astype("float32") + 1e-6

#     F = np.zeros([X, Z], dtype="float32")
#     fill_F(X, Y, Z, N, Mu, Var, F)
#     return X, Y, Z, F, Acsr_row, Acsr_col, Acsr_data


def prepare_constants(sthd_data, K=10): # Add K default
    X = sthd_data.adata.obs.shape[0]  
    Y = sthd_data.adata.shape[1]  
    Z = sthd_data.lambda_cell_type_by_gene_matrix.shape[0]  

    N = sthd_data.adata.to_df().values  
    Mu = sthd_data.lambda_cell_type_by_gene_matrix.astype("float32")  

    sq.gr.spatial_neighbors(
        sthd_data.adata, spatial_key="spatial", coord_type="generic", delaunay=True
    )
    A_csr = sthd_data.adata.obsp["spatial_connectivities"]  
    Acsr_row = A_csr.indptr
    Acsr_col = A_csr.indices
    Acsr_data = A_csr.data  

    Var = np.var(N, axis=0).astype("float32") + 1e-6
    F = np.zeros([X, Z], dtype="float32")
    fill_F(X, Y, Z, N, Mu, Var, F)
    
    return X, Y, Z, K, F, Acsr_row, Acsr_col, Acsr_data # Return K


# def prepare_training_weights(X, Y, Z):
#     # prepare training parameters
#     W = np.ones([X, Z]).astype("float32")  # initialization
#     # prepare derived parameters
#     eW = np.zeros([X, Z], dtype="float32")
#     P = np.zeros([X, Z], dtype="float32")
#     Phi = np.zeros([X], dtype="float32")
#     # prepare deriatives
#     ll_wat = np.zeros([X, Z], dtype="float32")
#     ce_wat = np.zeros([X, Z], dtype="float32")
#     # additional variables for adam
#     m = np.zeros([X, Z], dtype="float32")
#     v = np.zeros([X, Z], dtype="float32")
#     return W, eW, P, Phi, ll_wat, ce_wat, m, v


def prepare_training_weights(X, Y, Z, K):
    # Spatial Niche Weights
    W_niche = np.zeros([X, K], dtype="float32")
    eW = np.zeros([X, K], dtype="float32")
    P_niche = np.zeros([X, K], dtype="float32")
    Phi = np.zeros([X], dtype="float32")
    ll_wat = np.zeros([X, K], dtype="float32")
    ce_wat = np.zeros([X, K], dtype="float32")
    m = np.zeros([X, K], dtype="float32")
    v = np.zeros([X, K], dtype="float32")
    
    # NEW: Biological Composition Weights (Theta)
    # V = np.zeros([K, Z], dtype="float32") # Unnormalized Theta
    # Initialize with random noise to break perfect symmetry!
    np.random.seed(42) 
    V = (np.random.randn(K, Z) * 0.1).astype("float32")
    Theta = np.ones([K, Z], dtype="float32") / Z
    ll_wat_V = np.zeros([K, Z], dtype="float32")
    m_V = np.zeros([K, Z], dtype="float32")
    v_V = np.zeros([K, Z], dtype="float32")
    
    return W_niche, eW, P_niche, Phi, ll_wat, ce_wat, m, v, V, Theta, ll_wat_V, m_V, v_V

def early_stop_criteria_2(metrics, beta, n=10, threshold=0.01):
    if len(metrics) < n:
        return False
    else:
        metrics_check = metrics[-n:]
        loss = [-i[0] + beta * i[1] for i in metrics_check]
        loss_range = max(loss) - min(loss)
        loss_ratio = loss_range / np.abs(loss[-1])
        if loss_ratio < threshold:
            return True
        else:
            return False


# def train(
#     n_iter,
#     step_size,
#     beta,
#     constants,
#     weights,
#     early_stop=False,
#     beta1=0.9,
#     beta2=0.999,
#     epsilon=1e-8,
# ):
#     X, Y, Z, F, Acsr_row, Acsr_col, Acsr_data = constants
#     W, eW, P, Phi, ll_wat, ce_wat, m, v = weights

#     metrics = []
#     for i in range(n_iter):
#         update_eW(eW, W, X, Y, Z)
#         update_Phi(Phi, eW, X, Y, Z)
#         update_P(P, eW, Phi, X, Y, Z)
#         update_ll_wat(ll_wat, P, F, X, Y, Z)
#         update_ce_wat(ce_wat, P, Acsr_row, Acsr_col, Acsr_data, X, Y, Z)
#         update_m_v(m, v, beta1, beta2, beta, ll_wat, ce_wat, X, Y, Z)
#         update_W_adam(W, m, v, beta1, beta2, i + 1, step_size, epsilon, X, Y, Z)
#         ll = calculate_ll(P, F, X, Y, Z)
#         ce = calculate_ce(P, Acsr_row, Acsr_col, Acsr_data, X, Y, Z)
#         metrics.append((ll, ce))
#         print(i, -ll + beta * ce, ll, ce)

#         if early_stop and early_stop_criteria_2(metrics, beta):
#             return metrics

#     return metrics
def train(n_iter, step_size, beta, constants, weights, early_stop=False, beta1=0.9, beta2=0.999, epsilon=1e-8):
    X, Y, Z, K, F, Acsr_row, Acsr_col, Acsr_data = constants
    W_niche, eW, P_niche, Phi, ll_wat, ce_wat, m, v, V, Theta, ll_wat_V, m_V, v_V = weights

    metrics = []
    for i in range(n_iter):
        # 1. Forward Pass (Calculate Probabilities)
        update_eW(eW, W_niche, X, K, K) # Passing K for Z argument slot
        update_Phi(Phi, eW, X, K, K)
        update_P(P_niche, eW, Phi, X, K, K)
        update_Theta(Theta, V, K, Z)
        
        # 2. Backward Pass (Calculate Gradients)
        update_ll_wat(ll_wat, P_niche, Theta, F, X, K, Z)
        update_ce_wat(ce_wat, P_niche, Acsr_row, Acsr_col, Acsr_data, X, K)
        update_ll_wat_V(ll_wat_V, P_niche, Theta, F, X, K, Z)
        
        # 3. Optimize (Step the weights)
        update_m_v(m, v, beta1, beta2, beta, ll_wat, ce_wat, X, K)
        update_W_adam(W_niche, m, v, beta1, beta2, i + 1, step_size, epsilon, X, K)
        
        # Optimize the Biological Niche Compositions at a slower learning rate for stability
        update_V_adam(V, m_V, v_V, beta1, beta2, i + 1, step_size * 0.1, epsilon, K, Z, ll_wat_V)
        
        # 4. Score
        ll = calculate_ll(P_niche, Theta, F, X, K, Z)
        ce = calculate_ce(P_niche, Acsr_row, Acsr_col, Acsr_data, X, K)
        metrics.append((ll, ce))
        print(f"Iter {i} | Total Loss: {-ll + beta * ce:.4f} | LL: {ll:.4f} | CE: {ce:.4f}")

    return metrics, P_niche, Theta

# ----------------------- update trainable parameters -------------------

@njit(parallel=True, fastmath=True)
def update_Theta(Theta, V, K, Z):
    for k in prange(K):
        # Subtract max for mathematical stability against exploding exponentials
        max_V = np.max(V[k, :]) 
        sum_e = 0.0
        for t in range(Z):
            Theta[k, t] = np.exp(V[k, t] - max_V)
            sum_e += Theta[k, t]
        for t in range(Z):
            Theta[k, t] /= sum_e

@njit(parallel=True, fastmath=True)
def update_eW(eW, W, X, Y, Z):
    for a in prange(X):
        for t in range(Z):
            eW[a, t] = np.exp(W[a, t])


@njit(parallel=True, fastmath=True)
def update_Phi(Phi, eW, X, Y, Z):
    for a in prange(X):
        Phi[a] = 0
        for t in range(Z):
            Phi[a] = Phi[a] + eW[a, t]


@njit(parallel=True, fastmath=True)
def update_P(P, eW, Phi, X, Y, Z):
    for a in prange(X):
        for t in range(Z):
            P[a, t] = eW[a, t] / Phi[a]


# ----------------------- calculate losses -------------------
# CODEX MODIFICATION: Extract spatial weights alongside indices
@njit
def csr_obtain_column_index_and_data_for_row(row, column, data, i):
    row_start = row[i]
    row_end = row[i + 1]
    return column[row_start:row_end], data[row_start:row_end]


# @njit
# def calculate_ce(P, Acsr_row, Acsr_col, Acsr_data, X, Y, Z):
#     res = 0
#     for a in range(X):
#         neighbors, weights = csr_obtain_column_index_and_data_for_row(Acsr_row, Acsr_col, Acsr_data, a)
#         for t in range(Z):
#             cur = 0
#             for k, a_star in enumerate(neighbors):
#                 # CODEX MODIFICATION: Multiply cross-entropy by edge weight
#                 cur = cur + weights[k] * np.log(P[a_star, t])
#             res = res - P[a, t] * cur
#     res = res / X
#     return res


# @njit(parallel=True, fastmath=True)
# def calculate_ll(P, F, X, Y, Z):
#     res = 0
#     for a in prange(X):
#         for t in range(Z):
#             res = res + P[a, t] * F[a, t]
#     res = res / X
#     return res



@njit
def calculate_ce(P_niche, Acsr_row, Acsr_col, Acsr_data, X, K):
    res = 0
    for a in range(X):
        neighbors, weights = csr_obtain_column_index_and_data_for_row(Acsr_row, Acsr_col, Acsr_data, a)
        for k in range(K):
            cur = 0
            for i, a_star in enumerate(neighbors):
                cur = cur + weights[i] * np.log(P_niche[a_star, k])
            res = res - P_niche[a, k] * cur
    res = res / X
    return res

@njit(parallel=True, fastmath=True)
def calculate_ll(P_niche, Theta, F, X, K, Z):
    res = 0
    for a in prange(X):
        for t in range(Z):
            prob_t = 0
            for k in range(K):
                prob_t = prob_t + P_niche[a, k] * Theta[k, t]
            res = res + prob_t * F[a, t]
    res = res / X
    return res

# ----------------------- calculate gradients -------------------
# @njit(parallel=True, fastmath=True)
# def update_ll_wat(ll_wat, P, F, X, Y, Z):
#     for a_tilda in prange(X):
#         for t_tilda in range(Z):
#             cur = 0
#             for t in range(Z):
#                 cur = cur + P[a_tilda, t] * P[a_tilda, t_tilda] * F[a_tilda, t]
#             ll_wat[a_tilda, t_tilda] = (
#                 -1 * cur + P[a_tilda, t_tilda] * F[a_tilda, t_tilda]
#             )
#             ll_wat[a_tilda, t_tilda] = ll_wat[a_tilda, t_tilda] / X


# @njit(parallel=True, fastmath=True)
# def update_ce_wat(ce_wat, P, Acsr_row, Acsr_col, Acsr_data, X, Y, Z):
#     for a_tilda in prange(X):
#         neighbors, weights = csr_obtain_column_index_and_data_for_row(Acsr_row, Acsr_col, Acsr_data, a_tilda)
#         for t_tilda in range(Z):
#             comp1 = 0  
#             comp2 = 0  
#             for k, a_star in enumerate(neighbors):
#                 w = weights[k] # CODEX MODIFICATION: Apply edge weights to gradients
#                 cur1 = 0
#                 for t in range(Z):
#                     cur1 = cur1 + P[a_tilda, t] * np.log(P[a_star, t])
#                 comp1 = comp1 + w * (np.log(P[a_star, t_tilda]) - cur1)
#                 comp2 = comp2 + w * (P[a_star, t_tilda] - P[a_tilda, t_tilda])
#             ce_wat[a_tilda, t_tilda] = (-P[a_tilda, t_tilda] * comp1 - comp2) / X


@njit(parallel=True, fastmath=True)
def update_ll_wat(ll_wat, P_niche, Theta, F, X, K, Z):
    # Calculates the gradient of the Log-Likelihood with respect to W_niche
    for a in prange(X):
        # 1. Precompute the expected protein match (F) for each Niche for this specific cell
        expected_F = np.zeros(K, dtype=np.float32)
        total_expected_F = 0.0
        
        for k in range(K):
            for t in range(Z):
                expected_F[k] += Theta[k, t] * F[a, t]
            total_expected_F += P_niche[a, k] * expected_F[k]
        
        # 2. Apply the chain rule derivative of the Softmax function
        for k in range(K):
            ll_wat[a, k] = (P_niche[a, k] * (expected_F[k] - total_expected_F)) / X


@njit(parallel=True, fastmath=True)
def update_ll_wat_V(ll_wat_V, P_niche, Theta, F, X, K, Z):
    # 1. Calculate the raw gradient of the proteins given the spatial niches
    G = np.zeros((K, Z), dtype=np.float32)
    for k in prange(K):
        for t in range(Z):
            tmp = 0.0
            for a in range(X):
                tmp += P_niche[a, k] * F[a, t]
            G[k, t] = tmp / X
            
    # 2. Apply Softmax chain rule
    for k in prange(K):
        expected_G = 0.0
        for t in range(Z):
            expected_G += Theta[k, t] * G[k, t]
        for t in range(Z):
            ll_wat_V[k, t] = Theta[k, t] * (G[k, t] - expected_G)



@njit(parallel=True, fastmath=True)
def update_ce_wat(ce_wat, P_niche, Acsr_row, Acsr_col, Acsr_data, X, K):
    # Calculates the gradient of the spatial Cross-Entropy with respect to W_niche
    for a_tilda in prange(X):
        neighbors, weights = csr_obtain_column_index_and_data_for_row(Acsr_row, Acsr_col, Acsr_data, a_tilda)
        for k_tilda in range(K):
            comp1 = 0.0  
            comp2 = 0.0  
            for i, a_star in enumerate(neighbors):
                w = weights[i] # Distance weight (e.g., Delaunay edge length)
                cur1 = 0.0
                for k in range(K):
                    cur1 += P_niche[a_tilda, k] * np.log(P_niche[a_star, k])
                comp1 += w * (np.log(P_niche[a_star, k_tilda]) - cur1)
                comp2 += w * (P_niche[a_star, k_tilda] - P_niche[a_tilda, k_tilda])
                
            ce_wat[a_tilda, k_tilda] = (-P_niche[a_tilda, k_tilda] * comp1 - comp2) / X

# # ----------------------- Optimizer: ADAM -------------------
# @njit(parallel=True, fastmath=True)
# def update_m_v(m, v, beta1, beta2, beta, ll_wat, ce_wat, X, Y, Z):
#     for a in prange(X):
#         for t in range(Z):
#             cur_gredient = -ll_wat[a, t] + beta * ce_wat[a, t]
#             m[a, t] = m[a, t] * beta1 + (1 - beta1) * cur_gredient
#             v[a, t] = v[a, t] * beta2 + (1 - beta2) * cur_gredient**2


# @njit(parallel=True, fastmath=True)
# def update_W_adam(W, m, v, beta1, beta2, i, alpha, epsilon, X, Y, Z):
#     for a in prange(X):
#         for t in range(Z):
#             m_correct = m[a, t] / (1 - beta1**i)
#             v_correct = v[a, t] / (1 - beta2**i)
#             W[a, t] = W[a, t] - alpha * m_correct / (v_correct**0.5 + epsilon)



# ----------------------- Optimizer: ADAM -------------------
@njit(parallel=True, fastmath=True)
def update_m_v(m, v, beta1, beta2, beta, ll_wat, ce_wat, X, K): # Changed Z to K
    for a in prange(X):
        for k in range(K):
            cur_gredient = -ll_wat[a, k] + beta * ce_wat[a, k]
            m[a, k] = m[a, k] * beta1 + (1 - beta1) * cur_gredient
            v[a, k] = v[a, k] * beta2 + (1 - beta2) * cur_gredient**2

@njit(parallel=True, fastmath=True)
def update_W_adam(W, m, v, beta1, beta2, i, alpha, epsilon, X, K): # Changed Z to K
    for a in prange(X):
        for k in range(K):
            m_correct = m[a, k] / (1 - beta1**i)
            v_correct = v[a, k] / (1 - beta2**i)
            W[a, k] = W[a, k] - alpha * m_correct / (v_correct**0.5 + epsilon)

@njit(parallel=True, fastmath=True)
def update_V_adam(V, m_V, v_V, beta1, beta2, i, alpha, epsilon, K, Z, ll_wat_V):
    # NEW: Optimizes the Theta matrix
    for k in prange(K):
        for t in range(Z):
            cur_grad = -ll_wat_V[k, t] 
            m_V[k, t] = m_V[k, t] * beta1 + (1 - beta1) * cur_grad
            v_V[k, t] = v_V[k, t] * beta2 + (1 - beta2) * (cur_grad ** 2)
            m_correct = m_V[k, t] / (1 - beta1**i)
            v_correct = v_V[k, t] / (1 - beta2**i)
            V[k, t] = V[k, t] - alpha * m_correct / (v_correct**0.5 + epsilon)