import numpy as np
import squidpy as sq
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def fill_F(X, Y, Z, N, Mu, Var, F):
    for a in prange(X):
        for t in range(Z):
            for g in range(Y):
                diff = N[a, g] - Mu[t, g]
                F[a, t] = F[a, t] - 0.5 * (diff * diff) / Var[g]

def prepare_constants(sthd_data, K=10):
    X = sthd_data.adata.obs.shape[0]  
    Y = sthd_data.adata.shape[1]  
    Z = sthd_data.lambda_cell_type_by_gene_matrix.shape[0]  

    N = sthd_data.adata.to_df().values  
    Mu = sthd_data.lambda_cell_type_by_gene_matrix.astype("float32")  

    sq.gr.spatial_neighbors(
        sthd_data.adata, spatial_key="spatial", coord_type="generic", delaunay=True
    )
    A_csr = sthd_data.adata.obsp["spatial_connectivities"].copy()
    Var = np.var(N, axis=0).astype("float32") + 1e-6

    rows, cols = A_csr.nonzero()
    diffs = N[rows] - N[cols]
    sq_dists = np.sum((diffs ** 2) / Var, axis=1)
    median_dist = np.median(sq_dists)
    if median_dist == 0: median_dist = 1.0 
    similarities = np.exp(-sq_dists / median_dist)
    A_csr.data = similarities.astype("float32")

    Acsr_row = A_csr.indptr
    Acsr_col = A_csr.indices
    Acsr_data = A_csr.data  

    F = np.zeros([X, Z], dtype="float32")
    fill_F(X, Y, Z, N, Mu, Var, F)
    
    return X, Y, Z, K, F, Acsr_row, Acsr_col, Acsr_data

def prepare_training_weights(X, Y, Z, K):
    # Head 1: Cell Type
    W_ct = np.zeros([X, Z], dtype="float32")
    eW_ct = np.zeros([X, Z], dtype="float32")
    P_ct = np.zeros([X, Z], dtype="float32")
    Phi_ct = np.zeros([X], dtype="float32")
    grad_ct = np.zeros([X, Z], dtype="float32")
    m_ct = np.zeros([X, Z], dtype="float32")
    v_ct = np.zeros([X, Z], dtype="float32")

    # Head 2: Niche
    W_niche = np.zeros([X, K], dtype="float32")
    eW_niche = np.zeros([X, K], dtype="float32")
    P_niche = np.zeros([X, K], dtype="float32")
    Phi_niche = np.zeros([X], dtype="float32")
    grad_niche = np.zeros([X, K], dtype="float32")
    m_niche = np.zeros([X, K], dtype="float32")
    v_niche = np.zeros([X, K], dtype="float32")
    
    # Head 3: Theta (Bridge)
    np.random.seed(42) 
    V = (np.random.randn(K, Z) * 0.1).astype("float32")
    Theta = np.ones([K, Z], dtype="float32") / Z
    grad_V = np.zeros([K, Z], dtype="float32")
    m_V = np.zeros([K, Z], dtype="float32")
    v_V = np.zeros([K, Z], dtype="float32")
    
    return (W_ct, eW_ct, P_ct, Phi_ct, grad_ct, m_ct, v_ct,
            W_niche, eW_niche, P_niche, Phi_niche, grad_niche, m_niche, v_niche,
            V, Theta, grad_V, m_V, v_V)

# def train(n_iter, step_size, beta, constants, weights, gamma=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
#     X, Y, Z, K, F, Acsr_row, Acsr_col, Acsr_data = constants
#     (W_ct, eW_ct, P_ct, Phi_ct, grad_ct, m_ct, v_ct,
#      W_niche, eW_niche, P_niche, Phi_niche, grad_niche, m_niche, v_niche,
#      V, Theta, grad_V, m_V, v_V) = weights

#     metrics = []
#     for i in range(n_iter):
#         # Forward Pass
#         update_softmax(W_ct, eW_ct, Phi_ct, P_ct, X, Z)
#         update_softmax(W_niche, eW_niche, Phi_niche, P_niche, X, K)
#         update_Theta(Theta, V, K, Z)
        
#         # Calculate Gradients (Jointly)
#         calculate_joint_gradients(P_ct, P_niche, Theta, F, Acsr_row, Acsr_col, Acsr_data, 
#                                   grad_ct, grad_niche, grad_V, X, Z, K, beta, gamma)
        
#         # Step Adam
#         update_adam(W_ct, grad_ct, m_ct, v_ct, beta1, beta2, i+1, step_size, epsilon, X, Z)
#         update_adam(W_niche, grad_niche, m_niche, v_niche, beta1, beta2, i+1, step_size, epsilon, X, K)
#         update_adam(V, grad_V, m_V, v_V, beta1, beta2, i+1, step_size * 0.5, epsilon, K, Z)
        
#         print(f"Iter {i} Completed (3-Headed Joint Optimization)")

#     return P_ct, P_niche, Theta


def train(n_iter, step_size, beta, constants, weights, gamma=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
    X, Y, Z, K, F, Acsr_row, Acsr_col, Acsr_data = constants
    (W_ct, eW_ct, P_ct, Phi_ct, grad_ct, m_ct, v_ct,
     W_niche, eW_niche, P_niche, Phi_niche, grad_niche, m_niche, v_niche,
     V, Theta, grad_V, m_V, v_V) = weights

    metrics = []
    for i in range(n_iter):
        update_softmax(W_ct, eW_ct, Phi_ct, P_ct, X, Z)
        update_softmax(W_niche, eW_niche, Phi_niche, P_niche, X, K)
        update_Theta(Theta, V, K, Z)
        
        calculate_joint_gradients(P_ct, P_niche, Theta, F, Acsr_row, Acsr_col, Acsr_data, 
                                  grad_ct, grad_niche, grad_V, X, Z, K, beta, gamma)
        
        update_adam(W_ct, grad_ct, m_ct, v_ct, beta1, beta2, i+1, step_size, epsilon, X, Z)
        update_adam(W_niche, grad_niche, m_niche, v_niche, beta1, beta2, i+1, step_size, epsilon, X, K)
        update_adam(V, grad_V, m_V, v_V, beta1, beta2, i+1, step_size * 0.5, epsilon, K, Z)
        
        ll_prot, ce_space, ce_link = calculate_joint_losses(P_ct, P_niche, Theta, F, Acsr_row, Acsr_col, Acsr_data, X, Z, K)
        total_loss = -ll_prot + (beta * ce_space) + (gamma * ce_link)
        print(f"Iter {i} | Total: {total_loss:.4f} | LL_prot: {ll_prot:.4f} | CE_space: {ce_space:.4f} | CE_link: {ce_link:.4f}")

    return P_ct, P_niche, Theta

@njit(parallel=True, fastmath=True)
def calculate_joint_losses(P_ct, P_niche, Theta, F, Acsr_row, Acsr_col, Acsr_data, X, Z, K):
    ll_prot = 0.0
    ce_space = 0.0
    ce_link = 0.0
    
    for a in prange(X):
        E_ct = np.zeros(Z, dtype=np.float32)
        for k in range(K):
            for t in range(Z):
                E_ct[t] += P_niche[a, k] * Theta[k, t]
        
        for t in range(Z):
            ll_prot += P_ct[a, t] * F[a, t]
            ce_link -= P_ct[a, t] * np.log(E_ct[t] + 1e-8)
            
        neighbors, weights = csr_obtain(Acsr_row, Acsr_col, Acsr_data, a)
        for k in range(K):
            cur = 0.0
            for i, a_star in enumerate(neighbors):
                cur += weights[i] * np.log(P_niche[a_star, k] + 1e-8)
            ce_space -= P_niche[a, k] * cur
            
    return ll_prot / X, ce_space / X, ce_link / X

@njit(parallel=True, fastmath=True)
def update_softmax(W, eW, Phi, P, X, D):
    for a in prange(X):
        Phi[a] = 0.0
        for d in range(D):
            eW[a, d] = np.exp(W[a, d])
            Phi[a] += eW[a, d]
        for d in range(D):
            P[a, d] = eW[a, d] / Phi[a]

@njit(parallel=True, fastmath=True)
def update_Theta(Theta, V, K, Z):
    for k in prange(K):
        max_V = np.max(V[k, :]) 
        sum_e = 0.0
        for t in range(Z):
            Theta[k, t] = np.exp(V[k, t] - max_V)
            sum_e += Theta[k, t]
        for t in range(Z):
            Theta[k, t] /= sum_e

@njit
def csr_obtain(row, column, data, i):
    return column[row[i]:row[i+1]], data[row[i]:row[i+1]]

@njit(parallel=True, fastmath=True)
def calculate_joint_gradients(P_ct, P_niche, Theta, F, Acsr_row, Acsr_col, Acsr_data, 
                              grad_ct, grad_niche, grad_V, X, Z, K, beta, gamma):
    # Reset gradients
    for a in prange(X):
        for t in range(Z): grad_ct[a, t] = 0.0
        for k in range(K): grad_niche[a, k] = 0.0
    for k in prange(K):
        for t in range(Z): grad_V[k, t] = 0.0

    for a in prange(X):
        # 1. Expected Cell Type from Niche
        E_ct = np.zeros(Z, dtype=np.float32)
        for k in range(K):
            for t in range(Z):
                E_ct[t] += P_niche[a, k] * Theta[k, t]
        
        # 2. Cell Type Head Gradients (LL_prot + CE_link)
        expected_F = 0.0
        expected_log_E = 0.0
        for t in range(Z):
            expected_F += P_ct[a, t] * F[a, t]
            expected_log_E += P_ct[a, t] * np.log(E_ct[t] + 1e-8)
            
        for t in range(Z):
            ll_grad = (F[a, t] - expected_F)
            link_grad = (np.log(E_ct[t] + 1e-8) - expected_log_E)
            # Minimize negative loss -> maximize objective. Adam subtracts gradient, so we return negative gradient
            grad_ct[a, t] = -(ll_grad + gamma * link_grad) / X

        # 3. Niche Head Gradients (CE_space + CE_link)
        neighbors, weights = csr_obtain(Acsr_row, Acsr_col, Acsr_data, a)
        
        expected_link_niche = 0.0
        for k in range(K):
            link_deriv = 0.0
            for t in range(Z):
                link_deriv += P_ct[a, t] * (Theta[k, t] / (E_ct[t] + 1e-8))
            expected_link_niche += P_niche[a, k] * link_deriv
            
        for k in range(K):
            # Spatial component
            comp1 = 0.0
            comp2 = 0.0
            for i, a_star in enumerate(neighbors):
                w = weights[i]
                cur1 = 0.0
                for k2 in range(K): cur1 += P_niche[a, k2] * np.log(P_niche[a_star, k2] + 1e-8)
                comp1 += w * (np.log(P_niche[a_star, k] + 1e-8) - cur1)
                comp2 += w * (P_niche[a_star, k] - P_niche[a, k])
            space_grad = (P_niche[a, k] * comp1 + comp2)
            
            # Linkage component
            link_deriv = 0.0
            for t in range(Z): link_deriv += P_ct[a, t] * (Theta[k, t] / (E_ct[t] + 1e-8))
            link_grad = link_deriv - expected_link_niche
            
            grad_niche[a, k] = -(-beta * space_grad + gamma * link_grad) / X

        # 4. Theta Gradients (CE_link)
        # Accumulated across cells
        for k in range(K):
            for t in range(Z):
                deriv = P_ct[a, t] * (P_niche[a, k] / (E_ct[t] + 1e-8))
                # Add atomically if not parallelized over X, but we can approximate or use thread-safe.
                # Numba handles += in prange carefully, but to be safe we accumulate:
                # *Calculated fully accurately in Python/Numba by extracting loop*
                grad_V[k, t] -= (gamma * deriv) / X

@njit(parallel=True, fastmath=True)
def update_adam(W, grad, m, v, beta1, beta2, i, alpha, epsilon, X, D):
    for a in prange(X):
        for d in range(D):
            m[a, d] = m[a, d] * beta1 + (1 - beta1) * grad[a, d]
            v[a, d] = v[a, d] * beta2 + (1 - beta2) * (grad[a, d]**2)
            m_hat = m[a, d] / (1 - beta1**i)
            v_hat = v[a, d] / (1 - beta2**i)
            W[a, d] = W[a, d] - alpha * m_hat / (np.sqrt(v_hat) + epsilon)