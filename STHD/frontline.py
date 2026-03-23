from collections import Counter
import numpy as np
from numba import njit, prange
from tqdm import tqdm
from STHD import model

# CODEX CHANGE: Removed all `_binned` functions. Binning is obsolete for pre-segmented cells.

def get_neighbor_ct(adata, ctstr="cTNI", ctlst=[]):
    subset_cell_idx1 = np.where(adata.obs["STHD_pred_ct"].str.contains(ctstr))[0]
    subset_cell_idx2 = np.where(adata.obs["STHD_pred_ct"].isin(ctlst))[0]
    subset_cell_idx = np.array(list(set(np.append(subset_cell_idx1, subset_cell_idx2))))
    _, idx = adata.obsp["spatial_connectivities"][subset_cell_idx.T, :].nonzero()
    idx = np.array(list(set(np.append(idx, subset_cell_idx))))
    return adata.obs.iloc[idx]

def get_ambiguous_near_ct(adata, ctstr="cTNI", ctlst=[]):
    subset_cell_idx1 = np.where(adata.obs["STHD_pred_ct"].str.contains(ctstr))[0]
    subset_cell_idx2 = np.where(adata.obs["STHD_pred_ct"].isin(ctlst))[0]
    subset_cell_idx = np.array(list(set(np.append(subset_cell_idx1, subset_cell_idx2))))
    _, idx = adata.obsp["spatial_connectivities"][subset_cell_idx.T, :].nonzero()
    idx = np.array(list(set(np.append(idx, subset_cell_idx))))
    
    amb_near_subset = adata.obs.iloc[idx][adata.obs["STHD_pred_ct"].iloc[idx] == "ambiguous"].index
    return adata[amb_near_subset].copy()

def sthd_neighbor_ct_count(adata):
    sthd_p_cols = [t for t in adata.obs.columns if t[:5] == "p_ct_"]
    adj = adata.obsp["spatial_connectivities"]
    adj_row, adj_col = adj.indptr, adj.indices

    p = adata.obs[sthd_p_cols]
    cell_type_names = list(p.columns)
    pmap = np.array([cell_type_names[i] for i in p.values.argmax(1)])
    pmap_value = p.values.max(1)
    pmap[pmap_value == -1] = "filtered"

    neighbors = [model.csr_obtain_column_index_for_row(adj_row, adj_col, i) for i in range(len(p))]
    neighbor_celltypes = [Counter(pmap[n]) for n in tqdm(neighbors)]
    neighbor_celltype_count = np.array([len(n_ct) for n_ct in tqdm(neighbor_celltypes)])
    neighbor_celltype_name = np.array(["|".join(sorted(list(n_ct.keys()))) for n_ct in tqdm(neighbor_celltypes)])

    adata.obs["neighbor_celltype_name"] = neighbor_celltype_name
    adata.obs["neighbor_celltype_count"] = neighbor_celltype_count
    adata.obs["STHD_pred_ct_raw"] = pmap

def get_frontline(adata, A="Tumor cE", B="Macrophage", frontline_name="frontline_ctA_ctB"):
    itself_A_B = (adata.obs["STHD_pred_ct"].str.contains(A).values) | (adata.obs["STHD_pred_ct"].str.contains(B).values)
    subset = adata[(itself_A_B) & (adata.obs["neighbor_celltype_count"] <= 2)].copy()
    neighbor_A_B = (subset.obs["neighbor_celltype_name"].str.contains(A).values) & (subset.obs["neighbor_celltype_name"].str.contains(B).values)
    
    label = np.zeros(len(subset))
    label[(subset.obs["STHD_pred_ct"].str.contains(A).values) & (subset.obs["neighbor_celltype_count"] == 1)] = 1
    label[(subset.obs["STHD_pred_ct"].str.contains(B).values) & (subset.obs["neighbor_celltype_count"] == 1)] = 2
    label[(subset.obs["STHD_pred_ct"].str.contains(A).values) & neighbor_A_B] = 3
    label[(subset.obs["STHD_pred_ct"].str.contains(B).values) & neighbor_A_B] = 4

    subset.obs["label"] = label
    adata.obs[frontline_name] = 0
    labels = adata.obs[frontline_name].values
    for i in range(1, 5):
        labels[np.where(adata.obs.index.isin(subset[subset.obs["label"] == i].obs.index))[0]] = i
    adata.obs[frontline_name] = labels

def frontline_summarize(adata, frontlines):
    adata.obs["frontline_sum_type"] = "non_fl"
    labels = adata.obs["frontline_sum_type"].values
    for frontline in frontlines:
        labels[adata.obs[frontline].isin([3, 4])] = frontline
    adata.obs["frontline_sum_type"] = labels

    adata.obs["frontline_sum_type_AB"] = "non_fl"
    labels = adata.obs["frontline_sum_type_AB"].values
    for frontline in frontlines:
        labels[adata.obs[frontline] == 3] = frontline + "_A"
        labels[adata.obs[frontline] == 4] = frontline + "_B"
    adata.obs["frontline_sum_type_AB"] = labels

@njit(parallel=True, fastmath=True)
def min_pairwise_distance(X, Y):
    n, m = len(X), len(Y)
    res = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        x = X[i]
        d = np.inf
        for j in range(m):
            y = Y[j]
            cur_d = ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** (0.5)
            d = min(d, cur_d)
        res[i] = d
    return res

def calculate_distance(adata, frontline_name):
    # CODEX CHANGE: Switched from rigid grid indices (array_row/col) to continuous (x, y) coordinates.
    frontline_label = np.isin(adata.obs[frontline_name].values, [3, 4])
    location = adata.obs[["x", "y"]].values
    frontline_location = location[frontline_label]
    distance = min_pairwise_distance(location, frontline_location)
    adata.obs["dTo_" + frontline_name] = distance