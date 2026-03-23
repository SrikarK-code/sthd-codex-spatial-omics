import argparse
import os
import pandas as pd
from STHD import model, refscrna, sthdio
import numpy as np


# CODEX CHANGE: Removed `qcmask` imports and background filtering. 
# Segmented cell graphs do not contain empty "background" grid spots to filter.

def sthdata_match_refgene(sthd_data, refile):
    # CODEX CHANGE: Removed RNA-specific filtering and sum-to-1 normalization logic.
    genemeanpd_filtered = refscrna.load_scrna_ref(refile)
    sthd_data.match_refscrna(genemeanpd_filtered)
    genemeanpd_filtered = genemeanpd_filtered.loc[sthd_data.adata.var_names]
    return sthd_data, genemeanpd_filtered

# def train(sthd_data, n_iter, step_size, beta, debug=False, early_stop=False):
#     # CODEX CHANGE: Added Acsr_data to handle the distance-weighted edge penalties.
#     X, Y, Z, F, Acsr_row, Acsr_col, Acsr_data = model.prepare_constants(sthd_data)
#     W, eW, P, Phi, ll_wat, ce_wat, m, v = model.prepare_training_weights(X, Y, Z)
#     metrics = model.train(
#         n_iter=n_iter, step_size=step_size, beta=beta,
#         constants=(X, Y, Z, F, Acsr_row, Acsr_col, Acsr_data),
#         weights=(W, eW, P, Phi, ll_wat, ce_wat, m, v),
#         early_stop=early_stop,
#     )
#     if debug: return P, metrics
#     return P

# def predict(sthd_data, p, genemeanpd_filtered, mapcut=0.8):
#     adata = sthd_data.adata.copy()
#     for i, ct in enumerate(genemeanpd_filtered.columns):
#         adata.obs["p_ct_" + ct] = p[:, i]
#     adata.obs["x"] = adata.obsm["spatial"][:, 0]
#     adata.obs["y"] = adata.obsm["spatial"][:, 1]

#     STHD_prob = adata.obs[[t for t in adata.obs.columns if "p_ct_" in t]]
#     ct_max = STHD_prob.columns[STHD_prob.values.argmax(1)]
#     STHD_pred_ct = pd.DataFrame({"ct_max": ct_max}, index=STHD_prob.index)
#     STHD_pred_ct["ct"] = STHD_pred_ct["ct_max"]

#     ambiguous_mask = (STHD_prob.max(axis=1) < mapcut).values
#     STHD_pred_ct.loc[ambiguous_mask, "ct"] = "ambiguous"
    
#     # CODEX CHANGE: Removed the "filtered" state assignment. 
#     # All segmented cells are valid biological entities.

#     adata.obs["STHD_pred_ct"] = STHD_pred_ct["ct"]
#     sthd_data.adata = adata
#     return sthd_data



def train(sthd_data, n_iter, step_size, beta, K=10, debug=False, early_stop=False):
    # Added K to constants and unpack the 13 Dual-State variables
    X, Y, Z, K_out, F, Acsr_row, Acsr_col, Acsr_data = model.prepare_constants(sthd_data, K)
    W_niche, eW, P_niche, Phi, ll_wat, ce_wat, m, v, V, Theta, ll_wat_V, m_V, v_V = model.prepare_training_weights(X, Y, Z, K_out)
    
    metrics, P_niche_out, Theta_out = model.train(
        n_iter=n_iter, step_size=step_size, beta=beta,
        constants=(X, Y, Z, K_out, F, Acsr_row, Acsr_col, Acsr_data),
        weights=(W_niche, eW, P_niche, Phi, ll_wat, ce_wat, m, v, V, Theta, ll_wat_V, m_V, v_V),
        early_stop=early_stop,
    )
    return P_niche_out, Theta_out

def predict(sthd_data, p_niche, mapcut=0.0):
    adata = sthd_data.adata.copy()
    K = p_niche.shape[1]
    
    # Save Niche probabilities instead of Cell Type probabilities
    for i in range(K):
        adata.obs[f"p_niche_{i}"] = p_niche[:, i]
    
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    STHD_prob = adata.obs[[t for t in adata.obs.columns if "p_niche_" in t]]
    niche_max = STHD_prob.columns[STHD_prob.values.argmax(1)]
    
    STHD_pred_niche = pd.DataFrame({"niche_max": niche_max}, index=STHD_prob.index)
    STHD_pred_niche["STHD_pred_niche"] = STHD_pred_niche["niche_max"]

    ambiguous_mask = (STHD_prob.max(axis=1) < mapcut).values
    STHD_pred_niche.loc[ambiguous_mask, "STHD_pred_niche"] = "ambiguous"

    adata.obs["STHD_pred_niche"] = STHD_pred_niche["STHD_pred_niche"]
    sthd_data.adata = adata
    return sthd_data







def save_prediction_pdata(sthdata, file_path=None, prefix=""):
    # Target Niche columns
    predcols = ["x", "y", "STHD_pred_niche"] + [t for t in sthdata.adata.obs.columns if "p_niche_" in t]
    pdata = sthdata.adata.obs[predcols]
    if file_path is not None:
        pdata_path = os.path.join(file_path, prefix + "_pdata.tsv")
        pdata.to_csv(pdata_path, sep="\t")
    return pdata


# def save_prediction_pdata(sthdata, file_path=None, prefix=""):
#     predcols = ["x", "y", "STHD_pred_ct"] + [t for t in sthdata.adata.obs.columns if "p_ct_" in t]
#     pdata = sthdata.adata.obs[predcols]
#     if file_path is not None:
#         pdata_path = os.path.join(file_path, prefix + "_pdata.tsv")
#         pdata.to_csv(pdata_path, sep="\t")
#     return pdata

def load_data(file_path):
    sthd_data = sthdio.STHD(spatial_path=os.path.join(file_path, "adata.h5ad"), load_type="file")
    return sthd_data

def load_pdata(file_path, prefix=""):
    pdata_path = os.path.join(file_path, prefix + "_pdata.tsv")
    pdata = pd.read_table(pdata_path, index_col=0)
    pdata.index = pdata.index.astype(str)
    
    return pdata

def add_pdata(sthd_data, pdata):
    sthdata = sthd_data
    exist_cols = sthdata.adata.obs.columns.intersection(pdata.columns)
    for col in sthdata.adata.obs[exist_cols]:
        del sthdata.adata.obs[col]
    sthdata.adata.obs = sthdata.adata.obs.merge(pdata, how="left", left_index=True, right_index=True)
    return sthdata

def load_data_with_pdata(file_path, pdata_prefix=""):
    sthdata = load_data(file_path)
    pdata = load_pdata(file_path, pdata_prefix)
    return add_pdata(sthdata, pdata)

# def main(args):
#     for patch_path in args.patch_list:
#         sthdata = load_data(patch_path)
#         sthdata, genemeanpd_filtered = sthdata_match_refgene(sthdata, args.refile)
#         P = train(sthdata, args.n_iter, args.step_size, args.beta)
#         sthdata = predict(sthdata, P, genemeanpd_filtered, mapcut=args.mapcut)
#         save_prediction_pdata(sthdata, file_path=patch_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_iter", default=23, type=int)
#     parser.add_argument("--step_size", default=1, type=int)
#     parser.add_argument("--beta", default=0.1, type=float)
#     parser.add_argument("--mapcut", default=0.8, type=float)
#     parser.add_argument("--refile", type=str, required=True)
#     parser.add_argument("--patch_list", nargs="+", default=[])
#     args = parser.parse_args()
#     main(args)



def main(args):
    for patch_path in args.patch_list:
        sthdata = load_data(patch_path)
        sthdata, genemeanpd_filtered = sthdata_match_refgene(sthdata, args.refile)
        
        P_niche, Theta = train(sthdata, args.n_iter, args.step_size, args.beta, K=args.K)
        np.save(os.path.join(patch_path, "theta.npy"), Theta)
        
        sthdata = predict(sthdata, P_niche, mapcut=args.mapcut)
        save_prediction_pdata(sthdata, file_path=patch_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", default=25, type=int)
    parser.add_argument("--step_size", default=1.0, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--mapcut", default=0.0, type=float)
    parser.add_argument("--refile", type=str, required=True)
    parser.add_argument("--K", default=10, type=int) # NEW: Number of Niches
    parser.add_argument("--patch_list", nargs="+", default=[])
    args = parser.parse_args()
    main(args)