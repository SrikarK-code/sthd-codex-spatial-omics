import argparse
import os
import pandas as pd
from STHD import model, refscrna, sthdio
import numpy as np

def sthdata_match_refgene(sthd_data, refile):
    genemeanpd_filtered = refscrna.load_scrna_ref(refile)
    sthd_data.match_refscrna(genemeanpd_filtered)
    genemeanpd_filtered = genemeanpd_filtered.loc[sthd_data.adata.var_names]
    return sthd_data, genemeanpd_filtered

# def train(sthd_data, n_iter, step_size, beta, K=10):
#     X, Y, Z, K_out, F, Acsr_row, Acsr_col, Acsr_data = model.prepare_constants(sthd_data, K)
#     weights = model.prepare_training_weights(X, Y, Z, K_out)
    
#     P_ct_out, P_niche_out, Theta_out = model.train(
#         n_iter=n_iter, step_size=step_size, beta=beta,
#         constants=(X, Y, Z, K_out, F, Acsr_row, Acsr_col, Acsr_data),
#         weights=weights
#     )
#     return P_ct_out, P_niche_out, Theta_out

def train(sthd_data, n_iter, step_size, beta, K=10):
    X, Y, Z, K_out, F, Acsr_row, Acsr_col, Acsr_data = model.prepare_constants(sthd_data, K)
    weights = model.prepare_training_weights(X, Y, Z, K_out)
    
    P_ct_out, P_niche_out, Theta_out = model.train(
        n_iter=n_iter, step_size=step_size, beta=beta, gamma=0.1,  # <--- ADD GAMMA HERE
        constants=(X, Y, Z, K_out, F, Acsr_row, Acsr_col, Acsr_data),
        weights=weights
    )
    return P_ct_out, P_niche_out, Theta_out

def predict(sthd_data, p_ct, p_niche, genemeanpd_filtered, mapcut=0.0):
    adata = sthd_data.adata.copy()
    Z = p_ct.shape[1]
    K = p_niche.shape[1]
    
    # Save Cell Type probabilities
    for i, ct in enumerate(genemeanpd_filtered.columns):
        adata.obs[f"p_ct_{ct}"] = p_ct[:, i]
        
    # Save Niche probabilities
    for i in range(K):
        adata.obs[f"p_niche_{i}"] = p_niche[:, i]
    
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    # Predict CT
    CT_prob = adata.obs[[t for t in adata.obs.columns if "p_ct_" in t]]
    ct_max = CT_prob.columns[CT_prob.values.argmax(1)].str.replace('p_ct_', '')
    adata.obs["STHD_pred_ct"] = ct_max

    # Predict Niche
    Niche_prob = adata.obs[[t for t in adata.obs.columns if "p_niche_" in t]]
    niche_max = Niche_prob.columns[Niche_prob.values.argmax(1)]
    adata.obs["STHD_pred_niche"] = niche_max

    sthd_data.adata = adata
    return sthd_data

def save_prediction_pdata(sthdata, file_path=None, prefix=""):
    predcols = ["x", "y", "STHD_pred_ct", "STHD_pred_niche"] + \
               [t for t in sthdata.adata.obs.columns if "p_ct_" in t] + \
               [t for t in sthdata.adata.obs.columns if "p_niche_" in t]
    pdata = sthdata.adata.obs[predcols]
    if file_path is not None:
        pdata_path = os.path.join(file_path, prefix + "_pdata.tsv")
        pdata.to_csv(pdata_path, sep="\t")
    return pdata

def load_data(file_path):
    return sthdio.STHD(spatial_path=os.path.join(file_path, "adata.h5ad"), load_type="file")

def load_pdata(file_path, prefix=""):
    pdata = pd.read_table(os.path.join(file_path, prefix + "_pdata.tsv"), index_col=0)
    pdata.index = pdata.index.astype(str)
    return pdata

def add_pdata(sthd_data, pdata):
    exist_cols = sthd_data.adata.obs.columns.intersection(pdata.columns)
    sthd_data.adata.obs.drop(columns=exist_cols, inplace=True)
    sthd_data.adata.obs = sthd_data.adata.obs.merge(pdata, how="left", left_index=True, right_index=True)
    return sthd_data

def load_data_with_pdata(file_path, pdata_prefix=""):
    return add_pdata(load_data(file_path), load_pdata(file_path, pdata_prefix))

def main(args):
    for patch_path in args.patch_list:
        sthdata = load_data(patch_path)
        sthdata, genemeanpd_filtered = sthdata_match_refgene(sthdata, args.refile)
        
        P_ct, P_niche, Theta = train(sthdata, args.n_iter, args.step_size, args.beta, K=args.K)
        np.save(os.path.join(patch_path, "theta.npy"), Theta)
        
        sthdata = predict(sthdata, P_ct, P_niche, genemeanpd_filtered, mapcut=args.mapcut)
        save_prediction_pdata(sthdata, file_path=patch_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", default=25, type=int)
    parser.add_argument("--step_size", default=1.0, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--mapcut", default=0.0, type=float)
    parser.add_argument("--refile", type=str, required=True)
    parser.add_argument("--K", default=10, type=int) 
    parser.add_argument("--patch_list", nargs="+", default=[])
    args = parser.parse_args()
    main(args)