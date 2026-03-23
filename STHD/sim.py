import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CODEX CHANGE: Replaced rigid grid `meshgrid` logic with randomly scattered cell centroids.
def simulate_scattered_2cell(side_length=15, num_cells=300, center1=(4, 4), radius1=4, center2=(9, 9), radius2=4):
    """
    Simulates scattered segmented cells instead of a rigid square spot grid.
    """
    x = np.random.uniform(0, side_length, num_cells)
    y = np.random.uniform(0, side_length, num_cells)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color="blue", alpha=0.3)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    mask1 = ((x - center1[0]) ** 2 + (y - center1[1]) ** 2) <= radius1**2
    mask2 = ((x - center2[0]) ** 2 + (y - center2[1]) ** 2) <= radius2**2
    
    mask_overlap = mask1 & mask2
    mask1[mask_overlap] = False

    ax.scatter(x[mask1], y[mask1], color="orange")
    ax.scatter(x[mask2], y[mask2], color="green")
    plt.show()
    return x, y, mask1, mask2

# CODEX CHANGE: Replaced `np.random.poisson` discrete count simulation with `np.random.normal` for continuous fluorescence intensities.
def simulate_cell_expr_2cell(x, y, mask1, mask2, mu_ct1geneA=2.0, mu_ct1geneB=0.0, mu_ct2geneA=0.01, mu_ct2geneB=4.0, mu_geneC=4.0, noise_sigma=0.5):
    total_cells = len(x)
    barcode_lst = ["cell_" + str(t) for t in range(total_cells)]

    obs = pd.DataFrame({"cellid": range(total_cells)}, index=barcode_lst)
    var = pd.DataFrame(index=["geneA", "geneB", "geneC"])
    expr = pd.DataFrame(np.zeros([total_cells, 3]), index=obs.index, columns=var.index)

    obs["celltype"] = ""
    obs.loc[mask1, "celltype"] = "ct1"
    obs.loc[mask2, "celltype"] = "ct2"

    obs["x"] = x
    obs["y"] = y
    obsm = {"spatial": obs[["x", "y"]].values}

    ct1_ids = obs[obs["celltype"] == "ct1"].index
    ct2_ids = obs[obs["celltype"] == "ct2"].index

    # CODEX CHANGE: Gaussian continuous emission simulation
    expr.loc[ct1_ids, "geneA"] += np.random.normal(loc=mu_ct1geneA, scale=noise_sigma, size=len(ct1_ids))
    expr.loc[ct1_ids, "geneB"] += np.random.normal(loc=mu_ct1geneB, scale=noise_sigma, size=len(ct1_ids))
    expr.loc[ct2_ids, "geneA"] += np.random.normal(loc=mu_ct2geneA, scale=noise_sigma, size=len(ct2_ids))
    expr.loc[ct2_ids, "geneB"] += np.random.normal(loc=mu_ct2geneB, scale=noise_sigma, size=len(ct2_ids))

    # Background global noise and unrelated gene
    expr["geneA"] += np.random.normal(loc=0, scale=noise_sigma, size=total_cells)
    expr["geneB"] += np.random.normal(loc=0, scale=noise_sigma, size=total_cells)
    expr.loc[:, "geneC"] = np.random.normal(loc=mu_geneC, scale=noise_sigma, size=total_cells)
    
    # Clip to 0 since fluorescence intensity cannot be negative
    expr = expr.clip(lower=0)

    adata = anndata.AnnData(expr, obs=obs, var=var, obsm=obsm)
    return adata