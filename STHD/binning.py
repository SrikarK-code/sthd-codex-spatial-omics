import anndata
import pandas as pd
import scanpy as sc

# CODEX CHANGE: Completely removed `get_bin_adata` and `get_sthd_guided_bin_adata_v1`.
# Binning artificial spots into squares makes no sense when the data is already pre-segmented whole cells.

def cluster_cells(sthdata, resolution=0.5):
    """
    CODEX CHANGE: Renamed to `cluster_cells`. Removed RNA-specific `target_sum` 
    and `log1p` normalization, replacing it with standard scaling since CODEX 
    protein intensities are pre-normalized continuous values.
    """
    adata_cell = sthdata.adata.copy()
    
    if 'counts' not in adata_cell.layers.keys():
        adata_cell.layers['counts'] = adata_cell.X.copy()
        
    sc.pp.scale(adata_cell) 
    sc.tl.pca(adata_cell)
    sc.pp.neighbors(adata_cell)
    sc.tl.umap(adata_cell)
    sc.tl.leiden(adata_cell, resolution=resolution)
    
    return adata_cell