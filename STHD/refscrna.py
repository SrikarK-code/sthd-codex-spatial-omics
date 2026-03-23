import pandas as pd

def load_scrna_ref(refile):
    genemeanpd_filtered = pd.read_table(refile, index_col=0)
    return genemeanpd_filtered

def gene_lambda_by_ct(adata, ctcol='group'):
    """
    CODEX CHANGE: Calculate the mean continuous intensity per cell type. 
    Removed the discrete count normalization steps.
    """
    pdlst = []
    celltypes_group = list(set(adata.obs[ctcol].values))
    
    for ct in celltypes_group:
        ctbars = adata[adata.obs[ctcol] == ct].obs.index.tolist()
        
        if 'counts' in adata.layers:
            cellbygenecount = adata.layers['counts'][adata.obs.index.isin(ctbars), :]
        else:
            cellbygenecount = adata.X[adata.obs.index.isin(ctbars), :]
            
        if hasattr(cellbygenecount, 'todense'):
            cellbygenecount = cellbygenecount.todense()
            
        cellbygene = pd.DataFrame(cellbygenecount, index=ctbars, columns=adata.var_names)
        
        # Take the mean of continuous intensities directly
        genemean = cellbygene.mean(axis=0)
        pdlst.append(pd.DataFrame(genemean, columns=[ct], index=adata.var_names))

    genemeanpd = pd.concat(pdlst, axis=1)
    return genemeanpd