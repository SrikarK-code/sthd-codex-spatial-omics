##
## intestine


import pandas as pd
import anndata as ad

df_HuBMAP = pd.read_csv('/content/drive/MyDrive/Duke_Personal/Teaching/doi_10_5061_dryad_pk0p2ngrf__v20230913/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
df_metadata = pd.read_csv('/content/drive/MyDrive/Duke_Personal/Teaching/doi_10_5061_dryad_pk0p2ngrf__v20230913/donor_metadata.csv', index_col=0)

df_metadataT = df_metadata.T.reset_index()
df_metadataT.rename(columns={'index': 'donor'}, inplace=True)
df_merged = df_HuBMAP.merge(df_metadataT, on='donor', how='left')

marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

X = df_merged[marker_cols].values
obs = df_merged.drop(columns=marker_cols)
spatial = df_merged[['x', 'y']].values

adata_intestine = ad.AnnData(X=X, obs=obs)
adata_intestine.var_names = marker_cols
adata_intestine.obsm['spatial'] = spatial
adata_intestine.write_h5ad("intestine_sthd.h5ad")