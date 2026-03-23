import os
import json
import copy
import anndata
import numpy as np

class STHD:
    def __init__(self, spatial_path, load_type="anndata"):
        """
        CODEX CHANGE: Load a generic AnnData object containing fully segmented 
        cells and continuous protein intensities. High res background images are removed.
        """
        if load_type == "anndata":
            self.adata = spatial_path
        elif load_type == "file":
            self.adata = anndata.read_h5ad(spatial_path)

    def crop(self, x1, x2, y1, y2):
        """
        CODEX CHANGE: Filter segmented cells strictly by their spatial centroids 
        instead of cropping image pixels.
        """
        spatial_coords = self.adata.obsm["spatial"]
        mask_x = (spatial_coords[:, 0] >= x1) & (spatial_coords[:, 0] <= x2)
        mask_y = (spatial_coords[:, 1] >= y1) & (spatial_coords[:, 1] <= y2)
        adata_subset = self.adata[mask_x & mask_y].copy()
        
        return STHD(adata_subset, load_type="anndata")

    def match_refscrna(self, ref):
        """
        CODEX CHANGE: Removed Poisson noise addition and sum to 1 normalization. 
        The continuous Gaussian model requires raw mean intensities.
        """
        self.adata.var_names_make_unique()
        overlap_gs = self.adata.var.index.intersection(ref.index)
        
        self.adata = self.adata[:, self.adata.var_names.isin(overlap_gs)].copy()
        
        cell_type_by_gene_matrix = ref.loc[overlap_gs].T.values
        self.lambda_cell_type_by_gene_matrix = cell_type_by_gene_matrix

    def copy(self):
        return copy.deepcopy(self)

    def get_sequencing_data_region(self, adata=None):
        if not adata:
            adata = self.adata
        x1, y1 = np.nanmin(adata.obsm["spatial"], axis=0)
        x2, y2 = np.nanmax(adata.obsm["spatial"], axis=0)
        return int(x1), int(y1), int(x2), int(y2)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.adata.write_h5ad(os.path.join(path, "adata.h5ad"), compression="gzip")