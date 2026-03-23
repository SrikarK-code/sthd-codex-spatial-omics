"""Function for region of interest analysis in CODEX"""

def extract_roi(sthdata, x_min, x_max, y_min, y_max):
    """
    CODEX CHANGE: Extract a Region of Interest (ROI) using continuous 
    physical coordinate bounds instead of parsing Visium grid barcodes.
    """
    spatial_coords = sthdata.adata.obsm["spatial"]
    
    # Filter cells within the continuous bounding box
    mask_x = (spatial_coords[:, 0] >= x_min) & (spatial_coords[:, 0] <= x_max)
    mask_y = (spatial_coords[:, 1] >= y_min) & (spatial_coords[:, 1] <= y_max)
    
    roi_adata = sthdata.adata[mask_x & mask_y].copy()
    sthdata.adata = roi_adata
    
    return sthdata