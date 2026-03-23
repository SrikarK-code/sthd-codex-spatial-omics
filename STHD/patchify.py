
# import os
# import pathlib
# import numpy as np
# import pandas as pd
# from scipy.spatial import cKDTree
# from tqdm import tqdm
# from STHD import train
# from collections import defaultdict

# def partition_kdtree_rigid(spatial_coords, max_cells):
#     """Recursively splits the physical space WITHOUT calculating the halo yet."""
#     if len(spatial_coords) <= max_cells:
#         x_min, y_min = np.min(spatial_coords, axis=0)
#         x_max, y_max = np.max(spatial_coords, axis=0)
#         return [(x_min, x_max, y_min, y_max)]
    
#     spreads = np.ptp(spatial_coords, axis=0)
#     split_dim = np.argmax(spreads)
#     median_val = np.median(spatial_coords[:, split_dim])
    
#     left_mask = spatial_coords[:, split_dim] <= median_val
#     right_mask = spatial_coords[:, split_dim] > median_val
    
#     patches = []
#     if np.sum(left_mask) > 0:
#         patches.extend(partition_kdtree_rigid(spatial_coords[left_mask], max_cells))
#     if np.sum(right_mask) > 0:
#         patches.extend(partition_kdtree_rigid(spatial_coords[right_mask], max_cells))
        
#     return patches

# def patchify(sthd_data, save_path, max_cells=5000, halo=50.0):
#     allregion_path = f"{save_path}/all_region"
#     patch_path = f"{save_path}/patches"

#     pathlib.Path(allregion_path).mkdir(parents=True, exist_ok=True)
#     pathlib.Path(patch_path).mkdir(parents=True, exist_ok=True)

#     sthd_data.save(allregion_path)
    
#     coords = sthd_data.adata.obsm["spatial"]
    
#     print("Building cKDTree for instant spatial querying...")
#     tree = cKDTree(coords)
    
#     print("Calculating rigid partitions...")
#     rigid_boxes = list(set(partition_kdtree_rigid(coords, max_cells)))
    
#     print(f"Extracting {len(rigid_boxes)} patches with halo...")
#     for idx, (x1, x2, y1, y2) in enumerate(tqdm(rigid_boxes)):
#         # Calculate a generous circular radius to encompass the box + halo
#         center = [(x1 + x2) / 2, (y1 + y2) / 2]
#         search_radius = np.sqrt(((x2 - x1) / 2)**2 + ((y2 - y1) / 2)**2) + halo
        
#         # cKDTree instantly grabs the integer indices of cells in this local area
#         cell_indices = tree.query_ball_point(center, search_radius)
        
#         # Filter the circle down to the precise bounding box + halo
#         patch_coords = coords[cell_indices]
#         valid_mask = (patch_coords[:, 0] >= x1 - halo) & (patch_coords[:, 0] <= x2 + halo) & \
#                      (patch_coords[:, 1] >= y1 - halo) & (patch_coords[:, 1] <= y2 + halo)
        
#         final_indices = np.array(cell_indices)[valid_mask]
        
#         if len(final_indices) > 0:
#             from STHD import sthdio # Ensure sthdio is imported
#             # Slicing AnnData by integer index is vastly faster than boolean masking
#             crop_adata = sthd_data.adata[final_indices].copy()
#             crop_sthd = sthdio.STHD(crop_adata, load_type="anndata")
#             crop_sthd.save(f"{patch_path}/patch_{idx}")

# def _load_into_dict(res_dict, file, columns):
#     data = train.load_pdata(file)
#     indices = data.index.tolist()
#     values = data.values
#     for i, barcode in enumerate(indices):
#         res_dict[barcode].append(values[i])

# def _process_barcode(res_dict, columns):
#     id_STHD_pred_ct = columns.index("STHD_pred_ct")
#     for barcode in tqdm(res_dict):
#         data = np.array(res_dict[barcode])
#         data = np.delete(data, id_STHD_pred_ct, axis=1)
#         data_non_filtered = data[data[:, -1] != -1]
        
#         if len(data) == 1 or len(data_non_filtered) == 0:
#             res_dict[barcode] = data[0]
#         else:
#             res_dict[barcode] = data_non_filtered.mean(axis=0)

# def _combine_patch(patch_dir):
#     files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]
#     res_dict = defaultdict(list)
#     columns = train.load_pdata(files[0]).columns.tolist()

#     for file in tqdm(files):
#         _load_into_dict(res_dict, file, columns)

#     _process_barcode(res_dict, columns)

#     columns_remove_prediction = columns.copy()
#     columns_remove_prediction.remove("STHD_pred_ct")
#     pdata = pd.DataFrame.from_dict(res_dict, orient="index", columns=columns_remove_prediction)
#     pdata.index = pdata.index.astype(str)
#     return pdata

# def merge(save_path, refile):
#     allregion_path = f"{save_path}/all_region"
#     patch_path = f"{save_path}/patches"

#     sthdata = train.load_data(allregion_path)
#     # CODEX CHANGE: Removed obsolete RNA keyword arguments
#     sthdata, genemeanpd_filtered = train.sthdata_match_refgene(sthdata, refile)
#     pdata = _combine_patch(patch_path)

#     align_sthdata_pdata = train.add_pdata(sthdata, pdata)
#     pdata_reorder = align_sthdata_pdata.adata.obs[[t for t in align_sthdata_pdata.adata.obs.columns if "p_ct_" in t]]
    
#     sthdata_with_pdata = train.predict(sthdata, pdata_reorder.values, genemeanpd_filtered, mapcut=0.8)
#     train.save_prediction_pdata(sthdata_with_pdata, file_path=allregion_path)


import os
import pathlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
from STHD import train
from collections import defaultdict

def partition_kdtree_rigid(spatial_coords, max_cells):
    """Recursively splits the physical space WITHOUT calculating the halo yet."""
    if len(spatial_coords) <= max_cells:
        x_min, y_min = np.min(spatial_coords, axis=0)
        x_max, y_max = np.max(spatial_coords, axis=0)
        return [(x_min, x_max, y_min, y_max)]
    
    spreads = np.ptp(spatial_coords, axis=0)
    split_dim = np.argmax(spreads)
    median_val = np.median(spatial_coords[:, split_dim])
    
    left_mask = spatial_coords[:, split_dim] <= median_val
    right_mask = spatial_coords[:, split_dim] > median_val
    
    patches = []
    if np.sum(left_mask) > 0:
        patches.extend(partition_kdtree_rigid(spatial_coords[left_mask], max_cells))
    if np.sum(right_mask) > 0:
        patches.extend(partition_kdtree_rigid(spatial_coords[right_mask], max_cells))
        
    return patches

def patchify(sthd_data, save_path, max_cells=5000, halo=50.0):
    allregion_path = f"{save_path}/all_region"
    patch_path = f"{save_path}/patches"

    pathlib.Path(allregion_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(patch_path).mkdir(parents=True, exist_ok=True)

    sthd_data.save(allregion_path)
    
    coords = sthd_data.adata.obsm["spatial"]
    
    print("Building cKDTree for instant spatial querying...")
    tree = cKDTree(coords)
    
    print("Calculating rigid partitions...")
    rigid_boxes = list(set(partition_kdtree_rigid(coords, max_cells)))
    
    print(f"Extracting {len(rigid_boxes)} patches with halo...")
    for idx, (x1, x2, y1, y2) in enumerate(tqdm(rigid_boxes)):
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        search_radius = np.sqrt(((x2 - x1) / 2)**2 + ((y2 - y1) / 2)**2) + halo
        
        cell_indices = tree.query_ball_point(center, search_radius)
        
        patch_coords = coords[cell_indices]
        valid_mask = (patch_coords[:, 0] >= x1 - halo) & (patch_coords[:, 0] <= x2 + halo) & \
                     (patch_coords[:, 1] >= y1 - halo) & (patch_coords[:, 1] <= y2 + halo)
        
        final_indices = np.array(cell_indices)[valid_mask]
        
        if len(final_indices) > 0:
            from STHD import sthdio 
            crop_adata = sthd_data.adata[final_indices].copy()
            crop_sthd = sthdio.STHD(crop_adata, load_type="anndata")
            crop_sthd.save(f"{patch_path}/patch_{idx}")

def _load_into_dict(res_dict, file, columns):
    data = train.load_pdata(file)
    indices = data.index.tolist()
    values = data.values
    for i, barcode in enumerate(indices):
        res_dict[barcode].append(values[i])

def _process_barcode(res_dict, columns):
    # CHANGED to STHD_pred_niche
    id_STHD_pred_niche = columns.index("STHD_pred_niche")
    for barcode in tqdm(res_dict):
        data = np.array(res_dict[barcode])
        data = np.delete(data, id_STHD_pred_niche, axis=1)
        data_non_filtered = data[data[:, -1] != -1]
        
        if len(data) == 1 or len(data_non_filtered) == 0:
            res_dict[barcode] = data[0]
        else:
            res_dict[barcode] = data_non_filtered.mean(axis=0)

def _combine_patch(patch_dir):
    files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]
    res_dict = defaultdict(list)
    columns = train.load_pdata(files[0]).columns.tolist()

    for file in tqdm(files):
        _load_into_dict(res_dict, file, columns)

    _process_barcode(res_dict, columns)

    columns_remove_prediction = columns.copy()
    # CHANGED to STHD_pred_niche
    columns_remove_prediction.remove("STHD_pred_niche")
    pdata = pd.DataFrame.from_dict(res_dict, orient="index", columns=columns_remove_prediction)
    pdata.index = pdata.index.astype(str)
    return pdata

def merge(save_path, refile):
    allregion_path = f"{save_path}/all_region"
    patch_path = f"{save_path}/patches"

    sthdata = train.load_data(allregion_path)
    sthdata, genemeanpd_filtered = train.sthdata_match_refgene(sthdata, refile)
    pdata = _combine_patch(patch_path)

    align_sthdata_pdata = train.add_pdata(sthdata, pdata)
    
    # CHANGED to look for p_niche_
    pdata_reorder = align_sthdata_pdata.adata.obs[[t for t in align_sthdata_pdata.adata.obs.columns if "p_niche_" in t]]
    
    # CHANGED to match new predict signature (no genemeanpd_filtered needed)
    sthdata_with_pdata = train.predict(sthdata, pdata_reorder.values, mapcut=0.0)
    train.save_prediction_pdata(sthdata_with_pdata, file_path=allregion_path)