from rasterio.io import MemoryFile
import rasterio as rio
import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np


def crown_zonalstats(crown_path, chm_path, zonal_fname="zonal_cleaned", clean=True):
    """
    Perform zonal statistic on crown and the input CHM

    Args:
        crown_path (str or geoDataFrame): crown file path or crown geoDataFrame
        chm_path (str): path of the CHM file

    Returns:
        mod_in_crown (geoDataFrame): modified crown geoDataFrame
    """

    if type(crown_path) == str:
        in_crown = gpd.read_file(crown_path)
    else:
        in_crown = crown_path
    mod_in_crown = in_crown.copy()
    crown_list = list(mod_in_crown.geometry)
    mod_in_crown["height"] = 0.00
    mod_in_crown["th_median"] = 0.00
    mod_in_crown["th_std"] = 0.00
    mod_in_crown["category"] = ""


    for c_id, crown_poly in enumerate(crown_list):
        zstat_data = zonal_stats(crown_poly, chm_path, stats="max median std")
        tree_height = zstat_data[0]["max"]
        th_median = zstat_data[0]["median"]
        th_std = zstat_data[0]["std"]
        
        if tree_height == None:
            tree_height = 0
        if th_median == None:
            th_median = 0
        if th_std == None:
            th_std = 0

        mod_in_crown.loc[c_id, "height"] = tree_height
        mod_in_crown.loc[c_id, "th_median"] = th_median
        mod_in_crown.loc[c_id, "th_std"] = th_std

        if tree_height <= 0:
            mod_in_crown.loc[c_id, "category"] = "Not Tree"
        if th_median < 2:
            mod_in_crown.loc[c_id, "category"] = "Not Tree"
        else:
            mod_in_crown.loc[c_id, "category"] = "Tree"
        # mod_in_crown.loc[c_id, "height"] = tree_height
        # mod_in_crown.loc[c_id, "th_median"] = th_median
        # mod_in_crown.loc[c_id, "th_std"] = th_std

        # if tree_height <= 0:
        #     mod_in_crown.loc[c_id, "category"] = "Not Tree"
        # elif tree_height > 0 and (th_median < 2 or th_std < 2):
        #     mod_in_crown.loc[c_id, "category"] = "Not Tree"
        # else:
        #     mod_in_crown.loc[c_id, "category"] = "Tree"

    if clean:
        mod_in_crown = mod_in_crown[mod_in_crown["category"] == "Tree"]

    mod_in_crown.to_file(zonal_fname + ".gpkg")

    print("Crown has ben cleaned.")

    return mod_in_crown


def process_chm(dsm_path, dtm_path):
    """
    Generate CHM from DSM and DTM

    Args:
        dsm_path (str): path of the DSM file
        dtm_path (str): path of the DTM file

    Returns:
        ds_chm (array): CHM in np.ndarray format
        src_dsm.meta (dict): DSM metadata
    """

    src_dsm = rio.open(dsm_path)
    src_dtm = rio.open(dtm_path)

    ds_dsm = src_dsm.read()
    ds_dtm = src_dtm.read()

    ds_dsm_c = ds_dsm.copy()
    ds_dtm_c = ds_dtm.copy()

    ds_chm = ds_dsm_c - ds_dtm_c

    return ds_chm, src_dsm.meta


def zonal_cleaning(dsm_path, dtm_path, crown_path, to_file=False, file_name="temp_name", clean_crown=True):
    """
    Zonal cleaning handler and CHM generation

    Args:
        dsm_path (str): path of the DSM file
        dtm_path (str): path of the DTM file
        crown_path (str or geoDataFrame): crown file path or crown geoDataFrame
        to_file(boolean): output CHM to a file
        clean_crown(boolean): clean crown

    Returns:
        mod_in_crown (geoDataFrame): modified crown geoDataFrame from
                                     crown_zonalstats() function
    """

    # CHM generation from DSM and DTM
    ds_chm, out_meta = process_chm(dsm_path, dtm_path)

    print("Performing zonal cleaning...")
    # If user want CHM output
    if to_file:
        out_meta = out_meta

        # chm_path = site_path + "/CHM_4" + ".tif"
        chm_path = file_name + ".tif"
        with rio.open(chm_path, "w", **out_meta) as dest:
            dest.write(ds_chm)

        return crown_zonalstats(crown_path, chm_path, clean_crown)

    # Otherwise, chm only as temp file
    else:
        with MemoryFile() as temp_file:
          with temp_file.open(**out_meta) as dataset:
              dataset.write(ds_chm)

          return crown_zonalstats(crown_path, temp_file, clean_crown)