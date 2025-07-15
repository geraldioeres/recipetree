import geopandas as gpd
import math
from rasterstats import zonal_stats
import rasterio as rio
import numpy as np

def dbh_estimation(pred_poly):
    """
    Function to estimate tree DBH based on the crown diameter (cm).
    Crown diameter is derived from the crown area (polygon area) (m2).

    Args:
        pred_poly (shapely.Poygon): polygon of the corresponding tree crown

    Returns:
        dbh_est (float): estimated tree DBH
    """

    # Crown area
    crown_area = pred_poly.area

    # Crown diameter calculation
    crown_diameter = 2 * math.sqrt(crown_area / math.pi)

    # DBH estimation using sigmoidal function (R2 = 0.756, RMSE = 5.22 cm)
    # Coefficient used in this funciton are from (Iizuka, 2022)
    dbh_est = 20.098 + (37.61) / (1 + math.exp(-1.281 * (crown_diameter - 4.5)))
    dbh_est = round(dbh_est, 2)
    return dbh_est


def tree_height_zstats(pred_poly, chm_path):
    """
    Estimate highest CHM raster value on the area of interest (polygon).
    Max value on the polygon zone will be used as the tree height.

    Args:
        pred_poly (shapely.Polygon): polygon of the corresponding tree crown
        chm_path (String): path of the CHM (or subtracted DSM)

    Returns:
        tree_height (float): estimated tree height
    """

    # Zonal statistic of CHM, finding the tree top (maximum height)
    tree_height = zonal_stats(pred_poly, chm_path, stats="max")
    tree_height = round(tree_height[0]["max"], 2)
    return tree_height

def above_ground_biomass_est(dbh_est, height_est):
    """
    Estimate the AGB (Above Ground Biomass) based on estimated DBH and height
    Allometric using [Qin, 2021]

    Args:
        dbh_est (float): estimated tree DBH in cm
        height_est (float): estimated tree height in m

    Returns:
        total_AGB (float): total AGB of single tree in kg
    """

    # Above Ground Biomass (AGB) calculation
    stem_AGB = 0.0263 * pow((pow(dbh_est, 2) * height_est), 0.9695)
    branch_AGB = 0.0232 * pow((pow(dbh_est, 2) * height_est), 0.8055)
    leaf_AGB = 0.0075 * pow((pow(dbh_est, 2) * height_est), 0.8015)

    total_AGB = stem_AGB + branch_AGB + leaf_AGB
    total_AGB = round(total_AGB, 2)
    # print(total_AGB)
    return total_AGB

def carbon_estimations(total_AGB, carbon_coef=0.5):
    """
    Estimate carbon stock of a single tree and its carbon sequestration in kg

    Args:
        total_AGB (float): estimated total ABG of single tree in kg
        carbon_coef (float): carbon coefficient (default = 0.5)

    Returns:
       tuple: estimated carbon stock and sequestration in kg
    """

    # Carbon stock estimation
    carbon_coef = 0.495

    carbon_stock = total_AGB * carbon_coef

    carbon_seq = (44 / 12) * carbon_stock

    carbon_stock = round(carbon_stock, 2)
    carbon_seq = round(carbon_seq, 2)

    return carbon_stock, carbon_seq

def estimations_handler(pred_path, chm_path, out_path, out_file):
    """
    Handler for various tree parameters estimations that inlcudes:
        - Tree height estimation
        - DBH estimation
        - AGB estimation
        - Carbon stock estimation

    Args:
        pred_path (String): path of the predicted tree crown polygons
        chm_path (String): path of the CHM (subtracted DSM)

    Returns:
        ???
    """

    # Read crown file and
    pred_crowns = gpd.read_file(pred_path)
    pred_polys = list(pred_crowns.geometry)

    pred_mod = pred_crowns.copy()
    pred_crowns["dbh_est"] = 0.00
    pred_crowns["height_est"] = 0.00
    pred_crowns["total_agb"] = 0.00
    pred_crowns["carbon_stc"] = 0.00
    pred_crowns["carbon_seq"] = 0.00

    for p_idx, pred_poly in enumerate(pred_polys):
        dbh_est = dbh_estimation(pred_poly)
        tree_height = tree_height_zstats(pred_poly, chm_path)
        total_AGB = above_ground_biomass_est(dbh_est, tree_height)
        carbon_stock, carbon_seq = carbon_estimations(total_AGB)
        pred_mod.loc[p_idx, 'dbh_est'] = dbh_est # DBH estimation
        pred_mod.loc[p_idx, 'height_est'] = tree_height # Height estimation
        pred_mod.loc[p_idx, 'total_agb'] = total_AGB # Total AGB
        pred_mod.loc[p_idx, 'carbon_stc'] = carbon_stock # Carbon stock
        pred_mod.loc[p_idx, 'carbon_seq'] = carbon_seq # Carbon sequestration
        # print("====================================")
        # if p_idx == 20:
        #     break

    pred_mod.to_file(out_path + "/" + out_file)
