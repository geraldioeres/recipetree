from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor


def delineate_forest(ortho_path, tiles_path=""):
    """
    Delineate trees on the forest's orthomosaic

    Args:
        ortho_path (str): orthomosaic file path
        tiles_path (str): tiling_path (optional)

    Returns:
        cleaned_crown (geoDataFrame): cleaned overlapping delineated tree crowns
    """

    buffer = 20
    tile_width = 30
    tile_height = 30
    print("Tiling orthomosaic...")
    if tiles_path == "":
        tiles_path = "/tilespred_temp/"
    else:
        tiles_path = tiles_path
    tile_data(ortho_path, tiles_path, buffer, tile_width, tile_height, dtype_bool = True)
    print("Done tiling.")

    resnet_pr_model = "/content/drive/MyDrive/Research/model_resnet_pointrend/model_9.pth"
    base_resnetpr_model= "/content/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco.yaml"

    print("Delineating trees...")
    cfg = setup_cfg(model_source="point_rend", base_model=base_resnetpr_model, update_model=resnet_pr_model)
    predict_on_data(tiles_path, predictor=DefaultPredictor(cfg))
    project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")
    crowns = stitch_crowns(tiles_path + "predictions_geo/", 1)
    cleaned_crown = clean_crowns(crowns, 0.4, confidence=0.15)
    cleaned_crown.to_file("crown_pred" + ".gpkg")
    print("\nProcess completed.")

    return cleaned_crown



