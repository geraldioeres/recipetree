from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def delineate_forest(ortho_path, tiles_path="", file_name="pred_crown"):
    """
    Delineate trees on the forest's orthomosaic

    Args:
        ortho_path (str): orthomosaic file path
        tiles_path (str): tiling_path (optional)
        file_name (str): output file name (optional)

    Returns:
        cleaned_crown (geoDataFrame): cleaned overlapping delineated tree crowns
    """

    buffer = 20
    tile_width = 30
    tile_height = 30
    print("Tiling orthomosaic...")
    if tiles_path == "":
        tiles_path = id_generator() + "/"
    else:
        tiles_path = tiles_path
    tile_data(ortho_path, tiles_path, buffer, tile_width, tile_height, dtype_bool = True)
    print("Done tiling.")

    resnext_model = "/content/drive/MyDrive/Research/model_resnext/model_17.pth"
    base_resnext_model = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

    print("Delineating trees...")
    cfg = setup_cfg(base_model=base_resnext_model, update_model=resnext_model)
    predict_on_data(tiles_path, predictor=DefaultPredictor(cfg))
    project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")
    crowns = stitch_crowns(tiles_path + "predictions_geo/", 1)
    cleaned_crown = clean_crowns(crowns, 0.4, confidence=0.15)
    cleaned_crown.to_file(file_name + ".gpkg")
    print("\nProcess completed.")

    return cleaned_crown