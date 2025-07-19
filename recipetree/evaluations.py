import geopandas as gpd
from shapely.ops import unary_union


def calculate_precision_recall_f1(confussion_matix):
    """
    Calculate precisoin, recall, and F1 based on the confussion matrix (TP, FP, FN)

    Args:
        confussion_matrix (tuple): confussion matrix values consisting TP, FP, FN.

    Returns:
        tuple: Precision and Recall scores.
    """
    # Confussion matrix values slicing
    true_positives, false_positives, false_negatives = confussion_matix

    # Calculate precision, recall, and F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def calculate_iou(prediction, ground_truth):
    """
    Calculate IoU between prediction and ground truth polygons.

    Args:
        prediction (GeoDataFrame): Predicted polygons.
        ground_truth (GeoDataFrame): Ground truth polygons.

    Returns:
        float: IoU score between prediction and ground truth.
    """
    # Convert prediction CRS to ground truth CRS
    if prediction.crs != ground_truth.crs:
        prediction = prediction.to_crs(ground_truth.crs)

    # Merge all polygons in prediction and ground truth
    merged_prediction = unary_union(prediction.geometry)
    merged_ground_truth = unary_union(ground_truth.geometry)

    # Calculate intersection and union areas
    intersection_area = merged_prediction.intersection(merged_ground_truth).area
    union_area = merged_prediction.union(merged_ground_truth).area

    # IoU calculation
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def prediction_evaluation2(prediction, ground_truth, prcat_path, gtcat_path, iou_threshold=0.5):
    """
    Compare prediction results and ground truth polygons and returns confusion matrix values.
    It also creates a new column for the category of the polygon for the visualization purpose.

    Args:
        prediction (GeoDataFrame): predicted polygons.
        ground_truth (GeoDataFrame): ground truth polygons.
        prcat_path (String): categorized prediction shapefile output filename and path
        gtcat_path (String): categorized ground truth shapefile output filename and path
        iou_threshold (float): IoU threshold to consider a match (default = 0.5).

    Returns:
        tuple: confusion matrix values (TP, FP, FN).
    """
    # Convert prediction CRS to ground truth CRS
    if prediction.crs != ground_truth.crs:
        prediction = prediction.to_crs(ground_truth.crs)

    # Convert polygons to lists for pairwise comparison
    predicted_polys = list(prediction.geometry)
    ground_truth_polys = list(ground_truth.geometry)

    prediction_mod = prediction.copy()
    prediction_mod["cat"] = 0 # category column
    prediction_mod["iou"] = 0.0 # iou column
    prediction_mod["gt_match"]= -1 # match column
    groundtr_mod = ground_truth.copy()
    groundtr_mod["cat"] = 0 # category column
    groundtr_mod["iou"] = 0.0 # iou column
    groundtr_mod["gt_match"]= -1 # match column


    # Track matches
    matched_gt = set()  # Track matched ground truth indices
    duplicated_polys = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    iou_list = []

    # Check each predicted polygon
    for p_idx, pred_poly in enumerate(predicted_polys):
        matched = False
        duplicated = False
        for idx, gt_poly in enumerate(ground_truth_polys):
            # Calculate IoU
            intersection_area = pred_poly.intersection(gt_poly).area
            union_area = pred_poly.union(gt_poly).area
            iou = intersection_area / union_area if union_area > 0 else 0
            prediction_mod.loc[p_idx, 'iou'] = iou # For IoU

            # Check if IoU exceeds threshold
            if iou >= iou_threshold:
                matched = True
                iou_list.append(iou)
                matched_gt.add(idx)  # Mark this ground truth as matched
                groundtr_mod.loc[idx, 'cat'] = 2 # Category 2 for Matched
                if groundtr_mod.loc[idx, 'gt_match'] != -1:
                    duplicated = True
                if groundtr_mod.loc[idx, 'gt_match'] == -1 or groundtr_mod.loc[idx, 'iou'] < iou: #Store the matching prediction
                    groundtr_mod.loc[idx, 'iou'] = iou
                    groundtr_mod.loc[idx, 'gt_match'] = p_idx #
                    # duplicated_polys += 1
                prediction_mod.loc[p_idx, 'gt_match'] = idx # Store the matching ground truth polygon
                break


        if matched:
            true_positives += 1
            prediction_mod.loc[p_idx, 'cat'] = 2 # Category 2 for TP
        else:
            false_positives += 1
            # prediction_mod.loc[p_idx, 'cat'] = 1 # Category 1 for FP

        if duplicated:
            duplicated_polys += 1

    # Re-Count the True positives and False positives considering the duplication
    true_positives = true_positives - duplicated_polys
    false_positives = false_positives

    # Count false negatives (unmatched ground truth polygons)
    false_negatives = len(ground_truth_polys) - len(matched_gt)

    print("True Positives", true_positives)
    print("False Positives", false_positives)
    print("False Negatives", false_negatives)
    # print(iou_list)
    avg_iou = sum(iou_list)/len(iou_list)
    print(f"Average IoU: {avg_iou:.4f}")

    prediction_mod.to_file(prcat_path)
    groundtr_mod.to_file(gtcat_path)

    return true_positives, false_positives, false_negatives

def over_under(prediction, ground_truth, ovud_iou_th=0.1):
    """
    Check occurrence of oversegmentation and undersegmentation of the ground truth polygons.
    Creates a new column for the category of the polygon for visualization purpose.

    Args:
        prediction (GeoDataFrame): predicted polygons.
        ground_truth (GeoDataFrame): ground truth polygons.
        ovud_iou_th (float): IoU threshold to consider a oversegmentation or undersegmentation (default = 0.1).

    Returns:
        tuple: confusion matrix values (TP, FP, FN), and oversegmentation count.
    """
    # Convert prediction CRS to ground truth CRS
    if prediction.crs != ground_truth.crs:
        prediction = prediction.to_crs(ground_truth.crs)

    predicted_polys = list(prediction.geometry)
    ground_truth_polys = list(ground_truth.geometry)

    prediction_mod = prediction.copy()
    prediction_mod["cat"] = 0 # category column
    prediction_mod["num_match"] = 0  # How many GT this prediction matches
    groundtr_mod = ground_truth.copy()
    groundtr_mod["cat"] = 0 # category column
    groundtr_mod["num_match"] = 0 # count of matching predictions

    oversegmented = 0
    undersegmented = 0


    for p_idx, pred_poly in enumerate(predicted_polys):
        for idx, gt_poly in enumerate(ground_truth_polys):
            # Calculate IoU
            intersection_area = pred_poly.intersection(gt_poly).area
            union_area = pred_poly.union(gt_poly).area
            iou = intersection_area / union_area if union_area > 0 else 0


            if iou >= ovud_iou_th:
                groundtr_mod.loc[idx, 'num_match'] += 1
                prediction_mod.at[p_idx, "num_match"] += 1

    for idx, gt_poly in enumerate(ground_truth_polys):
        num_matches = groundtr_mod.loc[idx, 'num_match']
        if num_matches > 1:
            oversegmented += 1
            groundtr_mod.loc[idx, 'cat'] = 3  # oversegmented

    for idx, pred_poly in enumerate(predicted_polys):
        num_matches = prediction_mod.loc[idx, 'num_match']
        if num_matches > 1:
            undersegmented += 1
            prediction_mod.loc[idx, 'cat'] = 4  # undersegmented

    print("Oversegmented:", oversegmented)
    print("Undersegmented:", undersegmented)

    return oversegmented, undersegmented