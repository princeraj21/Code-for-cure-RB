import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion


def get_dims(file_names: pd.Series) -> dict:
    """get test image dimensions

    Args:
        file_names (pd.Series): test data filenames

    Returns:
        dict: dimensions for test data images
    """
    dim_dict = dict()
    print('Getting dimensions from Images')
    for file in tqdm(file_names):
        img = cv2.imread(f'../data/jpgs/{file}')
        h, w = img.shape[:2]
        dim_dict[file] = {'height': h, 'width': w}

    return dim_dict


def scale_bounding_boxes(model_result: pd.DataFrame, dim_dict: dict) -> pd.DataFrame:
    """scale the bounding boxes of the results

    Args:
        model_result (pd.DataFrame): un-scaled model results
        dim_dict (dict): image dimensions

    Returns:
        pd.DataFrame: scaled model results
    """
    model_result['xmin_scaled'] = model_result.apply(lambda x : x['xmin']/dim_dict[x['file_name']]['width'], axis = 1)
    model_result['ymin_scaled'] = model_result.apply(lambda x : x['ymin']/dim_dict[x['file_name']]['height'], axis = 1)
    model_result['xmax_scaled'] = model_result.apply(lambda x : x['xmax']/dim_dict[x['file_name']]['width'], axis = 1)
    model_result['ymax_scaled'] = model_result.apply(lambda x : x['ymax']/dim_dict[x['file_name']]['height'], axis = 1)

    return model_result


def merge_results(fisrt_model_result: pd.DataFrame, second_model_result: pd.DataFrame, file_names: list, 
                 weights: list, iou_thr: float, skip_box_thr: float) -> pd.DataFrame:
    """ensemble the results from two models

    Args:
        fisrt_model_result (pd.DataFrame): results from first model
        second_model_result (pd.DataFrame): results from second model
        file_names (list): list of all the images
        weights (list): list of weights for each model
        iou_thr (float): IoU value for boxes to be a match
        skip_box_thr (float): exclude boxes with score lower than this variable

    Returns:
        pd.DataFrame: combined results on the data
    """
    ensemble_results = pd.DataFrame()
    print('Merging Results from the two models')
    for file in tqdm(file_names):
        first_model_file = fisrt_model_result[fisrt_model_result['file_name']==file]
        second_model_file = second_model_result[second_model_result['file_name']==file]

        first_model_boxes = list(first_model_file[['xmin_scaled', 'ymin_scaled', 'xmax_scaled', 'ymax_scaled']].values)
        second_model_boxes = list(second_model_file[['xmin_scaled', 'ymin_scaled', 'xmax_scaled', 'ymax_scaled']].values)
    
        first_model_score = list(first_model_file['detection_score'].values)
        second_model_score = list(second_model_file['detection_score'].values)

        first_model_labels = np.ones(first_model_file.shape[0])
        second_model_labels = np.ones(second_model_file.shape[0])

        boxes_list = [first_model_boxes, second_model_boxes]
        scores_list = [first_model_score, second_model_score]
        labels_list = [first_model_labels, second_model_labels]

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        
        file_results = pd.DataFrame(boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax'])
        file_results['detection_score'] = scores
        file_results['file_name'] = file

        ensemble_results = pd.concat([ensemble_results, file_results], ignore_index=True)

    return ensemble_results


def rescale_bounding_boxes(model_result: pd.DataFrame, dim_dict: dict) -> pd.DataFrame:
    """rescaling the bounding boxes of the results

    Args:
        model_result (pd.DataFrame): scaled model results
        dim_dict (dict): image dimensions

    Returns:
        pd.DataFrame: rescaled model results
    """
    model_result['xmin'] = model_result.apply(lambda x : np.round(x['xmin']*dim_dict[x['file_name']]['width']), axis = 1)
    model_result['ymin'] = model_result.apply(lambda x : np.round(x['ymin']*dim_dict[x['file_name']]['height']), axis = 1)
    model_result['xmax'] = model_result.apply(lambda x : np.round(x['xmax']*dim_dict[x['file_name']]['width']), axis = 1)
    model_result['ymax'] = model_result.apply(lambda x : np.round(x['ymax']*dim_dict[x['file_name']]['height']), axis = 1)

    return model_result


def generate_test_results(first_model_result_path: str, second_model_result_path: str, file_names: list, files_dim: dict = None,
                         weights: list = [2,1], iou_thr: float = 0.5, skip_box_thr: float = 0.0001) -> pd.DataFrame:
    """combine results from two models to generate a single result on the test file

    Args:
        first_model_result_path (str): file path for results from the first model
        second_model_result_path (str): file path for results from the second model
        file_names (list): list of filenames for which boxes needs to be ensembled
        files_dim (dict): dictionary having filename as key and a dictionary of {'height': height_of_image, 'width': width_of_image} as value. Default: None
        weights (list): list of weights for each model. Default: [2,1]
        iou_thr (float): IoU value for boxes to be a match. Default: 0.5
        skip_box_thr (float): exclude boxes with score lower than this variable. Default: 0.0001

    Returns:
        pd.DataFrame: final result on the test data
    """
    fisrt_model_result = pd.read_csv(first_model_result_path, sep = ';')
    second_model_result = pd.read_csv(second_model_result_path, sep = ';')
    
    file_names = pd.unique(file_names)
    if files_dim:
        test_files_dim = files_dim
    else:
        test_files_dim = get_dims(file_names)
    fisrt_model_result = scale_bounding_boxes(fisrt_model_result, test_files_dim)
    second_model_result = scale_bounding_boxes(second_model_result, test_files_dim)

    ensemble_results = merge_results(fisrt_model_result, second_model_result, file_names, weights, iou_thr, skip_box_thr)
    ensemble_results = rescale_bounding_boxes(ensemble_results, test_files_dim)
    
    ensemble_results['section_id'] = ensemble_results.apply(lambda x : f"{x['file_name']}@{x['xmin']}-{x['xmax']}-{x['ymin']}-{x['ymax']}", axis=1)

    return ensemble_results
