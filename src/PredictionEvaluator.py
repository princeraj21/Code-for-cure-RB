import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from metrics import compute_iou
from gdsc_util import set_up_logging


class PredictionEvaluator:
    """
    This class was built to assist with evaluating the results from a trained model.

    Args:
        section_df (pd.DataFrame): Dataframe containing the ground truth test data of the model.

    """

    def __init__(self, section_df: pd.DataFrame) -> None:
        set_up_logging()
        self.section_df = section_df
        self._logger = logging.getLogger(__name__)

    def match_sections(self, prediction_df: pd.DataFrame, iou_threshold: float = 0.5) -> pd.DataFrame:
        """
        Takes the predictions of an e2e model and matches them to the ground truth.
        Returns a dataframe that contains all sections from either the ground truth or the predictions
        and how they match together.

        Args:
            prediction_df (pd.DataFrame): Dataframe containing the predictions of the e2e model.
            iou_threshold (float, optional): IoU threshold for section matching. Defaults to 0.5.

        Returns:
            pd.DataFrame: Matched sections.
        """
        self._logger.info('Matching sections')
        matched_sections = []
        ious = []
        sec_ids = []

        for f in np.unique(self.section_df.file_name):
            res_tmp = prediction_df[prediction_df.file_name == f]
            sec_tmp = self.section_df[self.section_df.file_name == f]

            for row in sec_tmp.itertuples():
                iou_max = -1
                best_match = pd.NA
                sec_coords = (row.xmin, row.ymin, row.xmax, row.ymax)
                for res in res_tmp.itertuples():
                    res_coords = (res.xmin, res.ymin, res.xmax, res.ymax)
                    iou = compute_iou(res_coords, sec_coords)
                    if iou > iou_max:
                        iou_max = iou
                        if iou_max >= iou_threshold:
                            best_match = res.Index  # Store the index of the best match
                sec_ids.append(row.Index)
                matched_sections.append(best_match)
                ious.append(iou_max)

        self._logger.info('Merging matched sections')

        # Create a dataframe with the sec_ids, matched_sections and ious
        ious_df = pd.DataFrame({'sec_id': sec_ids, 'match': matched_sections, 'iou': ious})

        # Merge ious_df to the section_df
        section_df_tmp = pd.merge(self.section_df, ious_df, left_index=True, right_on='sec_id')

        matched_section_df = pd.merge(section_df_tmp, prediction_df,
                                      how='outer', left_on='match', right_index=True,
                                      suffixes=('_gt', '_pred'))
        matched_section_df.set_index('sec_id', inplace=True)
        self._logger.info('Done matching sections')

        return matched_section_df

    def evaluate_predictions(self,
                             prediction_df: pd.DataFrame = None,
                             matched_section_df: pd.DataFrame = None,
                             iou_threshold: float = 0.5,
                             detailed_evaluation: bool = False) -> pd.DataFrame:
        """
        Returns a dataframe with the results matched to the ground truth.

        Args:
            prediction_df (pd.DataFrame, optional): Dataframe containing the predictions
                            of the e2e model. This or matched_section_df must not be None.
            matched_section_df (pd.DataFrame, optional): Dataframe containing the results of
                            match_sections(). This or prediction_df must not be None.
            iou_threshold (float, optional): IoU threshold for section matching. Defaults to 0.5.
            detailed_evaluation: (bool, optional): False returns a Dataframe with just the
                            overview scoring statistics. True returns a Dataframe with scoring
                            statistics for each file. Defaults to False.

        Returns:
            pd.DataFrame:Scoring statistics for the predictions.
        """
        if detailed_evaluation:
            self._logger.info('Evaluating detailed predictions')
        else:
            self._logger.info('Evaluating predictions')

        if matched_section_df is None:
            if prediction_df is None:
                raise ValueError(
                    'Either prediction_df or matched_section_df must be provided.')
            matched_section_df = self.match_sections(prediction_df, iou_threshold)

        self._logger.info('Computing overall scores')
        metrics = score_matched_sections(matched_section_df)
        metrics['file_name'] = 'overview'

        all_metrics = [metrics]
        if detailed_evaluation:
            nan_mask = matched_section_df.file_name_gt.isna()
            files = np.unique(matched_section_df.file_name_gt[~nan_mask])
            for f in files:
                self._logger.info('Evaluating predictions for file {}'.format(f))
                mask = (matched_section_df.file_name_gt == f) | (matched_section_df.file_name_pred == f)
                # Only consider sections in the file
                matched_section_tmp = matched_section_df[mask]
                metrics = score_matched_sections(matched_section_tmp)
                metrics['file_name'] = f
                all_metrics.append(metrics)

        ret = pd.DataFrame(all_metrics)
        return ret


def score_matched_sections(matched_section_df: pd.DataFrame) -> tuple:
    """
    Returns overall score along with accuracy results for detection predictions

    Args:
        matched_section_df (pd.DataFrame): Dataframe containing the results of
                        match_sections().
    Returns:
        tuple: detection_acc
    """
    # Compute detection metrics
    # Nr of predictions not matched to a ground truth section
    det_fp = matched_section_df.index.isna().sum()
    # Nr of ground truths not matched to a predicted section
    det_fn = matched_section_df.match.isna().sum()
    no_match = det_fp + det_fn
    matching_sections_nr = len(matched_section_df) - no_match

    detection_acc = round(100 * matching_sections_nr /
                          len(matched_section_df), 2)

    ret = dict(
        detection_acc=detection_acc,
        detection_tp=matching_sections_nr,
        detection_fp=det_fp,
        detection_fn=det_fn
    )

    return ret
