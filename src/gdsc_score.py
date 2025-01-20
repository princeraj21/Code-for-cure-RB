import logging


def get_leaderboard_score(prediction_df, thresholds, evaluator):
    logger = logging.getLogger(__name__)
    ret = {}
    score = 0
    for threshold in thresholds:
        logger.info(f'Computing results for threshold: {threshold}')
        matched_section_df = evaluator.match_sections(prediction_df, iou_threshold=threshold)
        results_df = evaluator.evaluate_predictions(matched_section_df=matched_section_df)
        assert len(results_df) == 1
        score += results_df['detection_acc'][0]  # Final score is the sum of all detection accuracies
        ret[f'detection_acc@iou{threshold}'] = results_df['detection_acc'][0]
        ret[f'detection_tp@iou{threshold}'] = results_df['detection_tp'][0]
        ret[f'detection_fp@iou{threshold}'] = results_df['detection_fp'][0]
        ret[f'detection_fn@iou{threshold}'] = results_df['detection_fn'][0]
    ret['score'] = score
    return ret


if __name__ == '__main__':
    from PredictionEvaluator import PredictionEvaluator
    from gdsc_util import load_sections_df
    prediction_df = load_sections_df('data/predictions_test.csv')
    section_df = load_sections_df('data/gdsc_test.csv')
    evaluator = PredictionEvaluator(section_df)
    thresholds = [0.5, 0.6, 0.7]
    ret = get_leaderboard_score(prediction_df, thresholds, evaluator)
    print(ret)
