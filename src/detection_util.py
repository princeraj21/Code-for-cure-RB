import pandas as pd
import logging
import mmcv

from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm


def create_predictions(file_names: list, cfg, checkpoint, device='cpu') -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    model = init_detector(cfg, checkpoint, device)

    logger.info(f'Creating predictions for {len(file_names)} files')

    predictions = []

    for file_name in tqdm(file_names):
        logger.info(f'Processing file: {file_name}')
        img = mmcv.imread(f'{cfg.data_root}/jpgs/{file_name}')
        detections = inference_detector(model, img)

        for box in detections[0]:
            xmin, ymin, xmax, ymax, score = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            box_dict = dict(
                section_id=f'{file_name}@{xmin}-{xmax}-{ymin}-{ymax}',
                file_name=file_name,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                detection_score=score
            )
            predictions.append(box_dict)

    prediction_df = pd.DataFrame(predictions)
    prediction_df.set_index('section_id', inplace=True)
    return prediction_df