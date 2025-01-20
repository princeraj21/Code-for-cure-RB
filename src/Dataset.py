import numpy as np
from mmcv.utils.logging import get_logger
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from gdsc_util import load_sections_df


@DATASETS.register_module()
class OnchoDataset(CustomDataset):
    CLASSES = (['section'])

    def load_annotations(self, ann_file):
        logger = get_logger(__name__)
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        try:
            section_df = load_sections_df(self.ann_file)
        except FileNotFoundError:
            logger.error(f'Missing file {self.ann_file}. Aborting')
            return

        data_infos = []
        # convert annotations to middle format        
        logger.info('Building Dataset')
        for image_id in np.unique(section_df.file_name):
            gt_bboxes = []
            gt_labels = []

            sec_tmp = section_df[section_df.file_name == image_id]
            height = sec_tmp.iloc[0].height
            width = sec_tmp.iloc[0].width
            data_info = dict(filename=f'{image_id}', width=width, height=height)
            for row in sec_tmp.itertuples():
                x_min = row.xmin
                y_min = row.ymin
                x_max = row.xmax
                y_max = row.ymax
                bbox = [x_min, y_min, x_max, y_max]
                gt_labels.append(cat2label['section'])
                gt_bboxes.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos