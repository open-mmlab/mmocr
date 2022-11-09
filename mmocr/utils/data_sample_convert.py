# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.structures import TextDetDataSample, TextSpottingDataSample


def det_to_spotting(det_data_sample: TextDetDataSample,
                    spotting_datga_sample: TextSpottingDataSample):
    """Convert detection data sample to spotting data sample.

    Args:
        det_data_sample (dict): Detection data sample.
        spotting_datga_sample (dict): Spotting data sample.

    Returns:
        dict: Spotting data sample.
    """
    spotting_datga_sample['pred_instances'] = det_data_sample['pred_instances']
    return spotting_datga_sample
