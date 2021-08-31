import glob

from mmcv import Config

from mmocr.datasets.pipelines.compose import Compose


def test_pipeline_composition():
    files = glob.glob('configs/**/*.py', recursive=True)
    for file in files:
        cfg = Config.fromfile(file)
        pipeline = cfg.get('train_pipeline', None)
        try:
            if pipeline:
                Compose(pipeline)
        except Exception as e:
            raise Exception(e, file)
        pipeline = cfg.get('test_pipeline', None)
        try:
            if pipeline:
                Compose(pipeline)
        except Exception as e:
            raise Exception(e, file)
