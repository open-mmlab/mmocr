# Copyright (c) OpenMMLab. All rights reserved.
"""MMOCR provides 20 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner',
    parent=MMENGINE_RUNNERS,
    # TODO: update the location when mmocr has its own runner
    locations=['mmocr.engine'])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    # TODO: update the location when mmocr has its own runner constructor
    locations=['mmocr.engine'])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop',
    parent=MMENGINE_LOOPS,
    # TODO: update the location when mmocr has its own loop
    locations=['mmocr.engine'])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['mmocr.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['mmocr.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=['mmocr.datasets.samplers'])
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmocr.datasets.transforms'])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmocr.models'])
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmocr.models'])
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmocr.models'])

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    # TODO: update the location when mmocr has its own optimizer
    locations=['mmocr.engine'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optimizer wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    # TODO: update the location when mmocr has its own optimizer wrapper
    locations=['mmocr.engine'])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    # TODO: update the location when mmocr has its own optimizer constructor
    locations=['mmocr.engine'])
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    # TODO: update the location when mmocr has its own parameter scheduler
    locations=['mmocr.engine'])
# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['mmocr.evaluation.metrics'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator',
    parent=MMENGINE_EVALUATOR,
    locations=['mmocr.evaluation.evaluator'])

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, locations=['mmocr.models'])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=['mmocr.visualization'])
# manage visualizer backend
VISBACKENDS = Registry(
    'visualizer backend',
    parent=MMENGINE_VISBACKENDS,
    locations=['mmocr.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'logger processor',
    parent=MMENGINE_LOG_PROCESSORS,
    # TODO: update the location when mmocr has its own log processor
    locations=['mmocr.engine'])
