# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine.registry import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmocr into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmocr default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmocr`, and all registries will build modules from mmocr's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmocr.datasets  # noqa: F401,F403
    import mmocr.engine  # noqa: F401,F403
    import mmocr.evaluation  # noqa: F401,F403
    import mmocr.models  # noqa: F401,F403
    import mmocr.structures  # noqa: F401,F403
    import mmocr.visualization  # noqa: F401,F403
    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmocr')
        if never_created:
            DefaultScope.get_instance('mmocr', scope_name='mmocr')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmocr':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmocr", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmocr". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmocr-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmocr')
