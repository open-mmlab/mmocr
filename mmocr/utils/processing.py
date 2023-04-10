# Copyright (c) OpenMMLab. All rights reserved.
import sys
from collections.abc import Iterable

from mmengine.utils.progressbar import ProgressBar, init_pool


def track_parallel_progress_multi_args(func,
                                       args,
                                       nproc,
                                       initializer=None,
                                       initargs=None,
                                       bar_width=50,
                                       chunksize=1,
                                       skip_first=False,
                                       file=sys.stdout):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (tuple[Iterable]): A tuple of tasks.
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    assert isinstance(args, tuple)
    for arg in args:
        assert isinstance(arg, Iterable)
    assert len(set([len(arg)
                    for arg in args])) == 1, 'args must have same length'
    task_num = len(args[0])
    tasks = zip(*args)

    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    gen = pool.starmap(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write('\n')
    pool.close()
    pool.join()
    return results
