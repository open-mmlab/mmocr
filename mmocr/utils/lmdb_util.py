# Copyright (c) OpenMMLab. All rights reserved.
import shutil
import sys
import time
from pathlib import Path

import lmdb

from mmocr.utils import list_from_file


def lmdb_converter(img_list_file, output, batch_size=1000, coding='utf-8'):
    # read img_list_file
    lines = list_from_file(img_list_file)

    # create lmdb database
    if Path(output).is_dir():
        while True:
            print('%s already exist, delete or not? [Y/n]' % output)
            Yn = input().strip()
            if Yn in ['Y', 'y']:
                shutil.rmtree(output)
                break
            if Yn in ['N', 'n']:
                return
    print('create database %s' % output)
    Path(output).mkdir(parents=True, exist_ok=False)
    # map_size should be small for windows
    env = lmdb.open(output, map_size=1024000)

    # build lmdb
    beg_time = time.strftime('%H:%M:%S')
    for beg_index in range(0, len(lines), batch_size):
        end_index = min(beg_index + batch_size, len(lines))
        sys.stdout.write('\r[%s-%s], processing [%d-%d] / %d' %
                         (beg_time, time.strftime('%H:%M:%S'), beg_index,
                          end_index, len(lines)))
        sys.stdout.flush()
        batch = [(str(index).encode(coding), lines[index].encode(coding))
                 for index in range(beg_index, end_index)]
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.putmulti(batch, dupdata=False, overwrite=True)
    sys.stdout.write('\n')
    with env.begin(write=True) as txn:
        key = 'total_number'.encode(coding)
        value = str(len(lines)).encode(coding)
        txn.put(key, value)
    print('done', flush=True)
