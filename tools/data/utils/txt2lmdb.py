import argparse
import shutil
import sys
import time
from pathlib import Path

import lmdb


def converter(imglist, output, batch_size=1000, coding='utf-8'):
    # read imglist
    with open(imglist) as f:
        lines = f.readlines()

    # create lmdb database
    if Path(output).is_dir():
        while True:
            print('%s already exist, delete or not? [Y/n]' % output)
            Yn = input().strip()
            if Yn in ['Y', 'y']:
                shutil.rmtree(output)
                break
            elif Yn in ['N', 'n']:
                return
    print('create database %s' % output)
    Path(output).mkdir(parents=True, exist_ok=False)
    env = lmdb.open(output, map_size=1099511627776)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--imglist', '-i', required=True, help='input imglist path')
    parser.add_argument(
        '--output', '-o', required=True, help='output lmdb path')
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=10000,
        help='processing batch size, default 10000')
    parser.add_argument(
        '--coding',
        '-c',
        default='utf8',
        help='bytes coding scheme, default utf8')
    opt = parser.parse_args()

    converter(
        opt.imglist, opt.output, batch_size=opt.batch_size, coding=opt.coding)


if __name__ == '__main__':
    main()
