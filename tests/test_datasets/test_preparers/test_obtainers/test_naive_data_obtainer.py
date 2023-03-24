# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers.obtainers import NaiveDataObtainer


class TestNaiveDataObtainer(unittest.TestCase):

    def setUp(self) -> None:
        """Create temporary directories and files for testing."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.cache_path = osp.join(self.tmp_dir.name, 'cache')
        self.data_root = osp.join(self.tmp_dir.name, 'data')
        self.obtainer = NaiveDataObtainer([], self.cache_path, self.data_root,
                                          'test')

    def tearDown(self) -> None:
        """Delete temporary directories and files used for testing."""
        self.tmp_dir.cleanup()

    def test_move(self):
        # create tmp files
        test_src = os.path.join(self.data_root, 'src')
        test_dst = os.path.join(self.data_root, 'dst')
        os.makedirs(test_src, exist_ok=True)
        os.makedirs(test_dst, exist_ok=True)
        # Create some test files/folders in src directory
        for i in range(3):
            with open(os.path.join(test_src, f'file{i}.txt'), 'w') as f:
                f.write('hello world\n')
            os.mkdir(os.path.join(test_src, f'dir{i}'))

        # Test moving file/dir
        mapping = [
            ('src/file0.txt', 'dst/file0_new.txt'),  # dst/file0_new.txt
            ('src/file1.txt', 'dst/abc/abc.txt'),  # dst/abc.txt
            ('src/file2.txt', 'dst/'),  # Not allowed
            ('src/dir0/', 'dst/dir0'),  # dst/dir0
            ('src/dir1', 'dst/abc/d2/'),  # dst/abc/d2
            ('src/dir2', 'dst/'),  # not allowed
        ]
        self.obtainer.move(mapping)

        mapping[2] = ['src/file2.txt', 'dst/file2.txt']
        mapping[5] = ['src/dir2', 'dst/dir2']
        mapping = [[osp.join(self.data_root, a),
                    osp.join(self.data_root, b)] for a, b in mapping]
        mapping[2] = mapping[2][::-1]
        mapping[5] = mapping[5][::-1]
        for a, b in mapping:
            self.assertFalse(os.path.exists(a))
            self.assertTrue(os.path.exists(b))

        # Test moving paths with wildcard
        mapping = [
            ('src/*.txt', 'dst/test2'),  # dst/test2/file2.txt
            ('src/*', 'dst/test2/file2.txt'),  # not allowed (file2.txt exists)
            ('src/*', 'dst/test2'),  # dst/dir2
        ]
        self.obtainer.move(mapping)

        mapping = [
            osp.join(self.data_root, p)
            for p in ['dst/test2/file2.txt', 'dst/test2/dir2']
        ]
        for a, b in mapping:
            self.assertFalse(os.path.exists(a))
            self.assertTrue(os.path.exists(b))
