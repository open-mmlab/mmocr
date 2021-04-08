import glob
import os
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mmocr/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    import sys
    # return short version for sdist
    if 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        return locals()['short_version']
    else:
        return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strip
    specific version information.

    Args:
        fname (str): Path to requirements file.
        with_version (bool, default=False): If True, include version specs.
    Returns:
        info (list[str]): List of requirements items.
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def get_rroi_align_extensions():

    extensions_dir = 'mmocr/models/utils/ops/rroi_align/csrc/csc'
    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))
    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {'cxx': []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = [
            '-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]

    print(sources)
    include_dirs = [extensions_dir]
    print('include_dirs', include_dirs, flush=True)
    ext = extension(
        name='mmocr.models.utils.ops.rroi_align.csrc',
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )

    return ext


if __name__ == '__main__':
    library_dirs = [
        lp for lp in os.environ.get('LD_LIBRARY_PATH', '').split(':')
        if len(lp) > 1
    ]
    cpp_root = 'mmocr/models/textdet/postprocess/'
    setup(
        name='mmocr',
        version=get_version(),
        description='Text Detection, OCR, and NLP Toolbox',
        long_description=readme(),
        keywords='Text Detection, OCR, KIE, NLP',
        url='https://github.com/open-mmlab/mmocr',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        package_data={'mmocr.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[
            CppExtension(
                name='mmocr.models.textdet.postprocess.pan',
                sources=[cpp_root + 'pan.cpp']),
            CppExtension(
                name='mmocr.models.textdet.postprocess.pse',
                sources=[cpp_root + 'pse.cpp']),
            get_rroi_align_extensions()
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
