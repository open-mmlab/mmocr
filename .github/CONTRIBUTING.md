<a id="markdown-contributing-to-mmocr" name="contributing-to-mmocr"></a>
# Contributing to mmocr

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components
<!-- TOC -->

- [Contributing to mmocr](#contributing-to-mmocr)
  - [Workflow](#workflow)
  - [Code style](#code-style)
    - [Python](#python)
    - [C++ and CUDA](#c-and-cuda)

<!-- /TOC -->
<a id="markdown-workflow" name="workflow"></a>
## Workflow

1. fork and pull the latest mmocr
2. checkout a new branch (do not use main branch for PRs)
3. commit your changes
4. create a PR

Note

- Please refer to [contributing.md](/docs/contributing.md) for detail steps.
- If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.
- If you are the author of some papers and would like to include your method to mmocr, please let us know (open an issue or contact the maintainers). We will much appreciate your contribution.
- For new features and new modules, unit tests are required to improve the code's robustness.


<a id="markdown-code-style" name="code-style"></a>
## Code style
<a id="markdown-python" name="python"></a>
### Python
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [setup.cfg](../setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`,
 fixes `end-of-files`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](../.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```shell
pip install -U pre-commit
```

From the repository folder

```shell
pre-commit install
```

If you are facing issue when installing markdown lint, you may install ruby for markdown lint by following

```shell
# install rvm
curl -L https://get.rvm.io | bash -s -- --autolibs=read-fail
# set up environment
# Note that you might need to edit ~/.bashrc, ~/.bash_profile.
rvm autolibs disable
# install ruby
rvm install 2.7.1
```

After this on every commit check code linters and formatter will be enforced.

>Before you create a PR, make sure that your code lints and is formatted by yapf.

<a id="markdown-c-and-cuda" name="c-and-cuda"></a>
### C++ and CUDA
We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
