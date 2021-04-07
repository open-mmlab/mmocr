<a id="markdown-contributing-to-mmocr" name="contributing-to-mmocr"></a>
# Contributing to mmocr

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components
<!-- TOC -->

- [Contributing to mmocr](#contributing-to-mmocr)
  - [Workflow](#workflow)
    - [Main Steps](#main-steps)
    - [Detailed Steps](#detailed-steps)
      - [Step 1: Create a Fork](#step-1-create-a-fork)
      - [Step 2: Develop a new feature](#step-2-develop-a-new-feature)
        - [Step 2.1: Keep your fork up to date](#step-21-keep-your-fork-up-to-date)
        - [Step 2.2: Create a feature branch](#step-22-create-a-feature-branch)
          - [Create an issue on github](#create-an-issue-on-github)
          - [Create branch](#create-branch)
      - [Step 3: Commit your changes](#step-3-commit-your-changes)
      - [Step 4: Prepare to Pull Request](#step-4-prepare-to-pull-request)
        - [Step 4.1: Merge official repo updates to your fork](#step-41-merge-official-repo-updates-to-your-fork)
        - [Step 4.2: Push <your_feature_branch> branch to your remote forked repo,](#step-42-push-your_feature_branch-branch-to-your-remote-forked-repo)
        - [Step 4.3: Create a Pull Request](#step-43-create-a-pull-request)
        - [Step 4.4: Review code](#step-44-review-code)
        - [Step 4.5: Revise <your_feature_branch>  (optional)](#step-45-revise-your_feature_branch--optional)
        - [Step 4.6: Delete <your_feature_branch> branch if your PR is accepted.](#step-46-delete-your_feature_branch-branch-if-your-pr-is-accepted)
  - [Code style](#code-style)
    - [Python](#python)
    - [C++ and CUDA](#c-and-cuda)

<!-- /TOC -->
<a id="markdown-workflow" name="workflow"></a>
## Workflow
<a id="markdown-main-steps" name="main-steps"></a>
### Main Steps
1. fork and pull the latest mmocr
2. checkout a new branch (do not use main branch for PRs)
3. commit your changes
4. create a PR

**Note**
- If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.
- If you are the author of some papers and would like to include your method to mmocr, please let us know (open an issue or contact the maintainers). We will much appreciate your contribution.
- For new features and new modules, unit tests are required to improve the code's robustness.

<a id="markdown-detailed-steps" name="detailed-steps"></a>
### Detailed Steps

The official public [repository](https://github.com/open-mmlab/mmocr) holds only one branch with an infinite lifetime: *main*

The *main* branch is the main branch where the source code of **HEAD** always reflects a state with the latest development changes for the next release.

Feature branches are used to develop new features for the upcoming or a distant future release.


All new developers to **MMOCR** need to follow the following steps:

<a id="markdown-step-1-create-a-fork" name="step-1-create-a-fork"></a>
#### Step 1: Create a Fork

1. Fork the repo on GitHub or GitLab to your personal account. Click the `Fork` button on the [project page](https://github.com/open-mmlab/mmocr).

2. Clone your new forked repo to your computer.
```
git clone https://github.com/<your name>/mmocr.git
```
3. Add the official repo as an upstream:
```
git remote add upstream https://github.com/open-mmlab/mmocr.git
```

<a id="markdown-step-2-develop-a-new-feature" name="step-2-develop-a-new-feature"></a>
#### Step 2: Develop a new feature

<a id="markdown-step-21-keep-your-fork-up-to-date" name="step-21-keep-your-fork-up-to-date"></a>
##### Step 2.1: Keep your fork up to date

Whenever you want to update your fork with the latest upstream changes, you need to fetch the upstream repo's branches and latest commits to bring them into your repository:

```
# Fetch from upstream remote
git fetch upstream

# Update your main branch
git checkout main
git rebase upstream/main
git push origin main
```

<a id="markdown-step-22-create-a-feature-branch" name="step-22-create-a-feature-branch"></a>
##### Step 2.2: Create a feature branch
<a id="markdown-create-an-issue-on-githubhttpsgithubcomopen-mmlabmmocr" name="create-an-issue-on-githubhttpsgithubcomopen-mmlabmmocr"></a>
###### Create an issue on [github](https://github.com/open-mmlab/mmocr)
- The title of the issue should be one of the following formats: `[Feature]: xxx`, `[Fix]: xxx`, `[Enhance]: xxx`, `[Refactor]: xxx`.
- More details can be written in the comments of the issue.

<a id="markdown-create-branch" name="create-branch"></a>
###### Create branch
```
git checkout -b feature/iss_<index> main
# index is the issue number above
```

<a id="markdown-step-3-commit-your-changes" name="step-3-commit-your-changes"></a>
#### Step 3: Commit your changes

Develop your new feature and test it to make sure it works well, then commit.

Please run
```
pre-commit run --all-files
pytest tests
```
and fix all failures before every git commit.
```
git commit -m "fix #<issue_index>: <commit_message>"
```

<a id="markdown-step-4-prepare-to-pull-request" name="step-4-prepare-to-pull-request"></a>
#### Step 4: Prepare to Pull Request
- Make sure to link your pull request to the related issue. Please refer to the [instructon](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue)


<a id="markdown-step-41-merge-official-repo-updates-to-your-fork" name="step-41-merge-official-repo-updates-to-your-fork"></a>
##### Step 4.1: Merge official repo updates to your fork

```
# fetch from upstream remote. i.e., the official repo
git fetch upstream

# update the main branch of your fork
git checkout main
git rebase upsteam/main
git push origin main

# update the <your_feature_branch> branch
git checkout <your_feature_branch>
git rebase main
# solve conflicts if any and Test
```

<a id="markdown-step-42-push-your_feature_branch-branch-to-your-remote-forked-repo" name="step-42-push-your_feature_branch-branch-to-your-remote-forked-repo"></a>
##### Step 4.2: Push <your_feature_branch> branch to your remote forked repo,
```
git checkout <your_feature_branch>
git push origin <your_feature_branch>
```
<a id="markdown-step-43-create-a-pull-request" name="step-43-create-a-pull-request"></a>
##### Step 4.3: Create a Pull Request

Go to the page for your fork on GitHub, select your new feature branch, and click the pull request button to integrate your feature branch into the upstream remote’s develop branch.

<a id="markdown-step-44-review-code" name="step-44-review-code"></a>
##### Step 4.4: Review code


<a id="markdown-step-45-revise-your_feature_branch--optional" name="step-45-revise-your_feature_branch--optional"></a>
##### Step 4.5: Revise <your_feature_branch>  (optional)
If PR is not accepted, pls follow steps above till your PR is accepted.

<a id="markdown-step-46-delete-your_feature_branch-branch-if-your-pr-is-accepted" name="step-46-delete-your_feature_branch-branch-if-your-pr-is-accepted"></a>
##### Step 4.6: Delete <your_feature_branch> branch if your PR is accepted.
```
git branch -d <your_feature_branch>
git push origin :<your_feature_branch>
```


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
