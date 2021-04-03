# Contributing to mmocr

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

This document describes the fork & merge request workflow that should be used when contributing to **MMOCR**.

The official public [repository](https://github.com/open-mmlab/mmocr) holds two branches with an infinite lifetime only:
+ master
+ develop

The *master* branch is the main branch where the source code of **HEAD** always reflects a *production-ready state*.

The *develop* branch is the branch where the source code of **HEAD** always reflects a state with the latest development changes for the next release.

Feature branches are used to develop new features for the upcoming or a distant future release.

![](res/git-workflow-master-develop.png)

All new developers to **MMOCR** need to follow the following steps:

### Step 1: creating a Fork

1. Fork the repo on GitHub or GitLab to your personal account. Click the `Fork` button on the [project page](https://github.com/open-mmlab/mmocr).

2. Clone your new forked repo to your computer.
```
git clone https://github.com/<your name>/mmocr.git
```
3. Add the official repo as an upstream:
```
git remote add upstream https://github.com/open-mmlab/mmocr.git
```

### Step 2: develop a new feature

#### Step 2.1: keeping your fork up to date

Whenever you want to update your fork with the latest upstream changes, you need to fetch the upstream repo's branches and latest commits to bring them into your repository:

```
# Fetch from upstream remote
git fetch upstream

# Update your master branch
git checkout master
git rebase upstream/master
git push origin master

# Update your develop branch
git checkout develop
git rebase upsteam/develop
git push origin develop
```

#### <span id = "step2.2">Step 2.2: creating a feature branch</span>
- Creating an issue on [github](https://github.com/open-mmlab/mmocr)
- The title of the issue should be one of the following formats: `[feature]: xxx`, `[fix]: xxx`, `[Enhance]: xxx`, `[Refactor]: xxx`.

- ```
  git checkout -b feature/iss_<index> develop
  # index is the issue number above
  ```
Till now, your fork has three branches as follows:

![](res/git-workflow-feature.png)

#### Step 2.3: develop and test <your_new_feature>

Develop your new feature and test it to make sure it works well.

Pls run
```
pre-commit run --all-files
pytest tests
```
and fix all failures before every git commit.
```
git commit -m "fix #<issue_index>: <commit_message>" --no-verify
```
**Note:**
- <issue_index> is the [issue](#step2.2) number.
- <commit_message> should be the same with the title of [issue](#step2.2).

#### Step 2.4: prepare to PR


##### Merge official repo updates to your fork

```
# fetch from upstream remote. i.e., the official repo
git fetch upstream

# update the develop branch of your fork
git checkout develop
git rebase upsteam/develop
git push origin develop

# update the <your_new_feature> branch
git checkout <your_new_feature>
git rebase develop
# solve conflicts if any and Test
```

##### Push <your_new_feature> branch to your remote forked repo,
```
git checkout <your_new_feature>
git push origin <your_new_feature>
```
#### Step 2.5: send PR

Go to the page for your fork on GitHub, select your new feature branch, and click the pull request button to integrate your feature branch into the upstream remote’s develop branch.

#### Step 2.6: review code


#### Step 2.7: revise <your_new_feature>  (optional)
If PR is not accepted, pls follow Step 2.1, 2.3, 2.4 and 2.5 till your PR is accepted.

#### Step 2.8: del <your_new_feature> branch if your PR is accepted.
```
git branch -d <your_new_feature>
git push origin :<your_new_feature>
```

## Code style

### Python
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

>Before you create a PR, make sure that your code lints and is formatted by yapf.

### C++ and CUDA
We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
