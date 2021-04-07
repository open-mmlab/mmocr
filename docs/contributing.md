This document describes the fork & merge request workflow that should be used when contributing to **MMOCR**.
<!-- TOC -->

- [Step 1: Create a Fork](#step-1-create-a-fork)
- [Step 2: Develop a new feature](#step-2-develop-a-new-feature)
  - [Step 2.1: Keep your fork up to date](#step-21-keep-your-fork-up-to-date)
  - [<span id = "step2.2">Step 2.2: Create a feature branch</span>](#step-22-create-a-feature-branch)
    - [Create an issue on github](#create-an-issue-on-github)
    - [Create branch](#create-branch)
  - [Step 2.3: Develop and test <your_new_feature>](#step-23-develop-and-test-your_new_feature)
  - [Step 2.4: Prepare to Pull Request](#step-24-prepare-to-pull-request)
    - [Merge official repo updates to your fork](#merge-official-repo-updates-to-your-fork)
    - [Push <your_new_feature> branch to your remote forked repo,](#push-your_new_feature-branch-to-your-remote-forked-repo)
  - [Step 2.5: Create a Pull Request](#step-25-create-a-pull-request)
  - [Step 2.6: Review code](#step-26-review-code)
  - [Step 2.7: Revise <your_new_feature>  (optional)](#step-27-revise-your_new_feature--optional)
  - [Step 2.8: Delete <your_new_feature> branch if your PR is accepted.](#step-28-delete-your_new_feature-branch-if-your-pr-is-accepted)

<!-- /TOC -->
The official public [repository](https://github.com/open-mmlab/mmocr) holds only one branch with an infinite lifetime only:
+ main

The *main* branch is the main branch where the source code of **HEAD** always reflects a *production-ready state*.

Feature branches are used to develop new features for the upcoming or a distant future release.


All new developers to **MMOCR** need to follow the following steps:

<a id="markdown-step-1-create-a-fork" name="step-1-create-a-fork"></a>
### Step 1: Create a Fork

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
### Step 2: Develop a new feature

<a id="markdown-step-21-keep-your-fork-up-to-date" name="step-21-keep-your-fork-up-to-date"></a>
#### Step 2.1: Keep your fork up to date

Whenever you want to update your fork with the latest upstream changes, you need to fetch the upstream repo's branches and latest commits to bring them into your repository:

```
# Fetch from upstream remote
git fetch upstream

# Update your main branch
git checkout main
git rebase upstream/main
git push origin main
```

<a id="markdown-span-id--step22step-22-create-a-feature-branchspan" name="span-id--step22step-22-create-a-feature-branchspan"></a>
#### <span id = "step2.2">Step 2.2: Create a feature branch</span>
<a id="markdown-create-an-issue-on-githubhttpsgithubcomopen-mmlabmmocr" name="create-an-issue-on-githubhttpsgithubcomopen-mmlabmmocr"></a>
##### Create an issue on [github](https://github.com/open-mmlab/mmocr)
- The title of the issue should be one of the following formats: `[Feature]: xxx`, `[Fix]: xxx`, `[Enhance]: xxx`, `[Refactor]: xxx`.
- More details can be written in the comments of the issue.

<a id="markdown-create-branch" name="create-branch"></a>
##### Create branch
```
git checkout -b feature/iss_<index> main
# index is the issue number above
```

<a id="markdown-step-23-develop-and-test-your_new_feature" name="step-23-develop-and-test-your_new_feature"></a>
#### Step 2.3: Develop and test <your_new_feature>

Develop your new feature and test it to make sure it works well.

Please run
```
pre-commit run --all-files
pytest tests
```
and fix all failures before every git commit.
```
git commit -m "fix #<issue_index>: <commit_message>"
```
**Note:**
- <issue_index> is the [issue](#step2.2) number.

<a id="markdown-step-24-prepare-to-pull-request" name="step-24-prepare-to-pull-request"></a>
#### Step 2.4: Prepare to Pull Request
- Make sure to link your pull request to the related issue. Please refer to the [instructon](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue)


<a id="markdown-merge-official-repo-updates-to-your-fork" name="merge-official-repo-updates-to-your-fork"></a>
##### Merge official repo updates to your fork

```
# fetch from upstream remote. i.e., the official repo
git fetch upstream

# update the main branch of your fork
git checkout main
git rebase upsteam/main
git push origin main

# update the <your_new_feature> branch
git checkout <your_new_feature>
git rebase main
# solve conflicts if any and Test
```

<a id="markdown-push-your_new_feature-branch-to-your-remote-forked-repo" name="push-your_new_feature-branch-to-your-remote-forked-repo"></a>
##### Push <your_new_feature> branch to your remote forked repo,
```
git checkout <your_new_feature>
git push origin <your_new_feature>
```
<a id="markdown-step-25-create-a-pull-request" name="step-25-create-a-pull-request"></a>
#### Step 2.5: Create a Pull Request

Go to the page for your fork on GitHub, select your new feature branch, and click the pull request button to integrate your feature branch into the upstream remote’s develop branch.

<a id="markdown-step-26-review-code" name="step-26-review-code"></a>
#### Step 2.6: Review code


<a id="markdown-step-27-revise-your_new_feature--optional" name="step-27-revise-your_new_feature--optional"></a>
#### Step 2.7: Revise <your_new_feature>  (optional)
If PR is not accepted, pls follow Step 2.1, 2.3, 2.4 and 2.5 till your PR is accepted.

<a id="markdown-step-28-delete-your_new_feature-branch-if-your-pr-is-accepted" name="step-28-delete-your_new_feature-branch-if-your-pr-is-accepted"></a>
#### Step 2.8: Delete <your_new_feature> branch if your PR is accepted.
```
git branch -d <your_new_feature>
git push origin :<your_new_feature>
```
