# Contribution Guide

OpenMMLab welcomes everyone who is interested in contributing to our projects and accepts contribution in the form of PR.

## What is PR

`PR` is the abbreviation of `Pull Request`. Here's the definition of `PR` in the [official document](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) of Github.

```
Pull requests let you tell others about changes you have pushed to a branch in a repository on GitHub. Once a pull request is opened, you can discuss and review the potential changes with collaborators and add follow-up commits before your changes are merged into the base branch.
```

## Basic Workflow

1. Get the most recent codebase
2. Checkout a new branch from `main` or `dev-1.x` branch, depending on the version of the codebase you want to contribute to (see [Maintenance Plan](../migration/overview.md) for more details)
3. Commit your changes ([Don't forget to use pre-commit hooks!](#3-commit-your-changes))
4. Push your changes and create a PR
5. Discuss and review your code
6. Merge your branch to `main` or `dev-1.x` branch

## Procedures in detail

### 1. Get the most recent codebase

- When you work on your first PR

  Fork the OpenMMLab repository: click the **fork** button at the top right corner of Github page
  ![avatar](https://user-images.githubusercontent.com/22607038/195038780-06a46340-8376-4bde-a07f-2577f231a204.png)

  Clone forked repository to local

  ```bash
  git clone git@github.com:XXX/mmocr.git
  ```

  Add source repository to upstream

  ```bash
  git remote add upstream git@github.com:open-mmlab/mmocr
  ```

- After your first PR

  Checkout the latest branch of the local repository and pull the latest branch of the source repository. Here we assume that you are working on the `dev-1.x` branch.

  ```bash
  git checkout dev-1.x
  git pull upstream dev-1.x
  ```

### 2. Checkout a new branch from the `main` branch or `dev-1.x` branch

```bash
git checkout -b branchname
```

```{tip}
To make commit history clear, we strongly recommend you checkout the `main` or `dev-1.x` branch before creating a new branch.
```

### 3. Commit your changes

- If you are a first-time contributor, please install and initialize pre-commit hooks from the repository root directory first.

  ```bash
  pip install -U pre-commit
  pre-commit install
  ```

- Commit your changes as usual. Pre-commit hooks will be triggered to stylize your code before each commit.

  ```bash
  # coding
  git add [files]
  git commit -m 'messages'
  ```

  ```{note}
  Sometimes your code may be changed by pre-commit hooks. In this case, please remember to re-stage the modified files and commit again.
  ```

### 4. Push your changes to the forked repository and create a PR

- Push the branch to your forked remote repository

  ```bash
  git push origin branchname
  ```

- Create a PR
  ![avatar](https://user-images.githubusercontent.com/22607038/195053564-71bd3cb4-b8d4-4ed9-9075-051e138b7fd4.png)

- Revise PR message template to describe your motivation and modifications made in this PR. You can also link the related issue to the PR manually in the PR message (For more information, checkout the [official guidance](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)).

- Specifically, if you are contributing to `dev-1.x`, you will have to change the base branch of the PR to `dev-1.x` in the PR page, since the default base branch is `main`.

  ![avatar](https://user-images.githubusercontent.com/22607038/195045928-f3ceedc8-0162-46a7-ae1a-7e22829fe189.png)

- You can also ask a specific person to review the changes you've proposed.

### 5. Discuss and review your code

- Modify your codes according to reviewers' suggestions and then push your changes.

### 6.  Merge your branch to the `main` / `dev-1.x` branch and delete the branch

- After the PR is merged by the maintainer, you can delete the branch you created in your forked repository.

  ```bash
  git branch -d branchname # delete local branch
  git push origin --delete branchname # delete remote branch
  ```

## PR Specs

1. Use [pre-commit](https://pre-commit.com) hook to avoid issues of code style

2. One short-time branch should be matched with only one PR

3. Accomplish a detailed change in one PR. Avoid large PR

   - Bad: Support Faster R-CNN
   - Acceptable: Add a box head to Faster R-CNN
   - Good: Add a parameter to box head to support custom conv-layer number

4. Provide clear and significant commit message

5. Provide clear and meaningful PR description

   - Task name should be clarified in title. The general format is: \[Prefix\] Short description of the PR (Suffix)
   - Prefix: add new feature \[Feature\], fix bug \[Fix\], related to documents \[Docs\], in developing \[WIP\] (which will not be reviewed temporarily)
   - Introduce main changes, results and influences on other modules in short description
   - Associate related issues and pull requests with a milestone
