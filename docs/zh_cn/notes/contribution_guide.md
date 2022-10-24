# 贡献指南

OpenMMLab 欢迎所有人参与我们项目的共建。本文档将指导您如何通过拉取请求为 OpenMMLab 项目作出贡献。

## 什么是拉取请求？

`拉取请求` (Pull Request), [GitHub 官方文档](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)定义如下。

```
拉取请求是一种通知机制。你修改了他人的代码，将你的修改通知原来作者，希望他合并你的修改。
```

## 基本的工作流：

1. 获取最新的代码库
2. 根据你打算贡献的版本，从最新的 `main` 或 `dev-1.x` 分支创建分支进行开发（可以阅读我们的[维护计划](../migration/overview.md)以获知详情）
3. 提交修改 ([不要忘记使用 pre-commit hooks!](#))
4. 推送你的修改并创建一个 `拉取请求`
5. 讨论、审核代码
6. 将开发分支合并到 `main` 或 `dev-1.x` 分支

## 具体步骤

### 1. 获取最新的代码库

- 当你第一次提 PR 时

  复刻 OpenMMLab 原代码库，点击 GitHub 页面右上角的 **Fork** 按钮即可
  ![avatar](https://user-images.githubusercontent.com/22607038/195038780-06a46340-8376-4bde-a07f-2577f231a204.png)

  克隆复刻的代码库到本地

  ```bash
  git clone git@github.com:XXX/mmocr.git
  ```

  添加原代码库为上游代码库

  ```bash
  git remote add upstream git@github.com:open-mmlab/mmocr
  ```

- 从第二个 PR 起

  检出本地代码库的主分支，然后从最新的原代码库的主分支拉取更新。这里假设你正基于 `dev-1.x` 开发。

  ```bash
  git checkout dev-1.x
  git pull upstream dev-1.x
  ```

### 2. 从 `main` 或 `dev-1.x` 分支创建一个新的开发分支

```bash
git checkout -b branchname
```

```{tip}
为了保证提交历史清晰可读，我们强烈推荐您先切换到 `main` 或 `dev-1.x` 分支，再创建新的分支。
```

### 3. 提交你的修改

- 如果你是第一次尝试贡献，请在 MMOCR 的目录下安装并初始化 pre-commit hooks。

  ```bash
  pip install -U pre-commit
  pre-commit install
  ```

- 提交修改。在每次提交前，pre-commit hooks 都会被触发并规范化你的代码格式。

  ```bash
  # coding
  git add [files]
  git commit -m 'messages'
  ```

  ```{note}
  有时你的文件可能会在提交时被 pre-commit hooks 自动修改。这时请重新添加并提交修改后的文件。
  ```

### 4. 推送你的修改到复刻的代码库，并创建一个拉取请求

- 推送当前分支到远端复刻的代码库

  ```bash
  git push origin branchname
  ```

- 创建一个拉取请求

  ![avatar](https://user-images.githubusercontent.com/22607038/195053564-71bd3cb4-b8d4-4ed9-9075-051e138b7fd4.png)

- 修改拉取请求信息模板，描述修改原因和修改内容。还可以在 PR 描述中，手动关联到相关的议题 (issue),（更多细节，请参考[官方文档](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)）。

- 另外，如果你正在往 `dev-1.x` 分支提交代码，你还需要在创建 PR 的界面中将基础分支改为 `dev-1.x`，因为现在默认的基础分支是 `main`。

  ![avatar](https://user-images.githubusercontent.com/22607038/195045928-f3ceedc8-0162-46a7-ae1a-7e22829fe189.png)

- 你同样可以把 PR 关联给相关人员进行评审。

### 5. 讨论并评审你的代码

- 根据评审人员的意见修改代码，并推送修改

### 6. `拉取请求`合并之后删除该分支

- 在 PR 合并之后，你就可以删除该分支了。

  ```bash
  git branch -d branchname # 删除本地分支
  git push origin --delete branchname # 删除远程分支
  ```

## PR 规范

1. 使用 [pre-commit hook](https://pre-commit.com)，尽量减少代码风格相关问题

2. 一个 PR 对应一个短期分支

3. 粒度要细，一个PR只做一件事情，避免超大的PR

   - Bad：实现 Faster R-CNN
   - Acceptable：给 Faster R-CNN 添加一个 box head
   - Good：给 box head 增加一个参数来支持自定义的 conv 层数

4. 每次 Commit 时需要提供清晰且有意义 commit 信息

5. 提供清晰且有意义的`拉取请求`描述

   - 标题写明白任务名称，一般格式:\[Prefix\] Short description of the pull request (Suffix)
   - prefix: 新增功能 \[Feature\], 修 bug \[Fix\], 文档相关 \[Docs\], 开发中 \[WIP\] (暂时不会被review)
   - 描述里介绍`拉取请求`的主要修改内容，结果，以及对其他部分的影响, 参考`拉取请求`模板
   - 关联相关的`议题` (issue) 和其他`拉取请求`
