# 分支迁移

在早期阶段，MMOCR 有三个分支：`main`、`1.x` 和 `dev-1.x`。随着 MMOCR 1.0.0 正式版的发布，我们也重命名了其中一些分支，下面提供了新旧分支的对照。

- `main` 分支包括了 MMOCR 0.x（例如 v0.6.3）的代码。现在已经被重命名为 `0.x`。
- `1.x` 包含了 MMOCR 1.x（例如 1.0.0rc6）的代码。现在它是 `main` 分支的别名，会在 2023 的年中删除。
- `dev-1.x` 是 MMOCR 1.x 的开发分支。现在保持不变。

有关分支的更多信息，请查看[分支](../notes/branches.md)。

## 升级 `main` 分支时解决冲突

对于希望从旧 `main` 分支（包含 MMOCR 0.x 代码）升级的用户，代码可能会导致冲突。要避免这些冲突，请按照以下步骤操作：

1. 请 commit 在 `main` 上的所有更改（若有），并备份您当前的 `main` 分支。

   ```bash
   git checkout main
   git add --all
   git commit -m 'backup'
   git checkout -b main_backup
   ```

2. 从远程存储库获取最新更改。

   ```bash
   git remote add openmmlab git@github.com:open-mmlab/mmocr.git
   git fetch openmmlab
   ```

3. 通过运行 `git reset --hard openmmlab/main` 将 `main` 分支重置为远程存储库上的最新 `main` 分支。

   ```bash
   git checkout main
   git reset --hard openmmlab/main
   ```

按照这些步骤，您可以成功升级您的 `main` 分支。
