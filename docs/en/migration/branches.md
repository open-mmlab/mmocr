# Branch Migration

At an earlier stage, MMOCR had three branches: `main`, `1.x`, and `dev-1.x`. Some of these branches have been renamed together with the official MMOCR 1.0.0 release, and here is the changelog.

- `main` branch housed the code for MMOCR 0.x (e.g., v0.6.3). Now it has been renamed to `0.x`.
- `1.x` contained the code for MMOCR 1.x (e.g., 1.0.0rc6). Now it is an alias of `main`, and will be removed in mid 2023.
- `dev-1.x` was the development branch for MMOCR 1.x. Now it remains unchanged.

For more information about the branches, check out [branches](../notes/branches.md).

## Resolving Conflicts When Upgrading the `main` branch

For users who wish to upgrade from the old `main` branch that has the code for MMOCR 0.x, the non-fast-forwarded-able nature of the upgrade may cause conflicts. To resolve these conflicts, follow the steps below:

1. Commit all the changes you have on `main` if you have any. Backup your current `main` branch by creating a copy.

   ```bash
   git checkout main
   git add --all
   git commit -m 'backup'
   git checkout -b main_backup
   ```

2. Fetch the latest changes from the remote repository.

   ```bash
   git remote add openmmlab git@github.com:open-mmlab/mmocr.git
   git fetch openmmlab
   ```

3. Reset the `main` branch to the latest `main` branch on the remote repository by running `git reset --hard openmmlab/main`.

   ```bash
   git checkout main
   git reset --hard openmmlab/main
   ```

By following these steps, you can successfully upgrade your `main` branch.
