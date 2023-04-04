# Branches

This documentation aims to provide a comprehensive understanding of the purpose and features of each branch in MMOCR.

## Branch Overview

### 1. `main`

The `main` branch serves as the default branch for the MMOCR project. It contains the latest stable version of MMOCR, currently housing the code for MMOCR 1.x (e.g. v1.0.0). The `main` branch ensures users have access to the most recent and reliable version of the software.

### 2. `dev-1.x`

The `dev-1.x` branch is dedicated to the development of the next major version of MMOCR. This branch will routinely undergo reliance tests, and the passing commits will be squashed in a release and published to the `main` branch. By having a separate development branch, the project can continue to evolve without impacting the stability of the `main` branch. **All the PRs should be merged into the `dev-1.x` branch.**

### 3. `0.x`

The `0.x` branch serves as an archive for MMOCR 0.x (e.g. v0.6.3). This branch will no longer actively receive updates or improvements, but it remains accessible for historical reference or for users who have not yet upgraded to MMOCR 1.x.

### 3. `1.x`

It's an alias of `main` branch, which is intended for a smooth transition from the compatibility period. It will be removed in mid 2023.

```{note}
The branches mapping has been changed in 2023.04.06. For the legacy branches mapping and the guide for migration, please refer to the [branch migration guide](../migration/branches.md).
```
