# Changelog

## Unreleased

- Replaced the classifier's leakage-prone train-to-test CLI workflow with immutable cohort manifests, repeated grouped cross-validation, development-pool refitting, and strict locked-test evaluation.
- Added provenance-complete schema-v2 checkpoints containing resolved configuration, dataset/filter hashes, split membership, seeds, code revision, and environment versions.
- Moved arbitrary dataset scoring to `evaluate-diagnostic`, whose reports are explicitly marked non-comparable.
- Configure the CuBLAS deterministic workspace before PyTorch import so official CUDA experiments can use strict deterministic algorithms.
