"""
Shared validation and error mapping for stratified data splitting.
"""

import math
from typing import Dict

import numpy as np


def validate_two_stage_stratified_feasibility(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, float]:
    """
    Validate whether two-stage stratified splitting is feasible.

    Returns a context dictionary that can be reused in wrapped errors.
    """
    labels = np.asarray(labels)
    n_samples = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    n_classes = len(unique_labels)
    min_class_count = int(np.min(counts))

    if min_class_count < 2:
        raise ValueError(
            f"Cannot stratify: some classes have fewer than 2 samples. "
            f"Minimum class count: {min_class_count}. "
            f"Add more data, merge rare classes, or reduce the number of classes."
        )

    # Two-stage stratification can fail when a class has only 2 members:
    # one sample may be allocated to test in stage 1, leaving 1 sample
    # for stage 2 (train/val), which is invalid for stratification.
    if min_class_count < 3:
        raise ValueError(
            f"Cannot perform two-stage stratified split: minimum class count is "
            f"{min_class_count}, but at least 3 samples per class are required. "
            f"Add more data for rare classes, merge classes, or adjust split ratios."
        )

    n_test = max(1, math.ceil(n_samples * test_ratio))
    if n_test < n_classes:
        raise ValueError(
            f"Test split too small for stratification: test_ratio={test_ratio} "
            f"yields ~{n_test} samples but {n_classes} classes require at least "
            f"{n_classes} samples. Increase test_ratio, add more data, or reduce "
            f"the number of classes."
        )

    ratio_denom = train_ratio + val_ratio
    if ratio_denom <= 0:
        raise ValueError(
            "Invalid split ratios for two-stage stratification: "
            "(train_ratio + val_ratio) must be > 0."
        )

    n_train_val = n_samples - n_test
    val_ratio_adjusted = val_ratio / ratio_denom
    n_val = max(1, math.ceil(n_train_val * val_ratio_adjusted))
    if n_val < n_classes:
        raise ValueError(
            f"Validation split too small for stratification: the train+val subset "
            f"({n_train_val} samples) with val_ratio_adjusted={val_ratio_adjusted:.3f} "
            f"yields ~{n_val} samples but {n_classes} classes require at least "
            f"{n_classes} samples. Increase val_ratio, add more data, or reduce "
            f"the number of classes."
        )

    n_train = n_train_val - n_val
    if n_train < n_classes:
        raise ValueError(
            f"Training split too small for stratification: ~{n_train} samples "
            f"for {n_classes} classes. Increase train_ratio, add more data, or "
            f"reduce the number of classes."
        )

    return {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "min_class_count": min_class_count,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }


def raise_friendly_stratify_error(
    stage: str, original_error: ValueError, context: Dict[str, float]
) -> None:
    """
    Raise a domain-friendly stratification error while preserving the cause.
    """
    raise ValueError(
        f"Stratified split failed during {stage}. "
        f"Class-space context: total_samples={context['n_samples']}, "
        f"num_classes={context['n_classes']}, "
        f"min_class_count={context['min_class_count']}, "
        f"planned_sizes(train={context['n_train']}, "
        f"val={context['n_val']}, test={context['n_test']}). "
        "Add more data for rare classes, merge classes, or adjust "
        "train/val/test ratios."
    ) from original_error
