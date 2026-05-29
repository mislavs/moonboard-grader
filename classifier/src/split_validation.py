"""
Shared validation and error mapping for stratified data splitting.
"""

import math
import warnings
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


def validate_grouped_split_feasibility(
    labels: np.ndarray,
    groups: np.ndarray,
    test_n_splits: int,
    val_n_splits: int,
) -> Dict[str, float]:
    """
    Validate whether two-stage stratified group splitting is feasible.

    Grouped splitting keeps identical layouts together, so rare grades with too
    few unique layout groups may not appear in every split. That condition emits
    a warning instead of failing so full-range datasets with rare top grades can
    still be baselined.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    if len(labels) != len(groups):
        raise ValueError(
            f"labels and groups must have the same length. "
            f"Got {len(labels)} labels and {len(groups)} groups."
        )

    if len(labels) == 0:
        raise ValueError("Cannot create grouped split from an empty dataset.")

    if test_n_splits < 2 or val_n_splits < 2:
        raise ValueError(
            "Grouped split requires at least 2 folds for both test and "
            f"validation stages. Got test_n_splits={test_n_splits}, "
            f"val_n_splits={val_n_splits}."
        )

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    required_folds = max(test_n_splits, val_n_splits)

    if n_groups < required_folds:
        raise ValueError(
            f"Cannot create grouped split: {n_groups} unique layout groups are "
            f"available, but at least {required_folds} are required for the "
            f"requested grouped folds (test_n_splits={test_n_splits}, "
            f"val_n_splits={val_n_splits}). Add more unique layouts, disable "
            "group_by_layout, or adjust train/val/test ratios."
        )

    largest_possible_test_fold = math.ceil(n_groups / test_n_splits)
    remaining_groups_after_test = n_groups - largest_possible_test_fold
    if remaining_groups_after_test < val_n_splits:
        raise ValueError(
            f"Cannot create grouped validation split: after reserving a test "
            f"fold, only ~{remaining_groups_after_test} unique layout groups "
            f"remain, but {val_n_splits} validation folds are required. Add "
            "more unique layouts, disable group_by_layout, or adjust split ratios."
        )

    unique_labels = np.unique(labels)
    per_class_group_counts = {
        int(label): int(len(np.unique(groups[labels == label])))
        for label in unique_labels
    }

    underrepresented = {
        label: count
        for label, count in per_class_group_counts.items()
        if count < 3
    }
    if underrepresented:
        details = ", ".join(
            f"class {label}: {count} layout group(s)"
            for label, count in sorted(underrepresented.items())
        )
        warnings.warn(
            "Some classes have too few unique layout groups to appear in every "
            f"train/validation/test split ({details}). Metrics for those rare "
            "classes may be noisy or missing in a split.",
            UserWarning,
            stacklevel=2,
        )

    min_class_group_count = min(per_class_group_counts.values()) if per_class_group_counts else 0
    return {
        "n_samples": len(labels),
        "n_groups": n_groups,
        "n_classes": len(unique_labels),
        "test_n_splits": test_n_splits,
        "val_n_splits": val_n_splits,
        "min_class_group_count": min_class_group_count,
    }


def raise_friendly_grouped_split_error(
    stage: str, original_error: ValueError, context: Dict[str, float]
) -> None:
    """
    Raise a domain-friendly grouped stratification error while preserving cause.
    """
    raise ValueError(
        f"Grouped stratified split failed during {stage}. "
        f"Context: total_samples={context['n_samples']}, "
        f"unique_layout_groups={context['n_groups']}, "
        f"num_classes={context['n_classes']}, "
        f"min_class_layout_groups={context['min_class_group_count']}, "
        f"test_n_splits={context['test_n_splits']}, "
        f"val_n_splits={context['val_n_splits']}. "
        "Add more data for rare classes, add more unique layouts, disable "
        "group_by_layout, or adjust train/val/test ratios."
    ) from original_error
