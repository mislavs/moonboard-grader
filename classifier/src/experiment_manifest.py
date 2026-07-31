"""Immutable cohort manifests for the classifier benchmark workflow."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from moonboard_core.data_processor import ProcessedProblem

from .experiment import (
    MANIFEST_SCHEMA_VERSION,
    canonical_cohort_rows,
    code_revision,
    cohort_filters,
    layout_hash,
    load_cohort,
    manifest_spec,
    manifest_spec_hash,
    sha256_json,
)


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return sha256_json(payload)


def _validate_group_counts(labels: np.ndarray, groups: np.ndarray, required: int) -> None:
    per_class = defaultdict(set)
    for label, group in zip(labels.tolist(), groups.tolist()):
        per_class[int(label)].add(str(group))
    failures = {label: len(values) for label, values in per_class.items() if len(values) < required}
    if failures:
        details = ", ".join(f"grade {label}: {count}" for label, count in sorted(failures.items()))
        raise ValueError(
            f"grouped splitting requires at least {required} unique layouts per grade; {details}"
        )


def _class_distribution_error(labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    all_counts = Counter(labels.tolist())
    test_counts = Counter(labels[test_idx].tolist())
    target_ratio = len(test_idx) / len(labels)
    return sum(
        abs((test_counts.get(label, 0) / count) - target_ratio)
        for label, count in all_counts.items()
    )


def _outer_split(
    labels: np.ndarray,
    groups: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n_splits = max(2, int(round(1.0 / test_ratio)))
    _validate_group_counts(labels, groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates = list(splitter.split(np.zeros(len(labels)), labels, groups))
    development_idx, test_idx = min(
        candidates,
        key=lambda item: (
            abs((len(item[1]) / len(labels)) - test_ratio),
            _class_distribution_error(labels, item[0], item[1]),
            tuple(item[1].tolist()),
        ),
    )
    return development_idx, test_idx, n_splits


def _inner_folds(
    development_idx: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    problem_ids: Sequence[str],
    folds: int,
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    dev_labels = labels[development_idx]
    dev_groups = groups[development_idx]
    _validate_group_counts(dev_labels, dev_groups, folds)
    repetitions = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        validation_folds = []
        for fold_index, (_, validation_relative_idx) in enumerate(
            splitter.split(np.zeros(len(development_idx)), dev_labels, dev_groups)
        ):
            validation_idx = development_idx[validation_relative_idx]
            validation_folds.append(
                {
                    "fold": fold_index,
                    "validation_ids": sorted(problem_ids[index] for index in validation_idx.tolist()),
                }
            )
        repetitions.append({"seed": int(seed), "folds": validation_folds})
    return repetitions


def create_manifest_document(
    config: Mapping[str, Any],
    records: Sequence[ProcessedProblem],
    dataset_identity: Mapping[str, Any],
    manifest_id: str,
) -> Dict[str, Any]:
    rows = canonical_cohort_rows(records)
    ids = [str(record.problem_id) for record in records]
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    groups = np.asarray([layout_hash(record.tensor) for record in records])
    experiment = config["experiment"]

    development_idx, test_idx, outer_folds = _outer_split(
        labels,
        groups,
        float(experiment["outer_test_ratio"]),
        int(experiment["outer_seed"]),
    )
    development_ids = sorted(ids[index] for index in development_idx.tolist())
    test_ids = sorted(ids[index] for index in test_idx.tolist())
    repetitions = _inner_folds(
        development_idx,
        labels,
        groups,
        ids,
        int(experiment["cv_folds"]),
        [int(seed) for seed in experiment["cv_seeds"]],
    )

    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "dataset": dict(dataset_identity),
        "cohort": {
            "filters": cohort_filters(config),
            "problem_count": len(records),
            "problems": rows,
        },
        "split": {
            "strategy": "stratified_group_kfold",
            "group_key": "layout_sha256_v1",
            "outer_seed": int(experiment["outer_seed"]),
            "requested_test_ratio": float(experiment["outer_test_ratio"]),
            "outer_fold_count": outer_folds,
            "actual_test_ratio": len(test_ids) / len(records),
            "development_ids": development_ids,
            "test_ids": test_ids,
        },
        "inner_cv": {
            "fold_count": int(experiment["cv_folds"]),
            "seeds": [int(seed) for seed in experiment["cv_seeds"]],
            "repetitions": repetitions,
        },
        "creation": {
            "split_spec": manifest_spec(config),
            "split_spec_sha256": manifest_spec_hash(config),
            "code_revision": code_revision(),
        },
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    validate_manifest_structure(manifest)
    return manifest


def validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported experiment manifest schema")
    expected_digest = manifest_digest(manifest)
    if manifest.get("manifest_sha256") != expected_digest:
        raise ValueError("manifest SHA-256 does not match its contents")

    problems = manifest.get("cohort", {}).get("problems", [])
    problem_ids = [str(problem.get("problem_id")) for problem in problems]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("manifest contains duplicate problem IDs")
    problem_set = set(problem_ids)
    if len(problem_set) != manifest.get("cohort", {}).get("problem_count"):
        raise ValueError("manifest problem count does not match problem entries")

    split = manifest.get("split", {})
    development = set(map(str, split.get("development_ids", [])))
    test = set(map(str, split.get("test_ids", [])))
    if development & test:
        raise ValueError("outer development and test memberships overlap")
    if development | test != problem_set:
        raise ValueError("outer split does not cover the cohort exactly")

    layout_by_id = {str(problem["problem_id"]): problem["layout_hash"] for problem in problems}
    development_layouts = {layout_by_id[problem_id] for problem_id in development}
    test_layouts = {layout_by_id[problem_id] for problem_id in test}
    if development_layouts & test_layouts:
        raise ValueError("layout leakage exists between outer development and test sets")

    inner = manifest.get("inner_cv", {})
    expected_seeds = list(map(int, inner.get("seeds", [])))
    repetitions = inner.get("repetitions", [])
    if [int(item.get("seed")) for item in repetitions] != expected_seeds:
        raise ValueError("inner CV seed membership does not match declared seeds")
    for repetition in repetitions:
        folds = repetition.get("folds", [])
        if len(folds) != int(inner.get("fold_count", 0)):
            raise ValueError("inner CV fold count is incomplete")
        seen = set()
        fold_layouts = []
        for fold in folds:
            validation = set(map(str, fold.get("validation_ids", [])))
            if not validation <= development:
                raise ValueError("inner validation includes a non-development problem")
            if seen & validation:
                raise ValueError("inner validation folds overlap within a seed")
            seen |= validation
            layouts = {layout_by_id[problem_id] for problem_id in validation}
            fold_layouts.append(layouts)
        if seen != development:
            raise ValueError("inner validation folds do not cover development exactly")
        for index, layouts in enumerate(fold_layouts):
            for other in fold_layouts[index + 1 :]:
                if layouts & other:
                    raise ValueError("layout leakage exists between inner folds")


def load_manifest(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    validate_manifest_structure(manifest)
    return manifest


def validate_manifest_for_config(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    data_path: Path,
) -> List[ProcessedProblem]:
    records, identity = load_cohort(config, data_path)
    if manifest["dataset"] != identity:
        raise ValueError("dataset hash, cohort hash, counts, path, or filters differ from the manifest")
    if manifest["cohort"]["filters"] != cohort_filters(config):
        raise ValueError("configuration filters differ from the manifest")
    if manifest["creation"]["split_spec_sha256"] != manifest_spec_hash(config):
        raise ValueError("cohort or split configuration differs from the manifest")
    if manifest["creation"]["split_spec"] != manifest_spec(config):
        raise ValueError("manifest split specification has changed")
    current_rows = canonical_cohort_rows(records)
    if current_rows != manifest["cohort"]["problems"]:
        raise ValueError("current cohort membership differs from the manifest")
    return records


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
