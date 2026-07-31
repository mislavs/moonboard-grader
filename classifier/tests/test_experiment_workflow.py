"""Tests for frozen manifests and provenance-complete experiment artifacts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from moonboard_core.data_processor import ProcessedProblem

from src.experiment import config_hash, resolve_experiment_config, seed_everything
from src.experiment_manifest import (
    create_manifest_document,
    manifest_digest,
    validate_manifest_structure,
)
from src.experiment_runner import run_refit


def _config(num_classes=3):
    return {
        "model": {
            "type": "fc",
            "num_classes": num_classes,
            "use_attention": True,
            "dropout_conv": 0.1,
            "dropout_fc1": 0.3,
            "dropout_fc2": 0.4,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 8,
            "num_epochs": 2,
            "early_stopping_patience": None,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "use_class_weights": False,
            "max_class_weight": 5.0,
            "use_balanced_sampling": False,
            "balanced_sampling_strategy": "sqrt",
            "loss_type": "ce",
            "focal_gamma": 2.0,
            "ordinal_weight": 0.5,
            "ordinal_alpha": 2.0,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
            "deterministic": True,
            "scheduler": {"type": "cosine", "horizon_epochs": 2, "eta_min": 1e-7},
        },
        "data": {
            "path": "problems.json",
            "filter_grades": False,
            "min_grade_index": 0,
            "max_grade_index": 18,
            "filter_repeats": False,
            "min_repeats": 1,
            "group_by_layout": True,
        },
        "checkpoint": {"dir": "models"},
        "experiment": {
            "outer_test_ratio": 0.15,
            "outer_seed": 42,
            "cv_folds": 5,
            "cv_seeds": [42, 43, 44],
            "refit_seed": 42,
            "primary_metric": "tolerance_1_accuracy",
        },
        "device": "cpu",
    }


def _records(classes=3, groups_per_class=14):
    records = []
    position = 0
    for label in range(classes):
        for _ in range(groups_per_class):
            tensor = np.zeros((3, 18, 11), dtype=np.float32)
            channel = position // (18 * 11)
            remainder = position % (18 * 11)
            row, column = divmod(remainder, 11)
            tensor[channel, row, column] = 1.0
            records.append(
                ProcessedProblem(
                    problem_id=1000 + position,
                    tensor=tensor,
                    label=label,
                    repeats=position,
                )
            )
            position += 1
    return records


def _identity(config, records):
    from src.experiment import canonical_cohort_rows, cohort_filters, sha256_json

    return {
        "source_path": config["data"]["path"],
        "source_sha256": "a" * 64,
        "source_count": len(records),
        "cohort_sha256": sha256_json(canonical_cohort_rows(records)),
        "cohort_count": len(records),
        "filters": cohort_filters(config),
    }


def test_manifest_is_deterministic_complete_and_layout_safe():
    config = _config()
    records = _records()

    first = create_manifest_document(config, records, _identity(config, records), "fixture-v1")
    second = create_manifest_document(config, records, _identity(config, records), "fixture-v1")

    assert first == second
    assert first["manifest_sha256"] == manifest_digest(first)
    development = set(first["split"]["development_ids"])
    locked_test = set(first["split"]["test_ids"])
    assert not development & locked_test
    assert len(development | locked_test) == len(records)
    assert len(first["inner_cv"]["repetitions"]) == 3
    for repetition in first["inner_cv"]["repetitions"]:
        seen = set()
        for fold in repetition["folds"]:
            validation = set(fold["validation_ids"])
            assert not seen & validation
            assert not locked_test & validation
            seen |= validation
        assert seen == development


def test_manifest_tampering_is_rejected():
    config = _config()
    records = _records()
    manifest = create_manifest_document(config, records, _identity(config, records), "fixture-v1")
    tampered = copy.deepcopy(manifest)
    tampered["split"]["test_ids"].append(tampered["split"]["development_ids"][0])

    with pytest.raises(ValueError, match="SHA-256"):
        validate_manifest_structure(tampered)


def test_manifest_can_be_reused_across_model_candidates(monkeypatch, tmp_path):
    from src import experiment_manifest as manifest_module

    config = _config()
    records = _records()
    identity = _identity(config, records)
    manifest = create_manifest_document(config, records, identity, "fixture-v1")
    candidate = copy.deepcopy(config)
    candidate["training"]["learning_rate"] = 0.0007
    candidate["model"]["dropout_fc1"] = 0.5

    # Model/training changes do not alter the frozen cohort and split contract.
    from src.experiment import manifest_spec, manifest_spec_hash

    assert manifest["creation"]["split_spec"] == manifest_spec(candidate)
    assert manifest["creation"]["split_spec_sha256"] == manifest_spec_hash(candidate)
    monkeypatch.setattr(manifest_module, "load_cohort", lambda *_: (records, identity))
    assert manifest_module.validate_manifest_for_config(
        manifest, candidate, tmp_path / "unused.json"
    ) == records


def test_resolved_config_rejects_legacy_split_fields(tmp_path):
    config = _config()
    config["data"]["train_ratio"] = 0.7
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy split fields"):
        resolve_experiment_config(path)


def test_seed_everything_returns_repeatable_generators():
    first = seed_everything(42)
    first_values = torch.rand(4, generator=first)
    second = seed_everything(42)
    second_values = torch.rand(4, generator=second)
    assert torch.equal(first_values, second_values)


def test_refit_checkpoint_contains_provenance_and_no_validation(tmp_path):
    config = _config(num_classes=2)
    config["training"]["num_epochs"] = 1
    config["training"]["scheduler"]["horizon_epochs"] = 2
    records = _records(classes=2, groups_per_class=5)
    ids = [str(record.problem_id) for record in records]
    manifest = {
        "manifest_id": "fixture-v1",
        "manifest_sha256": "b" * 64,
        "dataset": _identity(config, records),
        "cohort": {"filters": _identity(config, records)["filters"]},
    }
    output = tmp_path / "refit.pth"

    run_refit(
        records,
        config,
        manifest,
        {"development_ids": ids, "locked_test_ids": []},
        42,
        1,
        torch.device("cpu"),
        output,
        "c" * 64,
    )

    checkpoint = torch.load(output, map_location="cpu")
    assert checkpoint["checkpoint_schema_version"] == 2
    assert checkpoint["artifact_stage"] == "refit"
    assert checkpoint["config_sha256"] == config_hash(config)
    assert checkpoint["split_membership"]["development_ids"] == ids
    assert "validation" not in checkpoint["history"]
    assert checkpoint["source_cv_report_sha256"] == "c" * 64


def test_cli_exposes_strict_workflow_and_deprecates_train():
    import argparse

    from src.cli.commands import setup_parsers
    from src.cli.train import deprecated_train_command

    parser = argparse.ArgumentParser()
    setup_parsers(parser)
    train_args = parser.parse_args(["train"])
    assert train_args.func is deprecated_train_command
    with pytest.raises(RuntimeError, match="legacy train command is disabled"):
        train_args.func(train_args)

    evaluate_args = parser.parse_args(
        ["evaluate", "--checkpoint", "model.pth", "--manifest", "benchmark.json"]
    )
    assert not hasattr(evaluate_args, "data")


def test_completed_fold_resume_requires_matching_hashes(tmp_path):
    from src.cli.cross_validate import _load_completed_result
    from src.experiment import sha256_file

    fold_dir = tmp_path / "fold"
    fold_dir.mkdir()
    checkpoint = fold_dir / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    result_path = fold_dir / "result.json"
    result = {
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert _load_completed_result(result_path, "config", "manifest") == result
    with pytest.raises(ValueError, match="incompatible hashes"):
        _load_completed_result(result_path, "different", "manifest")
    with pytest.raises(ValueError, match="code revision"):
        _load_completed_result(
            result_path,
            "config",
            "manifest",
            {"commit": "abc", "dirty": False},
        )
