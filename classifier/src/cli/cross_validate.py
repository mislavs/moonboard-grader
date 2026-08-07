"""Repeated grouped cross-validation command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean, pstdev

import torch

from src.experiment import (
    config_hash,
    environment_metadata,
    require_clean_revision,
    resolve_experiment_config,
    sha256_file,
)
from src.experiment_manifest import load_manifest, validate_manifest_for_config
from src.experiment_runner import run_cv_fold

from .utils import console, print_completion_message, print_section_header, setup_device


def setup_cross_validate_parser(subparsers):
    parser = subparsers.add_parser(
        "cross-validate",
        help="Evaluate one candidate using the frozen grouped validation folds",
    )
    parser.add_argument("--config", default="config.yaml", help="Official experiment YAML")
    parser.add_argument("--manifest", required=True, help="Frozen cohort manifest JSON")
    parser.add_argument("--output-dir", help="Candidate artifact directory")
    parser.set_defaults(func=cross_validate_command)
    return parser


def _atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_completed_result(
    path: Path,
    expected_config: str,
    expected_manifest: str,
    expected_revision=None,
):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        result = json.load(file)
    if result.get("config_sha256") != expected_config or result.get("manifest_sha256") != expected_manifest:
        raise ValueError(f"completed fold artifact has incompatible hashes: {path}")
    if expected_revision is not None and result.get("code_revision") != expected_revision:
        raise ValueError(f"completed fold artifact has an incompatible code revision: {path}")
    checkpoint = path.parent / result["checkpoint"]
    if not checkpoint.exists() or sha256_file(checkpoint) != result.get("checkpoint_sha256"):
        raise ValueError(f"completed fold checkpoint is missing or changed: {checkpoint}")
    return result


def _aggregate(results):
    metric_names = (
        "tolerance_1_accuracy",
        "mean_absolute_error",
        "macro_accuracy",
        "exact_accuracy",
        "tolerance_2_accuracy",
        "avg_loss",
    )
    aggregate = {}
    for metric in metric_names:
        values = [float(result["metrics"][metric]) for result in results]
        aggregate[metric] = {"mean": mean(values), "std": pstdev(values)}
    return aggregate


def cross_validate_command(args):
    print_section_header("REPEATED GROUPED CROSS-VALIDATION")
    revision = require_clean_revision()
    config, data_path = resolve_experiment_config(args.config)
    manifest = load_manifest(args.manifest)
    records = validate_manifest_for_config(manifest, config, data_path)
    records_by_id = {str(record.problem_id): record for record in records}
    config_sha = config_hash(config)
    candidate_id = config_sha[:12]
    configured_dir = Path(config["checkpoint"]["dir"])
    output_dir = Path(args.output_dir) if args.output_dir else configured_dir / "experiments" / candidate_id / "cv"
    development = set(map(str, manifest["split"]["development_ids"]))
    test = sorted(map(str, manifest["split"]["test_ids"]))
    device, _ = setup_device(config["device"])

    results = []
    for repetition in manifest["inner_cv"]["repetitions"]:
        seed = int(repetition["seed"])
        for fold_definition in repetition["folds"]:
            fold = int(fold_definition["fold"])
            validation_ids = sorted(map(str, fold_definition["validation_ids"]))
            train_ids = sorted(development - set(validation_ids))
            fold_dir = output_dir / f"seed-{seed}" / f"fold-{fold}"
            result_path = fold_dir / "result.json"
            checkpoint_path = fold_dir / "checkpoint.pth"
            completed = _load_completed_result(
                result_path,
                config_sha,
                manifest["manifest_sha256"],
                revision,
            )
            if completed is not None:
                console.print(f"Reusing seed {seed}, fold {fold}")
                results.append(completed)
                continue

            console.print(f"Running seed {seed}, fold {fold}")
            result = run_cv_fold(
                [records_by_id[problem_id] for problem_id in train_ids],
                [records_by_id[problem_id] for problem_id in validation_ids],
                config,
                manifest,
                {
                    "train_ids": train_ids,
                    "validation_ids": validation_ids,
                    "locked_test_ids": test,
                },
                seed,
                fold,
                torch.device(device),
                checkpoint_path,
                revision=revision,
            )
            fold_result = {
                "seed": seed,
                "fold": fold,
                "selected_epoch": int(result["selected_epoch"]),
                "metrics": result["metrics"],
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "config_sha256": config_sha,
                "manifest_sha256": manifest["manifest_sha256"],
                "code_revision": revision,
            }
            _atomic_json(fold_result, result_path)
            results.append(fold_result)

    aggregate = _aggregate(results)
    report = {
        "report_schema_version": 1,
        "candidate_id": candidate_id,
        "config_sha256": config_sha,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "primary_metric": "tolerance_1_accuracy",
        "ranking_key": [
            aggregate["tolerance_1_accuracy"]["mean"],
            -aggregate["mean_absolute_error"]["mean"],
            aggregate["macro_accuracy"]["mean"],
            aggregate["exact_accuracy"]["mean"],
        ],
        "aggregate": aggregate,
        "selected_epochs": [int(result["selected_epoch"]) for result in results],
        "runs": results,
        "resolved_config": config,
        "environment": environment_metadata(),
        "code_revision": revision,
    }
    report_path = output_dir / "cv_report.json"
    _atomic_json(report, report_path)
    console.print(f"Candidate: {candidate_id}")
    console.print(
        f"Mean +-1 accuracy: {aggregate['tolerance_1_accuracy']['mean']:.2f}% "
        f"(std {aggregate['tolerance_1_accuracy']['std']:.2f})"
    )
    console.print(f"Report: {report_path}")
    print_completion_message("Cross-validation completed without test evaluation")
