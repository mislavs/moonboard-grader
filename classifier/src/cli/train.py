"""Single-run manifest-backed diagnostic training command."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from src.experiment import (
    code_revision,
    config_hash,
    environment_metadata,
    resolve_experiment_config,
    sha256_file,
)
from src.experiment_manifest import load_manifest, validate_manifest_for_config
from src.experiment_runner import run_cv_fold

from .utils import console, print_completion_message, print_section_header, setup_device


def setup_train_parser(subparsers):
    parser = subparsers.add_parser(
        "train",
        help="Train one non-promotable candidate on a frozen validation fold",
    )
    parser.add_argument("--config", default="config.yaml", help="Experiment YAML")
    parser.add_argument("--manifest", required=True, help="Frozen cohort manifest JSON")
    parser.add_argument("--seed", required=True, type=int, help="Manifest CV seed to use")
    parser.add_argument("--fold", required=True, type=int, help="Manifest validation fold to use")
    parser.add_argument("--output-dir", help="Diagnostic artifact directory")
    parser.set_defaults(func=train_command)
    return parser


def _atomic_json(payload, path: Path) -> None:
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


def _validation_membership(manifest, seed: int, fold: int):
    repetition = next(
        (item for item in manifest["inner_cv"]["repetitions"] if int(item["seed"]) == seed),
        None,
    )
    if repetition is None:
        available = ", ".join(str(value) for value in manifest["inner_cv"]["seeds"])
        raise ValueError(f"seed {seed} is not in the manifest; available seeds: {available}")
    fold_definition = next(
        (item for item in repetition["folds"] if int(item["fold"]) == fold),
        None,
    )
    if fold_definition is None:
        available = ", ".join(str(item["fold"]) for item in repetition["folds"])
        raise ValueError(
            f"fold {fold} is not defined for manifest seed {seed}; available folds: {available}"
        )
    return sorted(map(str, fold_definition["validation_ids"]))


def _print_epoch_progress(progress) -> None:
    metrics = progress["metrics"]
    best = " | best" if progress["is_best"] else ""
    console.print(
        f"Epoch {progress['epoch']}/{progress['max_epochs']}"
        f" | train loss {progress['train_loss']:.4f}"
        f" | val loss {metrics['avg_loss']:.4f}"
        f" | exact {metrics['exact_accuracy']:.2f}%"
        f" | +-1 {metrics['tolerance_1_accuracy']:.2f}%"
        f" | MAE {metrics['mean_absolute_error']:.3f}"
        f"{best}"
    )
    if progress["will_stop"]:
        console.print(
            f"Early stopping after epoch {progress['epoch']} "
            f"(no improvement for {progress['stale_epochs']} epochs)"
        )


def train_command(args):
    print_section_header("DIAGNOSTIC TRAINING")
    revision = code_revision()
    config, data_path = resolve_experiment_config(args.config)
    manifest = load_manifest(args.manifest)
    records = validate_manifest_for_config(manifest, config, data_path)
    records_by_id = {str(record.problem_id): record for record in records}

    seed = int(args.seed)
    fold = int(args.fold)
    validation_ids = _validation_membership(manifest, seed, fold)
    development = set(map(str, manifest["split"]["development_ids"]))
    locked_test_ids = sorted(map(str, manifest["split"]["test_ids"]))
    train_ids = sorted(development - set(validation_ids))
    if set(train_ids) & set(locked_test_ids) or set(validation_ids) & set(locked_test_ids):
        raise AssertionError("diagnostic training membership includes a locked test problem")

    config_sha = config_hash(config)
    candidate_id = config_sha[:12]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["checkpoint"]["dir"])
        / "experiments"
        / candidate_id
        / "train"
        / f"seed-{seed}"
        / f"fold-{fold}"
    )
    checkpoint_path = output_dir / "checkpoint.pth"
    report_path = output_dir / "report.json"
    for path in (checkpoint_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite diagnostic artifact: {path}")

    device, _ = setup_device(config["device"])
    console.print(f"Seed: {seed}")
    console.print(f"Fold: {fold}")
    console.print(f"Training problems: {len(train_ids)}")
    console.print(f"Validation problems: {len(validation_ids)}")
    result = run_cv_fold(
        [records_by_id[problem_id] for problem_id in train_ids],
        [records_by_id[problem_id] for problem_id in validation_ids],
        config,
        manifest,
        {
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "locked_test_ids": locked_test_ids,
        },
        seed,
        fold,
        torch.device(device),
        checkpoint_path,
        artifact_stage="diagnostic",
        revision=revision,
        progress_callback=_print_epoch_progress,
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    report = {
        "report_schema_version": 1,
        "evaluation_kind": "diagnostic_validation",
        "comparable": False,
        "promotion_eligible": False,
        "candidate_id": candidate_id,
        "config_sha256": config_sha,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "seed": seed,
        "fold": fold,
        "training_problem_count": len(train_ids),
        "validation_problem_count": len(validation_ids),
        "selected_epoch": int(result["selected_epoch"]),
        "metrics": result["metrics"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "resolved_config": config,
        "environment": environment_metadata(),
        "code_revision": revision,
    }
    _atomic_json(report, report_path)

    metrics = result["metrics"]
    console.print(f"Selected epoch: {result['selected_epoch']}")
    console.print(f"Exact accuracy: {metrics['exact_accuracy']:.2f}%")
    console.print(f"+-1 grade accuracy: {metrics['tolerance_1_accuracy']:.2f}%")
    console.print(f"Checkpoint: {checkpoint_path}")
    console.print(f"Report: {report_path}")
    console.print("Comparable: false (diagnostic validation only)")
    print_completion_message("Diagnostic training completed without test evaluation")
