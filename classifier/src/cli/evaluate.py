"""Strict evaluation of a promoted refit checkpoint on the locked test set."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from moonboard_core.grade_encoder import get_all_grades, get_filtered_grade_names

from src.dataset import MoonboardDataset
from src.evaluator import (
    calculate_mean_absolute_error,
    evaluate_model,
    generate_confusion_matrix,
    get_metrics_summary,
    plot_confusion_matrix,
)
from src.experiment import (
    CHECKPOINT_SCHEMA_VERSION,
    config_hash,
    require_clean_revision,
    sha256_file,
)
from src.experiment_manifest import load_manifest, validate_manifest_for_config
from src.predictor import Predictor

from .utils import console, print_completion_message, print_section_header


def setup_evaluate_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a promoted refit checkpoint on its locked test membership",
    )
    parser.add_argument("--checkpoint", required=True, help="Stage=refit checkpoint")
    parser.add_argument("--manifest", required=True, help="Matching frozen manifest")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--output", help="Immutable test report JSON path")
    parser.set_defaults(func=evaluate_command)
    return parser


def _resolve_dataset_path(config, manifest_path: Path, checkpoint_path: Path) -> Path:
    authored = Path(str(config["data"]["path"]))
    if authored.is_absolute() and authored.exists():
        return authored.resolve()
    candidates = [
        Path.cwd() / authored,
        manifest_path.parent / authored,
        manifest_path.parent.parent / authored,
        checkpoint_path.parent / authored,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Dataset from checkpoint configuration was not found: {authored}; run from the config directory"
    )


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


def _ensure_not_previously_evaluated(report_path: Path, checkpoint_sha: str):
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable test report: {report_path}")
    if not report_path.parent.exists():
        return
    for existing in report_path.parent.glob("*.test.json"):
        try:
            with open(existing, "r", encoding="utf-8") as file:
                report = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("checkpoint_sha256") == checkpoint_sha:
            raise FileExistsError(
                f"This checkpoint hash already has an official test report: {existing}"
            )


def evaluate_command(args):
    print_section_header("LOCKED OUTER TEST EVALUATION")
    current_revision = require_clean_revision()
    checkpoint_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("official evaluation requires a provenance-complete schema v2 checkpoint")
    if checkpoint.get("artifact_stage") != "refit":
        raise ValueError("official evaluation accepts only stage=refit checkpoints")
    if checkpoint.get("code_revision") != current_revision:
        raise ValueError("checkpoint code revision differs from the clean evaluation revision")

    manifest = load_manifest(manifest_path)
    if checkpoint.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("checkpoint and manifest hashes do not match")
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict) or checkpoint.get("config_sha256") != config_hash(config):
        raise ValueError("checkpoint resolved configuration is missing or has changed")
    data_path = _resolve_dataset_path(config, manifest_path, checkpoint_path)
    records = validate_manifest_for_config(manifest, config, data_path)
    locked_test_ids = sorted(map(str, manifest["split"]["test_ids"]))
    if sorted(map(str, checkpoint["split_membership"].get("locked_test_ids", []))) != locked_test_ids:
        raise ValueError("checkpoint locked-test membership differs from the manifest")

    checkpoint_sha = sha256_file(checkpoint_path)
    report_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".test.json")
    _ensure_not_previously_evaluated(report_path, checkpoint_sha)
    records_by_id = {str(record.problem_id): record for record in records}
    test_records = [records_by_id[problem_id] for problem_id in locked_test_ids]
    grade_offset = int(checkpoint.get("grade_offset", 0))
    tensors = np.stack([record.tensor for record in test_records])
    labels = np.asarray([record.label - grade_offset for record in test_records], dtype=np.int64)
    loader = DataLoader(
        MoonboardDataset(tensors, labels),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
    )

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    predictor = Predictor(checkpoint_path, device=device)
    metrics = evaluate_model(predictor.model, loader, device)
    if grade_offset:
        grade_names = get_filtered_grade_names(
            int(checkpoint["min_grade_index"]), int(checkpoint["max_grade_index"])
        )
    else:
        grade_names = get_all_grades()[: int(config["model"]["num_classes"])]
    summary = get_metrics_summary(
        np.asarray(metrics["predictions"]),
        np.asarray(metrics["labels"]),
        grade_names,
    )
    metrics_without_rows = {
        key: value for key, value in metrics.items() if key not in ("predictions", "labels")
    }
    metrics_without_rows["mean_absolute_error"] = calculate_mean_absolute_error(
        np.asarray(metrics["predictions"]), np.asarray(metrics["labels"])
    )
    confusion_path = report_path.with_suffix(".confusion.png")
    matrix = generate_confusion_matrix(
        metrics["predictions"], metrics["labels"], num_classes=len(grade_names)
    )
    plot_confusion_matrix(matrix, grade_names, str(confusion_path), normalize=True)

    report = {
        "report_schema_version": 1,
        "evaluation_kind": "locked_outer_test",
        "comparable": True,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": checkpoint["config_sha256"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "test_problem_count": len(locked_test_ids),
        "metrics": metrics_without_rows,
        "per_grade_metrics": summary["per_grade_metrics"],
        "confusion_matrix": confusion_path.name,
        "code_revision": current_revision,
    }
    _atomic_json(report, report_path)
    console.print(f"Exact accuracy: {metrics['exact_accuracy']:.2f}%")
    console.print(f"+-1 grade accuracy: {metrics['tolerance_1_accuracy']:.2f}%")
    console.print(f"Report: {report_path}")
    print_completion_message("Locked test evaluation completed")
