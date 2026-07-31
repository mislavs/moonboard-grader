"""Promotion refit command for a selected CV candidate."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import torch

from src.experiment import (
    config_hash,
    require_clean_revision,
    resolve_experiment_config,
    sha256_file,
)
from src.experiment_manifest import load_manifest, validate_manifest_for_config
from src.experiment_runner import run_refit

from .utils import console, print_completion_message, print_section_header, setup_device


def setup_refit_parser(subparsers):
    parser = subparsers.add_parser(
        "refit",
        help="Promote a selected CV candidate by refitting on all development data",
    )
    parser.add_argument("--config", default="config.yaml", help="Official experiment YAML")
    parser.add_argument("--manifest", required=True, help="Frozen cohort manifest JSON")
    parser.add_argument("--cv-report", required=True, help="Completed 15-run CV report")
    parser.add_argument("--output", help="Refit checkpoint path")
    parser.set_defaults(func=refit_command)
    return parser


def _load_cv_report(path: Path, config, manifest, current_revision):
    if not path.exists():
        raise FileNotFoundError(f"CV report not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)
    if report.get("report_schema_version") != 1:
        raise ValueError("unsupported CV report schema")
    if report.get("config_sha256") != config_hash(config):
        raise ValueError("CV report configuration hash does not match the refit configuration")
    if report.get("resolved_config") != config:
        raise ValueError("CV report resolved configuration does not match")
    if report.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("CV report manifest hash does not match")
    if report.get("code_revision") != current_revision:
        raise ValueError("CV report was produced by a different or dirty code revision")
    expected_runs = int(manifest["inner_cv"]["fold_count"]) * len(manifest["inner_cv"]["seeds"])
    if len(report.get("runs", [])) != expected_runs or len(report.get("selected_epochs", [])) != expected_runs:
        raise ValueError(f"CV report must contain all {expected_runs} completed runs")
    return report


def refit_command(args):
    print_section_header("PROMOTION REFIT")
    revision = require_clean_revision()
    config, data_path = resolve_experiment_config(args.config)
    manifest = load_manifest(args.manifest)
    records = validate_manifest_for_config(manifest, config, data_path)
    report_path = Path(args.cv_report)
    report = _load_cv_report(report_path, config, manifest, revision)
    selected_epoch = int(median([int(value) for value in report["selected_epochs"]]))
    development_ids = sorted(map(str, manifest["split"]["development_ids"]))
    test_ids = sorted(map(str, manifest["split"]["test_ids"]))
    records_by_id = {str(record.problem_id): record for record in records}
    seed = int(config["experiment"]["refit_seed"])
    candidate_id = report["candidate_id"]
    output = (
        Path(args.output)
        if args.output
        else Path(config["checkpoint"]["dir"])
        / "experiments"
        / candidate_id
        / f"refit_{candidate_id}_seed-{seed}.pth"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite refit checkpoint: {output}")
    device, _ = setup_device(config["device"])
    run_refit(
        [records_by_id[problem_id] for problem_id in development_ids],
        config,
        manifest,
        {"development_ids": development_ids, "locked_test_ids": test_ids},
        seed,
        selected_epoch,
        torch.device(device),
        output,
        sha256_file(report_path),
    )
    console.print(f"Refit epochs: {selected_epoch}")
    console.print(f"Training problems: {len(development_ids)}")
    console.print(f"Checkpoint: {output}")
    console.print(f"Checkpoint SHA-256: {sha256_file(output)}")
    print_completion_message("Candidate promoted to a refit checkpoint")
