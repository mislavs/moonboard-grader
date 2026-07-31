"""Explicitly non-comparable evaluation on an arbitrary dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from moonboard_core.data_processor import filter_dataset_by_grades, load_dataset
from moonboard_core.grade_encoder import remap_label

from src.dataset import MoonboardDataset
from src.evaluator import calculate_mean_absolute_error, evaluate_model
from src.predictor import Predictor

from .utils import print_completion_message, print_section_header


def setup_evaluate_diagnostic_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-diagnostic",
        help="Evaluate arbitrary data with results explicitly marked non-comparable",
    )
    parser.add_argument("--checkpoint", required=True, help="Compatible model checkpoint")
    parser.add_argument("--data", required=True, help="Arbitrary diagnostic JSON dataset")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--output", help="Optional diagnostic JSON report")
    parser.set_defaults(func=evaluate_diagnostic_command)
    return parser


def evaluate_diagnostic_command(args):
    print_section_header("NON-COMPARABLE DIAGNOSTIC EVALUATION")
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    predictor = Predictor(checkpoint_path, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset = load_dataset(data_path)
    grade_offset = int(checkpoint.get("grade_offset", 0))
    if grade_offset:
        dataset = filter_dataset_by_grades(
            dataset,
            int(checkpoint.get("min_grade_index", 0)),
            int(checkpoint.get("max_grade_index", 18)),
        )
        dataset = [(tensor, remap_label(label, grade_offset)) for tensor, label in dataset]
    if not dataset:
        raise ValueError("diagnostic filters produced an empty dataset")
    tensors = np.stack([item[0] for item in dataset])
    labels = np.asarray([item[1] for item in dataset], dtype=np.int64)
    loader = DataLoader(MoonboardDataset(tensors, labels), batch_size=32, shuffle=False)
    metrics = evaluate_model(predictor.model, loader, device)
    report_metrics = {
        key: value for key, value in metrics.items() if key not in ("predictions", "labels")
    }
    report_metrics["mean_absolute_error"] = calculate_mean_absolute_error(
        np.asarray(metrics["predictions"]), np.asarray(metrics["labels"])
    )
    report = {
        "report_schema_version": 1,
        "evaluation_kind": "diagnostic",
        "comparable": False,
        "warning": "Arbitrary-data diagnostics are not locked benchmark results.",
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "metrics": report_metrics,
    }
    print("WARNING: comparable=false; do not report these metrics as benchmark/test results.")
    print(f"Exact accuracy: {metrics['exact_accuracy']:.2f}%")
    print(f"+-1 grade accuracy: {metrics['tolerance_1_accuracy']:.2f}%")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True)
            file.write("\n")
        print(f"Diagnostic report: {output}")
    print_completion_message("Diagnostic evaluation completed")

