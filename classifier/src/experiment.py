"""Reproducible experiment configuration, hashing, and runtime provenance."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml

from moonboard_core.data_processor import ProcessedProblem, load_problem_records


CHECKPOINT_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CV_SEEDS = [42, 43, 44]
RELEVANT_GIT_PATHS = ("classifier", "moonboard_core", "pyproject.toml", "uv.lock")


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def layout_hash(tensor: np.ndarray) -> str:
    """Return the versioned, full-width layout digest used for grouping."""
    return hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()


def _setdefault(mapping: Dict[str, Any], key: str, value: Any) -> None:
    if key not in mapping:
        mapping[key] = copy.deepcopy(value)


def resolve_experiment_config(config_path: str | Path) -> Tuple[Dict[str, Any], Path]:
    """Load and fully resolve an official experiment configuration.

    Returns the portable resolved dictionary and the absolute dataset path.
    Paths stored in the dictionary remain as authored so the configuration hash
    is not tied to one workstation.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    config = copy.deepcopy(raw)
    for section in ("model", "training", "data", "checkpoint", "experiment"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"configuration section '{section}' must be a mapping")

    data = config["data"]
    obsolete_data = sorted(
        key for key in ("train_ratio", "val_ratio", "test_ratio", "random_seed") if key in data
    )
    if obsolete_data:
        raise ValueError(
            "official experiments do not accept legacy split fields: "
            + ", ".join(obsolete_data)
            + "; use the experiment section"
        )

    training = config["training"]
    obsolete_training = sorted(
        key
        for key in ("use_scheduler", "scheduler_factor", "scheduler_patience", "reproducibility_seed")
        if key in training
    )
    if obsolete_training:
        raise ValueError(
            "official experiments do not accept legacy scheduler/seed fields: "
            + ", ".join(obsolete_training)
        )

    model = config["model"]
    _setdefault(model, "type", "cnn")
    _setdefault(model, "num_classes", 19)
    _setdefault(model, "use_attention", True)
    _setdefault(model, "dropout_conv", 0.1)
    _setdefault(model, "dropout_fc1", 0.3)
    _setdefault(model, "dropout_fc2", 0.4)

    defaults = {
        "learning_rate": 0.0003,
        "batch_size": 64,
        "num_epochs": 150,
        "early_stopping_patience": 8,
        "optimizer": "adam",
        "weight_decay": 0.001,
        "use_class_weights": True,
        "max_class_weight": 5.0,
        "use_balanced_sampling": False,
        "balanced_sampling_strategy": "sqrt",
        "loss_type": "focal_ordinal",
        "focal_gamma": 2.0,
        "ordinal_weight": 0.5,
        "ordinal_alpha": 2.0,
        "label_smoothing": 0.0,
        "gradient_clip": 1.0,
        "deterministic": True,
    }
    for key, value in defaults.items():
        _setdefault(training, key, value)
    _setdefault(training, "scheduler", {})
    scheduler = training["scheduler"]
    if not isinstance(scheduler, dict):
        raise ValueError("training.scheduler must be a mapping")
    _setdefault(scheduler, "type", "cosine")
    _setdefault(scheduler, "horizon_epochs", training["num_epochs"])
    _setdefault(scheduler, "eta_min", 1e-7)
    if scheduler["type"] != "cosine":
        raise ValueError("official experiments require training.scheduler.type=cosine")
    if int(scheduler["horizon_epochs"]) < int(training["num_epochs"]):
        raise ValueError("scheduler horizon must be at least training.num_epochs")
    if not training["deterministic"]:
        raise ValueError("official experiments require training.deterministic=true")

    if "path" not in data:
        raise ValueError("data.path is required")
    _setdefault(data, "filter_grades", False)
    _setdefault(data, "min_grade_index", 0)
    _setdefault(data, "max_grade_index", 18)
    _setdefault(data, "filter_repeats", False)
    _setdefault(data, "min_repeats", 1)
    _setdefault(data, "group_by_layout", True)
    if not data["group_by_layout"]:
        raise ValueError("official experiments require data.group_by_layout=true")
    if data["filter_grades"]:
        expected = int(data["max_grade_index"]) - int(data["min_grade_index"]) + 1
        if expected <= 0 or expected != int(model["num_classes"]):
            raise ValueError(
                "model.num_classes must equal the inclusive filtered grade range"
            )

    experiment = config["experiment"]
    _setdefault(experiment, "outer_test_ratio", 0.15)
    _setdefault(experiment, "outer_seed", 42)
    _setdefault(experiment, "cv_folds", 5)
    _setdefault(experiment, "cv_seeds", DEFAULT_CV_SEEDS)
    _setdefault(experiment, "refit_seed", 42)
    _setdefault(experiment, "primary_metric", "tolerance_1_accuracy")
    if int(experiment["cv_folds"]) != 5 or list(experiment["cv_seeds"]) != DEFAULT_CV_SEEDS:
        raise ValueError("official experiments require 5 folds and cv_seeds [42, 43, 44]")
    if experiment["primary_metric"] != "tolerance_1_accuracy":
        raise ValueError("official experiments rank candidates by tolerance_1_accuracy")
    ratio = float(experiment["outer_test_ratio"])
    if not 0 < ratio < 0.5:
        raise ValueError("experiment.outer_test_ratio must be between 0 and 0.5")

    checkpoint = config["checkpoint"]
    _setdefault(checkpoint, "dir", "models")
    _setdefault(config, "device", "cpu")

    authored_data_path = Path(str(data["path"]))
    data_path = authored_data_path if authored_data_path.is_absolute() else config_path.parent / authored_data_path
    return config, data_path.resolve()


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256_json(config)


def manifest_spec(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only the cohort and split policy frozen by a manifest.

    Model and optimizer hyperparameters are intentionally excluded so multiple
    candidates can be compared on identical frozen memberships.
    """
    data = config["data"]
    experiment = config["experiment"]
    return {
        "data": {
            "path": data["path"],
            "group_by_layout": bool(data["group_by_layout"]),
            "filters": cohort_filters(config),
        },
        "split": {
            "outer_test_ratio": float(experiment["outer_test_ratio"]),
            "outer_seed": int(experiment["outer_seed"]),
            "cv_folds": int(experiment["cv_folds"]),
            "cv_seeds": [int(seed) for seed in experiment["cv_seeds"]],
        },
    }


def manifest_spec_hash(config: Mapping[str, Any]) -> str:
    return sha256_json(manifest_spec(config))


def cohort_filters(config: Mapping[str, Any]) -> Dict[str, Any]:
    data = config["data"]
    return {
        "grade": {
            "enabled": bool(data["filter_grades"]),
            "min_index": int(data["min_grade_index"]),
            "max_index": int(data["max_grade_index"]),
        },
        "repeats": {
            "enabled": bool(data["filter_repeats"]),
            "minimum": int(data["min_repeats"]),
        },
    }


def filter_records(records: Iterable[ProcessedProblem], config: Mapping[str, Any]) -> List[ProcessedProblem]:
    filters = cohort_filters(config)
    result = []
    for record in records:
        if filters["repeats"]["enabled"] and record.repeats < filters["repeats"]["minimum"]:
            continue
        grade = filters["grade"]
        if grade["enabled"] and not grade["min_index"] <= record.label <= grade["max_index"]:
            continue
        result.append(record)
    if not result:
        raise ValueError("filters produced an empty experiment cohort")
    return result


def canonical_cohort_rows(records: Sequence[ProcessedProblem]) -> List[Dict[str, Any]]:
    rows = [
        {
            "problem_id": str(record.problem_id),
            "label": int(record.label),
            "repeats": int(record.repeats),
            "layout_hash": layout_hash(record.tensor),
        }
        for record in records
    ]
    return sorted(rows, key=lambda item: item["problem_id"])


def load_cohort(config: Mapping[str, Any], data_path: Path) -> Tuple[List[ProcessedProblem], Dict[str, Any]]:
    source_records = load_problem_records(data_path)
    records = filter_records(source_records, config)
    rows = canonical_cohort_rows(records)
    identity = {
        "source_path": str(config["data"]["path"]),
        "source_sha256": sha256_file(data_path),
        "source_count": len(source_records),
        "cohort_sha256": sha256_json(rows),
        "cohort_count": len(records),
        "filters": cohort_filters(config),
    }
    return records, identity


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed every RNG controlled by the experiment process."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_tree_sha256(root: Path) -> str:
    """Hash experiment-affecting source/config files, including uncommitted files."""
    files = set()
    for base in (root / "classifier", root / "moonboard_core"):
        files.update(
            path
            for path in base.rglob("*.py")
            if "__pycache__" not in path.parts and ".pytest_cache" not in path.parts
        )
    files.update((root / "classifier").glob("config*.yaml"))
    files.add(root / "classifier" / "main.py")
    files.add(root / "pyproject.toml")
    files.add(root / "uv.lock")
    digest = hashlib.sha256()
    for path in sorted((path for path in files if path.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def code_revision() -> Dict[str, Any]:
    root = repository_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *RELEVANT_GIT_PATHS],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "commit": commit,
            "dirty": bool(status),
            "source_tree_sha256": _source_tree_sha256(root),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "dirty": True,
            "source_tree_sha256": _source_tree_sha256(root),
        }


def require_clean_revision() -> Dict[str, Any]:
    revision = code_revision()
    if revision["dirty"] or not revision["commit"]:
        raise RuntimeError(
            "official refit/evaluation requires a clean committed classifier and moonboard_core revision"
        )
    return revision


def environment_metadata() -> Dict[str, Any]:
    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
