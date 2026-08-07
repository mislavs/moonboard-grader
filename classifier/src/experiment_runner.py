"""Deterministic training primitives for official CV and refit artifacts."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from moonboard_core.data_processor import ProcessedProblem

from .dataset import MoonboardDataset
from .evaluator import calculate_mean_absolute_error, evaluate_model
from .experiment import (
    CHECKPOINT_SCHEMA_VERSION,
    code_revision,
    config_hash,
    environment_metadata,
    seed_everything,
)
from .losses import create_loss_function
from .models import create_model


def records_to_dataset(
    records: Sequence[ProcessedProblem],
    grade_offset: int,
) -> MoonboardDataset:
    tensors = np.stack([record.tensor for record in records])
    labels = np.asarray([record.label - grade_offset for record in records], dtype=np.int64)
    return MoonboardDataset(tensors, labels)


def _model_from_config(config: Mapping[str, Any]) -> nn.Module:
    model = config["model"]
    return create_model(
        model_type=model["type"],
        num_classes=int(model["num_classes"]),
        use_attention=bool(model["use_attention"]),
        dropout_conv=float(model["dropout_conv"]),
        dropout_fc1=float(model["dropout_fc1"]),
        dropout_fc2=float(model["dropout_fc2"]),
    )


def _optimizer_from_config(model: nn.Module, config: Mapping[str, Any]):
    training = config["training"]
    name = str(training["optimizer"]).lower()
    kwargs = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
    }
    if name == "adam":
        return optim.Adam(model.parameters(), **kwargs)
    if name == "sgd":
        return optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError(f"unsupported optimizer: {name}")


def _criterion_from_config(labels: np.ndarray, config: Mapping[str, Any], device: torch.device):
    training = config["training"]
    num_classes = int(config["model"]["num_classes"])
    class_weights = None
    if training["use_class_weights"]:
        unique = np.unique(labels)
        values = compute_class_weight(class_weight="balanced", classes=unique, y=labels)
        full = np.ones(num_classes, dtype=np.float32)
        full[unique] = values
        full = np.clip(full, 0.1, float(training["max_class_weight"]))
        class_weights = torch.as_tensor(full, dtype=torch.float32, device=device)

    loss_type = training["loss_type"]
    if loss_type == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(training["label_smoothing"]),
        )
    return create_loss_function(
        loss_type=loss_type,
        num_classes=num_classes,
        class_weights=class_weights,
        gamma=float(training["focal_gamma"]),
        ordinal_weight=float(training["ordinal_weight"]),
        ordinal_alpha=float(training["ordinal_alpha"]),
        ordinal_smoothing_kernel=training.get("ordinal_smoothing_kernel"),
        smoothing=float(training["label_smoothing"]),
    )


def _loaders(
    train_dataset: MoonboardDataset,
    validation_dataset: Optional[MoonboardDataset],
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    sampler = None
    shuffle = True
    if training["use_balanced_sampling"]:
        counts = np.bincount(train_dataset.labels, minlength=int(config["model"]["num_classes"]))
        counts = np.maximum(counts, 1)
        strategy = training["balanced_sampling_strategy"]
        if strategy == "sqrt":
            weights = 1.0 / np.sqrt(counts)
        elif strategy == "inverse":
            weights = 1.0 / counts
        else:
            raise ValueError(f"unsupported balanced sampling strategy: {strategy}")
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights[train_dataset.labels], dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: Optional[float],
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for tensors, labels in loader:
        tensors = tensors.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(tensors)
        loss = criterion(output, labels)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    if not batches:
        raise ValueError("training loader is empty")
    return total_loss / batches


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _base_checkpoint(
    stage: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    membership: Mapping[str, Sequence[str]],
    seeds: Mapping[str, Any],
    history: Mapping[str, Any],
    selected_epoch: int,
    grade_offset: int,
    revision: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "resolved_config": copy.deepcopy(dict(config)),
        "config_sha256": config_hash(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_id": manifest["manifest_id"],
        "dataset": copy.deepcopy(manifest["dataset"]),
        "filters": copy.deepcopy(manifest["cohort"]["filters"]),
        "split_membership": {name: list(values) for name, values in membership.items()},
        "seeds": copy.deepcopy(dict(seeds)),
        "code_revision": copy.deepcopy(dict(revision)) if revision is not None else code_revision(),
        "environment": environment_metadata(),
        "history": copy.deepcopy(dict(history)),
        "selected_epoch": int(selected_epoch),
        "epoch": int(selected_epoch - 1),
        "grade_offset": grade_offset,
        "min_grade_index": int(config["data"]["min_grade_index"]),
        "max_grade_index": int(config["data"]["max_grade_index"]),
    }


def run_cv_fold(
    train_records: Sequence[ProcessedProblem],
    validation_records: Sequence[ProcessedProblem],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    membership: Mapping[str, Sequence[str]],
    seed: int,
    fold: int,
    device: torch.device,
    checkpoint_path: Path,
    artifact_stage: str = "cv_fold",
    revision: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if artifact_stage not in ("cv_fold", "diagnostic"):
        raise ValueError(f"unsupported validation artifact stage: {artifact_stage}")
    generator = seed_everything(seed, deterministic=True)
    grade_offset = int(config["data"]["min_grade_index"]) if config["data"]["filter_grades"] else 0
    train_dataset = records_to_dataset(train_records, grade_offset)
    validation_dataset = records_to_dataset(validation_records, grade_offset)
    train_loader, validation_loader = _loaders(train_dataset, validation_dataset, config, generator)
    assert validation_loader is not None

    model = _model_from_config(config).to(device)
    optimizer = _optimizer_from_config(model, config)
    scheduler_config = config["training"]["scheduler"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_config["horizon_epochs"]),
        eta_min=float(scheduler_config["eta_min"]),
    )
    criterion = _criterion_from_config(train_dataset.labels, config, device)
    max_epochs = int(config["training"]["num_epochs"])
    patience = config["training"]["early_stopping_patience"]
    gradient_clip = config["training"]["gradient_clip"]

    history = {"train_loss": [], "validation": [], "learning_rate": []}
    best_key = None
    best_state = None
    best_metrics = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device, gradient_clip)
        metrics = evaluate_model(model, validation_loader, str(device), use_amp=False)
        metrics["mean_absolute_error"] = calculate_mean_absolute_error(
            np.asarray(metrics["predictions"]), np.asarray(metrics["labels"])
        )
        comparable_metrics = {
            key: value for key, value in metrics.items() if key not in ("predictions", "labels")
        }
        history["train_loss"].append(train_loss)
        history["validation"].append(comparable_metrics)
        history["learning_rate"].append(learning_rate)

        key = (
            float(metrics["tolerance_1_accuracy"]),
            -float(metrics["avg_loss"]),
            -epoch,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_metrics = comparable_metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        will_stop = patience is not None and stale_epochs >= int(patience)
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "max_epochs": max_epochs,
                    "learning_rate": learning_rate,
                    "train_loss": train_loss,
                    "metrics": comparable_metrics,
                    "is_best": best_epoch == epoch,
                    "stale_epochs": stale_epochs,
                    "patience": patience,
                    "will_stop": will_stop,
                }
            )
        scheduler.step()
        if will_stop:
            break

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    checkpoint = _base_checkpoint(
        artifact_stage,
        model,
        optimizer,
        scheduler,
        config,
        manifest,
        membership,
        {"training": seed, "cv_seed": seed, "fold": fold},
        history,
        best_epoch,
        grade_offset,
        revision,
    )
    checkpoint["validation_metrics"] = best_metrics
    if artifact_stage == "diagnostic":
        checkpoint["evaluation_kind"] = "diagnostic_validation"
        checkpoint["comparable"] = False
        checkpoint["promotion_eligible"] = False
    _atomic_torch_save(checkpoint, checkpoint_path)
    return {"selected_epoch": best_epoch, "metrics": best_metrics}


def run_refit(
    development_records: Sequence[ProcessedProblem],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    membership: Mapping[str, Sequence[str]],
    seed: int,
    epochs: int,
    device: torch.device,
    checkpoint_path: Path,
    source_cv_report_sha256: str,
) -> Dict[str, Any]:
    generator = seed_everything(seed, deterministic=True)
    grade_offset = int(config["data"]["min_grade_index"]) if config["data"]["filter_grades"] else 0
    dataset = records_to_dataset(development_records, grade_offset)
    train_loader, validation_loader = _loaders(dataset, None, config, generator)
    if validation_loader is not None:
        raise AssertionError("refit must not construct a validation loader")

    model = _model_from_config(config).to(device)
    optimizer = _optimizer_from_config(model, config)
    scheduler_config = config["training"]["scheduler"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_config["horizon_epochs"]),
        eta_min=float(scheduler_config["eta_min"]),
    )
    criterion = _criterion_from_config(dataset.labels, config, device)
    history = {"train_loss": [], "learning_rate": []}
    for _ in range(epochs):
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["train_loss"].append(
            _train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                config["training"]["gradient_clip"],
            )
        )
        scheduler.step()

    checkpoint = _base_checkpoint(
        "refit",
        model,
        optimizer,
        scheduler,
        config,
        manifest,
        membership,
        {"refit": seed},
        history,
        epochs,
        grade_offset,
    )
    checkpoint["source_cv_report_sha256"] = source_cv_report_sha256
    _atomic_torch_save(checkpoint, checkpoint_path)
    return checkpoint
