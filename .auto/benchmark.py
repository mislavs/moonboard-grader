"""Deterministic development benchmark for generator grade conditioning."""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "generator"
sys.path.insert(0, str(GENERATOR))
sys.path.insert(0, str(ROOT))
os.chdir(GENERATOR)

from classifier.src.predictor import Predictor
from moonboard_core import create_grid_tensor
from src.dataset import create_data_loaders
from src.generator import ProblemGenerator
from src.vae import ConditionalVAE
from src.vae_trainer import VAETrainer

TRAIN_SEED = 1729
SAMPLE_SEED = 2718
SAMPLES_PER_GRADE = 64
ARTIFACTS = ROOT / ".auto" / "artifacts"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main() -> None:
    shutil.rmtree(ARTIFACTS, ignore_errors=True)
    (ARTIFACTS / "models").mkdir(parents=True)
    seed_everything(TRAIN_SEED)

    source_config = yaml.safe_load((GENERATOR / "config.yaml").read_text(encoding="utf-8"))
    data = source_config["data"]
    train_loader, val_loader, dataset = create_data_loaders(
        data_path=str(ROOT / "data" / "problems.json"),
        batch_size=64,
        train_split=0.8,
        shuffle=True,
        num_workers=0,
        min_grade_index=2,
        max_grade_index=12,
    )
    model_config = source_config["model"]
    model = ConditionalVAE(
        latent_dim=model_config["latent_dim"],
        num_grades=dataset.get_num_model_grades(),
        grade_embedding_dim=model_config["grade_embedding_dim"],
        dropout_rate=model_config.get("dropout_rate", 0.1),
    )
    training = dict(source_config["training"])
    training.update(
        num_epochs=6,
        early_stopping_patience=None,
        checkpoint_dir=str(ARTIFACTS / "models"),
        log_dir=str(ARTIFACTS / "runs"),
        log_interval=100000,
    )
    trainer = VAETrainer(
        model,
        train_loader,
        val_loader,
        training,
        device="cuda" if torch.cuda.is_available() else "cpu",
        label_space_mode="remapped",
        grade_offset=dataset.grade_offset,
        min_grade_index=dataset.min_grade_index,
        max_grade_index=dataset.max_grade_index,
    )
    trainer.train()

    device = trainer.device
    checkpoint = torch.load(ARTIFACTS / "models" / "best_vae.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Deterministic posterior-mean reconstruction guardrail.
    intersections = unions = 0.0
    with torch.no_grad():
        for grids, grades in val_loader:
            grids, grades = grids.to(device), grades.to(device)
            mu, _ = model.encode(grids, grades)
            pred = torch.sigmoid(model.decode(mu, grades)) >= 0.5
            truth = grids >= 0.5
            intersections += (pred & truth).sum().item()
            unions += (pred | truth).sum().item()
    val_iou = intersections / unions if unions else 0.0

    seed_everything(SAMPLE_SEED)
    generator = ProblemGenerator(model, device=str(device))
    classifier = Predictor(ROOT / "classifier" / "models" / "best_model.pth", device=str(device))
    exact_by_grade = []
    within1_by_grade = []
    mean_predictions = []
    all_targets = []
    all_problem_keys = []
    generated = 0
    requested = SAMPLES_PER_GRADE * dataset.get_num_model_grades()

    for model_grade in range(dataset.get_num_model_grades()):
        global_grade = dataset.model_to_global_label(model_grade)
        problems = generator.generate_with_retry(
            grade_label=model_grade,
            num_samples=SAMPLES_PER_GRADE,
            max_attempts=10,
            temperature=1.0,
            gen_batch_size=SAMPLES_PER_GRADE,
        )
        generated += len(problems)
        if not problems:
            exact_by_grade.append(0.0)
            within1_by_grade.append(0.0)
            mean_predictions.append(float("nan"))
            continue
        grids = torch.tensor(
            np.asarray([create_grid_tensor(p["moves"]) for p in problems]),
            dtype=torch.float32,
        )
        predictions = np.asarray(
            [r["predicted_label"] for r in classifier.predict_from_tensor(grids)],
            dtype=np.int64,
        )
        differences = np.abs(predictions - global_grade)
        exact_by_grade.append(float(np.mean(differences == 0) * 100.0))
        within1_by_grade.append(float(np.mean(differences <= 1) * 100.0))
        mean_predictions.append(float(np.mean(predictions)))
        all_targets.append(global_grade)
        for problem in problems:
            key = tuple(sorted((m["description"], m.get("isStart", False), m.get("isEnd", False)) for m in problem["moves"]))
            all_problem_keys.append(key)

    finite = [(target, pred) for target, pred in zip(all_targets, mean_predictions) if np.isfinite(pred)]
    grade_correlation = float(np.corrcoef(np.asarray(finite).T)[0, 1]) if len(finite) > 1 else 0.0
    metrics = {
        "within1_accuracy": float(np.mean(within1_by_grade)),
        "exact_accuracy": float(np.mean(exact_by_grade)),
        "valid_rate": 100.0 * generated / requested,
        "unique_rate": 100.0 * len(set(all_problem_keys)) / len(all_problem_keys) if all_problem_keys else 0.0,
        "val_iou": val_iou,
        "grade_correlation": grade_correlation,
        "params_m": sum(p.numel() for p in model.parameters()) / 1_000_000.0,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("GRADE_MEANS " + json.dumps(mean_predictions))
    for name, value in metrics.items():
        print(f"METRIC {name}={value:.8f}")


if __name__ == "__main__":
    main()
