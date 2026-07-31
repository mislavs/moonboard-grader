# Moonboard Classifier Usage

## Official Experiment Workflow

Run commands from `classifier/`. Official experiments use one immutable manifest per exact dataset/filter cohort.

### 1. Create the manifest

```powershell
py main.py create-manifest `
  --config config.yaml `
  --output manifests/moonboard-masters-2017-all-v1.json
```

The command validates stable `problemId` values, applies the configured grade/repeat filters, hashes the source, semantic cohort, and experiment source tree, freezes an approximately 15% grouped outer test set, and records five validation folds for seeds `42`, `43`, and `44`. Existing manifests are never overwritten.

### 2. Cross-validate a candidate

```powershell
py main.py cross-validate `
  --config config.yaml `
  --manifest manifests/moonboard-masters-2017-all-v1.json
```

Each candidate runs 15 fresh fits from a clean committed revision. Fold artifacts are written atomically and matching completed folds are reused only when configuration, manifest, and code revision agree. Candidate reports rank by mean ±1-grade accuracy, then lower MAE, macro accuracy, and exact accuracy. The outer test membership is never loaded.

### 3. Refit the selected candidate

```powershell
py main.py refit `
  --config config.yaml `
  --manifest manifests/moonboard-masters-2017-all-v1.json `
  --cv-report models/experiments/<candidate-id>/cv/cv_report.json
```

Run refit only after selecting a candidate from validation results. It requires a clean committed code revision, trains from scratch on the complete development pool, and uses the median of the 15 selected CV epochs. The output filename contains candidate identity and seed, never accuracy.

### 4. Evaluate the promoted refit

```powershell
py main.py evaluate `
  --checkpoint models/experiments/<candidate-id>/refit_<candidate-id>_seed-42.pth `
  --manifest manifests/moonboard-masters-2017-all-v1.json
```

The command accepts only schema-v2 `stage=refit` checkpoints whose manifest, configuration, dataset, filters, membership, and clean code revision match. It writes an immutable `.test.json` report and confusion matrix. A checkpoint hash cannot receive a second official report in the same artifact directory.

## Non-comparable Diagnostics

Use this only for smoke checks or genuinely external datasets:

```powershell
py main.py evaluate-diagnostic `
  --checkpoint models/model.pth `
  --data ../data/external.json `
  --output diagnostic.json
```

Diagnostic reports always contain `evaluation_kind: diagnostic` and `comparable: false`. They are not benchmark or test scores.

## Prediction

```powershell
py main.py predict `
  --checkpoint models/model.pth `
  --input problem.json `
  --top-k 5 `
  --output prediction.json
```

Legacy checkpoints remain supported for prediction and diagnostic evaluation.

## Official Configuration

```yaml
model:
  type: cnn
  num_classes: 19

training:
  learning_rate: 0.0003
  batch_size: 64
  num_epochs: 150
  early_stopping_patience: 8
  optimizer: adam
  deterministic: true
  scheduler:
    type: cosine
    horizon_epochs: 150
    eta_min: 0.0000001

data:
  path: ../data/problems.json
  group_by_layout: true
  filter_grades: false
  min_grade_index: 0
  max_grade_index: 18
  filter_repeats: false
  min_repeats: 1

experiment:
  outer_test_ratio: 0.15
  outer_seed: 42
  cv_folds: 5
  cv_seeds: [42, 43, 44]
  refit_seed: 42
  primary_metric: tolerance_1_accuracy

checkpoint:
  dir: models
```

Official commands reject the old `train_ratio`, `val_ratio`, `test_ratio`, `random_seed`, `use_scheduler`, `scheduler_factor`, `scheduler_patience`, and `reproducibility_seed` fields instead of translating them silently.

## Dataset Contract

The JSON root must contain a `data` array. For a frozen manifest, every problem must have a unique stable ID repeated consistently on each move:

```json
{
  "data": [
    {
      "grade": "6B+",
      "repeats": 10,
      "moves": [
        {"problemId": 123, "description": "A5", "isStart": true, "isEnd": false},
        {"problemId": 123, "description": "K18", "isStart": false, "isEnd": true}
      ]
    }
  ]
}
```

Changing source bytes, cohort semantics, grade/repeat filters, or the outer seed intentionally invalidates the manifest. Create a new named manifest; do not overwrite the old benchmark.

## Verification

```powershell
uv run pytest
```

The test suite uses tiny synthetic fixtures. Do not launch the real 15-fit experiment without budgeting the required compute.
