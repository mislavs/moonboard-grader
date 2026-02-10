# MoonBoard Generator Technical Specification

## Overview

Conditional Variational Autoencoder (CVAE) for generating climbing problems at specified difficulty grades.

## Architecture

### Input/Output

- **Input**: 3×18×11 tensor (start holds, middle holds, end holds)
- **Conditioning**: Integer grade label (0-indexed, embedded to 32-dim vector)
- **Output**: 3×18×11 tensor with logits

### Encoder

```
3×18×11 input
→ Conv2d(3→32) + BN + ReLU → 32×18×11
→ Conv2d(32→64, s=2) + BN + ReLU → 64×9×6
→ Conv2d(64→128, s=2) + BN + ReLU → 128×5×3
→ Conv2d(128→256, s=2) + BN + ReLU → 256×3×2
→ Flatten → 1536
→ Linear(1536→128) → μ
→ Linear(1536→128) → log σ²
```

**Reparameterization**: `z = μ + σ ⊙ ε` where `ε ~ N(0, I)`

### Decoder

```
Latent (128) + Grade embedding (32) → 160
→ Linear(160→1536) → Reshape to 256×3×2
→ ConvTranspose2d(256→128, k=4, s=3, p=1, op=1) + BN + ReLU → 128×9×6
→ ConvTranspose2d(128→64, k=3, s=2, p=1, op=(1,0)) + BN + ReLU → 64×18×11
→ Conv2d(64→32, k=3, p=1) + BN + ReLU → 32×18×11
→ Conv2d(32→3, k=1) → 3×18×11
```

## Loss Function

```
L = L_recon + β × L_KL

L_recon = Σ BCE(x_recon, x)
L_KL = -0.5 × Σ(1 + log σ² - μ² - σ²)
```

**KL Annealing**: β linearly increases from 0 to 1.0 over first 10 epochs to prevent posterior collapse.

## Training

### Dataset

- Loads from `../data/problems.json`
- Converts moves to grids using `moonboard_core.build_grid()`
- Encodes grades to integer labels
- Returns `(grid_tensor, grade_label)` pairs
- 80/20 train/validation split

### Hyperparameters

| Parameter | Default |
|-----------|---------|
| `latent_dim` | 128 |
| `grade_embedding_dim` | 32 |
| `learning_rate` | 0.001 |
| `batch_size` | 64 |
| `num_epochs` | 50 |
| `kl_weight` | 1.0 |
| `kl_annealing_epochs` | 10 |

### Optimizer & Scheduler

- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

### Checkpoints

Saved to `models/`:
- `best_vae.pth`: Best validation loss
- `final_vae.pth`: Final model

Contains model state, optimizer state, config, and training history.

## Generation

### Loading

```python
generator = ProblemGenerator.from_checkpoint(
    'models/best_vae.pth',
    device='cuda',
    threshold=0.5
)
```

### Sampling Process

1. Sample `z ~ N(0, temperature² × I)`
2. Decode with grade: `x = decode(z, grade_embedding)`
3. Apply sigmoid: `p = sigmoid(x)`
4. Threshold to binary: `holds = (p > threshold)`
5. Convert to moves: `moonboard_core.grid_to_moves()`

### Generation Methods

**Basic:**
```python
problems = generator.generate(grade_label=5, num_samples=10, temperature=1.0)
```

**Batch (multiple grades):**
```python
problems = generator.generate_batch(grade_labels=[3,5,7], temperature=1.0)
```

**With retry (ensures valid problems):**
```python
problems = generator.generate_with_retry(grade_label=5, num_samples=10, max_attempts=10)
```

### Parameters

**Temperature**: Controls sampling randomness
- `0.5`: Conservative, similar to training data
- `1.0`: Standard (default)
- `1.5`: Creative, more diverse

**Threshold**: Binary hold detection
- `0.3`: Dense problems (more holds)
- `0.5`: Balanced (default)
- `0.7`: Sparse problems (fewer holds)

### Validation

Uses `moonboard_core.validate_moves()`:
- At least 1 start hold
- At least 1 end hold
- Valid positions (A-R, 1-11)
- No duplicates

Typical validation rate: 70-90%

## Evaluation Metrics

### Overview

The evaluation system provides comprehensive quality assessment through five metrics, prioritized by reliability and usefulness.

### Metric Implementations

#### 1. Reconstruction Quality

**Purpose**: Measure VAE encoding/decoding fidelity

**Method**: Intersection over Union (IoU)

```python
def evaluate_reconstruction(model, val_loader, threshold=0.5):
    # Encode and reconstruct validation problems
    # Calculate IoU = intersection / union
    # Per-channel: start, middle, end holds
    # Per-grade: breakdown by difficulty
```

**Output**:
- Overall mean_iou, std_iou
- Per-channel IoU (start, middle, end)
- Per-grade IoU statistics

**Target**: >0.85 (excellent), >0.70 (good)

**Technical notes**:
- Uses sigmoid activation + thresholding
- Batch processing for efficiency
- Grade labels decoded from dataset mapping

#### 2. Diversity

**Purpose**: Ensure generated problems are unique and varied

**Method**: Pairwise Hamming distance + uniqueness ratio

```python
def evaluate_diversity(generator, num_samples_per_grade=100):
    # Generate problems at each grade
    # Convert to grid representations
    # Calculate pairwise Hamming distances
    # Count exact duplicates
    # Aggregate per-grade and overall
```

**Output**:
- Overall mean diversity (Hamming distance)
- Overall uniqueness ratio (unique / total)
- Per-grade diversity statistics

**Target**: >95% unique (excellent), >80% unique (good)

**Technical notes**:
- Uses `scipy.spatial.distance.pdist` with 'hamming' metric
- Grids flattened to 1D arrays for comparison
- Filters invalid problems before analysis

#### 3. Statistical Similarity

**Purpose**: Compare generated vs real problem distributions

**Method**: Wasserstein distance on problem statistics

```python
def evaluate_statistical_similarity(generator, data_path, num_samples_per_grade=100):
    # Load real dataset and group by grade
    # Generate problems at matching grades
    # Extract statistics: num_holds, num_start, num_end, num_middle, vertical_spread
    # Calculate Wasserstein distance for each statistic
    # Aggregate across grades
```

**Statistics extracted**:
- `num_holds`: Total hold count
- `num_start`: Start hold count
- `num_end`: End hold count
- `num_middle`: Middle hold count
- `vertical_spread`: Range of row numbers (max - min)

**Output**:
- Overall mean distance (lower = better)
- Per-grade Wasserstein distances
- Per-statistic aggregation (mean, std, min, max)

**Target**: <1.5 (excellent), <2.5 (good)

**Technical notes**:
- Uses `scipy.stats.wasserstein_distance`
- Requires sufficient samples (≥10) per grade
- Gracefully skips grades with insufficient data

#### 4. Latent Space Quality

**Purpose**: Assess learned representation structure

**Method**: Silhouette score + grade separation

```python
def evaluate_latent_space(model, val_loader):
    # Encode validation set to latent space
    # Calculate silhouette score (grade clustering)
    # Compute per-grade centroids and variance
    # Measure distances between adjacent grade centroids
```

**Output**:
- Silhouette score (-1 to 1, higher = better clustering)
- Latent space statistics (mean, std)
- Grade separation (centroid distances)
- Per-grade centroids with sample counts

**Target**: >0.3 (excellent), >0.0 (acceptable)

**Technical notes**:
- Uses `sklearn.metrics.silhouette_score`
- Low/negative scores expected for reconstruction-focused VAEs
- Centroids are 128-dimensional vectors (latent_dim)
- Grade separation measures Euclidean distance between centroids

#### 5. Grade Conditioning (Optional)

**Purpose**: Validate grade-conditioned generation accuracy

**Method**: Classifier prediction on generated problems

⚠️ **WARNING**: Limited reliability metric due to classifier baseline (~35% exact, ~70% ±1)

```python
def evaluate_classifier_check(generator, classifier_checkpoint, num_samples_per_grade=100):
    # Load classifier predictor
    # Generate problems at each grade
    # Classify each generated problem
    # Calculate accuracy metrics (exact, ±1, ±2)
```

**Output**:
- Overall exact, ±1, ±2 accuracy
- Per-grade accuracy breakdowns
- Warnings about classifier limitations

**Interpretation**: Use for RELATIVE comparison between models, not absolute quality

**Technical notes**:
- Requires classifier checkpoint path
- Gracefully skips if classifier not available
- Uses `classifier.src.predictor.Predictor`
- Results reflect classifier weakness, not just generator quality

### Orchestrator Architecture

**File**: `src/evaluator/orchestrator.py`

**Design**:
- `METRIC_FUNCTIONS`: Dispatch table mapping metric names to functions
- `get_metrics()`: Auto-detects implemented metrics (checks for 'not_implemented' status)
- `run_evaluation()`: Delegates to individual metric functions, aggregates results

**Auto-detection logic**:
```python
# Metric is "ready" if it doesn't return {'status': 'not_implemented'}
result = metric_func(...)
is_ready = result.get('status') != 'not_implemented'
```

### CLI Integration

**Command**: `py main.py evaluate`

**Arguments**:
- `--checkpoint`: VAE model checkpoint (required)
- `--data`: Validation data path (default: `../data/problems.json`)
- `--classifier-checkpoint`: Classifier for grade conditioning
- `--metrics`: Comma-separated list (default: all available)
- `--num-samples`: Samples per grade for generation metrics
- `--output`: JSON output file
- `--cpu`: Force CPU usage

**Output formats**:
- **Console**: Human-readable tables with interpretations
- **JSON**: Complete nested structure for programmatic access

### Expected Performance

| Metric | Good | Excellent | Notes |
|--------|------|-----------|-------|
| Reconstruction IoU | >0.70 | >0.85 | Core VAE quality |
| Diversity uniqueness | >80% | >95% | No mode collapse |
| Statistical distance | <2.5 | <1.5 | Realistic problems |
| Latent silhouette | >0.0 | >0.3 | May be negative |
| Grade conditioning | N/A | N/A | Unreliable metric |

## CLI

### Train

```bash
py main.py train --config config.yaml [--resume checkpoint.pth]
```

### Generate

```bash
py main.py generate \
  --checkpoint models/best_vae.pth \
  --grade 6B+ \
  --num-samples 10 \
  [--temperature 1.0] \
  [--threshold 0.5] \
  [--output file.json] \
  [--retry] \
  [--include-grade]
```

### Evaluate

```bash
py main.py evaluate \
  --checkpoint models/best_vae.pth \
  [--data ../data/problems.json] \
  [--metrics reconstruction,diversity,statistical,latent_space] \
  [--classifier-checkpoint ../classifier/test_models/best_model.pth] \
  [--num-samples 100] \
  [--output results.json] \
  [--cpu]
```

**Available metrics**: `reconstruction`, `diversity`, `statistical`, `latent_space`, `classifier_check`

**Default behavior**: Runs all implemented metrics (auto-detected)

## Performance

### Expected Training Losses

- **Reconstruction**: 100-200 → 30-50
- **KL divergence**: 0 → 50-100 (with annealing)
- **Validation**: Should track training loss

### Generation Quality

- 70-90% validation rate without retry
- 100% with retry logic
- Problems should be diverse at high temperature
- Grade conditioning should produce appropriate difficulty

### Evaluation Metrics

See "Evaluation Metrics" section above for complete details.

**Target scores for well-trained model:**

| Metric | Target Range | Critical Threshold |
|--------|--------------|-------------------|
| Reconstruction IoU | 0.85-0.95 | >0.70 (minimum) |
| Diversity uniqueness | 95-100% | >80% (minimum) |
| Statistical distance | 0.5-1.5 | <2.5 (maximum) |
| Latent silhouette | -0.1 to 0.2 | >-0.5 (acceptable) |
| Grade conditioning | N/A | Unreliable metric |

**Metric priorities** (most to least reliable):
1. Reconstruction (core VAE quality)
2. Diversity (mode collapse check)
3. Statistical (realism check)
4. Latent space (representation quality)
5. Grade conditioning (limited reliability)

## Integration

Depends on `moonboard_core`:
- `build_grid()`: moves → grid conversion
- `grid_to_moves()`: grid → moves conversion
- `encode_grade()`, `decode_grade()`: grade handling
- `validate_moves()`: problem validation

## Troubleshooting

### Training

**KL loss → 0**: Enable/increase KL annealing epochs

**Poor reconstruction**: Reduce KL weight or increase model capacity

**Out of memory**: Reduce batch size or use CPU

### Generation

**Invalid problems**: Use `--retry` or lower threshold

**No diversity**: Increase temperature or check training data diversity

**Too many/few holds**: Adjust threshold (0.3 = more, 0.7 = fewer)

### Evaluation

**Low reconstruction IoU (<0.5)**: Model undertrained or KL weight too high - retrain with adjusted config

**Low diversity (<50% unique)**: Mode collapse - increase latent_dim or add more training data

**High statistical distance (>5.0)**: Generated problems unrealistic - may need more epochs or better preprocessing

**Very negative silhouette (<-0.5)**: Latent space not learning grade structure - check grade embedding or increase embedding_dim

**Grade conditioning errors**: Ensure classifier checkpoint path is correct and classifier version matches

**Evaluation OOM**: Reduce `--num-samples` (e.g., 50 or 20 instead of 100)

**Missing dependencies**: Install scikit-learn for latent_space metric (`pip install scikit-learn`)

**Classifier import errors**: Ensure classifier module is in Python path and dependencies installed
