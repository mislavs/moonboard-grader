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
→ ConvTranspose2d(256→128) + BN + ReLU → 128×6×4
→ ConvTranspose2d(128→64) + BN + ReLU → 64×11×7
→ ConvTranspose2d(64→32) + BN + ReLU → 32×22×14
→ Conv2d(32→3) → 3×22×14
→ Interpolate → 3×18×11
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

## Integration

Depends on `moonboard_core`:
- `build_grid()`: moves → grid conversion
- `grid_to_moves()`: grid → moves conversion
- `encode_grade()`, `decode_grade()`: grade handling
- `validate_moves()`: problem validation

## Troubleshooting

**KL loss → 0**: Enable/increase KL annealing epochs

**Poor reconstruction**: Reduce KL weight or increase model capacity

**Invalid problems**: Use `--retry` or lower threshold

**Out of memory**: Reduce batch size or use CPU

**No diversity**: Increase temperature or check training data diversity
