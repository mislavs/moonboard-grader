# MoonBoard Generator

A Variational Autoencoder (VAE) for generating synthetic MoonBoard climbing problems at specified difficulty grades.

## Installation

```bash
uv sync
pip install -e ../moonboard_core
```

## Quick Start

**Train a model:**

```bash
py main.py train --config config.yaml
```

**Generate problems:**

```bash
py main.py generate --checkpoint models/best_vae.pth --grade 6B+ --num-samples 10
```

**Save to file:**

```bash
py main.py generate --checkpoint models/best_vae.pth --grade 6B+ --num-samples 10 --output generated.json
```

**Generate only valid problems:**

```bash
py main.py generate --checkpoint models/best_vae.pth --grade 7A --num-samples 10 --retry
```

**Evaluate model quality:**

```bash
py main.py evaluate --checkpoint models/best_vae.pth --output results.json
```

## Commands

### Train

```bash
py main.py train [--config config.yaml] [--resume checkpoint.pth]
```

Training monitors validation loss with both LR-on-plateau scheduling and patience-based early stopping.

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs
```

### Generate

```bash
py main.py generate --checkpoint PATH --grade GRADE [OPTIONS]
```

**Key options:**

- `--grade`: Grade to generate (e.g., "6B+", "7A")
- `--grade-labels`: Comma-separated **global** grade labels (moonboard_core indices)
- `--num-samples`: Number of problems (default: 1)
- `--temperature`: Sampling randomness, 0.5-2.0 (default: 1.0)
- `--threshold`: Hold detection threshold, 0.0-1.0 (default: 0.5)
- `--output`: Save to JSON file
- `--include-grade`: Include grade info in output
- `--retry`: Generate only valid problems
- `--cpu`: Force CPU usage

**Examples:**

```bash
# Higher creativity
py main.py generate --checkpoint models/best_vae.pth --grade 7B --temperature 1.5

# Denser problems
py main.py generate --checkpoint models/best_vae.pth --grade 6C --threshold 0.3

# Multiple grades
py main.py generate --checkpoint models/best_vae.pth --grade-labels "0,2,4,6,8"
```

`--grade` and `--grade-labels` are always interpreted in global grade space.
Filtered checkpoints remap labels internally and map back at the CLI boundary.

### Evaluate

Assess generator quality with comprehensive metrics:

```bash
# Run all metrics (no classifier needed for primary metrics)
py main.py evaluate --checkpoint models/best_vae.pth --output results.json

# Run specific metrics
py main.py evaluate --checkpoint models/best_vae.pth --metrics reconstruction,diversity

# Include optional classifier check (requires classifier)
py main.py evaluate --checkpoint models/best_vae.pth --classifier-checkpoint ../classifier/test_models/best_model.pth --metrics classifier_check

# Customize number of samples
py main.py evaluate --checkpoint models/best_vae.pth --num-samples 50 --metrics diversity
```

#### Available Metrics

**High Priority (No Classifier Required):**

- **`reconstruction`**: IoU between original and reconstructed problems
  - Target: >0.85 (excellent), >0.70 (good)
  - Measures how well the VAE learns to encode/decode problems
  - Per-channel breakdown (start/middle/end holds)
  - Per-grade analysis

- **`diversity`**: Uniqueness of generated problems
  - Target: >95% unique (excellent), >80% unique (good)
  - Pairwise Hamming distance between generated problems
  - Ensures model produces varied problems, not duplicates

- **`statistical`**: Similarity to real problem distributions
  - Target: Wasserstein distance <1.5 (excellent), <2.5 (good)
  - Compares statistics: num_holds, start/end holds, vertical spread
  - Low distance = generated problems statistically similar to real ones

**Medium Priority:**

- **`latent_space`**: Quality of learned representations
  - Target: Silhouette score >0.3 (excellent), >0.0 (acceptable)
  - Measures grade clustering in latent space
  - Note: Low scores expected for reconstruction-focused VAEs

**Low Priority (Requires Classifier, Limited Reliability):**

- **`classifier_check`**: Grade accuracy via classifier
  - ⚠️ **WARNING**: Limited by classifier's 35% exact accuracy, 70% ±1 grade
  - Use for RELATIVE comparison between models only
  - Not a reliable absolute quality measure
  - Requires `--classifier-checkpoint` argument

#### Metric Options

- `--checkpoint PATH`: VAE model checkpoint (required)
- `--data PATH`: Path to validation data (default: `../data/problems.json`)
- `--metrics METRICS`: Comma-separated list (default: all available)
- Unknown metric names fail fast with a non-zero exit.
- `--num-samples N`: Samples per grade for generation metrics (default: 100)
- `--output FILE`: Save results to JSON file
- `--classifier-checkpoint PATH`: Classifier for `classifier_check` metric
- `--cpu`: Force CPU usage

#### Examples

```bash
# Quick quality check
py main.py evaluate --checkpoint models/best_vae.pth --metrics reconstruction

# Full evaluation without classifier
py main.py evaluate --checkpoint models/best_vae.pth

# Complete evaluation with all metrics
py main.py evaluate --checkpoint models/best_vae.pth --classifier-checkpoint ../classifier/test_models/best_model.pth --output full_eval.json

# Compare model checkpoints
py main.py evaluate --checkpoint models/epoch_10.pth --output epoch_10.json
py main.py evaluate --checkpoint models/best_vae.pth --output best.json
# Then compare the JSON files
```

#### Interpreting Results

**Console Output:**
- Human-readable tables for each metric
- Overall scores and per-grade breakdowns
- Warnings for metrics with limitations

**JSON Output:**
- Complete nested structure for programmatic access
- Includes all raw data (centroid vectors, per-grade details, etc.)
- Use for automated evaluation pipelines

**What Good Scores Look Like:**
- Reconstruction IoU: 0.85-0.95 (excellent model learning)
- Diversity uniqueness: 95-100% (no duplicate generation)
- Statistical distance: 0.5-1.5 (closely matches real problems)
- Latent silhouette: -0.1 to 0.2 (expected for reconstruction VAEs)
- Classifier check: Interpret with caution due to classifier limits

## Configuration

Edit `config.yaml` to customize training:

```yaml
data:
  data_path: "../data/problems.json"
  train_split: 0.8
  batch_size: 64

model:
  latent_dim: 128
  grade_embedding_dim: 32

training:
  num_epochs: 50
  learning_rate: 0.001
  max_grad_norm: 1.0
  kl_weight: 1.0
  kl_annealing: true
  kl_annealing_epochs: 10
  early_stopping_patience: 15
  early_stopping_min_delta: 0.0001

checkpoint:
  checkpoint_dir: "models"

logging:
  log_dir: "runs"
  log_interval: 100

device: "cuda"  # or "cpu"
```

## Data Format

**Input** (`../data/problems.json`):

```json
{
  "Grade": "6B+",
  "Moves": [
    {"Description": "A5", "IsStart": true, "IsEnd": false},
    {"Description": "K10", "IsStart": false, "IsEnd": true}
  ]
}
```

**Output** (generated problems):

```json
[
  {
    "moves": [...],
    "stats": {
      "num_holds": 10,
      "num_start_holds": 2,
      "num_end_holds": 1,
      "num_middle_holds": 7
    }
  }
]
```

## Generation Parameters

**Temperature**: Controls randomness
- `0.5`: Conservative
- `1.0`: Standard (default)
- `1.5`: Creative

**Threshold**: Controls hold density
- `0.3`: More holds
- `0.5`: Balanced (default)
- `0.7`: Fewer holds

## Python API

```python
from src.generator import ProblemGenerator
from moonboard_core import encode_grade

# Load generator
generator = ProblemGenerator.from_checkpoint('models/best_vae.pth')

# Generate problems
problems = generator.generate(
    grade_label=encode_grade('6B+'),
    num_samples=10,
    temperature=1.0
)

# Access moves
for problem in problems:
    print(f"Generated {len(problem['moves'])} holds")
```

`ConditionalVAE` sampling semantics:

- `decode()` and `forward()` return logits.
- `sample_logits()` returns logits sampled from the prior.
- `sample()` returns probabilities in `[0, 1]` (`sigmoid(sample_logits())`).

## Troubleshooting

**Training Issues:**

- **Training too slow**: Use GPU with `device: "cuda"` in config
- **KL loss goes to zero**: Enable KL annealing in config
- **Out of memory**: Reduce `batch_size` in config or use `--cpu`
- **Poor reconstruction**: Reduce KL weight or increase model capacity

**Generation Issues:**

- **Invalid generated problems**: Use `--retry` flag or lower `--threshold 0.3`
- **No diversity**: Increase temperature (try 1.5) or check training data diversity
- **Too sparse/dense**: Adjust `--threshold` (0.3 = more holds, 0.7 = fewer holds)

**Evaluation Issues:**

- **Low reconstruction IoU (<0.5)**: Model may be undertrained or KL weight too high
- **Low diversity (<50% unique)**: Increase latent_dim or check for mode collapse
- **High statistical distance (>5.0)**: Generated problems differ from real ones - may need more training or better data
- **Negative silhouette score**: Expected for reconstruction-focused VAEs (not a problem unless <-0.5)
- **Grade conditioning fails**: Classifier checkpoint not found or incompatible - check path and versions
- **Evaluation OOM**: Reduce `--num-samples` (try 50 or 20)

## Testing

```bash
pytest tests/
```

## Technical Details

See [`spec.md`](spec.md) for VAE architecture and implementation details.
