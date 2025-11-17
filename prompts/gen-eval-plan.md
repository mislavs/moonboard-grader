# Add Generator Evaluation System

## Overview

Implement an `evaluate` command for the generator CLI that calculates quality metrics for the trained VAE model. Build infrastructure first (CLI + orchestrator), then add metrics incrementally with each one immediately testable. Primary metrics focus on reconstruction quality, diversity, and statistical similarity (no classifier needed). Grade conditioning is optional due to classifier accuracy limitations (35% exact, 70% ±1).

## Implementation Strategy

Each step adds a complete, testable metric. After implementing each metric, you can run:

```bash
py main.py evaluate --checkpoint models/best_vae.pth --metrics <newly-added-metric>
```

## Implementation Steps

### 1. Create CLI Infrastructure and Orchestrator ✅ COMPLETE

**Files**: `generator/src/evaluator/` (module), `generator/main.py`

Create the evaluation framework with no metrics yet:

**evaluator module structure**:

```
generator/src/evaluator/
  __init__.py           # Public API exports
  orchestrator.py       # run_evaluation() function
  reconstruction.py     # Reconstruction metric
  diversity.py          # Diversity metric
  statistical.py        # Statistical similarity metric
  latent_space.py       # Latent space quality metric
  grade_conditioning.py # Grade conditioning metric
  utils.py              # Shared helpers
```

**Each metric file**:

- Contains a single `evaluate_<metric>()` function
- Returns placeholder "not implemented" message initially
- Will have its own dependencies and helper functions when implemented
- Can be worked on independently

**orchestrator.py**:

- Create `METRIC_FUNCTIONS` dispatch table mapping metric names to functions (single source of truth)
- Create `get_metrics()` function that dynamically detects which metrics are ready
- Create `run_evaluation()` function that accepts metric list and delegates to individual metric functions
- Validate requested metrics against dispatch table

**utils.py**:

- Add helper `load_data_loader()` for loading validation data
- Add helper `extract_problem_stats()` for statistics extraction (used by multiple metrics)

**main.py**:

- Add `evaluate_command()` function with full argument parsing
- Add evaluate subparser with all arguments: `--checkpoint`, `--data`, `--classifier-checkpoint`, `--metrics`, `--num-samples`, `--output`, `--cpu`
- Implement console output formatting (header, metric sections, summary)
- Implement JSON output handler
- Default to running all available (implemented) metrics

**Output format** (with no metrics ready yet):

```
=== GENERATOR EVALUATION ===
Model: generator/models/best_vae.pth

No metrics available yet.

No metrics to evaluate yet.
```

**Test after this step**: Run `py main.py evaluate --help` to verify CLI is set up correctly.

**Status**: ✅ Completed - CLI infrastructure and orchestrator created. Refactored into modular structure with each metric in its own file for better separation of concerns. **Metrics are auto-detected** from the `METRIC_FUNCTIONS` dispatch table - single source of truth, no manual tracking, no redundant lists. Clean, concise API with just `get_metrics()`. Tested and working. No metrics ready yet, which is expected at this stage.

### 2. Implement Reconstruction Quality Metric ✅ COMPLETE

**File**: `generator/src/evaluator/reconstruction.py`

Replace placeholder with full `evaluate_reconstruction()`:

- Load validation data using `load_data_loader()` from utils
- For each batch, encode and reconstruct through VAE
- Calculate IoU between original and reconstructed grids
- Calculate per-channel IoU (start/middle/end holds)
- Calculate per-grade IoU statistics
- Return comprehensive dict with mean_iou, std_iou, per_channel_iou, per_grade_iou

**Key logic**:

```python
def evaluate_reconstruction(model, val_loader, threshold=0.5, device='cpu'):
    model.eval()
    ious = []
    channel_ious = {'start': [], 'middle': [], 'end': []}
    
    with torch.no_grad():
        for grids, grades in val_loader:
            grids = grids.to(device)
            grades = grades.to(device)
            
            x_recon, mu, logvar = model(grids, grades)
            x_recon_binary = (torch.sigmoid(x_recon) > threshold).float()
            
            # Calculate IoU per sample
            for i in range(grids.size(0)):
                original = grids[i]
                reconstructed = x_recon_binary[i]
                
                intersection = (original * reconstructed).sum()
                union = ((original + reconstructed) > 0).float().sum()
                iou = (intersection / (union + 1e-8)).item()
                ious.append(iou)
                
                # Per channel
                for ch, name in enumerate(['start', 'middle', 'end']):
                    ch_intersection = (original[ch] * reconstructed[ch]).sum()
                    ch_union = ((original[ch] + reconstructed[ch]) > 0).float().sum()
                    ch_iou = (ch_intersection / (ch_union + 1e-8)).item()
                    channel_ious[name].append(ch_iou)
    
    return {
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'per_channel_iou': {k: np.mean(v) for k, v in channel_ious.items()},
        'num_samples': len(ious)
    }
```

**Update orchestrator**: Metric will be **automatically detected** once implementation no longer returns `{'status': 'not_implemented'}`.

**Test after this step**:

```bash
py main.py evaluate --checkpoint models/best_vae.pth --metrics reconstruction
```

**Status**: ✅ Completed - Reconstruction metric fully implemented with:
- IoU calculation for overall and per-channel (start/middle/end) reconstruction quality
- Per-grade IoU statistics displayed in clean table format with human-readable grade labels (e.g., "6A+", "7B")
- **Grade decoding fixed** to use dataset's label_to_grade mapping (correctly handles filtered/remapped labels)
- Auto-detection working correctly (fixed orchestrator logic to detect implemented metrics)
- Model loading fixed to use `checkpoint['model_config']` structure
- Tested successfully with mean IoU of 0.9534 on validation set (excellent reconstruction quality)
- Returns comprehensive metrics: mean_iou, std_iou, per_channel_iou, per_grade_iou, num_samples, threshold, interpretation
- Console output features formatted table for per-grade statistics (readable at a glance)
- JSON output preserves full nested structure for programmatic access
- Default data path set to `../data/problems.json` for convenience

### 3. Implement Diversity Metric ✅ COMPLETE

**File**: `generator/src/evaluator/diversity.py`

Add full `evaluate_diversity()`:

- Generate num_samples problems at each grade
- Convert valid problems to grid representations
- Calculate pairwise Hamming distances using `scipy.spatial.distance.pdist`
- Count unique problems (exact duplicates)
- Calculate diversity statistics per grade and overall
- Return dict with mean_diversity, std_diversity, unique_problems, uniqueness_ratio, per_grade_diversity

**Key logic**:

```python
def evaluate_diversity(generator, num_samples_per_grade=100, device='cpu'):
    from scipy.spatial.distance import pdist
    from moonboard_core import build_grid, get_num_grades
    
    results_per_grade = {}
    
    for grade_label in range(get_num_grades()):
        # Generate problems
        problems = generator.generate(
            grade_label=grade_label,
            num_samples=num_samples_per_grade,
            temperature=1.0,
            validate=True
        )
        
        # Extract valid problems and convert to grids
        valid_problems = [p for p in problems if p.get('validation', {}).get('valid', True)]
        if len(valid_problems) < 2:
            continue
            
        grids = []
        for problem in valid_problems:
            grid = build_grid(problem['moves'])
            grids.append(grid.flatten())
        
        grids = np.array(grids)
        
        # Calculate pairwise Hamming distances
        if len(grids) > 1:
            distances = pdist(grids, metric='hamming')
            mean_diversity = np.mean(distances)
            std_diversity = np.std(distances)
        else:
            mean_diversity = 0
            std_diversity = 0
        
        # Count unique problems
        unique = len(np.unique(grids, axis=0))
        
        results_per_grade[grade_label] = {
            'mean_diversity': mean_diversity,
            'std_diversity': std_diversity,
            'unique_problems': unique,
            'total_valid': len(valid_problems),
            'uniqueness_ratio': unique / len(valid_problems)
        }
    
    # Aggregate across grades
    all_diversities = [r['mean_diversity'] for r in results_per_grade.values()]
    all_uniqueness = [r['uniqueness_ratio'] for r in results_per_grade.values()]
    
    return {
        'overall_mean_diversity': np.mean(all_diversities),
        'overall_uniqueness_ratio': np.mean(all_uniqueness),
        'per_grade': results_per_grade
    }
```

**Update orchestrator**: Metric will be **automatically detected** once implementation no longer returns `{'status': 'not_implemented'}`.

**Test after this step**:

```bash
py main.py evaluate --checkpoint models/best_vae.pth --metrics diversity --num-samples 50
```

**Status**: ✅ Completed - Diversity metric fully implemented with:
- Pairwise Hamming distance calculation for generated problems at each grade
- Uniqueness ratio (percentage of unique problems vs total valid)
- Per-grade diversity statistics with detailed breakdown
- Graceful handling of grades with insufficient valid problems (skips with detailed reason)
- Overall aggregation across all valid grades
- Automatic detection working correctly (no longer returns 'not_implemented')
- Tested successfully with 20 samples per grade showing 100% uniqueness
- Returns comprehensive metrics: overall_mean_diversity, overall_std_diversity, overall_uniqueness_ratio, per_grade details, interpretation
- Console output displays nested per-grade statistics clearly
- JSON output preserves full nested structure
- Uses `create_grid_tensor` from moonboard_core for grid conversion
- Proper logging at each step for debugging and monitoring

### 4. Implement Statistical Similarity Metric

**File**: `generator/src/evaluator/statistical.py`

Add full `evaluate_statistical_similarity()`:

- Generate problems at each grade
- Extract statistics: num_holds, num_start, num_end, num_middle, vertical_spread, horizontal_spread
- Load real problems from dataset at same grades
- Calculate Wasserstein distance between generated and real distributions
- Compare mean/std for each statistic
- Return comprehensive similarity scores

**Key logic**:

```python
def evaluate_statistical_similarity(generator, data_path, num_samples_per_grade=100, device='cpu'):
    from scipy.stats import wasserstein_distance
    from moonboard_core import get_num_grades
    
    # Load real dataset
    with open(data_path, 'r') as f:
        real_data = json.load(f)
    
    # Group real problems by grade
    real_by_grade = {}
    for problem in real_data:
        grade_label = encode_grade(problem['Grade'])
        if grade_label not in real_by_grade:
            real_by_grade[grade_label] = []
        real_by_grade[grade_label].append(_extract_problem_stats(problem))
    
    results_per_grade = {}
    
    for grade_label in range(get_num_grades()):
        if grade_label not in real_by_grade:
            continue
        
        # Generate problems
        gen_problems = generator.generate(
            grade_label=grade_label,
            num_samples=num_samples_per_grade,
            validate=True
        )
        
        valid_gen = [p for p in gen_problems if p.get('validation', {}).get('valid', True)]
        gen_stats = [_extract_problem_stats({'moves': p['moves']}) for p in valid_gen]
        
        if len(gen_stats) < 10:  # Need enough samples
            continue
        
        # Calculate Wasserstein distances
        real_stats = real_by_grade[grade_label]
        
        distances = {}
        for stat_name in ['num_holds', 'num_start', 'num_end', 'num_middle', 'vertical_spread']:
            gen_values = [s[stat_name] for s in gen_stats]
            real_values = [s[stat_name] for s in real_stats]
            distances[stat_name] = wasserstein_distance(gen_values, real_values)
        
        results_per_grade[grade_label] = {
            'wasserstein_distances': distances,
            'mean_distance': np.mean(list(distances.values())),
            'num_generated': len(gen_stats),
            'num_real': len(real_stats)
        }
    
    # Overall score
    all_mean_distances = [r['mean_distance'] for r in results_per_grade.values()]
    
    return {
        'overall_mean_distance': np.mean(all_mean_distances),
        'interpretation': 'lower is better (0 = identical distributions)',
        'per_grade': results_per_grade
    }

def _extract_problem_stats(problem):
    """Helper to extract statistics from a problem."""
    moves = problem['moves'] if 'moves' in problem else problem.get('Moves', [])
    
    num_holds = len(moves)
    num_start = sum(1 for m in moves if m.get('isStart') or m.get('IsStart'))
    num_end = sum(1 for m in moves if m.get('isEnd') or m.get('IsEnd'))
    num_middle = num_holds - num_start - num_end
    
    # Vertical spread (range of row numbers)
    positions = [m.get('description') or m.get('Description') for m in moves]
    rows = [int(pos[1:]) for pos in positions if len(pos) > 1]
    vertical_spread = max(rows) - min(rows) if rows else 0
    
    return {
        'num_holds': num_holds,
        'num_start': num_start,
        'num_end': num_end,
        'num_middle': num_middle,
        'vertical_spread': vertical_spread
    }
```

**Update orchestrator**: Metric will be **automatically detected** once implementation no longer returns `{'status': 'not_implemented'}`.

**Test after this step**:

```bash
py main.py evaluate --checkpoint models/best_vae.pth --metrics statistical
```

### 5. Implement Latent Space Quality Metric

**File**: `generator/src/evaluator/latent_space.py`

Add full `evaluate_latent_space()`:

- Load validation data
- Encode all problems to latent space
- Calculate silhouette score for grade clustering
- Calculate per-grade centroids and variance
- Measure grade separation distances
- Return quality scores

**Key logic**:

```python
def evaluate_latent_space(model, val_loader, device='cpu'):
    from sklearn.metrics import silhouette_score
    
    model.eval()
    all_latents = []
    all_grades = []
    
    with torch.no_grad():
        for grids, grades in val_loader:
            grids = grids.to(device)
            mu, logvar = model.encode(grids)
            all_latents.append(mu.cpu().numpy())
            all_grades.extend(grades.tolist())
    
    latents = np.concatenate(all_latents, axis=0)
    grades = np.array(all_grades)
    
    # Silhouette score (how well grades cluster)
    if len(np.unique(grades)) > 1:
        silhouette = silhouette_score(latents, grades)
    else:
        silhouette = 0.0
    
    # Calculate per-grade centroids
    centroids = {}
    for grade in np.unique(grades):
        grade_latents = latents[grades == grade]
        centroids[int(grade)] = {
            'mean': grade_latents.mean(axis=0).tolist(),
            'std': grade_latents.std(axis=0).mean()
        }
    
    # Measure grade separation (distance between adjacent grade centroids)
    centroid_distances = []
    sorted_grades = sorted(centroids.keys())
    for i in range(len(sorted_grades) - 1):
        g1, g2 = sorted_grades[i], sorted_grades[i+1]
        c1 = np.array(centroids[g1]['mean'])
        c2 = np.array(centroids[g2]['mean'])
        dist = np.linalg.norm(c1 - c2)
        centroid_distances.append(dist)
    
    return {
        'silhouette_score': float(silhouette),
        'interpretation': 'higher is better (-1 to 1, >0.3 is good clustering)',
        'latent_mean': float(np.abs(latents).mean()),
        'latent_std': float(latents.std()),
        'grade_separation': float(np.mean(centroid_distances)) if centroid_distances else 0.0,
        'per_grade_centroids': centroids
    }
```

**Update orchestrator**: Metric will be **automatically detected** once implementation no longer returns `{'status': 'not_implemented'}`.

**Test after this step**:

```bash
py main.py evaluate --checkpoint models/best_vae.pth --metrics latent_space
```

### 6. Implement Optional Grade Conditioning Metric

**File**: `generator/src/evaluator/grade_conditioning.py`

Add `evaluate_grade_conditioning()` with prominent warnings:

- Check if classifier_checkpoint is provided
- Load classifier using `classifier.src.predictor.Predictor`
- Generate problems at each grade
- Classify generated problems
- Calculate accuracy metrics with warnings
- Return results with limitations section

**Key logic**:

```python
def evaluate_grade_conditioning(generator, classifier_checkpoint, num_samples_per_grade=100, device='cpu'):
    """
    Evaluate grade conditioning accuracy using classifier.
    
    ⚠️ WARNING: This metric has limited reliability due to classifier accuracy.
    Current classifier baseline: ~35% exact, ~70% ±1 grade.
    Use this for RELATIVE comparison between models, not absolute quality assessment.
    """
    from classifier.src.predictor import Predictor
    from moonboard_core import get_num_grades, build_grid
    
    if classifier_checkpoint is None:
        return {
            'error': 'Classifier checkpoint required for grade conditioning metric',
            'skipped': True
        }
    
    # Load classifier
    classifier = Predictor(classifier_checkpoint, device=device)
    
    results_per_grade = {}
    
    for grade_label in range(get_num_grades()):
        # Generate problems
        problems = generator.generate(
            grade_label=grade_label,
            num_samples=num_samples_per_grade,
            validate=True
        )
        
        valid_problems = [p for p in problems if p.get('validation', {}).get('valid', True)]
        if len(valid_problems) == 0:
            continue
        
        # Classify each problem
        predictions = []
        for problem in valid_problems:
            grid = build_grid(problem['moves'])
            result = classifier.predict_from_tensor(grid)
            predictions.append(result['predicted_label'])
        
        predictions = np.array(predictions)
        labels = np.full(len(predictions), grade_label)
        
        # Calculate accuracy
        exact_acc = np.mean(predictions == labels) * 100
        tolerance_1 = np.mean(np.abs(predictions - labels) <= 1) * 100
        
        results_per_grade[grade_label] = {
            'exact_accuracy': exact_acc,
            'tolerance_1_accuracy': tolerance_1,
            'num_valid': len(valid_problems),
            'num_generated': num_samples_per_grade
        }
    
    # Overall metrics
    all_exact = [r['exact_accuracy'] for r in results_per_grade.values()]
    all_tol1 = [r['tolerance_1_accuracy'] for r in results_per_grade.values()]
    
    return {
        'overall_exact_accuracy': np.mean(all_exact),
        'overall_tolerance_1_accuracy': np.mean(all_tol1),
        'per_grade': results_per_grade,
        'warnings': [
            'Classifier baseline accuracy: ~35% exact, ~70% ±1 grade',
            'Results limited by classifier performance',
            'Use for relative comparison between models only',
            'Low absolute scores may reflect classifier weakness, not generator issues'
        ],
        'interpretation': 'CAUTION: Limited reliability metric'
    }
```

**Update orchestrator**: Metric will be **automatically detected** once implementation no longer returns `{'status': 'not_implemented'}`.

**Test after this step**:

```bash
# Without classifier (should skip gracefully)
py main.py evaluate --checkpoint models/best_vae.pth --metrics grade_conditioning

# With classifier (should run with warnings)
py main.py evaluate --checkpoint models/best_vae.pth --classifier-checkpoint ../classifier/test_models/best_model.pth --metrics grade_conditioning
```

### 7. Add Documentation

**Files**: `generator/README.md`, `generator/spec.md`

Update documentation:

- Add Evaluate section to README with examples
- Document each metric with interpretation guides
- Emphasize metric priorities (reconstruction > diversity > statistical > latent_space > grade_conditioning)
- Add troubleshooting section
- Include expected value ranges

**README section**:

````markdown
### Evaluate

Assess generator quality with comprehensive metrics:

```bash
# Run all metrics (no classifier needed for primary metrics)
py main.py evaluate --checkpoint models/best_vae.pth --output results.json

# Run specific metrics
py main.py evaluate --checkpoint models/best_vae.pth --metrics reconstruction,diversity

# Include optional grade conditioning (requires classifier)
py main.py evaluate \
  --checkpoint models/best_vae.pth \
  --classifier-checkpoint ../classifier/test_models/best_model.pth \
  --metrics grade_conditioning
````

#### Available Metrics

**High Priority (No Classifier Required):**

- `reconstruction`: IoU between original and reconstructed problems (target: >0.7)
- `diversity`: Uniqueness of generated problems (target: >80% unique)
- `statistical`: Similarity to real problem distributions (target: distance <0.1)

**Medium Priority:**

- `latent_space`: Quality of learned representations (target: silhouette >0.2)

**Low Priority (Requires Classifier, Limited Reliability):**

- `grade_conditioning`: Grade accuracy via classifier (⚠️ limited by classifier's 35% accuracy)

```

## Key Design Decisions

1. **Infrastructure first**: CLI and orchestrator ready from step 1
2. **Incremental testing**: Each metric immediately usable after implementation
3. **Classifier-independent focus**: Primary metrics don't require classifier
4. **Clear metric priority**: Order reflects reliability and usefulness
5. **Comprehensive warnings**: Grade conditioning clearly marked as limited-reliability
6. **Flexible metric selection**: Can run all, subset, or individual metrics
7. **Modular architecture**: Each metric in its own file for isolation and maintainability
8. **Auto-detection**: Metrics are automatically detected when implemented - no manual tracking needed

## Testing Checklist

After each step:

- Run help: `py main.py evaluate --help`
- Test metric alone: `--metrics <new-metric>`
- Test with output: `--output test.json` and verify JSON structure
- Test with different num-samples
- Verify console formatting is clear and informative