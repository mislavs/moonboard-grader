# Changelog

All notable changes to the Moonboard Grade Prediction model will be documented in this file.

---

## [v0.3.1] - 2025-11-01 (Critical Bugfix - Class Weights)

### Fixed
- **CRITICAL**: Fixed class weights calculation that caused extreme loss values and 0% accuracy
  - **Root causes identified**:
    1. Wrong label indexing: `labels[:(len(train_dataset))]` instead of `labels[train_idx]`
    2. Tiny epsilon (`1e-6`) causing billion-scale weights for rare classes
    3. Manual formula producing uncapped extreme weights even after fixes
  - **Solution**: Replaced manual calculation with sklearn's `compute_class_weight('balanced')`
    - Uses proper balanced weighting formula
    - Added weight capping: `np.clip(weights, 0.1, max_weight)` with default max=5.0
    - Prevents extreme weights while still helping with class imbalance
  - **Results**:
    - Before fix: Loss ~97, Val Acc 0%
    - After fix: Loss ~2.2, Val Acc ~34% (normal behavior)
  - Added `max_class_weight` config parameter (default: 5.0)

### Changed
- Added import: `from sklearn.utils.class_weight import compute_class_weight`
- Improved debug output: Shows weight range instead of raw class counts

---

## [v0.3.0] - 2025-11-01 (Overfitting Fix)

Major regularization and training improvements to address severe overfitting observed in v0.2.0.

**v0.2.0 Results (Overfitting Issue):**
- Training stopped at epoch 18 (best at epoch 3)
- Validation loss increased from 2.18 → 3.11 (+43% degradation)
- Train/Val gap: 1.72 (severe overfitting)
- Exact Accuracy: 34.47%
- ±1 Grade Accuracy: 68.53%
- ±2 Grade Accuracy: 86.72%

### Problem Diagnosis
- Model was **memorizing training data** instead of learning generalizable patterns
- Validation loss increased dramatically after epoch 3 while training loss decreased
- Early stopping patience (15) was too long, wasting 12+ epochs on overfitting
- Insufficient regularization for 391k parameter model

### Added
- **Data Augmentation** module (`src/augmentation.py`):
  - Horizontal flip augmentation with configurable probability
  - Effectively doubles training dataset diversity
  - Simulates climbers with different dominant hands
  - Applied only to training set, not validation/test
- **Label Smoothing** (0.1) to prevent overconfident predictions
- **Gradient Clipping** (max_norm=1.0) to prevent exploding gradients
- Augmentation configuration options in `config.yaml`
- Transform parameter support in `MoonboardDataset` class

### Changed
- **Regularization Strengthened**:
  - Dropout: 0.4/0.3 → **0.5/0.5** (stronger regularization)
  - Weight decay: 0.0001 → **0.001** (10x increase in L2 regularization)
  - Added label smoothing: 0.0 → **0.1**
- **Learning Rate Strategy**:
  - Initial LR: 0.0005 → **0.0003** (more stable training)
  - Scheduler factor: 0.5 → **0.3** (more aggressive reduction)
  - Scheduler patience: 5 → **3** (faster response to plateaus)
  - Min LR: 1e-6 → **1e-7** (allow finer tuning)
  - Added verbose=True to scheduler
- **Early Stopping**:
  - Patience: 15 → **8** epochs (stop overfitting earlier)
- **Training Configuration** (config.yaml):
  - Added `label_smoothing: 0.1`
  - Added `gradient_clip: 1.0`
  - Added `augmentation: true`
  - Added `flip_probability: 0.5`

### Technical Details

**Data Augmentation Implementation:**
```python
class MoonboardAugmentation:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob
    
    def __call__(self, grid):
        if random.random() < self.flip_prob:
            grid = torch.flip(grid, dims=[2])  # Mirror left-right
        return grid
```

**Updated Loss Function:**
```python
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1  # NEW: prevents overconfidence
)
```

**Gradient Clipping in Training Loop:**
```python
loss.backward()
if self.gradient_clip is not None:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
self.optimizer.step()
```

### Expected Improvements
Based on regularization changes:
- Best model epoch: 3 → **8-12** (more training before convergence)
- Validation loss trend: ↗️ Increasing → **➡️ Stable**
- Train/Val gap: 1.72 → **< 0.5** (healthy generalization)
- Exact Accuracy: 34.47% → **38-42%** (estimated)
- ±1 Grade Accuracy: 68.53% → **72-76%** (estimated)
- ±2 Grade Accuracy: 86.72% → **88-91%** (estimated)

### Why These Changes Work
1. **Stronger Dropout (0.5)** - Prevents neuron co-adaptation, forces redundancy
2. **Higher Weight Decay (0.001)** - Penalizes complex models, prefers simpler solutions
3. **Label Smoothing (0.1)** - Prevents 100% confidence, better calibration
4. **Data Augmentation** - More diverse training samples, position invariance
5. **Gradient Clipping** - Stabilizes training, prevents gradient explosion
6. **Lower LR (0.0003)** - More careful weight updates, better convergence
7. **Aggressive LR Schedule** - Adapts quickly to plateaus (factor=0.3, patience=3)
8. **Earlier Stopping (patience=8)** - Prevents wasting epochs on overfitting

All techniques work together to encourage **generalizable patterns** instead of **memorizing examples**.

### Files Modified
- `src/models.py` - Increased dropout from 0.4/0.3 to 0.5/0.5
- `src/trainer.py` - Added gradient clipping support
- `src/dataset.py` - Added transform parameter for augmentation
- `src/augmentation.py` - **NEW**: Data augmentation module
- `src/__init__.py` - Exported augmentation functions
- `main.py` - Integrated label smoothing, gradient clipping, and augmentation
- `config.yaml` - Updated all training hyperparameters

Watch for these **good signs** during training:
- Val loss decreases steadily for first 8-15 epochs
- Train/val gap stays small (< 1.0 difference)
- LR reductions happen 1-2 times during training
- Best model saved around epoch 10-15, not epoch 3

### Further Tuning Options

**If Still Overfitting:**
- Reduce model capacity (halve filter counts: 32→16, 64→32, 128→64)
- Increase dropout to 0.6
- Temporarily disable augmentation to isolate issues

**If Underfitting (val loss > 2.5):**
- Lower dropout to 0.3
- Increase learning rate to 0.0005
- Reduce weight decay to 0.0005

---

## [v0.2.0] - 2025-11-01

Major model architecture overhaul and training improvements to address baseline performance issues.

**Baseline Performance (v0.1.0):**
- Exact Accuracy: 37.28%
- ±1 Grade Accuracy: 65.81%
- ±2 Grade Accuracy: 86.14%
- Model Size: ~40k parameters

### Added
- **Batch Normalization** layers after each convolutional layer for training stability
- **3rd convolutional layer** (Conv2d 64→128) for deeper feature extraction
- **Extra fully connected layer** (1024→256) for improved representation capacity
- **Class weighting** in loss function to handle severe class imbalance (Grade 2: 11,833 samples vs Grade 15: 11 samples)
- **Learning rate scheduling** with ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
- **L2 regularization** via weight_decay=0.0001 to prevent overfitting
- Scheduler parameter support in `Trainer` class

### Changed
- **CNN Architecture** (src/models.py):
  - Layer 1: Conv2d(3→16) → Conv2d(3→32) with BatchNorm
  - Layer 2: Conv2d(16→32) → Conv2d(32→64) with BatchNorm
  - Added Layer 3: Conv2d(64→128) with BatchNorm
  - FC layers: 256→128→19 → 1024→256→128→19
  - Model parameters: ~40k → **~305k** (7.6x increase)
- **Dropout rates**: Reduced from 0.5 to 0.4/0.3 in FC layers (less aggressive)
- **Training Configuration** (config.yaml):
  - Learning rate: 0.001 → 0.0005 (more stable for larger model)
  - Batch size: 32 → 64 (better gradient estimates)
  - Max epochs: 100 → 150 (more training time)
  - Early stopping patience: 10 → 15 epochs
- **Loss function**: CrossEntropyLoss() → CrossEntropyLoss(weight=class_weights)

### Technical Details

**ConvolutionalModel Architecture:**
```
Input: (batch, 3, 18, 11)
├─ Conv2d(3→32, k=3, p=1) → BatchNorm2d(32) → ReLU → MaxPool2d(2)  [→ 32×9×5]
├─ Conv2d(32→64, k=3, p=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2) [→ 64×4×2]
├─ Conv2d(64→128, k=3, p=1) → BatchNorm2d(128) → ReLU              [→ 128×4×2]
├─ Flatten                                                          [→ 1024]
├─ Linear(1024→256) → ReLU → Dropout(0.4)
├─ Linear(256→128) → ReLU → Dropout(0.3)
└─ Linear(128→19)
Output: (batch, 19) logits
```

**Class Weighting Formula:**
```python
weight[i] = total_samples / (num_classes × class_count[i])
```

### Expected Performance
Based on architecture improvements:
- Exact Accuracy: 37.28% → **42-48%** (estimated)
- ±1 Grade Accuracy: 65.81% → **70-75%** (estimated)
- ±2 Grade Accuracy: 86.14% → **88-92%** (estimated)
- Improved performance on rare grades (10-16)

### Files Modified
- `src/models.py` - Enhanced CNN architecture
- `src/trainer.py` - Added scheduler support
- `main.py` - Class weighting and scheduler integration
- `config.yaml` - Updated training hyperparameters

---

## [v0.1.0] - Initial Baseline

Initial implementation with basic CNN architecture.

### Features
- 2-layer CNN architecture (16→32 filters)
- Basic fully connected head (256→128→19)
- Adam optimizer with fixed learning rate
- Early stopping with patience=10
- Train/Val/Test split (70/15/15)

### Performance
- Exact Accuracy: 37.28%
- ±1 Grade Accuracy: 65.81%
- ±2 Grade Accuracy: 86.14%
- Model Size: ~40k parameters

---

## Future Tuning Options

If further improvements are needed:

### Architecture
- Add 4th convolutional layer (3→64→128→256)
- Increase filter counts further
- Try residual connections (ResNet-style)
- Experiment with attention mechanisms

### Training
- Lower learning rate: 0.0003 for more stable training
- Higher learning rate: 0.001 if training too slow
- Try SGD with momentum instead of Adam
- Increase batch size to 128 (if GPU available)

### Data
- Add data augmentation (horizontal flips, rotations)
- Hold position jittering
- Synthetic data generation for rare grades

### Loss Functions
- Focal loss for hard examples
- Label smoothing
- Ordinal regression (treat as ordered classes)

