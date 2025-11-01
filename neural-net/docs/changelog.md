# Changelog

All notable changes to the Moonboard Grade Prediction model will be documented in this file.

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

