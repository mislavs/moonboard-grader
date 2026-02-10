# Generator Project - Analysis Findings

Code review of the `generator/` project covering correctness, architecture, and quality.

---

## Critical - Architectural / Correctness Issues

### 1. Encoder is not grade-conditioned (fundamental CVAE design flaw)
- **File**: `generator/src/vae.py` — `encode()` method
- **Problem**: In a Conditional VAE, both the encoder and decoder should be conditioned on the label. Only the decoder receives the grade embedding. The encoder maps input grids to latent space without any knowledge of grade.
- **Impact**: The latent space cannot learn grade-specific structure. A 6A+ problem and an 8A problem with identical holds map to the same latent point. The decoder must do all grade differentiation from the embedding alone, limiting generation quality.
- **Fix**: Pass the grade embedding to the encoder — either concatenate to the flattened features before `fc_mu`/`fc_logvar`, or add the grade as a 4th input channel.
- **Status**: Done. Encoder now conditions on grade via feature+embedding concatenation before `fc_mu`/`fc_logvar`, with updated encode call sites and checkpoint compatibility handling for legacy architecture mismatch.
- [x] Add grade conditioning to the encoder

### 2. Decoder uses bilinear interpolation to fix dimension mismatch
- **File**: `generator/src/vae.py` — `decode()` method
- **Problem**: The decoder's transpose convolutions produce 32×22×14, which is resized to 3×18×11 via `F.interpolate(mode="bilinear")`. Bilinear interpolation creates smooth continuous values — exactly wrong for a sparse binary grid where ~590 of 594 cells should be 0. It smears activation across neighboring cells, making hold boundaries fuzzy.
- **Impact**: Reconstructed and generated grids have blurred hold positions. The 1×1 `output_adjust` conv cannot fully undo this damage.
- **Fix**: Redesign the decoder to output exactly 18×11 without interpolation. Options: asymmetric kernel/stride/padding, or a fully-connected layer producing exact spatial dimensions followed by refinement convs.
- **Status**: Done. Decoder now produces native `3×18×11` logits (`3×2 → 9×6 → 18×11`) with no interpolation or `output_adjust`, and legacy interpolation-decoder checkpoints fail fast with a dedicated compatibility error.
- [x] Redesign decoder to produce exact 18×11 output

### 3. `ConditionalVAE.sample()` returns raw logits, not probabilities
- **File**: `generator/src/vae.py` — `sample()` method
- **Problem**: The method returns the raw output of `decode()` (logits), but the docstring says it returns "grids." Anyone calling `model.sample()` directly would get logits, and thresholding would produce wrong results. The `ProblemGenerator` works around this by applying sigmoid in `_grids_from_latent`, so `sample()` is never used.
- **Fix**: Apply sigmoid in `sample()`, or clearly document the return as logits and rename accordingly.
- [ ] Fix `sample()` to return probabilities or document it as returning logits

---

## High - Design and Approach Issues

### 4. No gradient clipping in training
- **File**: `generator/src/vae_trainer.py` — `train_epoch()`
- **Problem**: With `reduction="sum"` in the loss, gradient magnitudes can spike, especially early in training. No `clip_grad_norm_` is applied before `optimizer.step()`.
- **Fix**: Add `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)` before the optimizer step.
- [ ] Add gradient clipping to training loop

### 5. `generate_with_retry` "max_attempts" is misleading
- **File**: `generator/src/generator.py` — `generate_with_retry()`
- **Problem**: Each "attempt" generates up to 10 problems in a batch. So `max_attempts=10` actually means up to 100 problems generated, not 10. The parameter name suggests individual attempts, not batch attempts.
- **Fix**: Rename to `max_batches` or change semantics to count individual samples.
- [ ] Clarify `max_attempts` semantics in retry generation

---

## Medium - Quality and Robustness

### 6. No early stopping mechanism
- **File**: `generator/src/vae_trainer.py` — `train()`
- **Problem**: Training runs for a fixed number of epochs. The LR scheduler reduces learning rate on plateau, but there's no mechanism to stop if validation loss hasn't improved for many epochs. With 70 epochs, the model may overfit.
- **Fix**: Add a patience-based early stopping check after validation.
- [ ] Add early stopping to training loop

### 7. No weight decay or dropout for regularization
- **File**: `generator/src/vae.py`, `generator/src/vae_trainer.py`
- **Problem**: No dropout layers in the model, and Adam optimizer has no weight decay. While KL divergence provides some regularization, additional regularization is standard for preventing overfitting.
- **Fix**: Add dropout after encoder/decoder layers and/or set `weight_decay` in Adam.
- [ ] Add dropout and/or weight decay

### 8. Code duplication between `train_epoch` and `_compute_losses`
- **File**: `generator/src/vae_trainer.py`
- **Problem**: `_compute_losses` was designed to share logic between training and validation, but `train_epoch` has its own nearly identical loop with added gradient computation. Changes to loss computation must be maintained in both places.
- **Fix**: Have `train_epoch` call `_compute_losses` with a training flag, or factor out the backpropagation step.
- **Status**: Done. Added shared `_run_batch` and training-aware `_compute_losses(...)`; `train_epoch` and `validate` now route through one loss path with mode-specific grad behavior.
- [x] Deduplicate training/validation loss computation

### 9. `load_checkpoint` doesn't set `weights_only`
- **File**: `generator/src/vae_trainer.py` — `load_checkpoint()`
- **Problem**: Uses `torch.load(checkpoint_path, map_location=self.device)` without `weights_only` parameter. Modern PyTorch warns about this being unsafe (arbitrary code execution via pickle). The `from_checkpoint` in `generator.py` correctly sets it.
- **Fix**: Add `weights_only=False` explicitly (or `True` if possible).
- [ ] Add explicit `weights_only` to `torch.load` calls

---

## Low - Minor Issues

### 10. Default `num_grades=17` in `ConditionalVAE.__init__` is wrong
- **File**: `generator/src/vae.py`
- **Problem**: Default is 17 but the system has 19 grades. Loading old checkpoints without `model_config` would create a mismatched model. Should be 19. Less impactful now that training and `from_checkpoint` always pass the correct value, but still misleading.
- [ ] Fix default `num_grades` to 19

### 11. `pin_memory=True` used unconditionally
- **File**: `generator/src/dataset.py` — `create_data_loaders()`
- **Problem**: `pin_memory=True` is only beneficial when using CUDA. On CPU it's a no-op or wastes memory. Should be conditional on device.
- [ ] Make `pin_memory` conditional on CUDA availability
