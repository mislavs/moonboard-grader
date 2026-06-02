# Generator: Fix Reconstruction Loss Class Imbalance

The conditional VAE's reconstruction term treats all 594 grid cells equally, but a MoonBoard problem only activates ~5-12 of them (roughly 1-2% positive across the 3x18x11 tensor). With unweighted summed binary cross-entropy, the loss-minimizing behavior is to predict near-zero everywhere, so the decoder systematically under-produces holds. The practical symptoms are sparse or empty generations, low validity at the default `threshold=0.5`, and heavy reliance on `--retry` and lowered `--threshold` to get usable problems.

This is the highest-impact change for generation validity and statistical realism. Current loss in `generator/src/vae.py` (`vae_loss`, lines 267-293):

```python
recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction="sum")
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
total_loss = recon_loss + kl_weight * kl_loss
```

---

## Tasks

### 1. Add per-channel `pos_weight` support to `vae_loss`

**Status:** Not started

**Severity:** Highest

`F.binary_cross_entropy_with_logits` accepts a `pos_weight` argument that scales the loss contribution of positive (hold) cells. A `pos_weight` tensor of shape `(3, 1, 1)` broadcasts across the 18x11 grid so each channel (start/middle/end) gets its own weight.

**Action:**
- Add an optional `pos_weight: Optional[torch.Tensor] = None` parameter to `vae_loss` and forward it to `F.binary_cross_entropy_with_logits`.
- Keep `reduction="sum"` so the trainer's per-sample normalization (`loss / batch_size` in `_run_batch`) keeps the same meaning; document that positive-cell loss now scales by `pos_weight`.
- Move the weight tensor to the input's device inside the loss (or in the trainer) to avoid device mismatches.

---

### 2. Compute per-channel positive weights from the training data

**Status:** Not started

**Severity:** High

Each channel has a different sparsity: start (~2 holds), middle (~6 holds), end (~1-2 holds) out of 198 cells per channel, so weights should reflect the per-channel negative/positive ratio.

**Action:**
- Reuse `moonboard_core.data_processor.get_dataset_stats` (returns `avg_start_holds`, `avg_middle_holds`, `avg_end_holds`) to compute `pos_weight_c = (198 - avg_holds_c) / avg_holds_c` for each channel.
- Compute on the training split only (avoid leaking validation statistics) in `create_data_loaders` (`generator/src/dataset.py`) or trainer setup, and pass the resulting tensor to `vae_loss`.
- Allow a global scale/cap (e.g., `sqrt` or clamp at a max such as 50): the raw end-channel ratio can exceed 150 and may over-bias toward dense grids.

---

### 3. Plumb `pos_weight` config through the trainer and CLI

**Status:** Not started

**Severity:** High

**Action:**
- Add a `recon_pos_weight` entry under `training:` in `generator/config.yaml` (default `auto`; also accept an explicit `[w_start, w_middle, w_end]` list or a scalar).
- Read it in `build_training_config` (`generator/main.py`, lines 67-90) and pass it to `VAETrainer`.
- In `VAETrainer.__init__`, resolve `auto` to the computed per-channel tensor (or parse the explicit list/scalar), store it, and pass it into the `vae_loss` call in `_run_batch` (`generator/src/vae_trainer.py`, line 197).
- Persist the resolved weights in the checkpoint `config`/`model_config` for reproducibility.

---

### 4. Re-tune `kl_weight` for the new reconstruction scale

**Status:** Not started

**Severity:** Medium

Adding `pos_weight` increases the magnitude of the reconstruction term, which shifts the recon-vs-KL balance and can cause posterior collapse if `kl_weight` is left unchanged.

**Action:**
- After enabling `pos_weight`, sweep `kl_weight` (e.g., 0.1, 0.25, 0.5) and confirm `Loss/val_kl` in TensorBoard stays clearly above 0.
- Keep `kl_annealing: true`; revisit `kl_annealing_epochs` if early training is unstable.

---

### 5. Update and extend tests

**Status:** Not started

**Severity:** Medium

**Action:**
- Extend `generator/tests/test_vae.py` `TestVAELoss` to cover `pos_weight`: weighting should increase the loss/gradient contribution of positive cells relative to the unweighted case.
- Add a test that a `(3, 1, 1)` `pos_weight` broadcasts correctly and does not change output shapes.

---

### 6. (Follow-up) Evaluate Dice/Focal loss as an alternative

**Status:** Not started

**Severity:** Low

**Action:**
- If `pos_weight` alone does not bring validity to target, prototype a soft Dice or focal term added to (or replacing) BCE and compare using the validation procedure below.

---

## Validation

Prerequisites: `../data/problems.json` present; run all commands from `generator/`; add `--cpu` if no GPU. Training is user-initiated only.

Baseline (run once, before the change): train with the current config, copy `models/best_vae.pth` to `models/baseline_vae.pth`, then record:
- `py main.py generate --checkpoint models/baseline_vae.pth --grade 7A --num-samples 200` and note `Valid: X/200`.
- `py main.py evaluate --checkpoint models/baseline_vae.pth --metrics reconstruction,statistical --output eval_baseline.json`.

After the change: retrain, then check:

1. Validity at `threshold=0.5` WITHOUT `--retry` (the key signal):
   - `py main.py generate --checkpoint models/best_vae.pth --grade 6B --num-samples 200`
   - `py main.py generate --checkpoint models/best_vae.pth --grade 7A --num-samples 200`
   - Read the printed `Valid: X/200`. Expect BIGGER -> target >= 70% valid (baseline is typically < 15%).
2. Statistical realism: `py main.py evaluate --checkpoint models/best_vae.pth --metrics statistical --output eval_poswt.json`
   - `metrics.statistical.overall_mean_distance`: SMALLER, target < 2.0 (ideally < 1.5).
   - `metrics.statistical.per_statistic.num_holds.mean` (Wasserstein): SMALLER, target drop >= 50% vs baseline.
3. Sparse-channel reconstruction: `py main.py evaluate --checkpoint models/best_vae.pth --metrics reconstruction`
   - `metrics.reconstruction.per_channel_iou.end` and `.start`: BIGGER vs baseline (these gain the most).
   - `metrics.reconstruction.mean_iou`: BIGGER, target > 0.7.
4. KL not collapsed: `tensorboard --logdir runs` -> `Loss/val_kl` stays clearly > 0.

Acceptance: validity rises sharply at the default threshold, num_holds Wasserstein drops, sparse-channel IoU rises, and KL stays healthy.

## Sequencing and dependencies

All three generator-accuracy plans interact through the recon-vs-KL balance. Recommended order: this plan first, then `generator-latent-capacity.md`, then `generator-grade-conditioning.md`, re-running the evaluation between changes. This plan changes the reconstruction scale, so it should land before KL/latent tuning.

## Open Questions

- The auto-computed end-channel weight can exceed ~150; do we cap or scale it (clamp at ~50 or use `sqrt`) to avoid over-biasing toward dense grids?
- Should weights be computed per filtered grade range or globally across the dataset?
- Keep `reduction="sum"` (current scale semantics) or move to a per-pixel mean to make `kl_weight` more transferable across configs?
