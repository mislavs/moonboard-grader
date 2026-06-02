# Generator: Reduce Latent Capacity and Fix Prior/Posterior Mismatch

A MoonBoard problem carries only tens of bits of information (~8 holds on a fixed board), but the VAE uses a 128-dimensional Gaussian latent (`latent_dim: 128` in `generator/config.yaml`, line 14; default in `generator/src/vae.py`, line 34). That over-capacity has two compounding effects: the latent can encode essentially everything, so the decoder has little incentive to rely on the grade embedding (worsening `generator-grade-conditioning.md`); and the aggregate posterior drifts far from the `N(0, I)` prior that sampling draws from, so prior samples land in regions the decoder never trained on, producing invalid or unrealistic generations even when reconstruction looks good.

Sampling always draws `z ~ N(0, I)` (`generator/src/generator.py`, `generate_batch`, lines 278-291), so closing the prior/posterior gap directly improves generated quality.

---

## Tasks

### 1. Reduce `latent_dim` and sweep

**Status:** Not started

**Severity:** High

**Action:**
- Lower `model.latent_dim` in `generator/config.yaml` to the 16-32 range and run a sweep over {16, 32, 64} (keeping 128 as the baseline reference).
- Confirm reconstruction stays acceptable (see validation) at the chosen size; pick the smallest latent that does not degrade reconstruction.

---

### 2. Retune KL weight / annealing and add a free-bits option

**Status:** Not started

**Severity:** Medium

Smaller latents change how much KL pressure is appropriate. Free-bits prevents collapse while still pulling the posterior toward the prior.

**Action:**
- Re-sweep `kl_weight` and `kl_annealing_epochs` for the new latent size (coordinate with `generator-recon-loss-class-imbalance.md`, which also changes the recon scale).
- Add an optional free-bits floor `lambda` (clamp per-dimension KL at a minimum before summing) in `vae_loss` (`generator/src/vae.py`) or `_run_batch`, exposed via config; this lets KL pressure shape the latent without driving any dimension to exactly zero.

---

### 3. (Optional) Select the checkpoint on a generation-side metric

**Status:** Not started

**Severity:** Medium

`_is_validation_improved` (`generator/src/vae_trainer.py`, lines 136-148) selects the "best" checkpoint purely on validation reconstruction loss, which can pick a model that autoencodes well but samples poorly.

**Action:**
- Periodically (e.g., every N epochs) generate a small batch from the prior and compute a cheap generation-side signal (validity rate, or a lightweight statistical distance) and incorporate it into checkpoint selection, or at minimum log it for manual selection.
- Keep the cost bounded (small sample count, infrequent cadence) so training time is not dominated by evaluation.

---

### 4. Config and tests

**Status:** Not started

**Severity:** Low

**Action:**
- Ensure `latent_dim` and any new free-bits / selection options are read from config and persisted in the checkpoint `model_config`.
- Confirm `generator/tests/test_vae.py` parametrized latent-dim tests still pass for the new default; add a free-bits unit test if implemented.

---

## Validation

Prerequisites: `../data/problems.json` present; a classifier checkpoint at `../classifier/test_models/best_model.pth` for `classifier_check`; run all commands from `generator/`; add `--cpu` if no GPU. Training is user-initiated only. Keep the 128-d baseline (`models/baseline_vae.pth`, `eval_baseline.json`) for comparison.

Sweep: retrain with `model.latent_dim` in {16, 32, 64}, saving each as `models/vae_l<dim>.pth`, then per dim:
- `py main.py evaluate --checkpoint models/vae_l<dim>.pth --classifier-checkpoint ../classifier/test_models/best_model.pth --output eval_l<dim>.json`

Compare across the JSONs (each vs the 128-d baseline):
1. Diversity (guard against collapse at small latent): from `--metrics diversity`
   - `metrics.diversity.overall_uniqueness_ratio`: must stay HIGH, target >= 0.8.
   - `metrics.diversity.overall_mean_diversity`: target > 0.3.
2. Sampling realism: `metrics.statistical.overall_mean_distance`: SMALLER-or-equal vs baseline, target < 2.0.
3. Grade fidelity not regressed: `metrics.classifier_check.overall_stats.exact_match_percent`: BIGGER-or-equal vs baseline.
4. KL health (no collapse): `tensorboard --logdir runs` -> `Loss/val_kl` stays clearly > 0 (not ~0).
5. Reconstruction retained: `metrics.reconstruction.mean_iou` should remain > 0.7.

Decision rule: pick the smallest `latent_dim` that keeps `mean_iou > 0.7` and `uniqueness_ratio >= 0.8` while maximizing `classifier_check` within +-1 and minimizing `statistical.overall_mean_distance`.

Acceptance: a smaller latent improves or holds the sampling-side metrics (statistical, classifier_check, diversity) without collapsing KL or materially degrading reconstruction.

## Sequencing and dependencies

All three generator-accuracy plans interact through the recon-vs-KL balance. Recommended order: `generator-recon-loss-class-imbalance.md` first, then this plan, then `generator-grade-conditioning.md`, re-running the evaluation between changes. Right-sizing the latent here makes grade conditioning (Doc 2) more effective, because a smaller latent forces more reliance on the grade embedding.

## Open Questions

- Free-bits floor vs a plain `kl_weight` retune - which gives a better prior/posterior match for this data without manual babysitting?
- Is the cost of a generation-side selection metric (task 3) worth it, or is logging it for manual checkpoint choice sufficient?
- How does the reduced latent interact with the `pos_weight` recon-scale change from Doc 1? The KL sweep should be redone after Doc 1 lands.
