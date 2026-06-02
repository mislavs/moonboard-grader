# Generator: Strengthen Grade Conditioning

For a conditional generator the core quality question is whether a problem generated for "7A" actually grades like a 7A. Today the only link between the target grade and the output is a 32-d embedding concatenated into the 160-d decoder input (and the encoder also sees the grade), with no loss term forcing the decoder to use it. With a high-capacity latent the model reconstructs almost entirely from `z` and treats grade as a faint nudge - the README itself reports a near-zero latent silhouette and weak `classifier_check`.

Current conditioning in `generator/src/vae.py` (`decode`, lines 157-186):

```python
grade_emb = self.grade_embedding(grade_labels)
z_cond = torch.cat([z, grade_emb], dim=1)
h = self.fc_decode(z_cond)
```

This plan raises grade fidelity. The approaches below are ordered by expected impact; Option A is the recommended primary lever and can be combined with B/C.

---

## Tasks

### 1. Option A (recommended): auxiliary grade loss via the frozen CNN classifier

**Status:** Not started

**Severity:** High

Add a term that pushes decoded grids to be classified as the target grade: `aux = CE(classifier(sigmoid(decode(z, g))), g_target)`, added to the VAE objective with weight `aux_grade_loss_weight`. This directly optimizes the quantity `classifier_check` measures. The classifier (`classifier/src/predictor.py`) uses CoordConv but accepts a 3-channel `(N, 3, 18, 11)` tensor (it prepends coordinate channels internally), so the decoder's continuous sigmoid probabilities can be fed straight in and gradients flow back to the generator.

**Action:**
- Load the classifier once in `VAETrainer` (reuse `Predictor` to get `predictor.model`), call `requires_grad_(False)` on its parameters, and keep it in `eval()` so BatchNorm/Dropout are fixed and its weights are never updated.
- In `_run_batch` (`generator/src/vae_trainer.py`, line 197), compute `aux` from `sigmoid(x_recon)` and add `aux_grade_loss_weight * aux` to the total loss (training only).
- Align label spaces: convert the generator target (model label) -> global label (via the trainer's `grade_offset`) -> classifier label (using `Predictor.grade_offset`/`min_grade_index`/`max_grade_index`). Skip/clip the aux term for grades outside the classifier's supported range.
- Add `aux_grade_loss_weight` and `classifier_checkpoint` under `training:` in `generator/config.yaml`; default weight 0 (feature off) so training without a classifier is unchanged. Consider annealing the weight in after a few epochs.

---

### 2. Option B: FiLM conditioning in the decoder

**Status:** Not started

**Severity:** High

Instead of a single concat, use the grade embedding to produce per-channel scale/shift (gamma, beta) applied after each decoder block (FiLM), so grade modulates every spatial location rather than being ~20% of one input vector.

**Action:**
- Add a small MLP `grade_embedding -> (gamma, beta)` per decoder stage in `generator/src/vae.py` and apply `gamma * h + beta` after each `BatchNorm`/activation in `self.decoder`.
- Note this changes the architecture; `generator/src/checkpoint_compat.py` already validates architecture, so old checkpoints will be rejected with a clear retrain message (acceptable).
- Optionally also FiLM-condition the encoder for symmetry (lower priority).

---

### 3. Option C: rebalance conditioning vs latent capacity

**Status:** Not started

**Severity:** Medium

Make grade a larger fraction of the decoder input.

**Action:**
- Raise `grade_embedding_dim` (e.g., 32 -> 64) and/or lower `latent_dim` in `generator/config.yaml`. The latent reduction is owned by `generator-latent-capacity.md`; coordinate the two so they are tuned together.

---

### 4. Tests and evaluation wiring

**Status:** Not started

**Severity:** Medium

**Action:**
- Add a unit test that the aux-loss path runs and is differentiable on a tiny synthetic batch with a stub/real classifier, and that `aux_grade_loss_weight=0` reproduces current loss.
- For FiLM, extend `generator/tests/test_vae.py` shape tests to the new module and confirm changing only the grade label changes the decoded output more than in the baseline (a conditioning-strength check).

---

## Validation

Prerequisites: `../data/problems.json` present; a classifier checkpoint at `../classifier/test_models/best_model.pth`; run all commands from `generator/`; add `--cpu` if no GPU. Training is user-initiated only. Capture the baseline (`models/baseline_vae.pth`) and its `eval_baseline.json` before the change.

After the change: retrain, then check:

1. Grade fidelity (primary): `py main.py evaluate --checkpoint models/best_vae.pth --classifier-checkpoint ../classifier/test_models/best_model.pth --metrics classifier_check --num-samples 100 --output eval_cond.json`
   - `metrics.classifier_check.overall_stats.exact_match_percent`: BIGGER, target >= +5 to +10 absolute points vs baseline (the classifier ceiling is ~35% exact / ~70% within +-1, so judge relative to baseline, not absolute).
   - `metrics.classifier_check.overall_stats.off_by_one_percent`: BIGGER; target combined within +-1 (exact + off-by-one) >= 60%.
   - Per-grade in `metrics.classifier_check.per_grade`: `mean_predicted_grade` should rise monotonically with the target grade, and each grade's `prediction_distribution` mode should sit on or adjacent to the target.
2. Latent structure: `py main.py evaluate --checkpoint models/best_vae.pth --metrics latent_space`
   - `metrics.latent_space.silhouette_score`: BIGGER (toward > 0 from a near-zero/negative baseline).
   - `metrics.latent_space.grade_separation`: BIGGER vs baseline.
3. Conditioning sanity (grades must differ): `py main.py evaluate --checkpoint models/best_vae.pth --metrics statistical --output eval_cond_stats.json`, then compare `metrics.statistical.per_grade` at the lowest vs highest available grade - `num_holds`/`vertical_spread` should differ in the expected direction rather than being near-identical.

Guardrail: confirm `metrics.reconstruction.mean_iou` (run `--metrics reconstruction`) does not regress materially (stay > 0.7), since a strong aux term can trade reconstruction for grade fit.

Acceptance: classifier exact/within +-1 up, silhouette and grade_separation up, per-grade statistics vary with grade, reconstruction not materially degraded.

## Sequencing and dependencies

All three generator-accuracy plans interact through the recon-vs-KL balance. Recommended order: `generator-recon-loss-class-imbalance.md` first, then `generator-latent-capacity.md`, then this plan, re-running the evaluation between changes. Grade fidelity is easiest to judge once generations are already valid (Doc 1) and the latent is right-sized (Doc 3). The aux-loss weight will likely need re-tuning after those land.

## Open Questions

- Which approach to adopt: A alone, B alone, or A+B? A is the most direct lever; B is a cleaner structural fix; they compose.
- Feed the classifier continuous `sigmoid` probabilities or sharpened/near-binary inputs? The classifier was trained on binary grids, so continuous inputs are mildly out of distribution - a sigmoid temperature or weight annealing may help.
- How to handle target grades outside the classifier's supported range (filtered classifier vs filtered generator) - skip the aux term, or require matching ranges?
- Does the aux term risk mode collapse toward "classifier-pleasing" prototypes? Track diversity (`generator-latent-capacity.md` validation) alongside grade fidelity.
