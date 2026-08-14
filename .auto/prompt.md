# Autoresearch: Improve Generator Grade Conditioning

## Objective
Improve the conditional VAE's sampling-side grade fidelity. Generated boards requested at each grade from 6A+ through 7C+ should be classified closer to that target by the repository's frozen classifier check. Prefer generic architectural/training improvements that strengthen conditioning; do not encode benchmark answers, alter classifier-check semantics, tune generation thresholds per grade, train against held-out samples, or otherwise game the classifier.

## Metrics
- **Primary**: `within1_accuracy` (percentage points, higher is better) — exact classifier matches plus predictions one grade away, macro-averaged over all 11 target grades.
- **Secondary**: `exact_accuracy` (must generally improve too), `valid_rate`, `unique_rate`, `val_iou`, `grade_correlation`, and `params_m`.
- A primary win that catastrophically harms validity, uniqueness, reconstruction, or exact accuracy must be discarded. Exact and within-one are both user goals.

## How to Run
`bash .auto/measure.sh` trains a fresh deterministic six-epoch development model and evaluates a fixed, balanced prior-sampling classifier check. It emits structured `METRIC` lines.

The six-epoch benchmark is a fast development proxy, not the locked product evaluation. Confirm substantial winners with a second seed or the standard longer training regime before finalization. Never inspect or use locked outer-test membership.

## Files in Scope
- `generator/src/vae.py` — conditional architecture and VAE objective.
- `generator/src/vae_trainer.py` — training objective and optimization.
- `generator/main.py` — configuration wiring if new generic options are needed.
- `generator/config.yaml` — product defaults after a robust improvement is established.
- `generator/src/checkpoint_compat.py` — compatibility handling for architecture changes.
- `generator/tests/` — tests for changed behavior.
- `.auto/benchmark.py`, `.auto/measure.sh`, `.auto/checks.sh` — benchmark/check harness; do not weaken or modify metrics to make an experiment pass.

## Off Limits
- `classifier/models/best_model.pth` and all classifier weights/code. (`classifier/test_models/best_model.pth` is a legacy 3-channel checkpoint rejected by the current CoordConv Predictor.)
- `data/problems.json`, result JSONs, model checkpoints, generated artifacts, and locked test cohorts.
- Classifier-check target labels, metric formulas, grade range, generation threshold, sample count, retry policy, seed, or benchmark epochs.
- Per-grade hand-authored output patterns, grade-dependent thresholds, benchmark-specific postprocessing, or direct optimization against the same frozen classifier used for scoring.

## Constraints
- Do not cheat or overfit to the benchmark. Structural conditioning improvements are preferred over narrow coefficient sweeps.
- Keep the frozen classifier strictly evaluation-only. An auxiliary loss against this same classifier would contaminate the benchmark and is not allowed in this session.
- Train from scratch for every run with the fixed development seed and split.
- Tests must pass. No new dependencies.
- Generated boards must remain diverse and valid; reconstruction should not materially collapse.
- Preserve checkpoint loading behavior or provide a clear compatibility error.

## What's Been Tried
- Previous reconstruction-focused session established the current sparse CVAE baseline: SiLU, positive BCE weight 4.25, BN momentum 0.2, posterior noise 0.75, asymmetric dropout, grade-embedding init scale 0.25, output bias -3.0.
- Grade embedding scale 0.5 was already worse for reconstruction; do not resume scalar tuning solely from that result because this target differs.
- Active plan `docs/exec-plans/active/generator-grade-conditioning.md` identifies FiLM decoder conditioning and conditioning/capacity rebalance as generic options. Its frozen-classifier auxiliary loss is explicitly excluded here because the same classifier defines the benchmark.
