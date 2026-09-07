# Classifier Experiments

This document records classifier experiment outcomes so future work can reference what worked, what did not, and what remains worth trying.

## Hold-Token Attention Architecture Comparison (2026-09-07)

**Finding:** this initial hold-token configuration did not improve the primary
metric. Within-one-grade accuracy decreased in all three paired validation runs,
and MAE increased in every pair. The mean exact-accuracy change was only +0.06
percentage points and was inconsistent across seeds. Keep the CNN as the
baseline.

### Hypothesis

Test whether representing a problem as a set of selected holds could learn
hold-specific properties and interactions between distant holds more effectively
than the CNN's progressively pooled grid. The attention model retains absolute
board coordinates and hold roles while making token order irrelevant. It uses
the same information as the CNN: learned hold embeddings change how position
identity is represented, without adding new problem metadata.

### Architecture And Protocol

- Baseline: existing CoordConv CNN, 392,019 parameters, trained afresh before
  comparing it with the new architecture.
- Candidate: small hold-token attention model, 85,779 parameters. Selected holds become
  tokens combining a learned position embedding, absolute coordinates, and three
  role flags. Two 64-dimensional attention blocks use four heads, 128-dimensional
  feed-forward layers, and dropout 0.1. A learned readout token and normalized
  log hold count feed the grade head. There is no sequence positional encoding,
  supplied beta sequence, foot-rule input, or relative-geometry feature.
- Manifest: `moonboard-masters-2017-all-v1`; fold 0 for each seed 42, 43, and 44.
  Each pair has identical train/validation IDs and disjoint layout groups. The
  development cohort has 38,076 problems; the locked test was not evaluated.
- Both architectures used the same training settings: Adam at 0.0003, batch size
  64, weight decay 0.001, class-weighted focal-ordinal loss, a 150-epoch cosine
  horizon, and early-stopping patience 8.
  Checkpoints are selected by validation within-one-grade accuracy, then lower
  cross-entropy loss, then earlier epoch. No architecture tuning was performed
  after observing candidate results.
- All six runs used the same CPU environment with four threads.
- This was an exploratory comparison of three paired validation runs, not the
  full 15-fit cross-validation protocol or a locked-test evaluation. The means
  below average three overlapping validation folds and do not establish
  statistical significance.

### Results

| Metric | CNN mean | Hold-token mean | Token minus CNN |
| --- | ---: | ---: | ---: |
| Exact accuracy | 37.4150% | 37.4763% | +0.0613 pp |
| Within one grade (primary) | 71.9977% | 70.8419% | -1.1558 pp |
| Within two grades | 88.7439% | 88.4418% | -0.3021 pp |
| Mean absolute error | 1.1076 | 1.1241 | +0.0165 grades |
| Macro accuracy | 19.6758% | 19.5066% | -0.1692 pp |

| Seed | Validation count | CNN exact | Token exact | CNN within one | Token within one |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 7,612 | 36.72% | 37.53% | 72.87% | 70.65% |
| 43 | 7,619 | 37.96% | 37.84% | 71.85% | 70.82% |
| 44 | 7,610 | 37.57% | 37.06% | 71.27% | 71.05% |

CNN selected epochs were 8, 7, and 15, with training stopping at 16, 15, and 23.
Hold-token selected epochs were 14, 20, and 12, with training stopping at 22, 28,
and 20. The smaller architecture saves about 78% of parameters, but this
experiment does not establish an accuracy improvement or an inference-speed gain.
The result applies to this architecture and training recipe, not every possible
hold-attention model.

### Interpretation Of The Effect

Within-one-grade accuracy fell by 0.22 to 2.22 percentage points across the three
pairs. Exact accuracy improved only for seed 42 and declined for seeds 43 and 44,
so the tiny positive mean does not demonstrate a consistent accuracy gain.
Higher MAE in every pair also means the candidate's predictions were farther
from the recorded grade on average. The reduction in model size was the clear
benefit measured here; prediction quality did not justify replacing the CNN.

This was a comparison of two complete architectures under the same training
recipe. Capacity, pooling, normalization, and architecture-specific dropout also
differ, so it does not isolate the effect of the attention operation alone.
Different attention designs or training settings remain untested. The practical
decision from this experiment was to retain the CNN baseline.

## Ordinal Smoothing

Ordinal smoothing trains the classifier with a soft target distribution instead of a one-hot grade label. Because Font grades are ordered, the correct class still gets most of the target probability, while nearby grades receive a small amount of probability mass. This tells the model that predicting a neighboring grade is less wrong than predicting a distant grade.

Tested kernel: `[0.025, 0.075, 0.8, 0.075, 0.025]`.

| Metric | Ordinal Smoothing | Focal Ordinal Reference |
| --- | ---: | ---: |
| Exact Accuracy | 35.93% | 38.96% |
| Macro Accuracy | 32.01% | 31.96% |
| +-1 Grade Accuracy | 72.03% | 72.89% |
| +-2 Grade Accuracy | 87.78% | 89.74% |
| Loss | 1.6411 | 1.5842 |

Finding: ordinal smoothing with this kernel did not improve over the focal ordinal reference. It slightly improved macro accuracy by 0.05 percentage points, but exact accuracy, +-1 accuracy, +-2 accuracy, and loss were all worse. Prefer focal ordinal over this ordinal smoothing setup unless a future run changes the kernel, combines it with another loss, or shows stronger validation/test behavior.

## Excluding Problems Without Repeats

This run excluded problems without repeats from the classifier experiment set. The comparison run included problems without repeats, so the comparison includes both loss-function and dataset-filter differences.

| Metric | Exclude Problems Without Repeats | Include Problems Without Repeats |
| --- | ---: | ---: |
| Exact Accuracy | 41.18% | 38.96% |
| Macro Accuracy | 32.64% | 31.96% |
| +-1 Grade Accuracy | 75.59% | 72.89% |
| +-2 Grade Accuracy | 92.36% | 89.74% |
| Loss | 1.4800 | 1.5842 |

Finding: excluding problems without repeats produced stronger test-set results than the earlier run that included those problems. Because the comparison run used a different dataset condition, treat this as evidence that the no-repeat filter is promising rather than as a clean isolated gain.
