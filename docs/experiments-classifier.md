# Classifier Experiments

This document records classifier experiment outcomes so future work can reference what worked, what did not, and what remains worth trying.

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
