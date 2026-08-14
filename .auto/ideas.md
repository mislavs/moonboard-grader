# Deferred ideas

- Confirm meaningful winners with a second fresh seed and, before finalization, the standard 70-epoch configuration plus official `classifier_check` evaluation.
- Explore an ordinal/smooth grade embedding parameterization if FiLM alone does not make adjacent labels behave monotonically.
- Consider a classifier auxiliary loss only with a separate non-scoring classifier or independent evaluation model; never optimize directly against the same frozen classifier used by this session's primary metric.
