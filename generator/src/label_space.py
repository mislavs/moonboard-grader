"""
Shared grade label-space utilities for generator training/inference/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from moonboard_core import decode_grade, get_num_grades, remap_label, unmap_label


def _safe_decode_grade(label: int) -> str:
    try:
        return decode_grade(label)
    except Exception:
        return str(label)


def infer_num_model_grades(checkpoint: Mapping[str, Any]) -> int:
    """
    Infer model class count from checkpoint contents.
    """
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, dict) and "num_grades" in model_config:
        return int(model_config["num_grades"])

    model_state = checkpoint.get("model_state_dict", {})
    if isinstance(model_state, dict) and "grade_embedding.weight" in model_state:
        return int(model_state["grade_embedding.weight"].shape[0])

    min_idx = checkpoint.get("min_grade_index")
    max_idx = checkpoint.get("max_grade_index")
    if min_idx is not None and max_idx is not None:
        return int(max_idx) - int(min_idx) + 1

    return get_num_grades()


def _infer_grade_offset(
    checkpoint: Mapping[str, Any],
    num_model_grades: int,
) -> int:
    """
    Infer grade offset with backward compatibility.
    """
    explicit_mode = checkpoint.get("label_space_mode")
    if explicit_mode == "global_legacy":
        return 0

    min_idx = checkpoint.get("min_grade_index")
    max_idx = checkpoint.get("max_grade_index")
    if min_idx is not None and max_idx is not None:
        expected = int(max_idx) - int(min_idx) + 1
        if expected == num_model_grades:
            return int(min_idx)

    return 0


@dataclass(frozen=True)
class EvaluationLabelContext:
    """
    Grade label mapping context loaded from checkpoint metadata.
    """

    grade_offset: int
    min_grade_index: Optional[int]
    max_grade_index: Optional[int]
    num_model_grades: int

    def __post_init__(self) -> None:
        if self.num_model_grades < 1:
            raise ValueError("num_model_grades must be >= 1")
        if self.grade_offset < 0:
            raise ValueError("grade_offset must be >= 0")
        if self.min_grade_index is None and self.max_grade_index is not None:
            raise ValueError("max_grade_index requires min_grade_index")
        if self.max_grade_index is None and self.min_grade_index is not None:
            raise ValueError("min_grade_index requires max_grade_index")
        if self.min_grade_index is not None and self.max_grade_index is not None:
            if self.min_grade_index < 0:
                raise ValueError("min_grade_index must be >= 0")
            if self.max_grade_index < self.min_grade_index:
                raise ValueError("max_grade_index must be >= min_grade_index")
            if self.max_grade_index >= get_num_grades():
                raise ValueError(
                    f"max_grade_index must be < {get_num_grades()}, got {self.max_grade_index}"
                )
            if self.grade_offset not in (0, self.min_grade_index):
                raise ValueError(
                    "grade_offset must be 0 (global) or min_grade_index (remapped) "
                    f"when explicit range is set, got {self.grade_offset}"
                )
            if self.grade_offset == self.min_grade_index:
                expected_range = self.max_grade_index - self.min_grade_index + 1
                if expected_range != self.num_model_grades:
                    raise ValueError(
                        "Remapped label space requires num_model_grades to match range size: "
                        f"{self.num_model_grades} vs {expected_range}"
                    )
            return

        max_global = self.grade_offset + self.num_model_grades - 1
        if max_global >= get_num_grades():
            raise ValueError(
                "Label space exceeds valid global grade bounds: "
                f"max_global={max_global}, max_allowed={get_num_grades() - 1}"
            )

    @property
    def label_space_mode(self) -> str:
        """
        Backward-compatible mode view derived from offset semantics.
        """
        return "remapped" if self.grade_offset > 0 else "global_legacy"

    @property
    def has_explicit_range(self) -> bool:
        return self.min_grade_index is not None and self.max_grade_index is not None

    def get_global_grade_bounds(self) -> Tuple[int, int]:
        if self.has_explicit_range:
            return self.min_grade_index, self.max_grade_index  # type: ignore[return-value]

        if self.grade_offset > 0:
            min_idx = self.grade_offset
            max_idx = min_idx + self.num_model_grades - 1
            return min_idx, max_idx

        return 0, self.num_model_grades - 1

    def get_global_grade_indices(self) -> List[int]:
        min_idx, max_idx = self.get_global_grade_bounds()
        return list(range(min_idx, max_idx + 1))

    def supports_global_grade(self, global_label: int) -> bool:
        min_idx, max_idx = self.get_global_grade_bounds()
        return min_idx <= global_label <= max_idx

    def _validate_model_label(self, model_label: int) -> None:
        if not 0 <= model_label < self.num_model_grades:
            raise ValueError(
                f"Model label {model_label} is out of range [0, {self.num_model_grades - 1}]"
            )

    def global_to_model_label(self, global_label: int) -> int:
        min_idx, max_idx = self.get_global_grade_bounds()
        if not min_idx <= global_label <= max_idx:
            raise ValueError(
                f"Global label {global_label} ({_safe_decode_grade(global_label)}) is out of checkpoint range "
                f"[{min_idx} ({_safe_decode_grade(min_idx)}), {max_idx} ({_safe_decode_grade(max_idx)})]"
            )

        model_label = remap_label(global_label, self.grade_offset)
        self._validate_model_label(model_label)
        return model_label

    def model_to_global_label(self, model_label: int) -> int:
        self._validate_model_label(model_label)
        return unmap_label(model_label, self.grade_offset)

    def model_label_to_grade_name(self, model_label: int) -> str:
        return decode_grade(self.model_to_global_label(model_label))


def build_label_context(
    checkpoint: Mapping[str, Any],
    num_model_grades: Optional[int] = None,
) -> EvaluationLabelContext:
    """
    Build a label context from checkpoint metadata with compatibility inference.
    """
    resolved_num_grades = (
        int(num_model_grades)
        if num_model_grades is not None
        else infer_num_model_grades(checkpoint)
    )
    inferred_offset = _infer_grade_offset(checkpoint, resolved_num_grades)
    explicit_mode = checkpoint.get("label_space_mode")

    min_idx_raw = checkpoint.get("min_grade_index")
    max_idx_raw = checkpoint.get("max_grade_index")
    min_idx = int(min_idx_raw) if min_idx_raw is not None else None
    max_idx = int(max_idx_raw) if max_idx_raw is not None else None

    default_offset = inferred_offset

    grade_offset = int(checkpoint.get("grade_offset", default_offset))
    if inferred_offset > 0 and explicit_mode is None and min_idx is not None:
        # Legacy inferred-remapped checkpoints should anchor offset to min index.
        grade_offset = min_idx

    # Legacy checkpoints may store odd offsets despite global labels; keep global mapping stable.
    if inferred_offset == 0:
        grade_offset = 0

    return EvaluationLabelContext(
        grade_offset=grade_offset,
        min_grade_index=min_idx,
        max_grade_index=max_idx,
        num_model_grades=resolved_num_grades,
    )


def label_context_to_metadata(context: EvaluationLabelContext) -> Dict[str, Any]:
    """
    Convert a label context to checkpoint metadata fields.
    """
    return {
        "label_space_mode": context.label_space_mode,
        "grade_offset": context.grade_offset,
        "min_grade_index": context.min_grade_index,
        "max_grade_index": context.max_grade_index,
    }
