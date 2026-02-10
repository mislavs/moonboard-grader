"""
Checkpoint compatibility helpers for model loading.
"""

from pathlib import Path
from typing import Mapping, Optional

import torch
from torch import nn


def _linear_in_features_from_state(
    state_dict: Mapping[str, torch.Tensor], key: str
) -> Optional[int]:
    weight = state_dict.get(key)
    if not isinstance(weight, torch.Tensor):
        return None
    if weight.ndim != 2:
        return None
    return int(weight.shape[1])


def _legacy_encoder_conditioning_message(
    checkpoint_path: Optional[str],
    expected_in_features: int,
    found_in_features: int,
) -> str:
    checkpoint_hint = (
        f" at '{Path(checkpoint_path)}'" if checkpoint_path else ""
    )
    return (
        "Checkpoint compatibility error"
        f"{checkpoint_hint}: this checkpoint appears to use the legacy CVAE "
        "architecture without encoder grade conditioning. "
        f"Current model expects `fc_mu`/`fc_logvar` input width {expected_in_features}, "
        f"but checkpoint provides {found_in_features}. "
        "Load a checkpoint trained with the current architecture or retrain."
    )


def _legacy_decoder_architecture_message(
    checkpoint_path: Optional[str],
) -> str:
    checkpoint_hint = (
        f" at '{Path(checkpoint_path)}'" if checkpoint_path else ""
    )
    return (
        "Checkpoint compatibility error"
        f"{checkpoint_hint}: this checkpoint appears to use the legacy CVAE "
        "decoder architecture that upsamples to 22x14 and applies interpolation "
        "to reach 18x11. "
        "Current model expects the native 18x11 decoder without interpolation. "
        "Load a checkpoint trained with the current architecture or retrain."
    )


def _detect_legacy_encoder_conditioning_mismatch(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> Optional[tuple[int, int]]:
    if not hasattr(model, "fc_mu") or not hasattr(model, "fc_logvar"):
        return None
    if not hasattr(model, "encoder_output_size") or not hasattr(
        model, "grade_embedding_dim"
    ):
        return None

    fc_mu = getattr(model, "fc_mu")
    fc_logvar = getattr(model, "fc_logvar")
    if not isinstance(fc_mu, nn.Linear) or not isinstance(fc_logvar, nn.Linear):
        return None

    expected_in = int(fc_mu.in_features)
    mu_in = _linear_in_features_from_state(state_dict, "fc_mu.weight")
    logvar_in = _linear_in_features_from_state(state_dict, "fc_logvar.weight")
    if mu_in is None or logvar_in is None:
        return None

    legacy_in = int(getattr(model, "encoder_output_size"))
    conditioned_expected = legacy_in + int(getattr(model, "grade_embedding_dim"))

    if (
        expected_in == conditioned_expected
        and mu_in == legacy_in
        and logvar_in == legacy_in
    ):
        return expected_in, legacy_in
    return None


def _is_legacy_decoder_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> bool:
    if "output_adjust.weight" in state_dict or "output_adjust.bias" in state_dict:
        return True

    decoder0 = state_dict.get("decoder.0.weight")
    if not isinstance(decoder0, torch.Tensor) or decoder0.ndim != 4:
        return False

    model_decoder = getattr(model, "decoder", None)
    if not isinstance(model_decoder, nn.Sequential) or len(model_decoder) == 0:
        return False

    first_layer = model_decoder[0]
    if not isinstance(first_layer, nn.ConvTranspose2d):
        return False

    expected_kernel = tuple(first_layer.kernel_size)
    found_kernel = tuple(decoder0.shape[2:])

    return expected_kernel == (4, 4) and found_kernel == (3, 3)


def load_state_dict_with_compatibility(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Load a model state_dict with explicit handling for known legacy incompatibilities.
    """
    if _is_legacy_decoder_checkpoint(model, state_dict):
        raise RuntimeError(_legacy_decoder_architecture_message(checkpoint_path))

    mismatch = _detect_legacy_encoder_conditioning_mismatch(model, state_dict)
    if mismatch is not None:
        expected_in, found_in = mismatch
        raise RuntimeError(
            _legacy_encoder_conditioning_message(
                checkpoint_path, expected_in, found_in
            )
        )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        if _is_legacy_decoder_checkpoint(model, state_dict):
            raise RuntimeError(
                _legacy_decoder_architecture_message(checkpoint_path)
            ) from exc

        mismatch = _detect_legacy_encoder_conditioning_mismatch(model, state_dict)
        if mismatch is not None:
            expected_in, found_in = mismatch
            raise RuntimeError(
                _legacy_encoder_conditioning_message(
                    checkpoint_path, expected_in, found_in
                )
            ) from exc
        raise
