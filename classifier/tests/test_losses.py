"""
Unit tests for custom loss functions.
"""

import pytest
import torch
import torch.nn.functional as F

from src.losses import (
    FocalLoss,
    FocalOrdinalLoss,
    OrdinalCrossEntropyLoss,
    OrdinalSmoothingCrossEntropy,
    create_loss_function,
)


def test_ordinal_loss_alpha_changes_value():
    """Changing alpha should change ordinal loss for identical logits/targets."""
    logits = torch.tensor([
        [2.0, 1.0, -0.5, -1.0, -2.0],
        [-1.5, -0.5, 0.0, 1.5, 2.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 4], dtype=torch.long)

    loss_alpha_0 = OrdinalCrossEntropyLoss(num_classes=5, alpha=0.0)(logits, targets)
    loss_alpha_2 = OrdinalCrossEntropyLoss(num_classes=5, alpha=2.0)(logits, targets)

    assert loss_alpha_2 > loss_alpha_0


def test_ordinal_loss_is_stable_differentiable_and_alpha_changes_gradients():
    """Ordinal loss should produce finite values/gradients and alpha-sensitive gradients."""
    targets = torch.tensor([1, 3], dtype=torch.long)

    logits_alpha_0 = torch.tensor([
        [0.5, 1.2, -0.8, -1.0, -2.0],
        [-1.0, -0.3, 0.2, 1.1, 0.4],
    ], dtype=torch.float32, requires_grad=True)
    loss_alpha_0 = OrdinalCrossEntropyLoss(num_classes=5, alpha=0.0)(logits_alpha_0, targets)
    loss_alpha_0.backward()
    grads_alpha_0 = logits_alpha_0.grad.detach().clone()

    logits_alpha_3 = logits_alpha_0.detach().clone().requires_grad_(True)
    loss_alpha_3 = OrdinalCrossEntropyLoss(num_classes=5, alpha=3.0)(logits_alpha_3, targets)
    loss_alpha_3.backward()
    grads_alpha_3 = logits_alpha_3.grad.detach().clone()

    assert torch.isfinite(loss_alpha_0)
    assert torch.isfinite(loss_alpha_3)
    assert torch.isfinite(grads_alpha_0).all()
    assert torch.isfinite(grads_alpha_3).all()
    assert not torch.allclose(grads_alpha_0, grads_alpha_3)


def test_ordinal_loss_penalizes_distant_errors_more_than_near_errors():
    """With the same confidence profile, far misclassification should incur more loss."""
    targets = torch.tensor([2], dtype=torch.long)
    criterion = OrdinalCrossEntropyLoss(num_classes=5, alpha=2.0)

    # Same logits values, but high-confidence wrong class is near vs far from target=2.
    near_error_logits = torch.tensor([[0.0, 5.0, -1.0, -4.0, -4.0]], dtype=torch.float32)
    far_error_logits = torch.tensor([[0.0, -4.0, -1.0, -4.0, 5.0]], dtype=torch.float32)

    near_loss = criterion(near_error_logits, targets)
    far_loss = criterion(far_error_logits, targets)

    assert far_loss > near_loss


def test_focal_loss_matches_cross_entropy_when_gamma_is_zero():
    """Non-ordinal losses should keep existing behavior."""
    logits = torch.tensor([
        [1.0, 0.0, -0.5],
        [-0.2, 0.1, 1.4],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 2], dtype=torch.long)

    focal = FocalLoss(gamma=0.0)(logits, targets)
    ce = F.cross_entropy(logits, targets)

    assert torch.allclose(focal, ce, atol=1e-6, rtol=1e-6)


def test_focal_ordinal_loss_changes_with_ordinal_alpha():
    """ordinal_alpha should affect combined focal+ordinal loss on same inputs."""
    logits = torch.tensor([
        [0.8, 1.1, -0.4, -1.2, -1.8],
        [-1.4, -0.3, 0.6, 1.3, 0.2],
    ], dtype=torch.float32)
    targets = torch.tensor([1, 3], dtype=torch.long)

    loss_alpha_0 = FocalOrdinalLoss(
        num_classes=5,
        gamma=2.0,
        ordinal_weight=0.7,
        ordinal_alpha=0.0,
    )(logits, targets)
    loss_alpha_3 = FocalOrdinalLoss(
        num_classes=5,
        gamma=2.0,
        ordinal_weight=0.7,
        ordinal_alpha=3.0,
    )(logits, targets)

    assert loss_alpha_3 > loss_alpha_0


def test_focal_ordinal_loss_ordinal_weight_controls_ordinal_component():
    """ordinal_weight=0 should match focal-only, and non-zero should differ."""
    logits = torch.tensor([
        [0.8, 1.1, -0.4, -1.2, -1.8],
        [-1.4, -0.3, 0.6, 1.3, 0.2],
    ], dtype=torch.float32)
    targets = torch.tensor([1, 3], dtype=torch.long)

    focal_only = FocalLoss(gamma=2.0)(logits, targets)
    combined_weight_0 = FocalOrdinalLoss(
        num_classes=5,
        gamma=2.0,
        ordinal_weight=0.0,
        ordinal_alpha=2.0,
    )(logits, targets)
    combined_weight_1 = FocalOrdinalLoss(
        num_classes=5,
        gamma=2.0,
        ordinal_weight=1.0,
        ordinal_alpha=2.0,
    )(logits, targets)

    assert torch.allclose(combined_weight_0, focal_only, atol=1e-6, rtol=1e-6)
    assert combined_weight_1 > combined_weight_0


def test_ordinal_smoothing_builds_centered_neighbor_targets():
    """Interior classes should receive the configured ordinal smoothing kernel."""
    criterion = OrdinalSmoothingCrossEntropy(
        kernel=[0.025, 0.075, 0.8, 0.075, 0.025]
    )

    soft_targets = criterion.build_soft_targets(
        targets=torch.tensor([2], dtype=torch.long),
        num_classes=5
    )

    expected = torch.tensor([[0.025, 0.075, 0.8, 0.075, 0.025]])
    assert torch.allclose(soft_targets, expected, atol=1e-6, rtol=1e-6)


def test_ordinal_smoothing_renormalizes_edge_targets():
    """Boundary classes should drop invalid neighbors and renormalize remaining mass."""
    criterion = OrdinalSmoothingCrossEntropy(
        kernel=[0.025, 0.075, 0.8, 0.075, 0.025]
    )

    soft_targets = criterion.build_soft_targets(
        targets=torch.tensor([0], dtype=torch.long),
        num_classes=5
    )

    expected = torch.tensor([[0.8, 0.075, 0.025, 0.0, 0.0]]) / 0.9
    assert torch.allclose(soft_targets, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(soft_targets.sum(dim=1), torch.tensor([1.0]))


def test_ordinal_smoothing_is_stable_and_differentiable():
    """Ordinal smoothing should produce finite values and gradients."""
    logits = torch.tensor([
        [0.8, 1.1, -0.4, -1.2, -1.8],
        [-1.4, -0.3, 0.6, 1.3, 0.2],
    ], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([1, 3], dtype=torch.long)

    loss = OrdinalSmoothingCrossEntropy()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_ordinal_smoothing_class_weights_change_loss():
    """Per-class weights should affect soft-target cross-entropy terms."""
    logits = torch.tensor([
        [0.8, 1.1, -0.4, -1.2, -1.8],
        [-1.4, -0.3, 0.6, 1.3, 0.2],
    ], dtype=torch.float32)
    targets = torch.tensor([1, 3], dtype=torch.long)

    unweighted = OrdinalSmoothingCrossEntropy()(logits, targets)
    weighted = OrdinalSmoothingCrossEntropy(
        class_weights=torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0])
    )(logits, targets)

    assert not torch.allclose(weighted, unweighted)


def test_ordinal_smoothing_rejects_invalid_kernels():
    """Kernels must be odd-length, non-negative, and have positive mass."""
    invalid_kernels = [
        [0.5, 0.5],
        [0.0, 0.0, 0.0],
        [0.5, -0.1, 0.6],
    ]

    for kernel in invalid_kernels:
        with pytest.raises(ValueError):
            OrdinalSmoothingCrossEntropy(kernel=kernel)


def test_create_loss_function_supports_ordinal_smoothing():
    """The loss factory should expose ordinal smoothing as a selectable loss type."""
    criterion = create_loss_function(
        loss_type='ordinal_smoothing',
        num_classes=5,
        class_weights=torch.ones(5),
        ordinal_smoothing_kernel=[0.025, 0.075, 0.8, 0.075, 0.025],
    )

    assert isinstance(criterion, OrdinalSmoothingCrossEntropy)


def test_create_loss_function_defaults_to_focal_ordinal():
    """The default loss factory choice should match the recommended config default."""
    criterion = create_loss_function(num_classes=5)

    assert isinstance(criterion, FocalOrdinalLoss)
