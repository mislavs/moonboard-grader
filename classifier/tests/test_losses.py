"""
Unit tests for custom loss functions.
"""

import torch
import torch.nn.functional as F

from src.losses import FocalLoss, OrdinalCrossEntropyLoss


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
