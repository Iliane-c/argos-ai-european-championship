"""Differentiable segmentation losses aligned with dual-class per-image mIoU."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDualIoULoss(nn.Module):
    """Differentiable surrogate directly matching the official mean(IoU_0, IoU_1) per image."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        g = target.float()
        inter_1 = (p * g).sum(dim=(-2, -1))
        union_1 = (p + g - p * g).sum(dim=(-2, -1))
        iou_1 = (inter_1 + self.eps) / (union_1 + self.eps)

        p0 = 1.0 - p
        g0 = 1.0 - g
        inter_0 = (p0 * g0).sum(dim=(-2, -1))
        union_0 = (p0 + g0 - p0 * g0).sum(dim=(-2, -1))
        iou_0 = (inter_0 + self.eps) / (union_0 + self.eps)
        return 1.0 - 0.5 * (iou_0 + iou_1).mean()


class SoftDiceLoss(nn.Module):
    """Per-image dual-class soft Dice loss."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        g = target.float()
        dice_1 = (2.0 * (p * g).sum(dim=(-2, -1)) + self.eps) / (p.sum(dim=(-2, -1)) + g.sum(dim=(-2, -1)) + self.eps)
        p0, g0 = 1.0 - p, 1.0 - g
        dice_0 = (2.0 * (p0 * g0).sum(dim=(-2, -1)) + self.eps) / (p0.sum(dim=(-2, -1)) + g0.sum(dim=(-2, -1)) + self.eps)
        return 1.0 - 0.5 * (dice_0 + dice_1).mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1.0)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    if labels_flat.numel() == 0:
        return logits_flat.sum() * 0.0
    signs = 2.0 * labels_flat.float() - 1.0
    errors = 1.0 - logits_flat * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels_flat[perm.data]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


class LovaszHingeLoss(nn.Module):
    """Per-image dual-class Lovasz-Hinge loss optimizing both watermarked and clean Jaccard."""

    def __init__(self, dual_class: bool = True):
        super().__init__()
        self.dual_class = dual_class

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bsz = logits.shape[0]
        losses = []
        for i in range(bsz):
            lf = logits[i].reshape(-1)
            tf = (target[i] > 0.5).float().reshape(-1)
            loss_i = _lovasz_hinge_flat(lf, tf)
            if self.dual_class:
                loss_i = 0.5 * (loss_i + _lovasz_hinge_flat(-lf, 1.0 - tf))
            losses.append(loss_i)
        return torch.stack(losses).mean()


class BoundaryLoss(nn.Module):
    """BCE weighted toward mask transition edges (morphological boundary cells)."""

    def __init__(self, edge_weight: float = 3.0):
        super().__init__()
        self.edge_weight = edge_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t4 = target.float().unsqueeze(1)
        dil = F.max_pool2d(t4, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-t4, kernel_size=3, stride=1, padding=1)
        edge = (dil - ero).squeeze(1).clamp(0.0, 1.0)
        weights = 1.0 + self.edge_weight * edge
        bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        return (bce * weights).mean()


class CompositeLoss(nn.Module):
    """Configurable weighted sum of mask losses and auxiliary kind / bit / residual heads."""

    def __init__(self, cfg_loss: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg_loss or {}
        self.w_bce = float(cfg.get("bce", 0.5))
        self.w_dice = float(cfg.get("dice", 0.5))
        self.w_dual_iou = float(cfg.get("dual_iou", 1.0))
        self.w_lovasz = float(cfg.get("lovasz", 0.0))
        self.w_boundary = float(cfg.get("boundary", 0.0))
        self.w_kind = float(cfg.get("aux_kind", 0.1))
        self.w_bits = float(cfg.get("aux_bits", 0.2))
        self.w_residual = float(cfg.get("aux_residual", 0.0))

        self.dual_iou = SoftDualIoULoss()
        self.dice = SoftDiceLoss()
        self.lovasz = LovaszHingeLoss(dual_class=True)
        self.boundary = BoundaryLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"]
        target = batch["mask128"].float()
        parts: dict[str, float] = {}
        total = logits.new_zeros(())

        if self.w_bce > 0.0:
            l_bce = F.binary_cross_entropy_with_logits(logits, target)
            total = total + self.w_bce * l_bce
            parts["bce"] = float(l_bce.detach())

        if self.w_dice > 0.0:
            l_dice = self.dice(logits, target)
            total = total + self.w_dice * l_dice
            parts["dice"] = float(l_dice.detach())

        if self.w_dual_iou > 0.0:
            l_iou = self.dual_iou(logits, target)
            total = total + self.w_dual_iou * l_iou
            parts["dual_iou"] = float(l_iou.detach())

        if self.w_lovasz > 0.0:
            l_lov = self.lovasz(logits, target)
            total = total + self.w_lovasz * l_lov
            parts["lovasz"] = float(l_lov.detach())

        if self.w_boundary > 0.0:
            l_bnd = self.boundary(logits, target)
            total = total + self.w_boundary * l_bnd
            parts["boundary"] = float(l_bnd.detach())

        if self.w_kind > 0.0 and "kind_logits" in outputs and "kind" in batch:
            valid = batch["kind"] >= 0
            if valid.any():
                l_kind = F.binary_cross_entropy_with_logits(
                    outputs["kind_logits"][valid], batch["kind"][valid].float()
                )
                total = total + self.w_kind * l_kind
                parts["aux_kind"] = float(l_kind.detach())

        if self.w_bits > 0.0 and "bit_logits" in outputs and "message_bits" in batch:
            bit_logits = outputs["bit_logits"]  # [B, 256, H, W]
            _, _, bh, bw = bit_logits.shape
            wm_mask = F.adaptive_avg_pool2d(target.unsqueeze(1), (bh, bw)) > 0.5  # [B, 1, H, W]
            msg_valid = batch["message_mask"][:, :, None, None]  # [B, 256, 1, 1]
            weight = wm_mask.float() * msg_valid
            denom = weight.sum().clamp(min=1.0)
            msg_tgt = batch["message_bits"][:, :, None, None].expand_as(bit_logits)
            bce_bits = F.binary_cross_entropy_with_logits(bit_logits, msg_tgt, reduction="none")
            l_bits = (bce_bits * weight).sum() / denom
            total = total + self.w_bits * l_bits
            parts["aux_bits"] = float(l_bits.detach())

        if self.w_residual > 0.0 and "residual" in outputs and "residual" in batch:
            l_res = F.l1_loss(outputs["residual"], batch["residual"])
            total = total + self.w_residual * l_res
            parts["aux_residual"] = float(l_res.detach())

        parts["total"] = float(total.detach())
        return total, parts
