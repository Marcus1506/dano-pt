"""This is a subset of the original torch_geometric.utils.metric from PyG 2.0.4.

PyG removed metrics in favor of torchmetrics, but we still need the PyG batching.
See: https://github.com/pyg-team/pytorch_geometric/discussions/7434
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean


def intersection_and_union(
    pred: Tensor, target: Tensor, num_classes: int, batch: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    r"""Computes intersection and union of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: (LongTensor, LongTensor)
    """
    pred, target = F.one_hot(pred, num_classes), F.one_hot(target, num_classes)

    if batch is None:
        i = (pred & target).sum(dim=0)
        u = (pred | target).sum(dim=0)
    else:
        i = scatter_add(pred & target, batch, dim=0)
        u = scatter_add(pred | target, batch, dim=0)

    return i, u


def mean_iou(
    pred: Tensor,
    target: Tensor,
    num_classes: int,
    batch: Optional[Tensor] = None,
    omitnans: bool = False,
) -> Tensor:
    r"""Computes the mean intersection over union score of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.
        omitnans (bool, optional): If set to True, will ignore any
            NaN values encountered during computation. Otherwise, will
            treat them as 1. (default: False)

    :rtype: Tensor
    """
    i, u = intersection_and_union(pred, target, num_classes, batch)
    iou = i.to(torch.float) / u.to(torch.float)

    if omitnans:
        iou = iou[~iou.isnan()].mean()
    else:
        iou[torch.isnan(iou)] = 1.0
        iou = iou.mean(dim=-1)

    return iou


def my_mse(pred, target, batch):
    """Compute MSE loss for each pytorch geometric batch.
    Args:
        pred: (n_nodes, num_features)
        target: (n_nodes, num_features)
        batch: (n_nodes,) integers denoting the batch index of each node

    Returns:
        loss: (num_batches,) tensor with MSE loss for each batch
    """
    loss = ((pred - target) ** 2).mean(-1)
    return scatter_mean(loss, batch, dim=0)

def calc_correlation(
        preds: torch.Tensor,
        target: torch.Tensor
        ) -> torch.Tensor:
    assert preds.shape == target.shape
    assert target.ndim == 3

    all_corr = []
    for window_idx in range(preds.shape[1]):
        preds_t = preds[:, window_idx]
        target_t = target[:, window_idx]
        preds_mean = torch.mean(preds_t, dim=0, keepdim=True)
        target_mean = torch.mean(target_t, dim=0, keepdim=True)
        preds_std = torch.std(preds_t, dim=0, unbiased=False)
        target_std = torch.std(target_t, dim=0, unbiased=False)
        # calculate mean correlation per timestep
        mean_corr_per_timestep = (
        torch.mean((preds_t - preds_mean) * (target_t - target_mean), dim=0)
        / (preds_std * target_std).clamp(min=1e-12)
        ).mean(dim=0)
        all_corr.append(mean_corr_per_timestep.item())
    return torch.tensor(all_corr)

def calc_mean_iou(
        self,
        rollouts: torch.Tensor,
        ground_truths: torch.Tensor,
        first_steps: int,
        n_compartments: int = 150,
        bounding_box: tuple | None = None,
    ) -> float:
    mious = torch.tensor(0.0)
    for rollout, ground_truth in zip(rollouts, ground_truths):
        miou = self.mean_iou(
            rollout=rollout[:first_steps],
            ground_truth=ground_truth[:first_steps],
            n_compartments=n_compartments,
            bounding_box=bounding_box,
        )
        mious += miou
    miou = mious / len(rollouts)
    return miou.item()

def mean_iou(
        rollout: torch.Tensor,
        ground_truth: torch.Tensor,
        n_compartments: int,
        bounding_box: tuple | None = None,
    ) -> torch.Tensor:
    """Compute the mean Intersection‑over‑Union (IoU) between two particle-rollout
    trajectories on a uniform 2‑D grid.

    Parameters
    ----------
    rollout : torch.Tensor
        Tensor of shape ``(T, N, 2)`` containing the *predicted* particle positions
        for *T* frames and *N* particles.
    ground_truth : torch.Tensor
        Tensor of shape ``(T, N, 2)`` containing the *ground‑truth* particle
        positions.
    n_compartments : int
        Number of equal‑sized grid cells along *each* spatial axis
        (total cells = ``n_compartments ** 2``).
    bounding_box : tuple[float, float, float, float] | None, optional
        Axis‑aligned bounding box ``(a_x, b_x, a_y, b_y)``. If *None*, the
        smallest bounding box that encloses **both** ``rollout`` and
        ``ground_truth`` across *all* frames is used.

    Returns
    -------
    torch.Tensor
        Scalar tensor – the mean IoU over all *T* frames.
    """

    if rollout.shape != ground_truth.shape:
        raise ValueError("rollout and ground_truth must share the same shape (T, N, 2).")
    if rollout.ndim != 3 or rollout.size(-1) != 2:
        raise ValueError("Input tensors must have shape (T, N, 2).")

    T, N, _ = rollout.shape
    device = rollout.device

    # ─────────────────────── Determine the bounding box ────────────────────────
    if bounding_box is None:
        combined = torch.cat((rollout, ground_truth), dim=1)  # (T, 2N, 2)
        min_xy = combined.amin(dim=(0, 1))  # (2,)
        max_xy = combined.amax(dim=(0, 1))  # (2,)
        a_x, a_y = min_xy.tolist()
        b_x, b_y = max_xy.tolist()
    else:
        a_x, b_x, a_y, b_y = bounding_box

    step_x = (b_x - a_x) / n_compartments
    step_y = (b_y - a_y) / n_compartments

    if step_x <= 0 or step_y <= 0:
        raise ValueError("Invalid bounding box – non‑positive side length.")

    # ────────────────── Helper: map coords → linear cell indices ───────────────
    def _to_indices(coords: torch.Tensor) -> torch.Tensor:
        """Map positions ``(T, N, 2)`` to linear grid indices ``(T, N)``."""
        x = ((coords[..., 0] - a_x) / step_x).floor().clamp_(0, n_compartments - 1).long()
        y = ((coords[..., 1] - a_y) / step_y).floor().clamp_(0, n_compartments - 1).long()
        return y * n_compartments + x  # row‑major order

    roll_idx = _to_indices(rollout)      # (T, N)
    gt_idx = _to_indices(ground_truth)   # (T, N)

    n_cells = n_compartments ** 2
    roll_mask = torch.zeros((T, n_cells), dtype=torch.bool, device=device)
    gt_mask = torch.zeros_like(roll_mask)

    # Vectorised scatter to mark occupied cells
    t_idxs = torch.arange(T, device=device).unsqueeze(1).expand(T, N)  # (T, N)
    roll_mask[t_idxs.reshape(-1), roll_idx.reshape(-1)] = True
    gt_mask[t_idxs.reshape(-1), gt_idx.reshape(-1)] = True

    # ───────────────────────────── IoU per frame ──────────────────────────────
    intersection = (roll_mask & gt_mask).sum(dim=1).float()  # (T,)
    union = (roll_mask | gt_mask).sum(dim=1).float().clamp_(min=1.0)

    return (intersection / union).mean()
