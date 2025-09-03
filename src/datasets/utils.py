from itertools import islice
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
from typing import Optional

from src.datasets.particle_datamodule import ParticleDataset

def get_subset_sclice(length: int, idx: int) -> slice:
    """
    Get a slice object for a subset of the dataset starting from given index.
    """
    left_positions = length - (idx % length)
    return slice(idx, idx + left_positions, 1)

def get_subset_iters(
        particle_ds: ParticleDataset,
        traj_idx: list[int],
        idx: list[list[int]],
        follow_batch: list[str] | None,
        stride: int = 1,
        fixed_length: Optional[int] = None
    ) -> list[iter]:
    """
    Get an iterator over a subset of the dataset indicating trajectories by `traj_idx` and starting indices of iterators inside those trajectories by `idx`.
    This iterator does yields unshuffled batches of data.
    """
    assert stride >= 1, "Stride must be at least 1."
    subset_iters = []
    for traj_i, idx_list in zip(traj_idx, idx):
        for i in idx_list:
            assert i <= particle_ds.n_per_traj, f"Index {i} exceeds the number of particles per trajectory {particle_ds.n_per_traj}."
            beginning_idx = i + traj_i * particle_ds.n_per_traj
            if fixed_length is None:
                slice_i = get_subset_sclice(particle_ds.n_per_traj, beginning_idx)
            else:
                slice_i = slice(beginning_idx, beginning_idx + fixed_length)
            loader = DataLoader(
                particle_ds.index_select(slice_i),
                batch_size=1,
                shuffle=False,
                follow_batch=follow_batch
            )
            subset_iters.append(islice(iter(loader), 0, None, stride))
    return subset_iters
