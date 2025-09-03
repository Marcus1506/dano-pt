import json
from pathlib import Path

import numpy as np
import torch
from numpy import array
from torch import Tensor


def random_rotation_matrix():
    """Generate a random 3x3 rotation matrix using PyTorch."""
    theta = 2 * torch.pi * torch.rand(1)  # Random rotation around the z-axis
    phi = torch.acos(2 * torch.rand(1) - 1)  # Random rotation around the y-axis
    psi = 2 * torch.pi * torch.rand(1)  # Random rotation around the x-axis

    Rz = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    Ry = torch.tensor(
        [[torch.cos(phi), 0, torch.sin(phi)], [0, 1, 0], [-torch.sin(phi), 0, torch.cos(phi)]]
    )
    Rx = torch.tensor(
        [[1, 0, 0], [0, torch.cos(psi), -torch.sin(psi)], [0, torch.sin(psi), torch.cos(psi)]]
    )
    R = torch.mm(Rz, torch.mm(Ry, Rx))  # Combined rotation matrix
    return R


def rotate_point_cloud(tensor: Tensor, R: Tensor) -> Tensor:
    """Rotate a tensor using a rotation matrix."""
    center = torch.mean(tensor, dim=0)
    tensor = tensor - center
    tensor = torch.mm(tensor, R.T)
    tensor = tensor + center
    return tensor


def load_metadata(path: Path):
    # Load metadata
    with open(path / "metadata.json") as f:
        metadata = json.loads(f.read())
    return metadata


def pos_init_cartesian_2d(box_size: array, dx: float):
    """Create a grid of particles in 2D.

    Particles are at the center of the corresponding Cartesian grid cells.
    Example: if box_size=np.array([1, 1]) and dx=0.1, then the first particle will be at
    position [0.05, 0.05].
    """
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")
    r = (np.vstack(list(map(np.ravel, grid))).T + 0.5) * dx
    return r


def pos_init_cartesian_3d(box_size: array, dx: float):
    """Create a grid of particles in 3D."""
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), range(n[2]), indexing="xy")
    r = (np.vstack(list(map(np.ravel, grid))).T + 0.5) * dx
    return r


class QuinticKernel:
    """The quintic kernel function of Morris."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h

        self._normalized_cutoff = 3.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 2:
            self._sigma = 7.0 / 478.0 / torch.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 3.0 / 359.0 / torch.pi * self._one_over_h**3

    def w(self, r):
        q = r * self._one_over_h
        zeros = torch.zeros_like(q)
        q1 = torch.maximum(zeros, 1.0 - q)
        q2 = torch.maximum(zeros, 2.0 - q)
        q3 = torch.maximum(zeros, 3.0 - q)

        return self._sigma * (q3**5 - 6.0 * q2**5 + 15.0 * q1**5)
