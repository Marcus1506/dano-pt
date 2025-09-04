import torch
from torch import Tensor
import einops
from torch_geometric.data import Data

def flatten_time(x: Tensor) -> Tensor:
    return einops.rearrange(
        x,
        "n_particles n_time n_dim -> n_particles (n_time n_dim)"
    )

def unflatten_time(x: Tensor, n_time: int, n_dim: int) -> Tensor:
    return einops.rearrange(
        x,
        "n_particles (n_time n_dim) -> n_particles n_time n_dim",
        n_time=n_time,
        n_dim=n_dim
    )

def update_position_cycle(
        old_position: Tensor,
        field_prediction: Tensor,
        type: str,
        n_redundant: int
    ) -> Tensor:
    """
    Takes in field prediction and old position and returns new position.
    Fields are either of shape (n, t, n_dim) or (n, n_jumps, t, n_dim).
    """
    if n_redundant != 0:
        field_preds = field_prediction[..., :-n_redundant, :]
    if field_preds.ndim > 2:
        field_preds = torch.sum(field_preds, dim=-2)
    else:
        field_preds = field_preds.squeeze(1)
    if type in ["velocity", "displacement"]:
        new_position = old_position + field_preds
    elif type == "acceleration":
        raise NotImplementedError
    return new_position
