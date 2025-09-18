import torch
from torch import Tensor
import einops
from typing import Dict, Any, Tuple

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
    if type in ["velocity", "displacement"]:
        if n_redundant != 0:
            field_prediction = field_prediction[..., :-n_redundant, :]
        if field_prediction.ndim > 2:
            field_prediction = torch.sum(field_prediction, dim=-2)
        else:
            field_prediction = field_prediction.squeeze(1)
        if type in ["velocity", "displacement"]:
            new_position = old_position + field_prediction
    elif type == "acceleration":
        assert n_redundant == 0, "n_redundant not 0 not supported for acceleration models"
        assert old_position.shape[1] == 2, "Acceleration needs two positions"
        new_vel_start = old_position[:, 1, :] - old_position[:, 0, :]
        new_vels = new_vel_start.unsqueeze(1) + field_prediction
        new_vels_sum = torch.sum(new_vels, dim=-2)
        new_position_1 = old_position[..., 0, :] + new_vels_sum
        new_position_0 = old_position[..., 0, :] + new_vels_sum - new_vels[..., -1, :]
        new_position = torch.stack([new_position_0, new_position_1], dim=-2)
    else:
        raise ValueError(f"Unknown model type '{type}'")
    return new_position

def rollout_iteration(
        model,
        input_data,
        target_data,
        jump_idx: int,
        pos: torch.Tensor,
        timestep: torch.Tensor,
        pred_latent: torch.Tensor,
        old_gt_latent: torch.Tensor,
        use_gt_field: bool,
        query_gt_pos: bool,
        n_time: int,
        n_dim: int,
        model_type: str,
        n_redundant: int,
        unnormalize,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    One rollout step after bootstrapping returns:
      - outputs: dict with per-step artifacts (to append to your lists)
      - state:   dict with updated state for the next iteration
    """

    # Optionally replace pred_latent with ground-truth latent for this step and refresh old_gt_latent
    if use_gt_field:
        pred_latent = old_gt_latent
        old_gt_latent = model.encode(
            enc_pos=input_data.input_enc_pos,
            enc_field=input_data.input_enc_field,
            enc_particle_type=input_data.input_enc_particle_type,
            enc_pos_batch_index=input_data.batch,
            supernode_index=input_data.supernode_index,
            supernode_batch_index=input_data.supernode_index_batch,
            timestep=input_data.input_timestep,
        )

    # Push forward with current timestep (before we overwrite it with input timestep)
    pred_latent = model.push_forward(
        latent=pred_latent,
        timestep=timestep
    )

    # Update timestep to the input timestep for decoding/targets
    timestep = input_data.input_timestep

    # Latent target + latent loss
    target_latent = model.encode(
        enc_pos=input_data.input_enc_pos,
        enc_field=input_data.input_enc_field,
        enc_particle_type=input_data.input_enc_particle_type,
        enc_pos_batch_index=input_data.batch,
        supernode_index=input_data.supernode_index,
        supernode_batch_index=input_data.supernode_index_batch,
        timestep=input_data.input_timestep,
    )
    latent_loss = model.loss_function(pred_latent, target_latent)

    # Decode to current pos and to the "correct" enc pos (used for field MSE)
    decode_pos = pos[:, 0] if model_type == "acceleration" else pos
    preds_field = model.decode(
        latent=pred_latent,
        dec_field_pos=decode_pos,
        timestep=timestep,
    )
    preds_field_correct_pos = model.decode(
        latent=pred_latent,
        dec_field_pos=input_data.input_enc_pos,
        timestep=timestep,
    )

    # Field MSE in normalized space (your current behavior)
    target_field = flatten_time(input_data.input_enc_field)
    field_mse = torch.nn.functional.mse_loss(preds_field_correct_pos, target_field)

    # Prepare field prediction for position update (unflatten -> unnormalize)
    preds_field = unflatten_time(preds_field, n_time, n_dim)
    preds_field_unnorm = unnormalize(preds_field)

    # Position update
    new_pos = update_position_cycle(
        old_position=pos,
        field_prediction=preds_field_unnorm,
        type=model_type,
        n_redundant=n_redundant
    )
    pos_next = target_data.target_pos[:, jump_idx] if query_gt_pos else new_pos
    if model_type == "acceleration":
        new_pos = new_pos[..., -1, :]
    
    # Build per-step outputs
    outputs = {
        "rollout_step": new_pos.detach().cpu(),
        "field_step": preds_field_unnorm.detach().cpu(),
        "GT_pos_step": target_data.target_pos[:, jump_idx].detach().cpu(),
        "GT_vel_field_normalized_step": target_data.target_field[:, jump_idx].detach().cpu(),
        "MSE_Field_normalized_step": field_mse.detach().cpu(),
        "MSE_Field_latent_step": latent_loss.detach().cpu(),
        "propagated_latent_step": pred_latent.detach().cpu(),
    }

    # Updated state for the next loop iteration
    state = {
        "pos_next": pos_next,
        "timestep_next": timestep,
        "pred_latent_next": pred_latent,
        "old_gt_latent_next": old_gt_latent,
    }

    return outputs, state

