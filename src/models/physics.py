import os
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import einops
import lightning as L
import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data

from src.utils.logging_utils import load_run_config_from_wb
from src.utils.metric import calc_correlation
from src.datasets.particle_datamodule import ParticleDataset, ParticleDataModule
from src.models.utils import (
    unflatten_time,
    flatten_time,
    update_position_cycle,
    rollout_iteration
)
from src.datasets.utils import get_subset_iters

class PhysicsLitModule(L.LightningModule):
    def __init__(
        self,
        latent_model: torch.nn.Module,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mode: str = "physics",
        model_type: str = "velocity",
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        first_stage_model_ckpt: str = None,
        first_stage_model_config: str = None,
        first_stage_model_wandb_entity: str = None,
        first_stage_model_wandb_project: str = None,
        first_stage_model_wandb_id: str = None,
        num_classes: int = 5,
        propagation: str = "normal"
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["latent_model", "loss_function"])
        self.latent_model: torch.nn.Module = latent_model()
        self.loss_function: torch.Module = loss_function

        self.initialize_first_stage_model()

        if mode == "physics":
            self.forward = self.forward_physics
            self.validation_step = self.validation_physics
        elif mode == "particle":
            self.forward = self.forward_particle
            self.validation_step = self.validation_particle

        self.propagation = propagation

    def initialize_first_stage_model(self):
        if self.hparams.first_stage_model_wandb_id is not None:
            cfg = load_run_config_from_wb(
                entity=self.hparams.first_stage_model_wandb_entity,
                project=self.hparams.first_stage_model_wandb_project,
                run_id=self.hparams.first_stage_model_wandb_id,
            )
        elif self.hparams.first_stage_model_config is not None:
            cfg = OmegaConf.load(self.hparams.first_stage_model_config)
        else:
            raise ValueError(
                "Please provide a value for first_stage_model_config or first_stage_model_wandb_id."
            )
        self.first_stage_model = instantiate(cfg.model)

        if self.hparams.first_stage_model_ckpt:
            ckpt_path = (
                self.hparams.first_stage_model_ckpt
                if self.hparams.first_stage_model_ckpt is not None
                else os.path.join(cfg.callbacks.model_checkpoint.dirpath, "last.ckpt")
            )
            checkpoint = torch.load(
                ckpt_path,
                map_location=self.first_stage_model.device,
                weights_only=False
            )
            self.first_stage_model.load_state_dict(checkpoint["state_dict"])

        self.first_stage_model.eval()
        self.first_stage_model.freeze()

    def forward_physics(self, batch: Data) -> Tensor:
        timestep = batch.input_timestep
        pred_latent = self.encode(
            enc_pos=batch.input_enc_pos,
            enc_field=batch.input_enc_field,
            enc_particle_type=batch.input_enc_particle_type,
            enc_pos_batch_index=batch.batch,
            supernode_index=batch.supernode_index,
            supernode_batch_index=batch.supernode_index_batch,
            timestep=timestep,
        )
        n_targets = batch.target_field.shape[1]
        loss = 0.0
        for jump_idx in range(n_targets):
            if self.propagation == "trick":
                pred_latent = pred_latent.detach()
            pred_latent = self.push_forward(
                latent=pred_latent,
                timestep=timestep
            )
            target_latent = self.encode(
                enc_pos=batch.target_pos[:, jump_idx],
                enc_field=batch.target_field[:, jump_idx],
                enc_particle_type=batch.target_particle_type,
                enc_pos_batch_index=batch.batch,
                supernode_index=batch.supernode_index,
                supernode_batch_index=batch.supernode_index_batch,
                timestep=batch.target_timestep[:, jump_idx],
            )
            # Update timestep for latent
            timestep = batch.target_timestep[:, jump_idx]
            # Calculate loss
            loss = loss + self.loss_function(pred_latent, target_latent)
        return loss

    def model_step(self, batch):
        loss = self.forward(batch=batch)
        return {"loss": loss}

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        loss = self.model_step(batch)
        # Log metrics
        batch_size = batch.batch.max().item() + 1
        self.log("train/loss", loss["loss"], prog_bar=True, batch_size=batch_size)
        return loss

    def validation_physics(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        input_data, target_data = batch
        timestep = input_data.input_timestep
        pred_latent = self.encode(
            enc_pos=input_data.input_enc_pos,
            enc_field=input_data.input_enc_field,
            enc_particle_type=input_data.input_enc_particle_type,
            enc_pos_batch_index=input_data.batch,
            supernode_index=input_data.supernode_index,
            supernode_batch_index=input_data.supernode_index_batch,
            timestep=input_data.input_timestep,
        )
        n_targets = target_data.target_field.shape[1]
        field_mse = 0.0
        for jump_idx in range(n_targets):
            pred_latent = self.push_forward(
                latent=pred_latent,
                timestep=timestep
            )
            timestep = target_data.target_timestep[:, jump_idx]
            # Decode
            preds_field = self.decode(
                latent=pred_latent,
                dec_field_pos=target_data.target_pos[:, jump_idx],
                timestep=timestep,
            )
            target_field = flatten_time(target_data.target_field[:, jump_idx])
            field_mse += nn.functional.mse_loss(preds_field, target_field)
        batch_size = input_data.batch.max().item() + 1
        self.log(
            "val/loss",
            field_mse,
            prog_bar=True,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
    
    @torch.no_grad()
    def rollout_physics(
            self,
            particle_dm: ParticleDataModule,
            traj_idx: list[int],
            idx: list[list[int]],
            query_gt_pos: bool = False,
            split: str = "test",
            use_gt_field: bool = False,
            fixed_length: Optional[int] = None
        ) -> tuple[
            Tensor | list[Tensor],
            Tensor | list[Tensor],
            Tensor | list[Tensor],
            Tensor | list[Tensor],
            Tensor | list[Tensor],
            Tensor | list[Tensor],
            Tensor | list[Tensor]
            ]:
        dataset = particle_dm.get_dataset(split=split)
        model_type = dataset.type
        if model_type == "acceleration":
            assert not use_gt_field, "use_gt_field not supported for acceleration models"
            assert not query_gt_pos, "query_gt_pos not supported for acceleration models"
        assert dataset.rollout
        unnormalize = dataset.unnormalize
        normalize = dataset.normalize
        subset_iters = get_subset_iters(
            dataset,
            traj_idx,
            idx,
            particle_dm.follow_batch,
            dataset.n_jump_ahead_timesteps,
            fixed_length,
        )
        device = next(self.parameters()).device

        GT_vel_fields_normalized = []
        GT_positions = []        
        rollouts = []
        fields = []
        MSE_Fields_normalized = []
        MSE_Fields_latent = []
        propagated_latents = []

        for subset_iter in subset_iters:
            GT_vel_field_normalized = []
            rollout = []
            field = []
            GT_pos = []
            MSE_Field_normalized = []
            MSE_Field_latent = []
            propagated_latent = []
            new_pos = None

            for i, batch in enumerate(subset_iter):
                input_data, target_data = batch
                input_data = input_data.to(device)
                target_data = target_data.to(device)
                
                if i == 0: # bootstrap iteration
                    n, n_time, n_dim = input_data.input_enc_field.shape

                    if model_type in ["velocity", "acceleration"]:
                        n_redundant = n_time - dataset.n_jump_ahead_timesteps
                    elif model_type == "displacement": n_redundant = 0

                    # we solve jumping in rollout by striding the dataloader:
                    jump_idx = 0

                    pred_latent = self.encode(
                        enc_pos=input_data.input_enc_pos,
                        enc_field=input_data.input_enc_field,
                        enc_particle_type=input_data.input_enc_particle_type,
                        enc_pos_batch_index=input_data.batch,
                        supernode_index=input_data.supernode_index,
                        supernode_batch_index=input_data.supernode_index_batch,
                        timestep=input_data.input_timestep,
                    )
                    old_gt_latent = pred_latent # keep GT latent from previous step
                    timestep = input_data.input_timestep
                    pos = target_data.target_pos[:, jump_idx]
                    if model_type == "acceleration":
                        pos = torch.stack((pos, target_data.second_pos), dim=1)

                    _rollout_iteration = partial(
                        rollout_iteration,
                        model=self,
                        model_type=model_type,
                        use_gt_field=use_gt_field,
                        query_gt_pos=query_gt_pos,
                        n_redundant=n_redundant,
                        unnormalize=unnormalize,
                        jump_idx=jump_idx,
                        n_time=n_time,
                        n_dim=n_dim,
                    )
                else:
                    outputs, state = _rollout_iteration(
                        input_data=input_data,
                        target_data=target_data,
                        pos=pos,
                        timestep=timestep,
                        pred_latent=pred_latent,
                        old_gt_latent=old_gt_latent,
                    )

                    rollout.append(outputs["rollout_step"])
                    field.append(outputs["field_step"])
                    GT_pos.append(outputs["GT_pos_step"])
                    GT_vel_field_normalized.append(
                        outputs["GT_vel_field_normalized_step"]
                    )
                    MSE_Field_normalized.append(outputs["MSE_Field_normalized_step"])
                    MSE_Field_latent.append(outputs["MSE_Field_latent_step"])
                    propagated_latent.append(outputs["propagated_latent_step"])

                    pos = state["pos_next"]
                    timestep = state["timestep_next"]
                    pred_latent = state["pred_latent_next"]
                    old_gt_latent = state["old_gt_latent_next"]

            # Grab last rollout step and GT explicitly
            if use_gt_field:
                pred_latent = old_gt_latent
            pred_latent = self.push_forward(
                latent=pred_latent,
                timestep=timestep
            )
            timestep = target_data.target_timestep[:, jump_idx]
            decode_pos = pos[:, 0] if model_type == "acceleration" else pos
            preds_field = self.decode(
                latent=pred_latent,
                dec_field_pos=decode_pos,
                timestep=timestep,
            )
            preds_field = unflatten_time(preds_field, n_time, n_dim)
            preds_field_unnorm = unnormalize(preds_field)
            pos = update_position_cycle(
                old_position=pos,
                field_prediction=preds_field_unnorm,
                type=model_type,
                n_redundant=n_redundant
            )
            last_pos = pos[:, 1] if model_type == "acceleration" else pos
            rollout.append(last_pos.cpu())
            GT_pos.append(target_data.last_pos.cpu())

            rollouts.append(torch.stack(rollout))
            fields.append(torch.stack(field))
            GT_positions.append(torch.stack(GT_pos))
            GT_vel_fields_normalized.append(
                torch.stack(GT_vel_field_normalized)
            )
            MSE_Fields_normalized.append(torch.stack(MSE_Field_normalized))
            MSE_Fields_latent.append(torch.stack(MSE_Field_latent))
            propagated_latents.append(torch.stack(propagated_latent))

        if len(subset_iters) == 1:
            rollouts = rollouts[0]
            fields = fields[0]
            GT_positions = GT_positions[0]
            GT_vel_fields_normalized = GT_vel_fields_normalized[0]
            MSE_Fields_normalized = MSE_Fields_normalized[0]
            MSE_Fields_latent = MSE_Fields_latent[0]
            propagated_latents = propagated_latents[0]
        
        return (
            rollouts,
            fields,
            GT_positions,
            GT_vel_fields_normalized,
            MSE_Fields_normalized,
            MSE_Fields_latent,
            propagated_latents
        )

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)
        # Freeze first stage model
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def configure_optimizers(self) -> Dict[str, Any]:
        # Exclude weight decay from normalization and bias parameters
        exclude = lambda n: "bias" in n or "norm" in n
        include = lambda n: not exclude(n)
        named_parameters = list(self.latent_model.named_parameters())
        norm_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
        optimizer = self.hparams.optimizer(
            [
                {"params": norm_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.hparams.optimizer.keywords["weight_decay"],
                },
            ]
        )
        if self.hparams.scheduler is not None:
            if (type(self.hparams.scheduler) is partial) and (
                self.hparams.scheduler.func.__name__ == "LinearWarmupCosineAnnealingLR"
            ):
                interval = "step"
            else:
                interval = "epoch"
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def encode(
        self,
        enc_pos: Tensor,
        enc_field: Tensor,
        enc_particle_type: Tensor,
        enc_pos_batch_index: Tensor,
        supernode_index: Tensor,
        supernode_batch_index: Tensor,
        timestep: Tensor,
    ) -> Dict:
        return self.first_stage_model.model.encode(
            enc_pos=enc_pos,
            enc_field=enc_field,
            enc_particle_type=enc_particle_type,
            enc_pos_batch_index=enc_pos_batch_index,
            supernode_index=supernode_index,
            supernode_batch_index=supernode_batch_index,
            timestep=timestep,
        )

    def decode(
        self,
        latent: Tensor,
        dec_field_pos: Tensor,
        timestep: Tensor,
    ):
        return self.first_stage_model.model.decode(
            latent=latent,
            dec_field_pos=dec_field_pos,
            timestep=timestep,
        )

    def condition(self, timestep: Tensor):
        if self.first_stage_model.model.conditioner is not None:
            condition = self.first_stage_model.model.conditioner(timestep)
        else:
            condition = None
        return condition

    def push_forward(
            self,
            latent: Tensor,
            timestep: Tensor,
        ) -> Tensor:
        condition = self.condition(timestep)
        next_latent = self.latent_model(latent, condition)
        return next_latent
