from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import einops
import lightning as L
import numpy as np
import torch
from torch import Tensor, nn
from torchmetrics.segmentation import MeanIoU
from torch_geometric.loader import DataLoader

from src.datasets.particle_datamodule import ParticleDataset, ParticleDataModule
from src.datasets.utils import get_subset_iters
from src.models.utils import update_position_cycle, flatten_time, unflatten_time

class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        conditioner: nn.Module,
        initialize_weights: Optional[Callable] = None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if conditioner is not None:
            self.conditioner = conditioner
        else:
            self.conditioner = None
        self.num_classes = (
            num_classes - 1
        )  # "no particle" class is not encoded in the encoder because inputs are always valid particles
        self.initialize_weights = initialize_weights

        if self.initialize_weights is not None:
            self.apply(self.initialize_weights)

    def forward(
        self,
        enc_pos: Tensor,
        enc_field: Tensor,
        enc_particle_type: Tensor,
        dec_pos: Tensor,
        enc_pos_batch_index: Tensor,
        supernode_index: Tensor,
        supernode_batch_index: Tensor,
        timestep: Tensor,
    ) -> Dict:
        latent = self.encode(
            enc_pos=enc_pos,
            enc_field=enc_field,
            enc_particle_type=enc_particle_type,
            enc_pos_batch_index=enc_pos_batch_index,
            supernode_index=supernode_index,
            supernode_batch_index=supernode_batch_index,
            timestep=timestep,
        )
        preds_field = self.decode(
            latent=latent,
            dec_field_pos=dec_pos,
            timestep=timestep,
        )
        return {"preds_field": preds_field, "latent": latent}

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
        # Stack particle types to field variable
        field = self.prepare_field(field=enc_field, particle_type=enc_particle_type)
        if self.conditioner is not None:
            condition = self.conditioner(timestep)
        else:
            condition = None
        return self.encoder(
            field=field,
            pos=enc_pos,
            batch_index=enc_pos_batch_index,
            supernode_index=supernode_index,
            supernode_batch_index=supernode_batch_index,
            condition=condition,
        )

    def decode(
        self,
        latent: Tensor,
        dec_field_pos: Tensor,
        timestep: Tensor,
    ):
        if self.conditioner is not None:
            condition = self.conditioner(timestep)
        else:
            condition = None
        return self.decoder(x=latent, pos=dec_field_pos, condition=condition)

    def prepare_field(
        self,
        field: Tensor,
        particle_type: Tensor,
    ):
        # Flatten past field values
        field = einops.rearrange(
            field,
            "n_particles n_timesteps n_dim -> n_particles (n_timesteps n_dim)",
        )
        # Add particle type to input features if there are multiple particle types
        if self.num_classes == 1:
            pass
        else:
            field = torch.concat(
                (
                    field,
                    nn.functional.one_hot(particle_type, num_classes=self.num_classes),
                ),
                dim=-1,
            )
        return field

class AutoencoderLitModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "loss_function"])
        self.model: torch.nn.Module = model(num_classes=num_classes)
        self.loss_function: torch.Module = loss_function

        self.mean_iou_train = MeanIoU(num_classes=self.hparams.num_classes)
        self.mean_iou_val = MeanIoU(num_classes=self.hparams.num_classes)

    def forward(
        self,
        enc_pos: Tensor,
        enc_field: Tensor,
        enc_particle_type: Tensor,
        dec_pos: Tensor,
        enc_pos_batch_index,
        supernode_index: Tensor,
        supernode_batch_index: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        return self.model(
            enc_pos=enc_pos,
            enc_field=enc_field,
            enc_particle_type=enc_particle_type,
            dec_pos=dec_pos,
            enc_pos_batch_index=enc_pos_batch_index,
            supernode_index=supernode_index,
            supernode_batch_index=supernode_batch_index,
            timestep=timestep,
        )

    def model_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        preds = self.forward(
            enc_pos=batch.enc_pos,
            enc_field=batch.enc_field,
            enc_particle_type=batch.enc_particle_type,
            dec_pos=batch.dec_pos,
            enc_pos_batch_index=batch.enc_pos_batch,
            supernode_index=batch.supernode_index,
            supernode_batch_index=batch.supernode_index_batch,
            timestep=batch.timestep,
        )
        preds_field = preds["preds_field"]
        target_field = flatten_time(batch.dec_field)
        loss = self.loss_function(
            input=preds_field,
            target=target_field,
        )
        return loss, preds

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        loss, preds = self.model_step(batch)
        # Log metrics
        batch_size = batch.batch.max().item() + 1
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        loss, preds = self.model_step(batch)
        batch_size = batch.batch.max().item() + 1
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        # Exclude weight decay from normalization and bias parameters
        exclude = lambda n: "bias" in n or "norm" in n
        include = lambda n: not exclude(n)
        named_parameters = list(self.model.named_parameters())
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

    @torch.no_grad()
    def GT_encode_decode(
            self,
            particle_dm: ParticleDataModule,
            traj_idx: list[int],
            idx: list[list[int]],
            query_gt_pos: bool = False,
            split: str = "test",
            fixed_length: Optional[int] = None
        ) -> tuple[
            Tensor | list[Tensor],
            Tensor | list[Tensor],
            Tensor | list[Tensor],
            Tensor | list[Tensor]
            ]:
        dataset = particle_dm.get_dataset(split)
        unnormalize = dataset.unnormalize
        assert dataset.rollout
        assert dataset.type == "velocity" # NOTE: For now
        device = next(self.parameters()).device

        subset_iters = get_subset_iters(
            dataset,
            traj_idx,
            idx,
            particle_dm.follow_batch,
            1, # Fixed stride for autoencoder
            fixed_length
        )
        n_redundant = dataset.n_fields - 1 # always use one new velocity

        GT_vel_fields_normalized = []
        GT_positions = []
        rollouts = []
        MSE_Fields_normalized = []
        latents = []

        for subset_iter in subset_iters:
            new_pos = None
            GT_vel = []
            GT_pos = []
            rollout = []
            MSE_Field_normalized = []
            latents_traj = []
            for batch in subset_iter:
                n, n_time, n_dim = batch.enc_field.shape
                batch = batch.to(device)
                pos = batch.dec_pos if (query_gt_pos or (new_pos is None)) else new_pos

                preds = self.forward(
                    enc_pos=batch.enc_pos,
                    enc_field=batch.enc_field,
                    enc_particle_type=batch.enc_particle_type,
                    dec_pos=pos,
                    enc_pos_batch_index=batch.enc_pos_batch,
                    supernode_index=batch.supernode_index,
                    supernode_batch_index=batch.supernode_index_batch,
                    timestep=batch.timestep,
                )
                retrieved_field = preds["preds_field"]
                latents_traj.append(preds["latent"])

                loss = self.loss_function(
                    input=retrieved_field,
                    target=flatten_time(batch.dec_field),
                )
                retrieved_field = unflatten_time(retrieved_field, n_time=n_time, n_dim=n_dim)

                new_pos = update_position_cycle(
                    old_position=pos,
                    field_prediction=unnormalize(retrieved_field),
                    type="velocity",
                    n_redundant=n_redundant
                )
                rollout.append(new_pos.cpu())
                #
                GT_vel.append(batch.enc_field.cpu())
                GT_pos.append(batch.enc_pos.cpu())
                MSE_Field_normalized.append(loss.cpu())
                #
            GT_vel_fields_normalized.append(torch.stack(GT_vel))
            GT_positions.append(torch.stack(GT_pos))
            rollouts.append(torch.stack(rollout))
            MSE_Fields_normalized.append(torch.stack(MSE_Field_normalized))
            latents.append(torch.stack(latents_traj))
        #
        if len(subset_iters) == 1:
            GT_vel_fields_normalized = GT_vel_fields_normalized[0]
            GT_positions = GT_positions[0]
            rollouts = rollouts[0]
            MSE_Fields_normalized = MSE_Fields_normalized[0]
            latents = latents[0]
        #
        return (
            rollouts,
            GT_positions,
            GT_vel_fields_normalized,
            MSE_Fields_normalized,
            latents
        )
