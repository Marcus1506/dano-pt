import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import einops
import h5pickle
import lightning as L
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.spatial
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import nearest, radius, radius_graph
from torch_geometric.transforms import KNNGraph
from torch_scatter import scatter_sum


@dataclass
class ParticleDataset(InMemoryDataset):
    n_fields: int = 2
    n_jump_ahead_timesteps: int = 1
    n_jumps: int = 1
    type: str = "velocity"
    split: str = "train"
    mode: str = "train_autoencoder"
    n_particle_types: int = None
    n_supernodes: int = None
    num_points_range: int = None
    global_root: str = None
    local_root: str = None
    seed: int = None
    num_points_decode: int = None
    dataset_rel_path: str = "multi_material"
    overfit_single_trajectory: bool = False,
    rollout: bool = False

    def __post_init__(self):
        super().__init__()
        assert self.split in [
            "train",
            "valid",
            "test",
        ], f"Split {self.split} not available."
        assert self.n_fields > 0, "n_fields must be greater than 0."
        assert self.mode in [
            "train_autoencoder",
            "train_physics",
            "val_physics",
            "particle_autoencoder",
            "particle"
        ]
        assert self.n_particle_types > 0
        if self.local_root is None:
            self.source_root = Path(self.global_root) / self.dataset_rel_path
        else:
            # load data from local_root
            self.source_root = Path(self.local_root) / self.dataset_rel_path
            # If local data does not exist, try global root
            if not self.source_root.exists():
                self.source_root = Path(self.global_root) / self.dataset_rel_path
        assert self.source_root.exists(), f"'{self.source_root.as_posix()}' doesn't exist"

        if self.overfit_single_trajectory:
            self.split = "train"
        self.trajectories = self.load_dataset(self.source_root, self.split)
        self.traj_keys = list(self.trajectories.keys())
        if self.overfit_single_trajectory:
            self.n_traj = 1
        else:
            self.n_traj = len(self.traj_keys)

        self.metadata = self.load_metadata(self.source_root)
        self.bounds = torch.tensor(self.metadata["bounds"])
        self.ndim = self.metadata["dim"]

        # Normalization stats
        if self.type == "velocity":
            self.vel_mean = torch.tensor(self.metadata["vel_mean"])
            self.vel_std = torch.tensor(self.metadata["vel_std"])
        elif self.type == "displacement":
            self.vel_mean = torch.tensor(
                self.metadata[f"displcmnt_{self.n_fields}_mean"]
            )
            self.vel_std = torch.tensor(
                self.metadata[f"displcmnt_{self.n_fields}_std"]
            )
        self.acc_mean = torch.tensor(self.metadata["acc_mean"])
        self.acc_std = torch.tensor(self.metadata["acc_std"])

        self.box = self.bounds[:, 1] - self.bounds[:, 0]
        self.pos_offset = self.bounds[:, 0]

        if self.mode == "train_autoencoder":
            # Every sample in the trajectory can be used
            self.n_seq = (
                self.metadata["sequence_length_train"]
                if "sequence_length_train" in self.metadata
                else self.metadata["sequence_length"]
            )
            # n_seq + 1 is the number of positions in the trajectory
            self.n_per_traj = self.n_seq + 1 - self.n_fields
            if self.type in ["velocity", "displacement"]:
                length = self.n_fields + 1
            elif self.type == "acceleration":
                self.n_per_traj -= 1
                length = self.n_fields + 2
            self.pos_getter = partial(self.get_positions, length=length)
            self.field_getter = partial(self.get_fields, type=self.type)
        elif self.mode in ["train_physics", "val_physics"]:
            self.n_seq = (
                self.metadata["sequence_length_val"]
                if "sequence_length_val" in self.metadata
                else self.metadata["sequence_length"]
            )
            assert self.n_jump_ahead_timesteps <= self.n_fields, "Temporal overlap needed."
            if self.type in ["velocity", "displacement"]:
                self.n_per_traj = (
                    self.n_seq - self.n_jump_ahead_timesteps * self.n_jumps - (self.n_fields - 1)
                )
                self.pos_getter = partial(
                    self.get_positions,
                    length=self.n_jump_ahead_timesteps * self.n_jumps + self.n_fields + 1,
                )
                self.field_getter = partial(self.get_fields, type=self.type)
                self.target_field_idx = torch.tensor(
                    [
                        list(
                            range(
                                i * self.n_jump_ahead_timesteps,
                                i * self.n_jump_ahead_timesteps + self.n_fields,
                            )
                        )
                        for i in range(1, self.n_jumps + 1)
                    ]
                )
                self.target_pos_idx = torch.tensor(
                    [
                        i * self.n_jump_ahead_timesteps
                        for i in range(1, self.n_jumps + 1)
                    ]
                )
                if self.rollout: # n_jumps irrelevant for rollout
                    self.n_per_traj = (
                        self.n_seq - self.n_jump_ahead_timesteps * 1 - (self.n_fields - 1)
                    )
                    self.pos_getter = partial(
                        self.get_positions,
                        length=self.n_jump_ahead_timesteps * 1 + self.n_fields + 1,
                    )
                    self.field_getter = partial(self.get_fields, type=self.type)
                    self.target_field_idx = self.target_field_idx[:1]
                    self.target_pos_idx = self.target_pos_idx[:1]
            elif self.type == "acceleration":
                self.n_per_traj = (
                    self.n_seq - self.n_jump_ahead_timesteps * self.n_jumps - (self.n_fields - 1) - 1
                )
                self.pos_getter = partial(
                    self.get_positions,
                    length=self.n_jump_ahead_timesteps * self.n_jumps + self.n_fields + 2,
                )
                self.field_getter = partial(self.get_fields, type=self.type)
                self.target_field_idx = torch.tensor( # same as velocity
                    [
                        list(
                            range(
                                i * self.n_jump_ahead_timesteps,
                                i * self.n_jump_ahead_timesteps + self.n_fields,
                            )
                        )
                        for i in range(1, self.n_jumps + 1)
                    ]
                )
                self.target_pos_idx = torch.tensor(
                    [
                        i * self.n_jump_ahead_timesteps
                        for i in range(1, self.n_jumps + 1)
                    ]
                )
                if self.rollout: # ugly
                    self.n_per_traj = (
                        self.n_seq - self.n_jump_ahead_timesteps * 1 - (self.n_fields - 1) - 1
                    )
                    self.pos_getter = partial(
                        self.get_positions,
                        length=self.n_jump_ahead_timesteps * 1 + self.n_fields + 2,
                    )
                    self.field_getter = partial(self.get_fields, type=self.type)
                    self.target_field_idx = self.target_field_idx[:1]
                    self.target_pos_idx = self.target_pos_idx[:1]
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

    def len(self) -> int:
        return self.n_traj * self.n_per_traj

    def __getitem__(self, idx: int):  # TODO: Add return type
        idx = self.indices()[idx]
        if self.mode == "train_autoencoder":
            return self.getitem_autoencoder(idx)
        elif self.mode == "train_physics":
            return self.getitem_train_physics(idx)
        elif self.mode == "val_physics":
            return self.getitem_val_physics(idx)
        elif self.mode == "particle_autoencoder":
            return self.getitem_particle_autoencoder(idx)
        elif self.mode == "particle":
            return self.getitem_particle(idx)

    def get_fields(
            self,
            positions: Tensor,
            perm_input: Tensor,
            perm_target: Tensor,
            type: str
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Takes in positions of needed timesteps and returns tuple of
        input_field, input_position, target_fields and target_positions
        """
        velocities = positions[:, 1:, :] - positions[:, :-1, :]
        if type == "velocity":
            input_velocities = velocities[:, :self.n_fields]
            if self.mode == "train_autoencoder":
                target_velocities = input_velocities
            elif self.mode in ["train_physics", "val_physics"]:
                target_velocities = velocities[:, self.target_field_idx]
            input_velocities = input_velocities[perm_input]
            target_velocities = target_velocities[perm_target]
            input_velocities = self.normalize_vel(input_velocities)
            target_velocities = self.normalize_vel(target_velocities)

            input_positions = positions[perm_input, 0]
            if self.mode in ["train_physics", "val_physics"]:
                target_positions = positions[:, self.target_pos_idx]
            else:
                target_positions = positions[:, 0]
            target_positions = target_positions[perm_target]
            return input_velocities, input_positions, target_velocities, target_positions
        elif type == "displacement":
            input_velocities = velocities[:, :self.n_fields]
            if self.mode == "train_autoencoder":
                target_velocities = input_velocities.sum(dim=-2, keepdim=True)
            elif self.mode in ["train_physics", "val_physics"]:
                target_velocities = velocities[:, self.target_field_idx]
                target_velocities = target_velocities.sum(dim=-2, keepdim=True)
            input_velocities = input_velocities[perm_input].sum(dim=-2, keepdim=True)
            target_velocities = target_velocities[perm_target]
            input_velocities = self.normalize_vel(input_velocities)
            target_velocities = self.normalize_vel(target_velocities)

            input_positions = positions[perm_input, self.n_fields - 1]
            if self.mode in ["train_physics", "val_physics"]:
                target_positions = positions[:, self.target_pos_idx]
            else:
                target_positions = positions[:, self.n_fields - 1]
            target_positions = target_positions[perm_target]
            return input_velocities, input_positions, target_velocities, target_positions
        elif type == "acceleration":
            accels = velocities[:, 1:, :] - velocities[:, :-1, :]
            input_accelerations = accels[:, :self.n_fields]
            if self.mode == "train_autoencoder":
                target_accelerations = input_accelerations
            elif self.mode in ["train_physics", "val_physics"]:
                target_accelerations = accels[:, self.target_field_idx]
            input_accelerations = input_accelerations[perm_input]
            target_accelerations = target_accelerations[perm_target]
            input_accelerations = self.normalize_acc(input_accelerations)
            target_accelerations = self.normalize_acc(target_accelerations)

            input_positions = positions[perm_input, 0]
            if self.mode in ["train_physics", "val_physics"]:
                target_positions = positions[:, self.target_pos_idx]
            else:
                target_positions = positions[:, 0]
            target_positions = target_positions[perm_target]
            return input_accelerations, input_positions, target_accelerations, target_positions
        else:
            raise ValueError(f"Unknown type: {type}. Expected 'velocity' or 'acceleration'.")

    def getitem_autoencoder(self, idx: int) -> Data:
        position_dict = self.pos_getter(idx)
        position = position_dict["position"]
        particle_type = position_dict["particle_type"]
        if self.rollout:
            perm_enc = torch.arange(position.shape[0])
            perm_dec = perm_enc
            supernode_index = self.get_permutation(
                n_particles=position.shape[0],
                n_perm=self.n_supernodes
            )
        else:
            perm_enc = self.get_permutation(n_particles=position.shape[0])
            perm_dec = self.get_permutation(
                n_particles=position.shape[0],
                n_perm=self.num_points_decode
            )
            supernode_index = torch.arange(self.n_supernodes)
        input_fields, input_positions, target_fields, target_positions = self.field_getter(
            position, perm_enc, perm_dec
        )
        return Data(
            enc_pos=input_positions,
            enc_field=input_fields,
            enc_particle_type=particle_type[perm_enc],
            dec_pos=target_positions,
            dec_field=target_fields,
            supernode_index=supernode_index,
            supernode_idx=supernode_index,
            timestep=position_dict["time_idx"][0],
            num_nodes=len(perm_enc),
            idx=idx,
        )

    def getitem_train_physics(self, idx: int) -> Data:
        position_dict = self.pos_getter(idx)
        position = position_dict["position"]
        particle_type = position_dict["particle_type"]
        if self.rollout:
            perm_input = torch.arange(position.shape[0])
            perm_target = perm_input
            supernode_index = self.get_permutation(
                n_particles=position.shape[0],
                n_perm=self.n_supernodes
            )
        else:
            perm_input = self.get_permutation(n_particles=position.shape[0])
            perm_target = self.get_permutation(
                n_particles=position.shape[0],
                n_perm=len(perm_input)
            )
            supernode_index = torch.arange(self.n_supernodes)

        input_fields, input_position, target_fields, target_position = self.field_getter(
            position, perm_input, perm_target
        )

        target_timestep = position_dict["time_idx"][self.target_pos_idx].unsqueeze(0)

        return Data(
            input_enc_pos=input_position,
            input_enc_field=input_fields,
            input_enc_particle_type=particle_type[perm_input],
            supernode_index=supernode_index,  # Is the same for input and target
            input_timestep=position_dict["time_idx"][0],
            target_pos=target_position,
            target_field=target_fields,
            target_particle_type=particle_type[perm_target],
            target_timestep=target_timestep,
            num_nodes=len(input_position),
            idx=idx,
        )

    def getitem_val_physics(self, idx: int) -> Tuple[Data, Data]:
        position_dict = self.pos_getter(idx)
        position = position_dict["position"]
        particle_type = position_dict["particle_type"]
        
        if self.rollout:
            perm_input = torch.arange(position.shape[0])
            perm_target = perm_input
            supernode_index = self.get_permutation(
                n_particles=position.shape[0],
                n_perm=self.n_supernodes
            )
        else:
            perm_input = self.get_permutation(n_particles=position.shape[0])
            perm_target = self.get_permutation(
                n_particles=position.shape[0],
                n_perm=self.num_points_decode
            )
            supernode_index = torch.arange(self.n_supernodes)

        input_fields, input_positions, target_fields, target_positions = self.field_getter(
            position, perm_input, perm_target
        )

        input_data = Data(
            input_enc_pos=input_positions,
            input_enc_field=input_fields,
            input_enc_particle_type=particle_type[perm_input],
            supernode_index=supernode_index,
            input_timestep=position_dict["time_idx"][0],
            num_nodes=len(input_positions),
            idx=idx,
            second_pos=position[perm_target, 1] if self.type == "acceleration" else None # needed for acceleration rollout
        )

        target_timestep = position_dict["time_idx"][self.target_pos_idx].unsqueeze(0)
        
        target_data = Data(
            target_pos=target_positions,
            target_field=target_fields,
            supernode_index=supernode_index,
            target_timestep=target_timestep,
            num_nodes=len(perm_target),
            last_pos=position[perm_target, -1], # this is needed for last GT in rollout
            second_pos=position[perm_target, self.n_fields + 1] if self.type == "acceleration" else None # needed for acceleration rollout
        )
        return input_data, target_data

    def load_dataset(self, path: Path, split):
        # Load dataset
        data = h5pickle.File(str(path / (split + ".h5")))
        return data

    def load_metadata(self, path: Path):
        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.loads(f.read())
        return metadata

    def get_positions(self, idx: int, length: int = 1) -> Dict[Tensor, Tensor]:
        # Index where to start in traj
        start_idx = idx % self.n_per_traj
        end_idx = start_idx + length
        time_idx = torch.arange(start_idx, end_idx)
        # Trajectory index
        i_traj = idx // self.n_per_traj
        traj = self.trajectories[self.traj_keys[i_traj]]

        position = traj["position"][time_idx]
        particle_type = traj["particle_type"][:]
        position = torch.tensor(position)
        particle_type = torch.tensor(particle_type)
        position = einops.rearrange(
            position, "n_timesteps n_particles n_dim -> n_particles n_timesteps n_dim"
        )
        return {
            "position": position,  # [185943, 3, 3]
            "particle_type": particle_type,  # [185943]
            "time_idx": time_idx,  # [3], e.g. [0, 1, 2]
        }

    def get_permutation(self, n_particles, n_perm=None):
        if n_perm is None:
            if self.num_points_range[0] == self.num_points_range[1]:
                # fixed num_points_range
                end = self.num_points_range[1]
            else:
                lb, ub = self.num_points_range
                ub = min(ub, n_particles)
                num_points_range = torch.rand(size=(1,), generator=None).item() * (ub - lb) + lb
                end = int(num_points_range)
            # uniform sampling
            perm = torch.randperm(n_particles, generator=None)[:end]
        else:
            perm = torch.randperm(n_particles, generator=None)[:n_perm]
        return perm

    def normalize_pos(self, pos):
        # pos = pos - self.pos_mean.to(pos.device)
        # pos = pos / self.pos_std.to(pos.device)
        return pos
    
    def unnormalize_pos(self, pos):
        # pos = pos * self.pos_std.to(pos.device)
        # pos = pos + self.pos_mean.to(pos.device)
        return pos

    def normalize_vel(self, vel):
        vel = vel - self.vel_mean.to(vel.device)
        vel = vel / self.vel_std.to(vel.device)
        return vel

    def unnormalize_vel(self, vel):
        vel = vel * self.vel_std.to(vel.device)
        vel = vel + self.vel_mean.to(vel.device)
        return vel

    def normalize_acc(self, acc):
        acc = acc - self.acc_mean.to(acc.device)
        acc = acc / self.acc_std.to(acc.device)
        return acc

    def unnormalize_acc(self, acc):
        acc = acc * self.acc_std.to(acc.device)
        acc = acc + self.acc_mean.to(acc.device)
        return acc
    
    def normalize_displcmnt(self, displcmnt):
        # displcmnt = displcmnt - self.displcmnt_mean.to(displcmnt.device)
        # displcmnt = displcmnt / self.displcmnt_std.to(displcmnt.device)
        return displcmnt
    
    def unnormalize_displcmnt(self, displcmnt):
        # displcmnt = displcmnt * self.displcmnt_std.to(displcmnt.device)
        # displcmnt = displcmnt + self.displcmnt_mean.to(displcmnt.device)
        return displcmnt
    
    @property
    def normalize(self):
        if self.type in ["velocity", "displacement"]:
            return self.normalize_vel
        elif self.type == "acceleration":
            return self.normalize_acc
    
    @property
    def unnormalize(self):
        if self.type in ["velocity", "displacement"]:
            return self.unnormalize_vel
        elif self.type == "acceleration":
            return self.unnormalize_acc

@dataclass
class ParticleDataModule(L.LightningDataModule):
    train_split: str = "train"
    eval_split: str = "valid"
    test_split: str = "test"
    n_fields: int = 2
    n_jump_ahead_timesteps: int = 1
    n_jumps: int = 1
    type: str = "velocity"
    stage: str = "autoencoder"
    n_particle_types: int = None
    n_supernodes: int = None
    num_points_range: int = None
    num_points_decode: int = None
    global_root: str = None
    local_root: str = None
    seed: int = None
    batch_size: int = 4
    num_workers: int = 1
    pin_memory: bool = True
    persistent_workers: bool = False
    overfit_single_trajectory: bool = False
    follow_batch: Optional[List[str]] = field(
        default_factory=lambda: ["enc_pos", "supernode_index"]
    )
    dataset_rel_path: str = "multi_material"

    def __post_init__(self):
        super().__init__()
        assert self.stage in [
            "autoencoder",
            "physics",
            "particle",
            "particle_autoencoder"
        ]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.stage == "autoencoder":
            mode_train = "train_autoencoder"
            mode_val = "train_autoencoder"
        elif self.stage == "physics":
            mode_train = "train_physics"
            mode_val = "val_physics"
        elif self.stage == "particle_autoencoder":
            mode_train = "particle_autoencoder"
            mode_val = "particle_autoencoder"
        elif self.stage == "particle":
            mode_train = "particle"
            mode_val = "particle"

        self.train_dataset = ParticleDataset(
            n_fields=self.n_fields,
            n_jump_ahead_timesteps=self.n_jump_ahead_timesteps,
            n_jumps=self.n_jumps,
            type=self.type,
            split=self.train_split,
            mode=mode_train,
            n_particle_types=self.n_particle_types,
            n_supernodes=self.n_supernodes,
            num_points_range=self.num_points_range,
            global_root=self.global_root,
            local_root=self.local_root,
            seed=self.seed,
            num_points_decode=self.num_points_decode,
            dataset_rel_path=self.dataset_rel_path,
            overfit_single_trajectory=self.overfit_single_trajectory,
        )
        self.val_dataset = ParticleDataset(
            n_fields=self.n_fields,
            n_jump_ahead_timesteps=self.n_jump_ahead_timesteps,
            n_jumps=self.n_jumps,
            type=self.type,
            split=self.eval_split,
            mode=mode_val,
            n_particle_types=self.n_particle_types,
            n_supernodes=self.n_supernodes,
            num_points_range=self.num_points_range,
            global_root=self.global_root,
            local_root=self.local_root,
            seed=self.seed,
            num_points_decode=self.num_points_decode,
            dataset_rel_path=self.dataset_rel_path,
            overfit_single_trajectory=self.overfit_single_trajectory,
        )
        self.test_dataset = ParticleDataset(
            n_fields=self.n_fields,
            n_jump_ahead_timesteps=self.n_jump_ahead_timesteps,
            n_jumps=self.n_jumps,
            type=self.type,
            split=self.test_split,
            mode=mode_val,
            n_particle_types=self.n_particle_types,
            n_supernodes=self.n_supernodes,
            num_points_range=self.num_points_range,
            global_root=self.global_root,
            local_root=self.local_root,
            seed=self.seed,
            num_points_decode=self.num_points_decode,
            dataset_rel_path=self.dataset_rel_path,
            overfit_single_trajectory=self.overfit_single_trajectory,
            rollout=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            follow_batch=self.follow_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            follow_batch=self.follow_batch,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            follow_batch=self.follow_batch,
        )
    
    def debug_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            follow_batch=self.follow_batch,
        )

    def get_dataset(self, split: str) -> ParticleDataset:
        assert split in ["train", "val", "test"], \
            f"Invalid split: {split}. Expected 'train', 'valid' or 'test'."
        attr_name = f"{split}_dataset"
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        else:
            raise ValueError(f"Dataset for split '{split}' not found.")
