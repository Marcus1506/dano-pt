import os
from pathlib import Path
import time
import numpy as np
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
import torch
import random
import rootutils

# Ensure repository root is on PYTHONPATH so `src` can be imported from anywhere
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.physics import PhysicsLitModule
from src.models.utils import (
    unflatten_time,
    update_position_cycle,
)
from src.datasets.utils import get_subset_iters

GlobalHydra.instance().clear()
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PROJECT_ROOT"] = str(REPO_ROOT)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Force deterministic algorithms (and fail if a nondet op is hit)
torch.use_deterministic_algorithms(True)          # raises on nondeterministic ops
torch.set_deterministic_debug_mode("error")       # PyTorch ≥2.1 alias: set_deterministic_debug
# cuDNN settings
torch.backends.cudnn.deterministic = True         # force det conv algos
torch.backends.cudnn.benchmark = False            # disable autotune nondet selection
# Keep math mode consistent
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

HYDRA_CONFIG = str(REPO_ROOT / "logs/train/runs/2025-09-12_18-29-07/.hydra/config.yaml")
CHKPT = str(REPO_ROOT / "logs/train/runs/2025-09-12_18-29-07/waterdrop_physics/9odrn4lu/checkpoints/epoch=27-step=403620.ckpt")

if __name__ == "__main__":
    cfg          = OmegaConf.load(
        HYDRA_CONFIG
    )

    latent_model_factory = instantiate(cfg.model.latent_model, _partial_=True)
    model_factory        = instantiate(cfg.model,            _partial_=True)
    loss_function        = instantiate(cfg.model.loss_function)

    lit_model = PhysicsLitModule.load_from_checkpoint(
        CHKPT,
        latent_model = latent_model_factory,
        model        = model_factory,
        loss_function= loss_function,
        strict       = True,
        map_location = "cuda"
    )
    lit_model.eval()
    dataset_PH = instantiate(cfg.data)
    dataset_PH.setup(stage="physics")
    dataset = dataset_PH.get_dataset(split="test")

    TRAJ_IDX = list(range(dataset.n_traj))
    n_per_traj_PH = dataset.n_per_traj
    n_jump_ahead_timesteps = dataset.n_jump_ahead_timesteps
    offset = 2 * n_jump_ahead_timesteps
    start_idx = 16 - offset
    IDX_PH = [[start_idx]] * len(TRAJ_IDX) # Same start index for all trajs

    # Number of repetitions for averaging
    N = 4

    # Pre rollout
    model_type = dataset.type
    unnormalize = dataset.unnormalize
    subset_iters = get_subset_iters(
        dataset,
        traj_idx=TRAJ_IDX,
        idx=IDX_PH,
        follow_batch=dataset_PH.follow_batch
    )
    device = next(lit_model.parameters()).device

    n_push_forward_steps = n_per_traj_PH // n_jump_ahead_timesteps

    particle_counts = []
    rollout_times = []

    with torch.no_grad():
        for _ in range(N):
            for subset_iter in subset_iters:
                batch = next(subset_iter)
                input_data, target_data = batch
                input_data = input_data.to(device)
                target_data = target_data.to(device)

                n, n_time, n_dim = input_data.input_enc_field.shape
                if model_type == "velocity":
                    n_redundant = n_time - n_jump_ahead_timesteps
                elif model_type == "displacement":
                    n_redundant = 0

                particle_counts.append(n)

                # Start timer
                t0 = time.perf_counter()

                pred_latent = lit_model.encode(
                    enc_pos=input_data.input_enc_pos,
                    enc_field=input_data.input_enc_field,
                    enc_particle_type=input_data.input_enc_particle_type,
                    enc_pos_batch_index=input_data.batch,
                    supernode_index=input_data.supernode_index,
                    supernode_batch_index=input_data.supernode_index_batch,
                    timestep=input_data.input_timestep,
                )

                timestep = input_data.input_timestep
                pos = target_data.target_pos[:, 0]

                for i in range(start_idx // n_jump_ahead_timesteps, n_push_forward_steps):
                    pred_latent = lit_model.push_forward(
                        latent=pred_latent,
                        timestep=timestep
                    )
                    timestep = timestep + n_jump_ahead_timesteps

                    preds_field = lit_model.decode(
                        latent=pred_latent,
                        dec_field_pos=pos,
                        timestep=timestep,
                    )
                    preds_field = unflatten_time(
                        preds_field,
                        n_time=n_time,
                        n_dim=n_dim
                    )
                    preds_field_unnorm = unnormalize(preds_field)

                    pos = update_position_cycle(
                        old_position=pos,
                        field_prediction=preds_field_unnorm,
                        type=model_type,
                        n_redundant=n_redundant
                    )
                
                # End timer
                dt = time.perf_counter() - t0
                rollout_times.append(dt)

    rollout_times = np.array(rollout_times)
    particle_counts = np.array(particle_counts)

    time_mean = np.mean(rollout_times)
    time_std  = np.std(rollout_times)
    particles_mean = np.mean(particle_counts)
    particles_std  = np.std(particle_counts)

    print(f"Rollout time for n={n_time}, tA={n_jump_ahead_timesteps} model over {len(rollout_times)} runs: {time_mean:.4f} ± {time_std:.4f} s")
    print(f"Particles per run: {particles_mean:.1f} ± {particles_std:.1f}")
