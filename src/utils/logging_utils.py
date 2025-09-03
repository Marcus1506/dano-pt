from typing import Any, Dict

import wandb
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

    # Surface select hyperparameters directly in W&B config and summary
    # so they appear as columns in the run table without opening the config.
    # Keys are prefixed with "KEY/" for easy discovery and pinning in the UI.
    try:
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        n_jumps = data_cfg.get("n_jumps")
        n_jump_ahead_timesteps = data_cfg.get("n_jump_ahead_timesteps")
        n_fields = data_cfg.get("n_fields")

        surfaced = {
            "KEY/n_jumps": n_jumps,
            "KEY/n_jump_ahead_timesteps": n_jump_ahead_timesteps,
            "KEY/n_fields": n_fields,
        }
        # Drop unset values to avoid cluttering runs that don't define these fields
        surfaced = {k: v for k, v in surfaced.items() if v is not None}

        if surfaced:
            for logger in trainer.loggers:
                experiment = getattr(logger, "experiment", None)
                if experiment is None:
                    continue
                # Update config for filtering and default visibility
                try:
                    experiment.config.update(surfaced, allow_val_change=True)
                except Exception:
                    pass
                # Update summary so they show as columns in the runs table
                try:
                    experiment.summary.update(surfaced)
                except Exception:
                    pass
    except Exception:
        # Never fail training because of surfacing convenience fields
        pass


def load_run_config_from_wb(entity: str, project: str, run_id: str) -> OmegaConf:
    """Retrieve the run configuration from a wandb run.

    :param entity: The entity to retrieve the run from.
    :type entity: str
    :param project: The project name to retrieve the run from.
    :type project: str
    :param run_id: The run id to retrieve the run from.
    :type run_id: str
    :return: The run configuration.
    :rtype: OmegaConf
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    config = run.config
    return OmegaConf.create(config)
