import torch
from lightning import Callback


class PeakMemory(Callback):
    """Get the maximum memory used during training."""

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset stats
        if "cuda" in str(pl_module.device):
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        # Log the maximum memory consumption
        if "cuda" in str(pl_module.device):
            max_memory_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            self.log("train/max_memory", max_memory_gb, prog_bar=True)
