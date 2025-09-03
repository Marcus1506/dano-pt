from lightning import Callback


class ConfigLRScheduler(Callback):
    """Count up every gradient update step rather than every epoch."""

    def on_train_start(self, trainer, pl_module):
        # Access the scheduler from the trainer
        self.scheduler = trainer.lr_scheduler_configs[0].scheduler
        assert self.scheduler.__class__.__name__ == "LinearWarmupCosineAnnealingLR"

        # configure the scheduler
        self.scheduler.set_steps_per_epoch(
            len(trainer.train_dataloader) // trainer.accumulate_grad_batches
        )
