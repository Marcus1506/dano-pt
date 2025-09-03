import lightning as L


class HyperparamTimeseriesLogger(L.Callback):

	def __init__(self, log_on: str = "epoch") -> None:
		assert log_on in ["epoch", "step"], "log_on must be 'epoch' or 'step'"
		self.log_on = log_on

	@staticmethod
	def _collect(trainer: L.Trainer) -> dict:
		dm = getattr(trainer, "datamodule", None)
		if dm is None:
			return {}
		values = {}
		for key in ("n_fields", "n_jump_ahead_timesteps", "n_jumps"):
			if hasattr(dm, key):
				values[f"hparam/{key}"] = getattr(dm, key)
		return values

	@staticmethod
	def _log_with_lightning(pl_module: L.LightningModule, values: dict, *, on_step: bool, on_epoch: bool) -> None:
		if not values:
			return
		# Use Lightning's logging so the logger step handling stays consistent.
		for key, value in values.items():
			try:
				pl_module.log(key, value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, batch_size=1)
			except Exception:
				pass

	def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
		if self.log_on == "epoch":
			self._log_with_lightning(pl_module, self._collect(trainer), on_step=False, on_epoch=True)

	def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
		if self.log_on == "epoch":
			self._log_with_lightning(pl_module, self._collect(trainer), on_step=False, on_epoch=True)

	def on_train_batch_end(
		self,
		trainer: L.Trainer,
		pl_module: L.LightningModule,
		outputs,
		batch,
		batch_idx: int,
	) -> None:
		if self.log_on == "step":
			self._log_with_lightning(pl_module, self._collect(trainer), on_step=True, on_epoch=False)


