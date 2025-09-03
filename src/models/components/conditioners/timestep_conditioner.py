from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from torch import nn

from src.modules.kappa import MLP


class TimestepConditioner(nn.Module):
    def __init__(
        self,
        dim,
        timestep_mlp_hidden,
        condition_dim,
        num_total_timesteps,
        init_weights="truncnormal",
    ):
        super().__init__()
        self.num_total_timesteps = num_total_timesteps
        self.dim = dim
        self.condition_dim = condition_dim
        # buffer/modules
        self.register_buffer(
            "timestep_embed",
            get_sincos_1d_from_seqlen(seqlen=self.num_total_timesteps, dim=dim),
        )
        self.timestep_mlp = MLP(
            input_dim=dim,
            output_dim=condition_dim,
            hidden_dims=timestep_mlp_hidden,
            init_weights=init_weights,
        )

    def forward(self, timestep):
        # checks + preprocess
        assert timestep.numel() == len(timestep)
        timestep = timestep.flatten()
        # embed
        timestep_embed = self.timestep_mlp(self.timestep_embed[timestep])
        return timestep_embed
