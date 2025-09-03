import torch
import torch.nn as nn
from kappamodules.layers import LinearProjection


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        init_weights: str = "truncnormal",
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        self.init_weights = init_weights
        self.act = act
        layers = [
            LinearProjection(input_dim, self.hidden_dims[0], init_weights=init_weights),
            act,
        ]
        for i in range(1, len(self.hidden_dims)):
            layers.append(
                LinearProjection(
                    self.hidden_dims[i - 1], self.hidden_dims[i], init_weights=init_weights
                )
            )
            layers.append(act)
        layers.append(
            LinearProjection(self.hidden_dims[-1], output_dim, init_weights=init_weights)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
