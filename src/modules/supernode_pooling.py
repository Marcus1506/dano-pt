import numpy as np
import torch
from kappamodules.layers import LinearProjection
from torch import nn
from torch_cluster import radius
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation, MeanAggregation

from src.modules.kappa import MLP


class SupernodeGnn(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        input_proj_hidden: list,
        message_net_hidden: list,
        pos_embed: nn.Module,
        aggr: Aggregation = MeanAggregation(),
        ndim: int = 3,
        relative_pos_embed: nn.Module = None,
        init_weights="truncnormal",
    ):
        super().__init__(aggr=aggr)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj_hidden = input_proj_hidden
        self.message_net_hidden = message_net_hidden
        self.init_weights = init_weights

        self.input_proj = MLP(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dims=input_proj_hidden,
            init_weights=init_weights,
        )
        # We concatenate the features of x_i and x_j in the message
        # Additionally if we use relative_pos_embed we also concat the relative_pos_embed
        # which also has hidden_dim
        input_dim_message = hidden_dim * 3 if relative_pos_embed else hidden_dim * 2

        self.message_net = MLP(
            input_dim=input_dim_message,
            output_dim=hidden_dim,
            hidden_dims=message_net_hidden,
            init_weights=init_weights,
        )
        self.pos_embed = pos_embed(dim=hidden_dim, ndim=ndim)
        if relative_pos_embed:
            self.relative_pos_embed = relative_pos_embed(dim=hidden_dim, ndim=ndim)
        else:
            self.relative_pos_embed = None

    def forward(self, x, pos, edge_index):
        x = self.input_proj(x) + self.pos_embed(pos)
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        if self.relative_pos_embed is not None:
            rel_pos_embed = self.relative_pos_embed(pos_i - pos_j)
            x = torch.cat([x_i, x_j, rel_pos_embed], dim=-1)
        else:
            x = torch.cat([x_i, x_j], dim=-1)
        return self.message_net(x)


class SupernodePooling(nn.Module):
    def __init__(
        self,
        net: MessagePassing,
        supernodes_radius: float,
        supernodes_max_neighbours: int,
    ):
        super().__init__()
        self.net = net
        self.supernodes_radius = supernodes_radius
        self.supernodes_max_neighbours = supernodes_max_neighbours

    def forward(
        self,
        x,
        pos,
        batch_index,
        supernode_index,
        super_node_batch_index,
    ):
        edges = radius(
            x=pos,
            y=pos[supernode_index],
            r=self.supernodes_radius,
            max_num_neighbors=self.supernodes_max_neighbours,
            batch_x=batch_index,
            batch_y=super_node_batch_index,
        )
        # Change direction from nodes to supernodes
        edges = edges[[1, 0], :]
        # Correct indexing of supernodes
        edges[1, :] = supernode_index[edges[1, :]]
        x = self.net(x, pos, edges)
        return x[supernode_index]
