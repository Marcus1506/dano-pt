from functools import partial

import einops
import torch
from einops import rearrange
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import DitBlock, DitPerceiverBlock, PerceiverBlock
from kappamodules.vit import VitBlock
from torch import nn

from src.modules.kappa import MLP


class TransformerPerceiver(nn.Module):
    def __init__(
        self,
        input_dim,
        transformer_dim,
        transformer_depth,
        transformer_attn_heads,
        pos_embed: nn.Module,
        query_mlp_hidden: list,
        perceiver_dim,
        perceiver_attn_heads,
        feature_dim,
        init_weights="truncnormal",
        condition_dim=None,
        ndim=2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.transformer_dim = transformer_dim
        self.transformer_depth = transformer_depth
        self.transformer_attn_heads = transformer_attn_heads
        self.query_mlp_hidden = query_mlp_hidden
        self.perceiver_dim = perceiver_dim
        self.perceiver_attn_heads = perceiver_attn_heads
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.ndim = ndim

        # Projection from input to transformer
        self.input_proj = LinearProjection(input_dim, transformer_dim, init_weights=init_weights)

        # Transformer
        if self.condition_dim is not None:
            block_ctor = partial(DitBlock, cond_dim=condition_dim)
        else:
            block_ctor = VitBlock
        self.blocks = nn.ModuleList(
            [
                block_ctor(
                    dim=transformer_dim,
                    num_heads=transformer_attn_heads,
                    init_weights=init_weights,
                )
                for _ in range(transformer_depth)
            ]
        )

        # Positional embedding + MLP to get the queries for the perceiver
        self.pos_embed = pos_embed(dim=perceiver_dim, ndim=ndim)
        self.query_mlp = MLP(
            input_dim=perceiver_dim,
            output_dim=perceiver_dim,
            hidden_dims=query_mlp_hidden,
            init_weights=init_weights,
        )

        if self.condition_dim is not None:
            block_ctor = partial(DitPerceiverBlock, cond_dim=condition_dim)
        else:
            block_ctor = PerceiverBlock

        # Perceiver for features
        self.transformer_to_feature_perceiver = LinearProjection(
            transformer_dim, perceiver_dim, init_weights=init_weights
        )
        self.feature_perceiver = block_ctor(
            dim=perceiver_dim, num_heads=perceiver_attn_heads, init_weights=init_weights
        )
        self.perceiver_to_feature = nn.Sequential(
            nn.LayerNorm(perceiver_dim, eps=1e-6),
            LinearProjection(perceiver_dim, feature_dim, init_weights=init_weights),
        )

    def forward(self, x, pos=None, condition=None):
        batch_size = x.shape[0]
        assert x.ndim == 3
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        # input projection
        x = self.input_proj(x)

        # apply blocks
        for blk in self.blocks:
            x = blk(x, **block_kwargs)

        pos_embed = self.pos_embed(pos)
        pos_embed = rearrange(
            pos_embed,
            "(batch_size num_points) dim -> batch_size num_points dim",
            batch_size=batch_size,
        )

        feature_query = self.query_mlp(pos_embed)
        feature_out = self.transformer_to_feature_perceiver(x)
        feature_out = self.feature_perceiver(q=feature_query, kv=feature_out, **block_kwargs)
        feature_out = self.perceiver_to_feature(feature_out)
        feature_out = einops.rearrange(
            feature_out,
            "batch_size max_num_points dim -> (batch_size max_num_points) dim",
        )
        return feature_out
