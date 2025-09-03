from functools import partial

import torch
from einops import rearrange
from kappamodules.layers import LinearProjection
from kappamodules.transformer import (
    DitBlock,
    DitPerceiverPoolingBlock,
    PerceiverPoolingBlock,
    PrenormBlock,
)
from torch import nn


class PoolTransformerPerceiver(nn.Module):
    def __init__(
        self,
        supernode_pooling,
        transformer_dim,
        transformer_depth,
        transformer_attn_heads,
        perceiver_dim,
        perceiver_attn_heads,
        num_latent_tokens,
        init_weights="truncnormal",
        condition_dim=None,
        output_ln=False,
    ):
        super().__init__()
        self.supernode_pooling = supernode_pooling
        self.transformer_dim = transformer_dim
        self.transformer_depth = transformer_depth
        self.transformer_attn_heads = transformer_attn_heads
        self.perceiver_dim = perceiver_dim
        self.perceiver_attn_heads = perceiver_attn_heads
        self.num_latent_tokens = num_latent_tokens
        self.init_weights = init_weights
        self.condition_dim = condition_dim
        self.output_ln = output_ln

        # Projection from supernodes to transformer
        self.supernode_to_transformer = LinearProjection(
            self.supernode_pooling.net.hidden_dim, transformer_dim, init_weights=init_weights
        )

        # Transformer to process global information in supernodes
        if self.condition_dim is not None:
            block_ctor = partial(DitBlock, cond_dim=self.condition_dim)
        else:
            block_ctor = PrenormBlock
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

        # Projection from transformer to perceiver pooling
        self.transformer_to_perceiver = LinearProjection(
            transformer_dim, perceiver_dim, init_weights=init_weights
        )
        # Perceiver pooling
        if self.condition_dim is not None:
            block_ctor = partial(
                DitPerceiverPoolingBlock,
                perceiver_kwargs=dict(
                    cond_dim=self.condition_dim,
                    init_weights=init_weights,
                ),
            )
        else:
            block_ctor = partial(
                PerceiverPoolingBlock,
                perceiver_kwargs=dict(init_weights=init_weights),
            )
        self.perceiver = block_ctor(
            dim=perceiver_dim,
            num_heads=perceiver_attn_heads,
            num_query_tokens=num_latent_tokens,
        )

    def forward(
        self,
        field,
        pos,
        batch_index,
        supernode_index,
        supernode_batch_index,
        condition=None,
    ):
        batch_size = batch_index.max().item() + 1
        x = self.supernode_pooling(
            x=field,
            pos=pos,
            batch_index=batch_index,
            supernode_index=supernode_index,
            super_node_batch_index=supernode_batch_index,
        )
        x = rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )

        # apply blocks
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        x = self.supernode_to_transformer(x)
        for blk in self.blocks:
            x = blk(x, **block_kwargs)

        # perceiver
        x = self.transformer_to_perceiver(x)
        x = self.perceiver(kv=x, **block_kwargs)

        if self.output_ln:
            x = nn.functional.layer_norm(x, (self.perceiver_dim,), eps=1e-6)

        return x
