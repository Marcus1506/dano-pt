from functools import partial

import torch
import einops
import torch.nn.functional as F
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PrenormBlock
from kappamodules.attention import DotProductAttention1d
from kappamodules.init import (
    init_norms_as_noaffine,
    init_xavier_uniform_zero_bias,
    init_xavier_uniform_merged_linear,
    init_truncnormal_zero_bias,
)
from kappamodules.layers import DropPath
from kappamodules.modulation import Dit
from kappamodules.modulation.functional import modulate_scale_shift, modulate_gate
from kappamodules.mlp import Mlp
from kappamodules.functional.pos_embed import relative_position_indices
from torch import nn

class DotProductAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            proj_bias=True,
            rel_pos_bias="none",
            seqlens=None,
            channel_first=False,
            init_weights="truncnormal002",
            init_last_proj_zero=False,
            grounding=False
    ):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rel_pos_bias = rel_pos_bias
        self.seqlens = seqlens
        self.channel_first = channel_first
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.grounding = grounding

        self.qkv = nn.Linear(2 * dim if grounding else dim, dim * 3, bias=qkv_bias)

        if rel_pos_bias == "none":
            self.rel_pos_bias_table = None
            self.rel_pos_idx = None
        elif rel_pos_bias == "learnable":
            assert seqlens is not None
            rel_pos_idx, num_distinct_distances = relative_position_indices(seqlens=seqlens, num_aux_tokens=1)
            self.register_buffer("rel_pos_idx", rel_pos_idx)
            self.rel_pos_bias_table = nn.Parameter(torch.empty(num_distinct_distances, num_heads))
        else:
            raise NotImplementedError
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            init_xavier_uniform_merged_linear(self.qkv, num_layers=3)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.rel_pos_bias_table is not None:
            nn.init.zeros_(self.rel_pos_bias_table)
        if self.init_last_proj_zero:
            nn.init.zeros_(self.proj.weight)
            # init_weights == "torch" has no zero bias init
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def to_channel_last(self, x):
        raise NotImplementedError

    def to_channel_first(self, x, og_shape):
        raise NotImplementedError

    def forward(self, x, attn_mask=None, grounding=None):
        if self.channel_first:
            og_shape = x.shape
            x = self.to_channel_last(x)
        else:
            og_shape = None

        if self.grounding:
            assert grounding is not None, "Grounding must be provided when grounding is True"
            x = torch.cat([x, grounding], dim=-1)

        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        
        if self.rel_pos_bias_table is not None:
            assert attn_mask is None
            seqlen = x.size(1)
            assert self.rel_pos_idx.shape == (seqlen, seqlen), \
                f"invalid input seqlen {seqlen} (expected {self.rel_pos_idx.shape[0]})"
            attn_mask = self.rel_pos_bias_table[self.rel_pos_idx.view(-1)].view(*self.rel_pos_idx.shape, -1)
            attn_mask = einops.rearrange(attn_mask, "... num_heads -> 1 num_heads ...").contiguous().to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)

        if self.channel_first:
            x = self.to_channel_first(x, og_shape=og_shape)
        return x

class DotProductAttention1d(DotProductAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c l -> b l c")

    def to_channel_first(self, x, og_shape):
        return einops.rearrange(x, "b l c -> b c l")

class DitBlock(nn.Module):
    """ adaptive norm block (https://github.com/facebookresearch/DiT) """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_hidden_dim=None,
            cond_dim=None,
            qkv_bias=True,
            drop_path=0.,
            attn_ctor=DotProductAttention1d,
            eps=1e-6,
            init_weights="xavier_uniform",
            init_norms="nonaffine",
            init_last_proj_zero=False,
            init_gate_zero=False,
            grounding=False
    ):
        super().__init__()
        # DiT uses non-affine LayerNorm and GELU with tanh-approximation
        norm_ctor = partial(nn.LayerNorm, elementwise_affine=False)
        act_ctor = partial(nn.GELU, approximate="tanh")
        # properties
        self.init_norms = init_norms
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        cond_dim = cond_dim or dim
        # modulation
        self.modulation = Dit(
            cond_dim=cond_dim,
            out_dim=dim,
            init_weights=init_weights,
            num_outputs=6,
            gate_indices=[2, 5],
            init_gate_zero=init_gate_zero,
        )
        # attn
        self.norm1 = norm_ctor(dim, eps=eps)
        self.attn = attn_ctor(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
            grounding=grounding,
        )
        self.drop_path1 = DropPath(drop_prob=drop_path)
        # mlp
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp = Mlp(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            act_ctor=act_ctor,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.drop_path2 = DropPath(drop_prob=drop_path)
        #
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_norms == "torch":
            pass
        elif self.init_norms == "nonaffine":
            init_norms_as_noaffine(self.norm1)
            init_norms_as_noaffine(self.norm2)
        else:
            raise NotImplementedError

    def _attn_residual_path(self, x, scale, shift, gate, attn_mask, grounding):
        x = modulate_scale_shift(self.norm1(x), scale=scale, shift=shift)
        x = self.attn(x, attn_mask=attn_mask, grounding=grounding)
        return modulate_gate(x, gate=gate)

    def _mlp_residual_path(self, x, scale, shift, gate):
        return modulate_gate(self.mlp(modulate_scale_shift(self.norm2(x), scale=scale, shift=shift)), gate=gate)

    def forward(self, x, cond, attn_mask=None, grounding=None):
        scale1, shift1, gate1, scale2, shift2, gate2 = self.modulation(cond)
        x = self.drop_path1(
            x,
            residual_path=self._attn_residual_path,
            residual_path_kwargs=dict(
                scale=scale1,
                shift=shift1,
                gate=gate1,
                attn_mask=attn_mask,
                grounding=grounding,
            ),
        )
        x = self.drop_path2(
            x,
            residual_path=self._mlp_residual_path,
            residual_path_kwargs=dict(
                scale=scale2,
                shift=shift2,
                gate=gate2,
            ),
        )
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        latent_dim,
        transformer_dim,
        transformer_depth,
        transformer_attn_heads,
        input_proj=True,
        output_proj=False,
        input_ln=False,
        output_ln=False,
        init_weights="truncnormal",
        init_last_proj_zero=True,
        full_residual=True,
        condition_dim=None,
        grounding=False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.transformer_dim = transformer_dim
        self.transformer_depth = transformer_depth
        self.transformer_attn_heads = transformer_attn_heads
        self.input_proj = input_proj
        self.output_proj = output_proj
        self.input_ln = input_ln
        self.output_ln = output_ln
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.full_residual = full_residual
        self.condition_dim = condition_dim
        self.grounding = grounding

        if self.input_proj:
            self.input_proj = LinearProjection(
                latent_dim, transformer_dim, init_weights=init_weights
            )
        else:
            self.input_proj = nn.Identity()

        # blocks
        if self.condition_dim is not None:
            block_ctor = partial(
                DitBlock,
                cond_dim=self.condition_dim,
                grounding=self.grounding
            )
        else:
            block_ctor = PrenormBlock
        self.blocks = nn.ModuleList(
            [
                block_ctor(
                    dim=transformer_dim,
                    num_heads=transformer_attn_heads,
                    init_weights=init_weights,
                    init_last_proj_zero=init_last_proj_zero,
                )
                for i in range(transformer_depth)
            ]
        )

        if output_proj:
            self.output_proj = LinearProjection(
                transformer_dim, latent_dim, init_weights=init_weights
            )
        else:
            self.output_proj = nn.Identity()

    def forward(self, x, condition=None, grounding=None):
        assert x.ndim == 3

        if self.input_ln:
            x = nn.functional.layer_norm(x, (self.latent_dim,), eps=1e-6)

        og_x = x

        # input projection
        x = self.input_proj(x)

        # apply blocks
        blk_kwargs = dict(cond=condition, grounding=grounding) if condition is not None else dict()
        for blk in self.blocks:
            x = blk(x, **blk_kwargs)

        # output projection
        x = self.output_proj(x)

        if self.full_residual:
            x = og_x + x

        if self.output_ln:
            x = nn.functional.layer_norm(x, (self.latent_dim,), eps=1e-6)

        return x
