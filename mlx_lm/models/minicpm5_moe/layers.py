# Copyright © 2026 MLX Contributors

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.models.deepseek_v3.layers import group_expert_select

__all__ = ["Attention", "DecoderLayer", "MLP", "MoE"]


def make_rope(args):
    parameters = args.rope_parameters or args.rope_scaling or {}
    if parameters.get("rope_type", parameters.get("type", "default")) != "default":
        raise ValueError("MiniCPM5 MoE requires full-head RoPE without scaling.")
    return nn.RoPE(
        args.head_dim,
        traditional=False,
        base=parameters.get("rope_theta", args.rope_theta),
    )


class ActivationDtypeLinear(nn.QuantizedLinear):
    def __call__(self, x):
        # The conversion stores exact affine offsets in float32.
        return super().__call__(x).astype(x.dtype)


class Linear(nn.Linear):
    def to_quantized(self, group_size=None, bits=None, mode="affine"):
        return ActivationDtypeLinear.from_linear(self, group_size, bits, mode)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x):
        args = self.args
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return group_expert_select(
            logits,
            self.e_score_correction_bias,
            args.num_experts_per_tok,
            args.n_group,
            args.topk_group,
            1.0,
            args.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            bias=args.mlp_bias,
        )
        self.shared_experts = (
            MLP(
                args.hidden_size,
                args.moe_intermediate_size * args.n_shared_experts,
                args.mlp_bias,
            )
            if args.n_shared_experts
            else None
        )
        self.routed_scaling_factor = args.routed_scaling_factor

    def __call__(self, x):
        indices, scores = self.gate(x)
        experts = self.switch_mlp(x, indices)
        routed = (experts * scores[..., None]).sum(-2) * self.routed_scaling_factor
        routed = routed.astype(x.dtype)
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(x)
        return routed


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.gated = args.use_gated_attention
        self.scale = self.head_dim**-0.5
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = Linear(
            args.hidden_size, q_dim * (2 if self.gated else 1), bias=args.attention_bias
        )
        self.k_proj = Linear(args.hidden_size, kv_dim, bias=args.attention_bias)
        self.v_proj = Linear(args.hidden_size, kv_dim, bias=args.attention_bias)
        self.o_proj = Linear(q_dim, args.hidden_size, bias=args.attention_bias)
        self.rope = make_rope(args)

    def __call__(self, x, mask, cache):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, -1)
        if self.gated:
            # Checkpoint rows alternate query and gate within each head.
            q, gate = mx.split(q, 2, axis=-1)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        offset = cache.offset if cache is not None else 0
        q = self.rope(q.transpose(0, 2, 1, 3), offset=offset)
        k = self.rope(k.transpose(0, 2, 1, 3), offset=offset)
        v = v.transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache, self.scale, mask)
        out = out.transpose(0, 2, 1, 3)
        if self.gated:
            out = out * mx.sigmoid(gate)
        return self.o_proj(out.reshape(batch, length, -1))


class DecoderLayer(nn.Module):
    def __init__(self, args, attention, mlp):
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask, cache):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))
