# Copyright © 2026 MLX Contributors

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = ["ModelArgs"]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    routed_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: dict | None = None
    rope_scaling: dict | None = None
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    use_gated_attention: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
