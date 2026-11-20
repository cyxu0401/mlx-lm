# Copyright © 2026 MLX Contributors

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.minicpm5_moe.layers import MLP, Attention, DecoderLayer, MoE


class MiniCPM5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args,
                Attention(args),
                (
                    MLP(args.hidden_size, args.intermediate_size, args.mlp_bias)
                    if index < args.first_k_dense_replace
                    else MoE(args)
                ),
            )
            for index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache, capture_layers, input_embeddings=None):
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        captured = {}
        for index, (layer, entry) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, entry)
            if index in capture_layers:
                captured[index] = h
        return self.norm(h), tuple(captured[index] for index in capture_layers)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.model_type != "minicpm5_moe" or args.hidden_act != "silu":
            raise ValueError("Expected a MiniCPM5 MoE checkpoint with SwiGLU.")
        self.args = args
        self.model_type = args.model_type
        self.model = MiniCPM5(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def project_logits(self, h):
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h, _ = self.model(inputs, cache, (), input_embeddings)
        return self.project_logits(h).astype(mx.float32)

    def capture(self, inputs, cache, layers, last_only=False):
        h, features = self.model(inputs, cache, layers)
        if last_only:
            h = h[:, -1:]
        return self.project_logits(h).astype(mx.float32), features
