# Copyright © 2026 MLX Contributors

"""Single-request DSpark generation with a Qwen3 GQA draft."""

import json
import logging
import math
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import KVCache, make_prompt_cache, trim_prompt_cache
from mlx_lm.models.minicpm5_moe.layers import MLP, DecoderLayer, make_rope
from mlx_lm.models.minicpm5_moe.model import Model
from mlx_lm.sample_utils import make_sampling_filter


@dataclass
class DraftArgs(BaseModelArgs):
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    target_layer_ids: list[int]
    block_size: int
    mask_token_id: int
    markov_rank: int
    markov_head_type: str
    num_target_layers: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: dict | None = None
    rope_scaling: dict | None = None
    sample_from_anchor: bool = True
    attention_bias: bool = False


class DraftAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        dim = args.hidden_size
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, kv_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, kv_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(q_dim, dim, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = make_rope(args)

    def project_kv(self, x, offset):
        batch, length, _ = x.shape
        shape = (batch, length, self.n_kv_heads, self.head_dim)
        k = self.k_norm(self.k_proj(x).reshape(shape)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        return self.rope(k, offset=offset), v

    def __call__(self, x, _mask, cache):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        q = self.rope(q, offset=cache.offset)
        k, v = self.project_kv(x, cache.offset)
        # Every draft position attends to the committed prefix and the full block.
        k = mx.concatenate([cache.keys[..., : cache.offset, :], k], axis=2)
        v = mx.concatenate([cache.values[..., : cache.offset, :], v], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MarkovHead(nn.Module):
    def __init__(self, vocab_size, rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def __call__(self, previous_token):
        return self.markov_w2(self.markov_w1(previous_token))


class Draft(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layers = [
            DecoderLayer(
                args,
                DraftAttention(args),
                MLP(args.hidden_size, args.intermediate_size),
            )
            for _ in range(args.num_hidden_layers)
        ]
        self.fc = nn.Linear(
            len(args.target_layer_ids) * args.hidden_size, args.hidden_size, bias=False
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.markov_head = MarkovHead(args.vocab_size, args.markov_rank)

    def append_context(self, features, cache):
        context = self.hidden_norm(self.fc(mx.concatenate(features, axis=-1)))
        for layer, entry in zip(self.layers, cache):
            k, v = layer.self_attn.project_kv(context, entry.offset)
            entry.update_and_fetch(k, v)

    def propose(self, target, anchor, cache, count, sampling):
        start = int(not self.args.sample_from_anchor)
        length = self.args.block_size + start
        tokens = mx.array([[anchor] + [self.args.mask_token_id] * (length - 1)])
        h = target.model.embed_tokens(tokens)
        for layer, entry in zip(self.layers, cache):
            h = layer(h, None, entry)
        base_logits = target.project_logits(self.norm(h[:, start : start + count]))[0]

        previous = mx.array(anchor)
        proposals, distributions = [], []
        for row in range(count):
            logits = base_logits[row] + self.markov_head(previous)
            if sampling.greedy:
                previous = mx.argmax(logits)
            else:
                log_q = sampling.log_distribution(logits)
                previous = mx.random.categorical(log_q)
                distributions.append(log_q)
            proposals.append(previous)
        return mx.stack(proposals), distributions

    @staticmethod
    def speculative_sampler(sampling):
        return Sampling(sampling.temperature, sampling.top_k, sampling.top_p or 1.0)

    def speculative_generate_step(
        self, prompt, target, stream, *, dspark_stats=None, **kwargs
    ):
        unsupported = [
            key
            for key in ("input_embeddings", "kv_bits", "max_kv_size")
            if kwargs.pop(key, None) is not None
        ]
        if kwargs.pop("logits_processors", None):
            unsupported.append("logits_processors")
        if unsupported:
            raise ValueError(f"DSpark does not support {', '.join(unsupported)}.")
        for key in ("num_draft_tokens", "kv_group_size", "quantized_kv_start"):
            kwargs.pop(key, None)
        stats = dspark_stats if dspark_stats is not None else DraftStats()
        steps = generate_step(prompt, target, self, stream, stats=stats, **kwargs)
        count = emitted = 0
        start = time.perf_counter()
        try:
            with closing(steps):
                for record in steps:
                    count += 1
                    emitted += record[2]
                    yield record
        finally:
            logging.info(
                "DSpark %s",
                json.dumps(
                    dict(
                        asdict(stats),
                        emitted_draft=emitted,
                        prompt_tokens=prompt.size,
                        completion_tokens=count,
                        elapsed_seconds=round(time.perf_counter() - start, 3),
                    )
                ),
            )


def load_draft(path: Path, target):
    if not isinstance(target, Model):
        raise ValueError("DSpark requires a MiniCPM5 MoE target.")
    config = json.loads((path / "config.json").read_text())
    args = DraftArgs.from_dict(config)
    if (
        "Qwen3DSparkModel" not in config.get("architectures", [])
        or args.markov_head_type != "vanilla"
        or args.markov_rank <= 0
        or args.block_size <= 0
        or config.get("attention_mode", "gqa") != "gqa"
        or config.get("use_sliding_window")
        or set(config.get("layer_types", ["full_attention"])) != {"full_attention"}
    ):
        raise ValueError(
            "Expected a Qwen3 DSpark GQA draft with a vanilla Markov head."
        )
    vocab_size = target.args.vocab_size
    if (
        args.hidden_size != target.args.hidden_size
        or args.vocab_size != vocab_size
        or config.get("draft_vocab_size", vocab_size) != vocab_size
        or args.num_target_layers != len(target.layers)
        or not 0 <= args.mask_token_id < vocab_size
        or not args.target_layer_ids
        or len(args.target_layer_ids) != len(set(args.target_layer_ids))
        or any(not 0 <= layer < len(target.layers) for layer in args.target_layer_ids)
    ):
        raise ValueError("DSpark draft does not match the target configuration.")

    draft = Draft(args)
    weights = mx.load(str(path / "model.safetensors"))
    # Fixed verification width does not use confidence-based scheduling.
    weights.pop("confidence_head.proj.weight", None)
    weights.pop("confidence_head.proj.bias", None)
    draft.load_weights(list(weights.items()))
    draft.eval()
    mx.eval(draft.parameters())
    return draft


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    _filter_logprobs: Callable[[mx.array], mx.array] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if (
            not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.top_k < 0
            or not 0 < self.top_p <= 1
        ):
            raise ValueError(
                "Expected temperature >= 0, top_k >= 0 and 0 < top_p <= 1."
            )

        object.__setattr__(
            self,
            "_filter_logprobs",
            make_sampling_filter(top_p=self.top_p, top_k=self.top_k),
        )

    @property
    def greedy(self):
        return self.temperature == 0

    def log_distribution(self, logits):
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        scores = self._filter_logprobs(logprobs) * (1.0 / self.temperature)
        scores = scores - scores.max(axis=-1, keepdims=True)
        return scores - mx.logsumexp(scores, axis=-1, keepdims=True)

    def __call__(self, logits):
        if self.greedy:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(self.log_distribution(logits))


@dataclass
class DraftStats:
    steps: int = 0
    proposed: int = 0
    accepted: int = 0


def verify(logits, proposals, draft_distributions, sampling):
    """Return the accepted prefix length and the correction or bonus token."""
    count = proposals.size
    if sampling.greedy:
        target = mx.argmax(logits, axis=-1)
        accept = target[:-1] == proposals
        accepted = next(
            (index for index, match in enumerate(accept.tolist()) if not match), count
        )
        return accepted, target[accepted].item()

    log_p = sampling.log_distribution(logits)
    log_q = mx.stack(draft_distributions)
    rows = mx.arange(count)
    log_ratio = log_p[rows, proposals] - log_q[rows, proposals]
    accept = mx.log(mx.random.uniform(shape=(count,))) < mx.minimum(log_ratio, 0)
    accepted = next(
        (index for index, match in enumerate(accept.tolist()) if not match), count
    )
    if accepted == count:
        return accepted, mx.random.categorical(log_p[-1]).item()

    # exp(log_p) - exp(log_q), evaluated with expm1 near equality.
    p, q = log_p[accepted], log_q[accepted]
    positive = p > q
    difference = mx.where(positive, q - p, 0)
    residual = mx.where(positive, p + mx.log(-mx.expm1(difference)), -mx.inf)
    return accepted, mx.random.categorical(residual).item()


def _logprobs(logits):
    return logits - mx.logsumexp(logits)


def generate_step(
    prompt,
    target,
    draft,
    stream,
    *,
    max_tokens,
    sampler=None,
    stats=None,
    eos_token_ids=(),
    prompt_cache=None,
    prefill_step_size=128,
    prompt_progress_callback=None,
):
    """Yield tokens, target log probabilities and draft acceptance flags."""
    sampling = Sampling() if sampler is None else sampler
    if not isinstance(sampling, Sampling):
        raise ValueError("DSpark requires a Sampling sampler.")
    if stats is None:
        stats = DraftStats()
    if (
        prompt.ndim != 1
        or prompt.size == 0
        or max_tokens <= 0
        or prefill_step_size <= 0
    ):
        raise ValueError(
            "Expected a nonempty 1-D prompt and positive token/step limits."
        )
    if prompt_cache is None:
        target_cache = make_prompt_cache(target)
        draft_cache = [KVCache() for _ in draft.layers]
    else:
        if len(prompt_cache) != len(target.layers) + len(draft.layers) or any(
            cache.offset for cache in prompt_cache
        ):
            raise ValueError("DSpark requires an empty target and draft KV cache.")
        target_cache = prompt_cache[: len(target.layers)]
        draft_cache = prompt_cache[len(target.layers) :]
    capture = draft.args.target_layer_ids

    with mx.stream(stream):
        # Match MLX LM's prefill split so the anchor uses the same input shape.
        for start in range(0, prompt.size - 1, prefill_step_size):
            end = min(start + prefill_step_size, prompt.size - 1)
            _, features = target.capture(
                prompt[None, start:end], target_cache, capture, last_only=True
            )
            draft.append_context(features, draft_cache)
            mx.eval([entry.state for entry in (*target_cache, *draft_cache)])
            if prompt_progress_callback:
                prompt_progress_callback(end, prompt.size)
        logits, features = target.capture(
            prompt[None, -1:], target_cache, capture, last_only=True
        )
        anchor = sampling(logits[0, -1]).item()
        if prompt_progress_callback:
            prompt_progress_callback(prompt.size, prompt.size)
        yield anchor, _logprobs(logits[0, -1]), False
        if anchor in eos_token_ids or max_tokens == 1:
            return
        draft.append_context(features, draft_cache)

        remaining = max_tokens - 1
        while remaining > 1:
            count = min(draft.args.block_size, remaining - 1)
            proposals, draft_distributions = draft.propose(
                target, anchor, draft_cache, count, sampling
            )
            block = mx.concatenate(
                [mx.array([anchor], dtype=proposals.dtype), proposals]
            )
            logits, features = target.capture(block[None], target_cache, capture)
            accepted, anchor = verify(
                logits[0], proposals, draft_distributions, sampling
            )
            trim_prompt_cache(target_cache, count - accepted)
            stats.steps += 1
            stats.proposed += count
            stats.accepted += accepted

            for index, token in enumerate(proposals[:accepted].tolist()):
                yield token, _logprobs(logits[0, index]), True
                if token in eos_token_ids:
                    return
            yield anchor, _logprobs(logits[0, accepted]), False
            remaining -= accepted + 1
            if anchor in eos_token_ids or remaining == 0:
                return

            # Commit the old anchor and accepted prefix. The new anchor is pending.
            draft.append_context(
                tuple(h[:, : accepted + 1] for h in features), draft_cache
            )

        if remaining:
            logits = target(mx.array([[anchor]]), cache=target_cache)
            yield sampling(logits[0, -1]).item(), _logprobs(logits[0, -1]), False
