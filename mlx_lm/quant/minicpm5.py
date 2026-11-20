# Copyright © 2026 MLX Contributors

"""Convert a MiniCPM5 CT asymmetric INT4 g128 checkpoint to MLX affine."""

import argparse
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

BITS = 4
GROUP_SIZE = 128
EXPERT = re.compile(r"(.+\.mlp)\.experts\.(\d+)\.(.+)")
CT_FIELDS = ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")


def repack_projection(packed, scales, zeros, shape):
    """CT and MLX use the same low-nibble-first weight codes."""
    rows, cols = shape.tolist()
    groups = cols // GROUP_SIZE
    if (
        rows % 8
        or cols % GROUP_SIZE
        or packed.dtype != mx.int32
        or zeros.dtype != mx.int32
        or scales.dtype != mx.bfloat16
        or packed.shape != (rows, cols // 8)
        or zeros.shape != (rows // 8, groups)
        or scales.shape != (rows, groups)
    ):
        raise ValueError("Invalid CT INT4 g128 projection tensors.")

    shifts = mx.arange(0, 32, 4, dtype=mx.uint32)[None, :, None]
    zero_codes = ((zeros.view(mx.uint32)[:, None, :] >> shifts) & 15).reshape(
        rows, groups
    )
    scales = scales.astype(mx.float32)
    # Both CT codes include the same +8 shift. Their difference is unchanged.
    return {
        "weight": packed.view(mx.uint32),
        "scales": scales,
        "biases": -zero_codes.astype(mx.float32) * scales,
    }


def repack_shard(weights, expert_count):
    converted = {}
    for key in tuple(weights):
        if not key.endswith(".weight_packed"):
            continue
        prefix = key.removesuffix(".weight_packed")
        keys = [f"{prefix}.{field}" for field in CT_FIELDS]
        if not all(name in weights for name in keys):
            raise ValueError(f"Incomplete CT projection {prefix}.")
        projected = repack_projection(*(weights.pop(name) for name in keys))
        converted.update(
            {f"{prefix}.{name}": value for name, value in projected.items()}
        )
    if any(
        key.endswith(tuple(f".{field}" for field in (*CT_FIELDS, "weight_g_idx")))
        for key in weights
    ):
        raise ValueError("Unmatched CT metadata or activation-order groups.")
    count = len(converted) // 3
    converted.update(weights)

    experts = defaultdict(dict)
    output = {}
    for key, tensor in converted.items():
        if match := EXPERT.fullmatch(key):
            prefix, expert, parameter = match.groups()
            experts[f"{prefix}.switch_mlp.{parameter}"][int(expert)] = tensor
        else:
            output[key] = tensor
    for key, parts in experts.items():
        if set(parts) != set(range(expert_count)):
            raise ValueError(f"All experts for {key} must be in the same source shard.")
        output[key] = mx.stack([parts[index] for index in range(expert_count)])
    return output, count


def converted_config(config):
    quant = config.get("quantization_config", {})
    groups = list(quant.get("config_groups", {}).values())
    expected = {
        "num_bits": BITS,
        "group_size": GROUP_SIZE,
        "symmetric": False,
        "strategy": "group",
        "type": "int",
        "dynamic": False,
    }
    if (
        config.get("model_type") != "minicpm5_moe"
        or quant.get("quant_method") != "compressed-tensors"
        or quant.get("format") != "pack-quantized"
        or len(groups) != 1
        or groups[0].get("input_activations") is not None
        or groups[0].get("weights", {}).get("actorder") not in (None, False)
        or any(
            groups[0]["weights"].get(key) != value for key, value in expected.items()
        )
    ):
        raise ValueError("Expected MiniCPM5 CT weight-only asymmetric INT4 g128.")
    result = dict(config)
    result.pop("quantization_config")
    result.pop("auto_map", None)
    result["quantization"] = {"bits": BITS, "group_size": GROUP_SIZE, "mode": "affine"}
    return result


def repack(source, output):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    config = converted_config(json.loads((source / "config.json").read_text()))
    index = json.loads((source / "model.safetensors.index.json").read_text())
    by_shard = defaultdict(set)
    for key, shard in index["weight_map"].items():
        if Path(shard).name != shard:
            raise ValueError(f"Invalid shard name {shard!r}.")
        by_shard[shard].add(key)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}."
    ) as directory:
        stage = Path(directory)
        weight_map = {}
        total_size = 0
        projections = 0
        for part, (shard, keys) in enumerate(sorted(by_shard.items()), 1):
            with mx.stream(mx.cpu):
                weights = mx.load(str(source / shard))
                if weights.keys() != keys:
                    raise ValueError(f"Weight index does not match {shard}.")
                weights, count = repack_shard(weights, config["n_routed_experts"])
                if weight_map.keys() & weights.keys():
                    raise ValueError(f"Duplicate MLX weight in {shard}.")
                name = f"model-{part:05d}-of-{len(by_shard):05d}.safetensors"
                mx.save_safetensors(
                    str(stage / name), weights, metadata={"format": "mlx"}
                )
                total_size += sum(weight.nbytes for weight in weights.values())
                weight_map.update(dict.fromkeys(weights, name))
                projections += count
                del weights
            print(f"{shard} -> {name} ({count} INT4 projections)", flush=True)
        if projections == 0:
            raise ValueError("No CT projections in checkpoint.")
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "chat_template.jinja",
        ):
            if (source / name).is_file():
                shutil.copyfile(source / name, stage / name)
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (stage / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
            )
            + "\n"
        )
        stage.rename(output)
    print(f"Converted {projections} INT4 projections to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repack(args.source, args.output)


if __name__ == "__main__":
    main()
