#!/usr/bin/env python3
"""Export a Dreamer training checkpoint to inference-only format.

Strips optimizer state, value/reward/cont/decoder heads, and frozen copies,
keeping only encoder, RSSM, and actor weights. Also exports the model config
as a JSON file for use by DreamerInference.

Usage:
    python export_dreamer_checkpoint.py \
        --checkpoint /path/to/latest.pt \
        --config /path/to/config.yaml \
        --output /path/to/output_dir/dreamer_inference.pt

This produces:
    - dreamer_inference.pt  (~150-200 MB vs ~552 MB full checkpoint)
    - dreamer_config.json   (model architecture config)
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Export Dreamer checkpoint for inference.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Hydra training config YAML (optional, for extracting model params)",
    )
    parser.add_argument("--output", type=str, required=True, help="Path for output inference checkpoint (.pt)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "agent_state_dict" not in checkpoint:
        print("Error: Checkpoint does not contain 'agent_state_dict' key.", file=sys.stderr)
        print(f"Available keys: {list(checkpoint.keys())}", file=sys.stderr)
        sys.exit(1)

    full_state_dict = checkpoint["agent_state_dict"]
    full_size = sum(v.numel() * v.element_size() for v in full_state_dict.values())

    # Filter to only inference-relevant components
    inference_prefixes = ("encoder.", "rssm.", "actor.")
    inference_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith(inference_prefixes)}
    inference_size = sum(v.numel() * v.element_size() for v in inference_state_dict.values())

    print(f"Full checkpoint: {len(full_state_dict)} keys, {full_size / 1e6:.1f} MB")
    print(f"Inference-only:  {len(inference_state_dict)} keys, {inference_size / 1e6:.1f} MB")
    print(f"Reduction:       {(1 - inference_size / full_size) * 100:.1f}%")

    # Save inference checkpoint
    torch.save(inference_state_dict, output_path)
    print(f"Saved inference checkpoint: {output_path}")

    # Export config if Hydra YAML is provided
    if args.config:
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(args.config)

            # Build config dict matching DreamerModelConfig structure
            model = cfg.get("model", cfg)
            env = cfg.get("env", {})

            # Extract training environment params for inference
            env_size = env.get("size", [64, 64])
            image_size = int(env_size[0]) if isinstance(env_size, (list, tuple)) else int(env_size)
            max_velocity = float(env.get("max_velocity", 0.6))

            # Derive obs_mode from encoder mlp_keys: "$^" means image-only
            encoder_mlp_keys = str(env.get("encoder", {}).get("mlp_keys", "$^"))
            obs_mode = "image" if encoder_mlp_keys == "$^" else "image_and_state"

            config_dict = {
                "image_size": image_size,
                "obs_mode": obs_mode,
                "max_velocity": max_velocity,
                "rssm": {
                    "stoch": int(model.get("rssm", {}).get("stoch", 32)),
                    "deter": int(model.get("deter", model.get("rssm", {}).get("deter", 4096))),
                    "hidden": int(model.get("hidden", model.get("rssm", {}).get("hidden", 512))),
                    "discrete": int(model.get("discrete", model.get("rssm", {}).get("discrete", 32))),
                    "img_layers": int(model.get("rssm", {}).get("img_layers", 2)),
                    "obs_layers": int(model.get("rssm", {}).get("obs_layers", 1)),
                    "dyn_layers": int(model.get("rssm", {}).get("dyn_layers", 1)),
                    "blocks": int(model.get("rssm", {}).get("blocks", 8)),
                    "act": str(model.get("act", "SiLU")),
                    "norm": True,
                    "unimix_ratio": float(model.get("rssm", {}).get("unimix_ratio", 0.01)),
                    "initial": "learned",
                },
                "encoder": {
                    "mlp_keys": str(cfg.get("env", {}).get("encoder", {}).get("mlp_keys", "$^")),
                    "cnn_keys": str(cfg.get("env", {}).get("encoder", {}).get("cnn_keys", "image")),
                    "mlp": {
                        "layers": 3,
                        "units": int(model.get("units", 512)),
                        "act": str(model.get("act", "SiLU")),
                        "norm": True,
                        "symlog_inputs": True,
                        "name": "mlp_encoder",
                    },
                    "cnn": {
                        "act": str(model.get("act", "SiLU")),
                        "norm": True,
                        "kernel_size": 5,
                        "minres": 4,
                        "depth": int(model.get("depth", 32)),
                        "mults": [2, 3, 4, 4],
                    },
                },
                "actor": {
                    "shape": [2],
                    "layers": 3,
                    "units": int(model.get("units", 512)),
                    "act": str(model.get("act", "SiLU")),
                    "norm": True,
                    "dist": {
                        "name": "bounded_normal",
                        "min_std": 0.1,
                        "max_std": 1.0,
                    },
                    "outscale": 0.01,
                    "symlog_inputs": False,
                    "name": "actor",
                },
            }

            config_path = output_dir / f"{checkpoint_path.stem}_config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            print(f"Saved config: {config_path}")
        except ImportError:
            print("Warning: omegaconf not installed, skipping config export.")
            print("You can manually create a dreamer_config.json or rely on default_size50m().")
    else:
        print("No --config provided, skipping config JSON export.")
        print("DreamerInference will use default_size50m() configuration.")


if __name__ == "__main__":
    main()
