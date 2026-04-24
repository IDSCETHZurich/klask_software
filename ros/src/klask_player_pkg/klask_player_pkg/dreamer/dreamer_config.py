"""Lightweight config dataclasses for Dreamer inference.

Mirrors the Hydra config structure used by r2dreamer but without any
Hydra/OmegaConf dependency.  Provides attribute-access config objects
that the network code in dreamer_network.py expects.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class MLPConfig:
    """Configuration for MLP encoder / decoder."""

    shape: Optional[list] = None
    layers: int = 3
    units: int = 512
    act: str = "SiLU"
    norm: bool = True
    device: str = "cpu"
    outscale: Optional[float] = None
    symlog_inputs: bool = True
    name: str = "mlp_encoder"


@dataclass
class CNNConfig:
    """Configuration for CNN encoder."""

    act: str = "SiLU"
    norm: bool = True
    kernel_size: int = 5
    minres: int = 4
    depth: int = 32
    mults: List[int] = field(default_factory=lambda: [2, 3, 4, 4])


@dataclass
class EncoderConfig:
    """Configuration for the MultiEncoder."""

    mlp_keys: str = "$^"  # regex matching nothing by default
    cnn_keys: str = "image"
    mlp: MLPConfig = field(default_factory=MLPConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)


@dataclass
class RSSMConfig:
    """Configuration for the Recurrent State Space Model."""

    stoch: int = 32
    deter: int = 4096
    hidden: int = 512
    discrete: int = 32
    img_layers: int = 2
    obs_layers: int = 1
    dyn_layers: int = 1
    blocks: int = 8
    act: str = "SiLU"
    norm: bool = True
    unimix_ratio: float = 0.01
    initial: str = "learned"
    device: str = "cpu"


@dataclass
class ActorDistConfig:
    """Distribution configuration for the actor head."""

    name: str = "bounded_normal"
    min_std: float = 0.1
    max_std: float = 1.0


@dataclass
class ActorConfig:
    """Configuration for the actor (policy) head."""

    shape: List[int] = field(default_factory=lambda: [2])
    layers: int = 3
    units: int = 512
    act: str = "SiLU"
    norm: bool = True
    device: str = "cpu"
    dist: ActorDistConfig = field(default_factory=ActorDistConfig)
    outscale: float = 0.01
    symlog_inputs: bool = False
    name: str = "actor"


@dataclass
class DreamerModelConfig:
    """Top-level model configuration for Dreamer inference."""

    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)

    @classmethod
    def default_size50m(cls, device: str = "cpu", obs_mode: str = "image") -> "DreamerModelConfig":
        """Create configuration matching the size50M training preset.

        Args:
            device: Torch device string.
            obs_mode: "image" (default) or "image_and_state".
        """
        mlp_keys = "policy" if obs_mode == "image_and_state" else "$^"

        return cls(
            rssm=RSSMConfig(
                stoch=32, deter=4096, hidden=512, discrete=32,
                img_layers=2, obs_layers=1, dyn_layers=1, blocks=8,
                act="SiLU", norm=True, unimix_ratio=0.01,
                initial="learned", device=device,
            ),
            encoder=EncoderConfig(
                mlp_keys=mlp_keys,
                cnn_keys="image",
                mlp=MLPConfig(
                    layers=3, units=512, act="SiLU", norm=True,
                    device=device, symlog_inputs=True, name="mlp_encoder",
                ),
                cnn=CNNConfig(
                    act="SiLU", norm=True, kernel_size=5,
                    minres=4, depth=32, mults=[2, 3, 4, 4],
                ),
            ),
            actor=ActorConfig(
                shape=[2], layers=3, units=512, act="SiLU", norm=True,
                device=device,
                dist=ActorDistConfig(name="bounded_normal", min_std=0.1, max_std=1.0),
                outscale=0.01, symlog_inputs=False, name="actor",
            ),
        )

    @classmethod
    def from_json(cls, path: str, device: str = "cpu") -> "DreamerModelConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        def _set_device(d, dev):
            if isinstance(d, dict):
                if "device" in d:
                    d["device"] = dev
                for v in d.values():
                    _set_device(v, dev)

        _set_device(data, device)

        return cls(
            rssm=RSSMConfig(**data.get("rssm", {})),
            encoder=EncoderConfig(
                mlp_keys=data.get("encoder", {}).get("mlp_keys", "$^"),
                cnn_keys=data.get("encoder", {}).get("cnn_keys", "image"),
                mlp=MLPConfig(**data.get("encoder", {}).get("mlp", {})),
                cnn=CNNConfig(**data.get("encoder", {}).get("cnn", {})),
            ),
            actor=ActorConfig(
                **{
                    k: (ActorDistConfig(**v) if k == "dist" else v)
                    for k, v in data.get("actor", {}).items()
                }
            ),
        )

    def to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
