"""Standalone Dreamer network definitions for inference.

Adapted from r2dreamer (third_party/r2dreamer/) with all dependencies inlined.
Only includes components needed for inference: encoder, RSSM, and actor.
No dependency on tensordict, einops, hydra, or omegaconf.
"""

import math
import re
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as torchd
from torch import nn
from torch.nn import init as nn_init


# ============================================================================
# Utility functions (from tools.py)
# ============================================================================


def to_f32(x):
    return x.to(dtype=torch.float32)


def rpad(x, pad):
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def weight_init_(m, fan_type="in"):
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return

    weight = getattr(m, "weight", None)
    if weight is None:
        return
    if weight.numel() == 0:
        return

    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)

    with torch.no_grad():
        fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


# ============================================================================
# Distributions (from distributions.py)
# ============================================================================


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        probs = F.softmax(to_f32(logits), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs, dtype=torch.float32) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self):
        _mode = F.one_hot(torch.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=(), temperature=1.0):
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError


def bounded_normal(x, min_std, max_std, **kwargs):
    mean, std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    dist = torchd.normal.Normal(torch.tanh(to_f32(mean)), to_f32(std))
    return torchd.independent.Independent(dist, 1)


# ============================================================================
# Network building blocks (from networks.py)
# ============================================================================


class LambdaLayer(nn.Module):
    """Wrap an arbitrary callable into an nn.Module."""

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BlockLinear(nn.Module):
    """Block-wise linear layer."""

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class Conv2dSamePad(nn.Conv2d):
    """A Conv2d layer that emulates TensorFlow's 'SAME' padding."""

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        i_div_s_ceil = (i + s - 1) // s
        return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over channel-last format applied to 4D tensors."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ============================================================================
# Encoder networks (from networks.py)
# ============================================================================


class ConvEncoder(nn.Module):
    def __init__(self, config, input_shape):
        super().__init__()
        act = getattr(torch.nn, config.act)
        h, w, input_ch = input_shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        self.kernel_size = int(config.kernel_size)
        in_dim = input_ch
        layers = []
        for i, depth in enumerate(self.depths):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=depth,
                    kernel_size=self.kernel_size,
                    stride=1,
                    bias=True,
                )
            )
            layers.append(nn.MaxPool2d(2, 2))
            if config.norm:
                layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth
            h, w = h // 2, w // 2

        self.out_dim = self.depths[-1] * h * w
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        """Encode image-like observations with a CNN."""
        # (B, T, H, W, C) or (B, H, W, C)
        obs = obs - 0.5
        x = obs.reshape(-1, *obs.shape[-3:])
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        return x.reshape(*obs.shape[:-3], x.shape[-1])


class MLP(nn.Module):
    def __init__(self, config, inp_dim):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._symlog_inputs = bool(config.symlog_inputs)
        self.layers = nn.Sequential()
        for i in range(config.layers):
            self.layers.add_module(f"{config.name}_linear{i}", nn.Linear(inp_dim, config.units, bias=True))
            self.layers.add_module(f"{config.name}_norm{i}", nn.RMSNorm(config.units, eps=1e-04, dtype=torch.float32))
            self.layers.add_module(f"{config.name}_act{i}", act())
            inp_dim = config.units
        self.out_dim = config.units

    def forward(self, x):
        if self._symlog_inputs:
            x = symlog(x)
        return self.layers(x)


class MLPHead(nn.Module):
    def __init__(self, config, inp_dim):
        super().__init__()
        self.mlp = MLP(config, inp_dim)
        self._dist_name = str(config.dist.name)
        self._outscale = float(config.outscale)

        # Resolve the distribution function
        _dist_map = {
            "bounded_normal": bounded_normal,
        }
        self._dist = _dist_map[self._dist_name]

        if self._dist_name == "bounded_normal":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0] * 2, bias=True)
            kwargs = {"min_std": float(config.dist.min_std), "max_std": float(config.dist.max_std)}
        else:
            raise NotImplementedError(f"Distribution '{self._dist_name}' not supported for inference")

        self._dist = partial(self._dist, **kwargs)

        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        if self._outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(self._outscale)

    def forward(self, x):
        """Produce a distribution head."""
        return self._dist(self.last(self.mlp(x)))


class MultiEncoder(nn.Module):
    def __init__(self, config, shapes):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded and not k.startswith("log_")}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}

        self.out_dim = 0
        self._cnn_keys = list(self.cnn_shapes.keys())
        self._mlp_keys = list(self.mlp_shapes.keys())
        self.encoders = nn.ModuleList()

        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self.encoders.append(ConvEncoder(config.cnn, input_shape))
            self.out_dim += self.encoders[-1].out_dim
        if self.mlp_shapes:
            inp_dim = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoders.append(MLP(config.mlp, inp_dim))
            self.out_dim += self.encoders[-1].out_dim

        self.apply(weight_init_)

    def forward(self, obs):
        """Encode a dict of observations."""
        outputs = []
        enc_idx = 0
        if self.cnn_shapes:
            cnn_input = torch.cat([obs[k] for k in self._cnn_keys], -1)
            outputs.append(self.encoders[enc_idx](cnn_input))
            enc_idx += 1
        if self.mlp_shapes:
            mlp_input = torch.cat([obs[k] for k in self._mlp_keys], -1)
            outputs.append(self.encoders[enc_idx](mlp_input))

        if len(outputs) > 1:
            return torch.cat(outputs, dim=-1)
        return outputs[0]


# ============================================================================
# RSSM — Recurrent State Space Model (from rssm.py)
# ============================================================================


class Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        """Deterministic state transition (block-GRU style)."""
        B = action.shape[0]
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)

        x = torch.cat([x0, x1, x2], -1)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))
        x = self._dyn_hid(x)
        x = self._dyn_gru(x)

        gates = torch.chunk(self.flat2group(x), 3, dim=-1)
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter

        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
        )

        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        self.apply(weight_init_)

    def initial(self, batch_size):
        """Return an initial latent state."""
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step."""
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        deter = self._deter_net(stoch, deter, prev_action)
        x = torch.cat([deter, embed], dim=-1)
        logit = self._obs_net(x)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)
