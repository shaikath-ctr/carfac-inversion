"""Phase 4 direct causal NAP-to-audio architectures and inference utilities.

Supports both AFHI ``arcfac``'s baseline causal dilated TCN (``architecture="tcn"``)
and our Phase 4 ``cochlea_hybrid`` model adding:
  - dual linear + log-compressed cochleotopic input representation for 30-95 dB SPL;
  - a causal 2D tonotopic stem preserving ERB channel neighborhood phase;
  - a strictly causal frame-rate (2 ms) dynamic AGC & envelope branch with FiLM
    conditioning + time-varying sub-clip gain modulation over >128 ms context;
  - a causal 4th-order 20 Hz Butterworth high-pass output stage preventing DC/infrasound drift.
"""

import dataclasses
import functools
import json
import pathlib
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from hp_acoustic.models import cochlea

from carfac_inversion import neural
from carfac_inversion import vocoder_losses


@dataclasses.dataclass(frozen=True)
class DirectInverterConfig:
  """Architecture and calibration configuration for :class:`CochleaDirectInverter`."""

  architecture: str = "cochlea_hybrid"
  width: int = 64
  kernel_size: int = 3
  num_blocks: int = 13
  coarse_dim: int = 96
  coarse_hop: int = 64
  coarse_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
  input_scale: float = 1.0
  output_scale: float = 1.0
  lookahead: int = 256
  output_highpass_hz: float = 20.0
  fs: float = 32000.0
  nap_log_scale: float = 0.02

  def __post_init__(self) -> None:
    """Validates architectural hyperparameters."""
    if self.architecture not in ("tcn", "cochlea_hybrid"):
      raise ValueError(
        f"architecture must be 'tcn' or 'cochlea_hybrid', got {self.architecture!r}."
      )
    if self.width < 1 or self.kernel_size < 1 or self.num_blocks < 1:
      raise ValueError("width, kernel_size, and num_blocks must be positive.")
    if self.lookahead < 0:
      raise ValueError(f"lookahead must be non-negative, got {self.lookahead}.")
    if self.coarse_hop < 1:
      raise ValueError(f"coarse_hop must be positive, got {self.coarse_hop}.")

  @property
  def receptive_field(self) -> int:
    """Temporal sample receptive field of the causal dilated TCN stack."""
    reach = (self.kernel_size - 1) * (2**self.num_blocks - 1)
    return 1 + reach + (self.kernel_size - 1)

  def to_json(self) -> str:
    """Serializes the configuration to a sorted JSON string."""
    data = dataclasses.asdict(self)
    data["coarse_dilations"] = list(self.coarse_dilations)
    return json.dumps(data, sort_keys=True)

  @classmethod
  def from_json(cls, payload: str) -> "DirectInverterConfig":
    """Deserializes a configuration saved via :meth:`to_json`."""
    data = json.loads(payload)
    if "coarse_dilations" in data and isinstance(
      data["coarse_dilations"], Sequence
    ):
      data["coarse_dilations"] = tuple(int(d) for d in data["coarse_dilations"])
    return cls(**data)


class _CausalConv1D(nn.Module):
  """Strictly left-padded 1D dilated convolution along axis 1."""

  features: int
  kernel_size: int = 3
  dilation: int = 1
  stride: int = 1
  feature_group_count: int = 1

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    pad = (self.kernel_size - 1) * self.dilation
    if pad > 0:
      x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
    return nn.Conv(
      features=self.features,
      kernel_size=(self.kernel_size,),
      strides=(self.stride,),
      kernel_dilation=(self.dilation,),
      feature_group_count=self.feature_group_count,
      padding="VALID",
    )(x)


def _tonotopic_basis(kernel_freq: int, n_freq: int) -> np.ndarray:
  """Builds static Toeplitz masks ``(kernel_freq, n_freq, n_freq)`` for 1D lowering."""
  pad_f = kernel_freq // 2
  basis = np.zeros((kernel_freq, n_freq, n_freq), dtype=np.float32)
  f_out = np.arange(n_freq)
  for m in range(kernel_freq):
    f_in = f_out + (m - pad_f)
    valid = (f_in >= 0) & (f_in < n_freq)
    basis[m, f_in[valid], f_out[valid]] = 1.0
  return basis


def _causal_tonotopic_conv(
  module: nn.Module,
  x_flat: jnp.ndarray,
  *,
  name: str,
  n_freq: int,
  in_ch: int,
  out_ch: int,
  kernel_time: int,
  kernel_freq: int,
) -> jnp.ndarray:
  """Computes local 2D (time x freq) conv via 1D conv to avoid TPU 2D padding."""
  w = module.param(
    f"{name}_kernel",
    nn.initializers.lecun_normal(),
    (kernel_time, kernel_freq, in_ch, out_ch),
  )
  b = module.param(f"{name}_bias", nn.initializers.zeros_init(), (out_ch,))
  basis = jnp.asarray(_tonotopic_basis(kernel_freq, n_freq))
  k_1d = jnp.einsum("mfg,tmio->tfigo", basis, w).reshape(
    (kernel_time, n_freq * in_ch, n_freq * out_ch)
  )
  if kernel_time > 1:
    x_flat = jnp.pad(x_flat, ((0, 0), (kernel_time - 1, 0), (0, 0)))
  y = jax.lax.conv_general_dilated(
    x_flat,
    k_1d,
    window_strides=(1,),
    padding="VALID",
    dimension_numbers=("NWC", "WIO", "NWC"),
  )
  return y + jnp.tile(b, n_freq)[None, None, :]


class _CausalTonotopicStem(nn.Module):
  """Causal 2D (time x channel) front-end preserving CF neighborhood phase."""

  width: int
  kernel_time: int = 3
  kernel_freq: int = 5
  hidden: int = 16

  @nn.compact
  def __call__(self, nap_feats: jnp.ndarray) -> jnp.ndarray:
    # nap_feats: (batch, samples, channels, feat_ch)
    b, t, n_freq, in_ch = nap_feats.shape
    x_flat = nap_feats.reshape((b, t, n_freq * in_ch))
    h = _causal_tonotopic_conv(
      self,
      x_flat,
      name="conv1",
      n_freq=n_freq,
      in_ch=in_ch,
      out_ch=self.hidden,
      kernel_time=self.kernel_time,
      kernel_freq=self.kernel_freq,
    )
    h = nn.gelu(h)
    h = _causal_tonotopic_conv(
      self,
      h,
      name="conv2",
      n_freq=n_freq,
      in_ch=self.hidden,
      out_ch=self.hidden,
      kernel_time=1,
      kernel_freq=self.kernel_freq,
    )
    h = nn.gelu(h)
    return nn.Dense(self.width)(h)


class _CausalTcnResidualBlock(nn.Module):
  """Checkpointed causal dilated residual block for the baseline TCN."""

  width: int
  kernel_size: int
  dilation: int

  @nn.compact
  def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
    y = _CausalConv1D(self.width, self.kernel_size, dilation=self.dilation)(h)
    y = nn.Dense(self.width)(nn.gelu(y))
    return h + y


class _HybridConditionedBlock(nn.Module):
  """Checkpointed FiLM-modulated causal dilated residual block for cochlea_hybrid."""

  width: int
  kernel_size: int
  dilation: int

  @nn.compact
  def __call__(
    self,
    h: jnp.ndarray,
    film_mul: jnp.ndarray,
    film_shift: jnp.ndarray,
  ) -> jnp.ndarray:
    d_out = _CausalConv1D(self.width, self.kernel_size, dilation=self.dilation)(
      h
    )
    d_mod = d_out * film_mul + film_shift
    return h + nn.Dense(self.width)(nn.gelu(d_mod))


_RematTonotopicStem = cast(
  type[_CausalTonotopicStem], nn.remat(_CausalTonotopicStem)
)
_RematTcnBlock = cast(
  type[_CausalTcnResidualBlock], nn.remat(_CausalTcnResidualBlock)
)
_RematHybridBlock = cast(
  type[_HybridConditionedBlock], nn.remat(_HybridConditionedBlock)
)


class CochleaDirectInverter(nn.Module):
  """Phase 4 causal direct NAP-to-audio inverter supporting multiple backbones."""

  config: DirectInverterConfig = dataclasses.field(
    default_factory=DirectInverterConfig
  )

  @nn.compact
  def __call__(self, nap: jnp.ndarray) -> jnp.ndarray:
    """Inverts centered/raw cochleotopic NAP ``(batch, samples, ch)`` to audio."""
    cfg = self.config
    nap_scaled = nap / max(float(cfg.input_scale), 1e-8)

    if cfg.architecture == "tcn":
      h = _CausalConv1D(cfg.width, cfg.kernel_size)(nap_scaled)
      for block in range(cfg.num_blocks):
        h = _RematTcnBlock(
          width=cfg.width, kernel_size=cfg.kernel_size, dilation=2**block
        )(h)
      y = nn.Dense(1)(nn.gelu(h))[..., 0] * float(cfg.output_scale)
      if cfg.output_highpass_hz > 0.0:
        y = vocoder_losses.causal_highpass(
          y, cutoff_hz=cfg.output_highpass_hz, fs=cfg.fs
        )
      return y

    # Phase 4 cochlea_hybrid:
    # 1) Dual linear + log-compressed input so 30-95 dB SPL stay well-scaled.
    t_len = nap.shape[1]
    nap_log = jnp.sign(nap) * jnp.log1p(jnp.abs(nap) / cfg.nap_log_scale)
    nap_dual = jnp.stack([nap_scaled, nap_log], axis=-1)

    h_stem = _RematTonotopicStem(width=cfg.width)(nap_dual)

    # 2) Strictly causal coarse frame-rate AGC / envelope branch (2 ms @ 32 kHz).
    #    Frame f looks only at sample <= f * hop <= t for t in [f*hop, (f+1)*hop).
    hop = cfg.coarse_hop
    win = 2 * hop
    x_coarse_in = nn.gelu(
      nn.Dense(cfg.coarse_dim)(jnp.concatenate([nap_scaled, nap_log], axis=-1))
    )
    x_coarse_pad = jnp.pad(x_coarse_in, ((0, 0), (win - 1, 0), (0, 0)))
    c_feat = nn.Conv(
      features=cfg.coarse_dim,
      kernel_size=(win,),
      strides=(hop,),
      feature_group_count=cfg.coarse_dim,
      padding="VALID",
    )(x_coarse_pad)
    c_feat = nn.gelu(nn.Dense(cfg.coarse_dim)(c_feat))
    for dil in cfg.coarse_dilations:
      c_res = c_feat
      c_y = nn.LayerNorm(epsilon=1e-5)(c_feat)
      c_y = _CausalConv1D(
        cfg.coarse_dim,
        kernel_size=5,
        dilation=dil,
        feature_group_count=cfg.coarse_dim,
      )(c_y)
      c_y = nn.Dense(2 * cfg.coarse_dim)(nn.gelu(c_y))
      gate_a, gate_b = jnp.split(c_y, 2, axis=-1)
      c_y = gate_a * jax.nn.sigmoid(gate_b)
      c_y = nn.Dense(cfg.coarse_dim, kernel_init=nn.initializers.zeros_init())(
        c_y
      )
      c_feat = c_res + c_y
    c_norm = nn.LayerNorm(epsilon=1e-5)(c_feat)

    # Strictly causal linear interpolation between c_norm[f - 1] and c_norm[f],
    # both ending at <= t, eliminating block edges without peeking ahead.
    n_frames = c_norm.shape[1]
    c_prev = jnp.pad(c_norm[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    t_idx = jnp.arange(t_len)
    f_idx = jnp.minimum(t_idx // hop, n_frames - 1)
    frac = ((t_idx % hop).astype(jnp.float32) / float(hop))[None, :, None]
    c_up = (1.0 - frac) * c_prev[:, f_idx, :] + frac * c_norm[:, f_idx, :]

    # Dynamic time-varying gain head centered around 1.0 at init.
    log_gain_t = nn.Dense(
      1,
      kernel_init=nn.initializers.zeros_init(),
      bias_init=nn.initializers.zeros_init(),
    )(c_up)[..., 0]
    gain_t = jnp.exp(jnp.clip(log_gain_t, -2.5, 2.5))

    # 3) Sample-rate causal dilated TCN conditioned dynamically by coarse AGC branch.
    film_proj = nn.Dense(2 * cfg.width)(c_up)
    film_scale, film_shift = jnp.split(film_proj, 2, axis=-1)
    film_mul = 1.0 + 0.5 * jnp.tanh(film_scale)

    h = h_stem
    for block in range(cfg.num_blocks):
      h = _RematHybridBlock(
        width=cfg.width, kernel_size=cfg.kernel_size, dilation=2**block
      )(h, film_mul, film_shift)

    y_fine = nn.Dense(1)(nn.gelu(h))[..., 0]
    y_coarse = nn.Dense(
      1,
      kernel_init=nn.initializers.zeros_init(),
      bias_init=nn.initializers.zeros_init(),
    )(c_up)[..., 0]
    y = (y_fine + y_coarse) * gain_t * float(cfg.output_scale)
    if cfg.output_highpass_hz > 0.0:
      y = vocoder_losses.causal_highpass(
        y, cutoff_hz=cfg.output_highpass_hz, fs=cfg.fs
      )
    return y


def save_direct_checkpoint(
  path: str | pathlib.Path,
  params: Mapping[str, Any],
  config: DirectInverterConfig,
) -> None:
  """Saves Phase 4 ``params`` and ``config`` into a compressed ``.npz`` file."""
  out_path = pathlib.Path(path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  flat = traverse_util.flatten_dict(params, sep="/")
  payload: dict[str, np.ndarray] = {
    k: np.asarray(v, dtype=np.float32) for k, v in flat.items()
  }
  payload["__config_json__"] = np.asarray(config.to_json())
  np.savez_compressed(out_path, **cast(dict[str, Any], payload))


@functools.lru_cache(maxsize=8)
def _load_direct_cached(
  resolved_path: str, mtime_ns: int
) -> tuple[CochleaDirectInverter, dict[str, Any], DirectInverterConfig]:
  del mtime_ns
  with np.load(resolved_path, allow_pickle=False) as data:
    config = DirectInverterConfig.from_json(str(data["__config_json__"].item()))
    flat = {
      k: jnp.asarray(data[k], dtype=jnp.float32)
      for k in data.files
      if k != "__config_json__"
    }
  params = traverse_util.unflatten_dict(flat, sep="/")
  return CochleaDirectInverter(config=config), params, config


def load_direct_checkpoint(
  path: str | pathlib.Path,
) -> tuple[CochleaDirectInverter, dict[str, Any], DirectInverterConfig]:
  """Loads a saved Phase 4 ``.npz`` checkpoint, caching by resolved path."""
  p = pathlib.Path(path).resolve()
  if not p.is_file():
    raise FileNotFoundError(f"Phase 4 checkpoint not found: {path}")
  return _load_direct_cached(str(p), p.stat().st_mtime_ns)


@functools.partial(jax.jit, static_argnames=("model",))
def _apply_direct_jit(
  model: CochleaDirectInverter,
  params: Mapping[str, Any],
  nap: jnp.ndarray,
) -> jnp.ndarray:
  return cast(jnp.ndarray, model.apply(params, nap))


def invert_nap_direct(
  target_nap: np.ndarray | jnp.ndarray,
  checkpoint_path: str | pathlib.Path,
  *,
  align_lookahead: bool = True,
  cochlea_model: cochlea.CochleaModel | None = None,
) -> np.ndarray:
  """Inverts ``target_nap`` directly in a single causal pass using Phase 4 weights.

  Args:
    target_nap: Input cochleotopic NAP ``(samples, channels)`` or
      ``(samples, channels, ears)``.
    checkpoint_path: Saved Phase 4 ``.npz`` checkpoint.
    align_lookahead: When ``True``, pads input by ``config.lookahead`` trailing
      resting/zero samples and trims the leading ``lookahead`` samples so the
      returned waveform aligns sample-for-sample with the target timeline.
    cochlea_model: Optional cochlea used strictly to validate channel/rate
      alignment or fill trailing resting NAP padding when aligning lookahead.

  Returns:
    Reconstructed audio waveform matching input time length and ear count.
  """
  model, params, cfg = load_direct_checkpoint(checkpoint_path)
  if (
    cochlea_model is not None
    and abs(neural.cochlea_sample_rate(cochlea_model) - cfg.fs) > 1.0
  ):
    raise ValueError(
      f"Cochlea sample rate {neural.cochlea_sample_rate(cochlea_model)} Hz "
      f"does not match checkpoint sample rate {cfg.fs} Hz."
    )
  nap_np = np.asarray(target_nap, dtype=np.float32)
  is_2d = nap_np.ndim == 2
  nap_3d = nap_np[:, :, None] if is_2d else nap_np
  if nap_3d.ndim != 3:
    raise ValueError(f"target_nap must be 2D or 3D, got shape {nap_np.shape}.")
  t_len, _, _ = nap_3d.shape
  batch_nap = np.transpose(nap_3d, (2, 0, 1))
  if align_lookahead and cfg.lookahead > 0:
    tail = np.repeat(batch_nap[:, -1:, :], cfg.lookahead, axis=1)
    batch_in = np.concatenate([batch_nap, tail], axis=1)
  else:
    batch_in = batch_nap
  y_batch = np.asarray(
    _apply_direct_jit(model, params, jnp.asarray(batch_in, dtype=jnp.float32)),
    dtype=np.float32,
  )
  start = cfg.lookahead if align_lookahead and cfg.lookahead > 0 else 0
  y_aligned = y_batch[:, start : start + t_len]
  out_2d = np.transpose(y_aligned, (1, 0))
  return out_2d[:, 0] if is_2d else out_2d
