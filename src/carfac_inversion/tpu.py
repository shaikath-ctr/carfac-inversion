"""TPU/MLCR runtime configuration, sharding, and multi-step scan utilities."""

import functools
import os
import pathlib
import queue
import threading
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from typing import TypeVar

import jax
import numpy as np
from absl import flags
from jax.experimental.compilation_cache import compilation_cache

_T = TypeVar("_T")
_P = TypeVar("_P")
_O = TypeVar("_O")
_B = TypeVar("_B")
_M = TypeVar("_M")

_VALID_MATMUL_PRECISIONS = ("default", "high", "highest", "float32")


def enable_mlcr_env(
  base_slice: str = "vlp:1x1",
  *,
  backend_target: str = "blade:mlcr-colab",
) -> None:
  """Exports environment variables for zero-code MLCR TPU auto-connection.

  Must be invoked before Python starts or before JAX initializes backends.

  Args:
    base_slice: MLCR virtual TPU topology string (e.g. ``'vlp:1x1'``,
      ``'vlp:2x4'``, ``'vf:2x2x4'``).
    backend_target: Pathways MLCR coordinator target.
  """
  os.environ["PATHWAYS_VIRTUAL_SLICES"] = base_slice
  os.environ["JAX_PLATFORMS"] = "mlcr,cpu"
  os.environ["JAX_BACKEND_TARGET"] = backend_target


def connect_mlcr_jax_config(
  base_slice: str = "vlp:1x1",
  *,
  backend_target: str = "blade:mlcr-colab",
) -> None:
  """Configures JAX config flags to attach to an MLCR shared TPU slice.

  Must be called prior to triggering ``jax.devices()`` or compiling XLA graphs.

  Args:
    base_slice: MLCR virtual TPU topology string.
    backend_target: Pathways MLCR coordinator target.
  """
  if "pathways_virtual_slices" in flags.FLAGS:
    flags.FLAGS.set_default("pathways_virtual_slices", base_slice)
  os.environ.setdefault("PATHWAYS_VIRTUAL_SLICES", base_slice)
  jax.config.update("jax_platforms", "mlcr,cpu")
  jax.config.update("jax_backend_target", backend_target)


def configure_runtime(
  *,
  compilation_cache_dir: str | None = None,
  matmul_precision: str = "high",
) -> tuple[str, int]:
  """Applies persistent XLA compilation cache and matmul precision settings.

  Keeps explicit ``float32`` arrays (including CARFAC states, FFTs, and losses)
  in ``float32`` while setting the default XLA matmul precision policy.

  Args:
    compilation_cache_dir: Optional directory (or GCS URI) for persistent XLA
      compiled executables across runs.
    matmul_precision: JAX default matmul precision policy; one of
      ``'default'``, ``'high'``, ``'highest'``, or ``'float32'``.

  Returns:
    ``(backend_platform, device_count)`` after configuration.

  Raises:
    ValueError: If ``matmul_precision`` is not recognized.
  """
  if matmul_precision not in _VALID_MATMUL_PRECISIONS:
    raise ValueError(
      f"Unsupported matmul_precision {matmul_precision!r}; expected one of "
      f"{_VALID_MATMUL_PRECISIONS}."
    )
  jax.config.update("jax_default_matmul_precision", matmul_precision)
  if compilation_cache_dir:
    if "://" not in compilation_cache_dir:
      pathlib.Path(compilation_cache_dir).mkdir(parents=True, exist_ok=True)
    compilation_cache.set_cache_dir(compilation_cache_dir)
  return str(jax.default_backend()), int(jax.device_count())


def make_data_mesh() -> jax.sharding.Mesh:
  """Builds a 1D ``('data',)`` device mesh across all available JAX devices."""
  return jax.make_mesh(
    (jax.device_count(),),
    ("data",),
    axis_types=(jax.sharding.AxisType.Auto,),
  )


def replicated_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
  """Returns a fully replicated ``NamedSharding`` on ``mesh``."""
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def batch_sharding(
  mesh: jax.sharding.Mesh,
  ndim: int,
  *,
  stacked: bool = False,
) -> jax.sharding.NamedSharding:
  """Returns data-parallel ``NamedSharding`` for unstacked or ``K``-stacked batches.

  Args:
    mesh: 1D data mesh built by :func:`make_data_mesh`.
    ndim: Rank of the global batch array.
    stacked: When ``True``, treats axis ``0`` as loop step ``K`` (replicated)
      and shards axis ``1`` across ``'data'`` (`PartitionSpec(None, 'data')`).
      When ``False``, shards axis ``0`` across ``'data'``.

  Raises:
    ValueError: If ``ndim`` is too small for the requested layout.
  """
  min_ndim = 2 if stacked else 1
  if ndim < min_ndim:
    raise ValueError(
      f"Expected array rank >= {min_ndim} for stacked={stacked}, got {ndim}."
    )
  spec = (
    jax.sharding.PartitionSpec(None, "data")
    if stacked
    else jax.sharding.PartitionSpec("data")
  )
  return jax.sharding.NamedSharding(mesh, spec)


def replicate_tree(tree: _T, mesh: jax.sharding.Mesh) -> _T:
  """Replicates every leaf of ``tree`` onto ``replicated_sharding(mesh)``."""
  rep = replicated_sharding(mesh)
  multi_process = jax.process_count() > 1

  def _place(leaf: Any) -> Any:
    if isinstance(leaf, jax.Array) and leaf.sharding == rep:
      return leaf
    if multi_process:
      return jax.make_array_from_process_local_data(rep, np.asarray(leaf))
    return jax.device_put(leaf, rep)

  return jax.tree.map(_place, tree)


def shard_batch(
  batch_tree: _T,
  mesh: jax.sharding.Mesh,
  *,
  stacked: bool = False,
) -> _T:
  """Shards host batch leaves across ``mesh`` along the batch dimension.

  Args:
    batch_tree: PyTree whose array leaves have shape ``[B_local, ...]`` (or
      ``[K, B_local, ...]`` when ``stacked=True``).
    mesh: 1D ``('data',)`` device mesh.
    stacked: Whether leaves have leading multi-step dimension ``K``.

  Returns:
    PyTree of device-backed ``jax.Array``s sharded across ``'data'``.

  Raises:
    ValueError: If any leaf has insufficient rank or batch size not divisible
      by ``len(mesh.local_devices)``.
  """
  batch_axis = 1 if stacked else 0
  n_local = max(1, len(mesh.local_devices))

  def _place(leaf: Any) -> jax.Array:
    arr = np.asarray(leaf)
    min_ndim = 2 if stacked else 1
    if arr.ndim < min_ndim:
      raise ValueError(
        f"Batch leaf rank {arr.ndim} is smaller than required {min_ndim}."
      )
    local_b = int(arr.shape[batch_axis])
    if local_b <= 0 or local_b % n_local != 0:
      raise ValueError(
        f"Local batch size {local_b} (axis {batch_axis} of shape {arr.shape}) "
        f"must be a positive multiple of local device count {n_local}."
      )
    sharding = batch_sharding(mesh, arr.ndim, stacked=stacked)
    return jax.make_array_from_process_local_data(sharding, arr)

  return jax.tree.map(_place, batch_tree)


def to_host_tree(tree: _T) -> _T:
  """Copies every leaf of a device PyTree into detached host NumPy arrays."""
  return jax.tree.map(np.asarray, tree)


def delete_tree(tree: Any) -> None:
  """Deletes live ``jax.Array`` device buffers inside ``tree`` immediately."""

  def _free(leaf: Any) -> None:
    if isinstance(leaf, jax.Array) and not leaf.is_deleted():
      leaf.delete()

  jax.tree.map(_free, tree)


class ThreadedDevicePrefetcher(Iterator[_T]):
  """Prefetches, stacks ``K`` batches, and transfers them to device memory.

  Offloads host batch sampling, ``K``-step leaf stacking (`[K, B, ...]`), and
  host-to-device sharded placement (`PartitionSpec(None, 'data')`) onto a
  bounded daemon thread so accelerator compute overlaps DMA transfers.
  Incomplete trailing chunks shorter than ``steps_per_loop`` are dropped cleanly
  to prevent end-of-epoch XLA recompilation.
  """

  def __init__(
    self,
    batch_iter: Iterator[_T],
    mesh: jax.sharding.Mesh,
    *,
    steps_per_loop: int = 1,
    max_prefetch: int = 2,
  ) -> None:
    """Starts background prefetching for ``batch_iter`` onto ``mesh``.

    Args:
      batch_iter: Host iterator yielding per-step PyTree batches `[B, ...]`.
      mesh: Target 1D ``('data',)`` mesh for sharded placement.
      steps_per_loop: Number of consecutive batches ``K`` to stack per device
        transfer. Values ``> 1`` produce ``[K, B, ...]`` stacked batches.
      max_prefetch: Maximum device-ready chunks buffered ahead in device memory.
    """
    if steps_per_loop < 1:
      raise ValueError(f"steps_per_loop must be >= 1, got {steps_per_loop}.")
    if max_prefetch < 1:
      raise ValueError(f"max_prefetch must be >= 1, got {max_prefetch}.")
    self._iter = batch_iter
    self._mesh = mesh
    self._steps_per_loop = steps_per_loop
    self._queue: queue.Queue[_T | None] = queue.Queue(maxsize=max_prefetch)
    self._stop_event = threading.Event()
    self._exhausted = False
    self._error: BaseException | None = None
    self._worker = threading.Thread(target=self._worker_loop, daemon=True)
    self._worker.start()

  def _put_with_stop(self, item: _T | None) -> bool:
    while not self._stop_event.is_set():
      try:
        self._queue.put(item, timeout=0.05)
        return True
      except queue.Full:
        continue
    return False

  def _worker_loop(self) -> None:
    try:
      while not self._stop_event.is_set():
        batches: list[_T] = []
        for _ in range(self._steps_per_loop):
          if self._stop_event.is_set():
            break
          try:
            batches.append(next(self._iter))
          except StopIteration:
            break
        if self._stop_event.is_set() or len(batches) < self._steps_per_loop:
          self._put_with_stop(None)
          break
        if self._steps_per_loop > 1:
          host_chunk = jax.tree.map(
            lambda *leaves: np.stack(leaves, axis=0), *batches
          )
          dev_chunk = shard_batch(host_chunk, self._mesh, stacked=True)
        else:
          dev_chunk = shard_batch(batches[0], self._mesh, stacked=False)
        if not self._put_with_stop(dev_chunk):
          break
    except BaseException as exc:
      self._error = exc
      self._put_with_stop(None)

  def close(self) -> None:
    """Signals the worker thread to stop and drains any buffered items."""
    self._stop_event.set()
    while not self._queue.empty():
      try:
        self._queue.get_nowait()
      except queue.Empty:
        break
    if self._worker.is_alive():
      self._worker.join(timeout=1.0)

  def __enter__(self) -> "ThreadedDevicePrefetcher[_T]":
    """Enters the prefetcher context manager."""
    return self

  def __exit__(self, *_: object) -> None:
    """Stops the background prefetcher thread on context exit."""
    self.close()

  def __iter__(self) -> "ThreadedDevicePrefetcher[_T]":
    """Returns self as an iterator over prefetched device chunks."""
    return self

  def __next__(self) -> _T:
    """Dequeues the next device-sharded chunk or raises ``StopIteration``."""
    if self._exhausted:
      if self._error is not None:
        raise self._error
      raise StopIteration
    item = self._queue.get()
    if item is None:
      self._exhausted = True
      if self._error is not None:
        raise self._error
      raise StopIteration
    return item


def make_multi_step_fns(
  step_impl: Callable[[_P, _O, _B], tuple[_P, _O, _M]],
  *,
  donate_argnums: tuple[int, ...] = (0, 1),
) -> tuple[
  Callable[[_P, _O, _B], tuple[_P, _O, _M]],
  Callable[[_P, _O, _B], tuple[_P, _O, _M]],
]:
  """Builds single-step and fused ``jax.lax.scan`` multi-step JIT functions.

  Both returned functions donate ``params`` and ``opt_state`` buffers
  (according to ``donate_argnums``) and return stacked metrics with leading step
  dimension ``[1, ...]`` or ``[K, ...]`` so callers consume them uniformly.

  Args:
    step_impl: Pure Python step ``(params, opt_state, batch) ->
      (next_params, next_opt_state, metrics)``.
    donate_argnums: Argument indices donated in-place on device.

  Returns:
    ``(single_step_fn, multi_step_fn)`` compiled with ``jax.jit``.
  """

  @functools.partial(jax.jit, donate_argnums=donate_argnums)
  def single_step_fn(
    params: _P,
    opt_state: _O,
    batch: _B,
  ) -> tuple[_P, _O, _M]:
    next_p, next_os, metrics = step_impl(params, opt_state, batch)
    return next_p, next_os, jax.tree.map(lambda x: x[None, ...], metrics)

  @functools.partial(jax.jit, donate_argnums=donate_argnums)
  def multi_step_fn(
    params: _P,
    opt_state: _O,
    stacked_batch: _B,
  ) -> tuple[_P, _O, _M]:
    def _scan_body(
      carry: tuple[_P, _O],
      batch_elem: _B,
    ) -> tuple[tuple[_P, _O], _M]:
      cur_p, cur_os = carry
      next_p, next_os, step_metrics = step_impl(cur_p, cur_os, batch_elem)
      return (next_p, next_os), step_metrics

    (final_p, final_os), stacked_metrics = jax.lax.scan(
      _scan_body,
      (params, opt_state),
      stacked_batch,
    )
    return final_p, final_os, stacked_metrics

  return single_step_fn, multi_step_fn
