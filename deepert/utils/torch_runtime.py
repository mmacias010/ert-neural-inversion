"""Torch-first array helpers used throughout Deepert."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math
from typing import Any
import warnings

import numpy as np
import torch

Array = torch.Tensor


def _ensure_tensor(values: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if dtype is not None and values.dtype != dtype:
            return values.to(dtype=dtype)
        return values
    if isinstance(values, np.ndarray) and not values.flags.writeable:
        values = values.copy()
    return torch.as_tensor(values, dtype=dtype)


def _tensor_astype(self: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return self.to(dtype=dtype)


def _tensor_copy(self: torch.Tensor) -> torch.Tensor:
    return self.clone()


class _AtUpdate:
    def __init__(self, tensor: torch.Tensor, index: Any):
        self.tensor = tensor
        self.index = index

    @staticmethod
    def _index_tensor(index: Any, *, device: torch.device) -> torch.Tensor:
        return _ensure_tensor(index, dtype=torch.long).to(device=device)

    def add(self, values: Any) -> torch.Tensor:
        result = self.tensor.clone()
        update = _ensure_tensor(values, dtype=result.dtype).to(device=result.device)
        index = self.index
        if isinstance(index, tuple) and index and index[0] == slice(None):
            if len(index) < 2:
                raise IndexError("missing indexed dimension for at.add")
            dim_index = self._index_tensor(index[1], device=result.device).reshape(-1)
            return result.index_add_(1, dim_index, update)
        if isinstance(index, tuple):
            result[index] += update
            return result
        dim_index = self._index_tensor(index, device=result.device).reshape(-1)
        return result.index_add_(0, dim_index, update.reshape(dim_index.shape[0], *result.shape[1:]))

    def set(self, values: Any) -> torch.Tensor:
        result = self.tensor.clone()
        result[self.index] = _ensure_tensor(values, dtype=result.dtype).to(device=result.device)
        return result


class _AtIndexer:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, index: Any) -> _AtUpdate:
        return _AtUpdate(self.tensor, index)


def _tensor_at(self: torch.Tensor) -> _AtIndexer:
    return _AtIndexer(self)


if not hasattr(torch.Tensor, "astype"):
    torch.Tensor.astype = _tensor_astype  # type: ignore[attr-defined]
if not hasattr(torch.Tensor, "copy"):
    torch.Tensor.copy = _tensor_copy  # type: ignore[attr-defined]
if not hasattr(torch.Tensor, "at"):
    torch.Tensor.at = property(_tensor_at)  # type: ignore[attr-defined]



def to_numpy(values: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Convert Torch tensors or array-likes to NumPy arrays."""

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
        if dtype is not None:
            return np.asarray(array, dtype=dtype)
        return array
    return np.asarray(values, dtype=dtype)


def block_until_ready(values: Any) -> Any:
    """Synchronize CUDA tensors in a nested value and return the original value."""

    if isinstance(values, torch.Tensor):
        if values.is_cuda:
            torch.cuda.synchronize(values.device)
        return values
    if isinstance(values, dict):
        for item in values.values():
            block_until_ready(item)
        return values
    if isinstance(values, (tuple, list)):
        for item in values:
            block_until_ready(item)
        return values
    return values


class _Config:
    @property
    def torch_enable_float64(self) -> bool:
        return torch.get_default_dtype() == torch.float64

    def update(self, key: str, value: Any) -> None:
        if key == "torch_enable_float64":
            torch.set_default_dtype(torch.float64 if bool(value) else torch.float32)
            return
        raise ValueError(f"unsupported Torch runtime config key: {key}")


class _Dlpack:
    @staticmethod
    def from_dlpack(values: Any) -> torch.Tensor:
        return torch.from_dlpack(values)


class _MapNamespace:
    @staticmethod
    def map(function: Callable[[tuple[torch.Tensor, ...]], torch.Tensor], xs: Iterable[torch.Tensor]) -> torch.Tensor:
        arrays = tuple(xs)
        if not arrays:
            raise ValueError("torch_runtime.map requires at least one input")
        outputs = [function(tuple(array[index] for array in arrays)) for index in range(int(arrays[0].shape[0]))]
        return torch.stack(outputs, dim=0)


class _TorchRuntime:
    config = _Config()
    dlpack = _Dlpack()
    _map = _MapNamespace()

    @staticmethod
    def map(function: Callable[[tuple[torch.Tensor, ...]], torch.Tensor], xs: Iterable[torch.Tensor]) -> torch.Tensor:
        return _TorchRuntime._map.map(function, xs)

    @staticmethod
    def block_until_ready(values: Any) -> Any:
        return block_until_ready(values)

    @staticmethod
    def device_get(values: Any) -> Any:
        return to_numpy(values) if isinstance(values, torch.Tensor) else values


class _LinalgNamespace:
    @staticmethod
    def inv(values: Any) -> torch.Tensor:
        return torch.linalg.inv(_ensure_tensor(values))

    @staticmethod
    def norm(values: Any, axis: int | tuple[int, ...] | None = None, **kwargs: Any) -> torch.Tensor:
        return torch.linalg.norm(_ensure_tensor(values), dim=axis, **kwargs)


class _TorchArrayNamespace:
    float64 = torch.float64
    float32 = torch.float32
    int32 = torch.int32
    pi = math.pi
    linalg = _LinalgNamespace()

    @staticmethod
    def asarray(values: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
        return _ensure_tensor(values, dtype=dtype)

    array = asarray

    @staticmethod
    def zeros(shape: tuple[int, ...] | int, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape: tuple[int, ...] | int, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype)

    @staticmethod
    def empty(shape: tuple[int, ...] | int, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype)

    @staticmethod
    def arange(*args: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.arange(*args, dtype=dtype)

    @staticmethod
    def eye(n: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.eye(n, dtype=dtype)

    @staticmethod
    def concatenate(values: Iterable[Any], axis: int = 0) -> torch.Tensor:
        return torch.cat(tuple(_ensure_tensor(value) for value in values), dim=axis)

    @staticmethod
    def stack(values: Iterable[Any], axis: int = 0) -> torch.Tensor:
        return torch.stack(tuple(_ensure_tensor(value) for value in values), dim=axis)

    @staticmethod
    def sort(values: Any, axis: int = -1) -> torch.Tensor:
        return torch.sort(_ensure_tensor(values), dim=axis).values

    @staticmethod
    def unique(values: Any, axis: int | None = None, return_counts: bool = False):
        return torch.unique(_ensure_tensor(values), dim=axis, return_counts=return_counts)

    @staticmethod
    def diff(values: Any, axis: int = -1) -> torch.Tensor:
        return torch.diff(_ensure_tensor(values), dim=axis)

    @staticmethod
    def broadcast_to(values: Any, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.broadcast_to(_ensure_tensor(values), shape)

    @staticmethod
    def mean(values: Any, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
        return torch.mean(_ensure_tensor(values), dim=axis)

    @staticmethod
    def sum(values: Any, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
        return torch.sum(_ensure_tensor(values), dim=axis)

    @staticmethod
    def any(values: Any, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
        return torch.any(_ensure_tensor(values), dim=axis)

    @staticmethod
    def all(values: Any, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
        return torch.all(_ensure_tensor(values), dim=axis)

    @staticmethod
    def where(condition: Any, x: Any, y: Any) -> torch.Tensor:
        return torch.where(_ensure_tensor(condition), _ensure_tensor(x), _ensure_tensor(y))

    @staticmethod
    def maximum(x: Any, y: Any) -> torch.Tensor:
        left = _ensure_tensor(x)
        return torch.maximum(left, _ensure_tensor(y, dtype=left.dtype))

    @staticmethod
    def take(values: Any, indices: Any, axis: int | None = None) -> torch.Tensor:
        array = _ensure_tensor(values)
        index = _ensure_tensor(indices, dtype=torch.long)
        if axis is None:
            return torch.take(array, index)
        if index.ndim != 1:
            return torch.index_select(array, dim=axis, index=index.reshape(-1)).reshape(
                *array.shape[:axis],
                *index.shape,
                *array.shape[axis + 1 :],
            )
        return torch.index_select(array, dim=axis, index=index)

    @staticmethod
    def einsum(equation: str, *operands: Any) -> torch.Tensor:
        return torch.einsum(equation, *(_ensure_tensor(operand) for operand in operands))

    @staticmethod
    def tensordot(a: Any, b: Any, axes: int | tuple[Any, Any] = 2) -> torch.Tensor:
        dims = axes
        if isinstance(axes, tuple) and len(axes) == 2 and all(isinstance(axis, int) for axis in axes):
            dims = ((axes[0],), (axes[1],))
        return torch.tensordot(_ensure_tensor(a), _ensure_tensor(b), dims=dims)

    abs = staticmethod(torch.abs)
    exp = staticmethod(torch.exp)
    isfinite = staticmethod(torch.isfinite)
    log = staticmethod(torch.log)
    sign = staticmethod(torch.sign)
    sqrt = staticmethod(torch.sqrt)
    square = staticmethod(torch.square)


class BCOO:
    """Minimal BCOO-like wrapper backed by a coalesced Torch COO tensor."""

    def __init__(self, args: tuple[Any, Any], *, shape: tuple[int, int], unique_indices: bool = False):
        data, indices = args
        self.data = _ensure_tensor(data)
        self.indices = _ensure_tensor(indices, dtype=torch.long)
        self.shape = shape
        self.unique_indices = unique_indices

    def sum_duplicates(self) -> "BCOO":
        tensor = torch.sparse_coo_tensor(self.indices.T, self.data, self.shape).coalesce()
        result = BCOO((tensor.values(), tensor.indices().T), shape=self.shape, unique_indices=True)
        result._tensor = tensor
        return result

    def todense(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(self.indices.T, self.data, self.shape).coalesce().to_dense()


class CSR:
    """Minimal CSR wrapper with matrix multiplication support."""

    def __init__(self, args: tuple[Any, Any, Any], *, shape: tuple[int, int]):
        data, indices, indptr = args
        self.data = _ensure_tensor(data)
        self.indices = _ensure_tensor(indices, dtype=torch.long)
        self.indptr = _ensure_tensor(indptr, dtype=torch.long)
        self.shape = shape

    def _tensor(self) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
            warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled.*")
            return torch.sparse_csr_tensor(self.indptr, self.indices, self.data, size=self.shape)

    def __matmul__(self, other: Any) -> torch.Tensor:
        return torch.matmul(self._tensor(), _ensure_tensor(other))


torch_np = _TorchArrayNamespace()
torch_runtime = _TorchRuntime()

__all__ = [
    "Array",
    "BCOO",
    "CSR",
    "block_until_ready",
    "torch_np",
    "to_numpy",
    "torch_runtime",
]
