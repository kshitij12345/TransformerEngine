# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure-Python, torch.compile-traceable quantized-tensor allocation.

``make_empty_traceable`` is the data-free counterpart of ``Quantizer.make_empty``:
it builds an (uninitialized) quantized tensor -- or, under ``FakeTensorMode`` /
``register_fake``, a fake one -- purely from the quantizer's Python primitives
(``alloc_tensors`` / ``create_metadata`` / the storage's ``__tensor_unflatten__``),
with no C++ kernel involved. A fake impl calls it directly on its real (fake)
inputs, instead of converting them to a separate descriptor first, so the fake
and real impls share the same attribute-access code
(``inp.is_quantized``, ``weight._quantizer``, ...).

Why the result stashes ``_te_flat_names`` / ``_te_flat_ctx``
--------------------------------------------------------------
``forward_fn`` (in ``custom_op.py``) runs inside torch.compile's trace. It needs
the flat inner-tensor names and unflatten context to decode the custom op's flat
``Tensor[]`` return back into a structured quantized tensor. Calling
``__tensor_flatten__()`` for this would graph-break (it returns non-Tensor
Python objects -- a name list and a dict -- that Dynamo cannot represent as
graph nodes). Stashing them as plain attributes sidesteps that: Dynamo treats
non-callable attributes on a traceable wrapper subclass as constant metadata,
so ``forward_fn`` reads them without calling any method.

Allocation cost: when ``forward_fn`` calls the fake impl to obtain these
templates, the ``torch.empty`` calls appear as nodes in the initial Dynamo FX
graph. Because the tensors themselves are never used (only the stashed metadata
is read), AOT Autograd's dead-code elimination removes them before any kernel
code is generated -- they do not appear in the final compiled graph.
"""

from __future__ import annotations
import copy as _copy
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch._prims_common import make_contiguous_strides_for


def make_empty_traceable(
    quantizer: Optional[Any],
    shape: Tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: bool = False,
) -> Any:
    """Allocate a tensor purely in Python (traceable under torch.compile).

    When ``quantizer`` is not ``None``, produces a quantized tensor via
    ``quantizer.alloc_tensors`` + ``create_metadata`` + the storage's
    ``__tensor_unflatten__`` -- the compile-friendly equivalent of
    ``Quantizer.make_empty``. The quantizer is copied first (mirroring
    ``Quantizer.make_empty``), so the caller's instance is never mutated by
    allocation internals.

    When ``quantizer`` is ``None``, falls back to a plain ``torch.empty``.
    """
    device = torch.device(device) if device is not None else torch.device("cuda")
    shape = tuple(shape)
    if quantizer is None:
        # ``requires_grad=True`` passed directly to a tensor-creation function
        # is unsupported under Dynamo tracing; set it after the fact instead
        # (mirroring the quantized branch below).
        result = torch.empty(shape, dtype=dtype, device=device)
        if requires_grad:
            result.requires_grad_(True)
        return result

    # Copy so the caller's quantizer is not mutated by alloc_tensors internals.
    # The caller is expected to have already called set_usage() on the quantizer
    # before passing it here -- the copy captures the post-``set_usage`` state,
    # so the stashed _te_flat_names reflects the correct buffer layout.
    q = quantizer.copy() if hasattr(quantizer, "copy") else _copy.copy(quantizer)
    ctx = q.create_metadata(shape, dtype=dtype, requires_grad=requires_grad)
    inner = q.alloc_tensors(shape, device=device)
    storage_cls = ctx["cls"]
    result = storage_cls.__tensor_unflatten__(inner, ctx, shape, make_contiguous_strides_for(shape))
    if requires_grad and hasattr(result, "requires_grad_"):
        result.requires_grad_(True)
    result._te_flat_names = tuple(inner.keys())
    result._te_flat_ctx = ctx
    return result


# --------------------------------------------------------------------------- #
# Slot counting and reassembly for the custom-op flat Tensor[] protocol.
# --------------------------------------------------------------------------- #


def flat_slot_count(value: Any) -> int:
    """Number of flat ``Tensor[]`` slots ``value`` occupies in an op's return.

    Reads ``_te_flat_names`` stashed by :func:`make_empty_traceable`, which is
    safe to access at Dynamo trace time (treated as constant metadata on a
    traceable wrapper subclass). ``None`` and plain tensors (no stashed names)
    occupy exactly one slot.
    """
    if value is None:
        return 1
    names = getattr(value, "_te_flat_names", None)
    if names is not None:
        return len(names)
    return 1


def reassemble_from_flat(
    template: Optional[Any], chunk: Sequence[Optional[torch.Tensor]]
) -> Optional[Union[torch.Tensor, Any]]:
    """Rebuild a value from its flat inner tensors, using ``template`` for geometry.

    ``template`` is a value produced by (or passed through) :func:`make_empty_traceable`
    -- the fake impl's twin of the real output. Reassembly uses the stashed
    ``_te_flat_names`` / ``_te_flat_ctx`` attributes (trace-safe, unlike calling
    ``__tensor_flatten__``). A plain tensor / ``None`` template (no stashed
    metadata) takes its single chunk element directly.
    """
    if template is None:
        return None
    names = getattr(template, "_te_flat_names", None)
    ctx = getattr(template, "_te_flat_ctx", None)
    if names is None or ctx is None:
        return chunk[0]
    inner = dict(zip(names, chunk))
    shape = tuple(template.shape)
    return type(template).__tensor_unflatten__(
        inner, ctx, shape, make_contiguous_strides_for(shape)
    )
