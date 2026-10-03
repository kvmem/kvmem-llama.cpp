"""Grouped symmetric quantization used by NInfer artifact converters.

The persistent numeric format fixes the code range, group size, and binary16
scale.  Model-specific recipes decide which tensors use those formats; this
module only performs the registered numeric transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.artifact.layouts import (
    row_split_geometry,
)
from tools.artifact.formats import QuantFormat, get_format

_FP16_MIN_SUBNORMAL = 2.0**-24


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


def _canonical_scale_words(
    max_abs: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical binary16 scales and binary32 reciprocals on the host.

    CUDA division is not correctly rounded at every binary16 scale boundary.
    The host oracle performs the specified division in binary64, explicitly
    rounds through binary32 and binary16, then computes the reciprocal in the
    same way.  A binary32 input divided by these small integer denominators has
    enough binary64 precision for the final binary32 rounding to be exact.
    """

    host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32)
        scale = raw_scale.astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")

    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _mse_scale_candidates(
    s_hi: np.ndarray,
    s_ls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group candidate binary16 scales for the MSE-optimal scale search.

    ``s_hi`` is the canonical max-abs scale (index 0 of the result, so ties keep
    the ``grouped_absmax`` outcome) and ``s_ls`` the least-squares estimate from
    the max-abs codes.  Candidates are exact binary16 values: the two scales,
    power-of-two multiples around them, and ±2-ulp neighbourhoods in word space.
    Words outside the positive finite binary16 range (zero, NaN, negative) are
    marked invalid; a zero group therefore keeps only its zero candidate.
    """
    def words_of(values: np.ndarray) -> np.ndarray:
        return values.astype(np.float16).view(np.uint16)

    def scaled(words: np.ndarray, exponent: int) -> np.ndarray:
        # Bit-cast the binary16 words to their values, scale by the exact
        # power of two, and round back to binary16.  Multiples beyond the
        # binary16 range become infinity; those words are masked out below.
        values = words.view(np.float16).astype(np.float64) * (2.0**exponent)
        with np.errstate(over="ignore", invalid="ignore"):
            return words_of(values)

    def shifted(words: np.ndarray, offset: int) -> np.ndarray:
        return (words.astype(np.int32) + offset).astype(np.uint16)

    w_hi = words_of(s_hi)
    w_ls = words_of(s_ls)
    raw = np.stack(
        (
            w_hi,
            scaled(w_hi, -2),
            scaled(w_hi, -1),
            scaled(w_hi, 1),
            scaled(w_hi, 2),
            w_ls,
            scaled(w_ls, -1),
            scaled(w_ls, 1),
            shifted(w_hi, -2),
            shifted(w_hi, -1),
            shifted(w_hi, 1),
            shifted(w_hi, 2),
            shifted(w_ls, -2),
            shifted(w_ls, 2),
        ),
        axis=1,
    )
    # Word 0 (scale 0) stays a candidate: for an all-zero group it is the
    # index-0 tie-break keeping the grouped_absmax result, and for a nonzero
    # group its error is the full sum of squares, so it never wins.  Infinity
    # (0x7C00), NaN (0x7C01 and up) and negative words are excluded.
    valid = raw <= 0x7BFF
    raw = raw.view(np.float16).astype(np.float64)
    return raw, valid


def quantize_matrix_mse(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Grouped quantisation choosing the per-group scale that minimises the
    encoded squared error.

    Each group's scale is the best binary16 candidate (see
    ``_mse_scale_candidates``) under the exact engine encoding -- codes are
    ``round(w * binary32(1 / scale))`` clamped to the code range -- with the
    canonical max-abs scale breaking ties.  The search and code evaluation run
    on the host in ordered arithmetic, so results are bit-identical across
    devices; the stored codes and scales are exactly what the row-split codec
    and the runtime dequantisation consume.
    """
    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(
            f"grouped quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError("grouped quantization requires floating-point values")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    max_abs = grouped.abs().amax(dim=2)
    base_scales, base_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    base_codes = torch.clamp(
        torch.round(grouped * base_reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    )

    host_values = grouped.detach().cpu().numpy().astype(np.float64)
    host_max = max_abs.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    host_base_codes = base_codes.detach().cpu().numpy().astype(np.float64)

    flat_w = host_values.reshape(-1, spec.group_size)
    flat_base_scales = base_scales.detach().cpu().numpy().astype(np.float64).reshape(-1)
    flat_base_codes = host_base_codes.reshape(-1, spec.group_size)

    # Least-squares scale estimate: the scale fitting the max-abs code pattern.
    code_norm = (flat_base_codes**2).sum(axis=1)
    s_ls = np.where(
        code_norm > 0,
        (flat_w * flat_base_codes).sum(axis=1) / np.maximum(code_norm, 1e-300),
        flat_base_scales,
    )
    candidates, candidate_valid = _mse_scale_candidates(flat_base_scales, s_ls)

    # True encoded error per candidate, using the exact engine code formula.
    # A valid zero scale encodes every value to code 0 (reciprocal 0), and an
    # invalid word never competes, so no NaN can enter the search.
    values32 = flat_w.astype(np.float32)
    best_error = np.full(candidates.shape, np.inf)
    for index in range(candidates.shape[1]):
        scale_c = candidates[:, index]
        valid_c = candidate_valid[:, index]
        positive = valid_c & (scale_c > 0.0)
        reciprocal = np.zeros(candidates.shape[0], dtype=np.float32)
        reciprocal[positive] = (1.0 / scale_c[positive]).astype(np.float32)
        codes_c = np.clip(
            np.rint(values32 * reciprocal[:, None]), spec.qmin, spec.qmax
        ).astype(np.float64)
        # Invalid rows keep a zero scale so the error product stays finite;
        # their SSE is discarded below.
        error = flat_w - codes_c * np.where(valid_c, scale_c, 0.0)[:, None]
        sse = (error**2).sum(axis=1)
        best_error[:, index] = np.where(valid_c, sse, np.inf)
    best = best_error.argmin(axis=1)  # first minimum wins ties: index 0 = max-abs
    chosen = candidates[np.arange(candidates.shape[0]), best]

    # Emit codes through the same formula the runtime dequantisation assumes.
    scales = chosen.astype(np.float16)
    reciprocal = np.zeros(candidates.shape[0], dtype=np.float32)
    positive = scales > 0
    reciprocal[positive] = (1.0 / scales[positive].astype(np.float64)).astype(
        np.float32
    )
    codes = np.clip(
        np.rint(values32 * reciprocal[:, None]), spec.qmin, spec.qmax
    ).astype(np.int8)

    return QuantizedMatrix(
        codes=torch.from_numpy(codes)
        .reshape(geometry.n, geometry.groups_per_row, spec.group_size)
        .to(target),
        scales=torch.from_numpy(scales)
        .reshape(geometry.n, geometry.groups_per_row)
        .to(target),
    )


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values, including registered K padding.

    Scales are rounded to binary16 before codes are selected because those are
    the exact scales consumed after loading.  Padding values are zero and do
    not affect a partially populated final group.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(
            f"grouped quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    max_abs = grouped.abs().amax(dim=2)
    host_scales, host_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    scales = host_scales.to(target)
    reciprocal = host_reciprocal.to(target)
    codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales)
