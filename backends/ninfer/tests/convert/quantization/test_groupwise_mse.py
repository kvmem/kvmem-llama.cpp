from __future__ import annotations

import numpy as np
import torch

from tools.artifact.codecs.row_split import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
)
from tools.artifact.layouts import row_split_geometry
from tools.convert.quantization.groupwise import (
    quantize_matrix,
    quantize_matrix_mse,
)


def _f16_words(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).view(np.uint16)


def reference_mse_matrix(weight: torch.Tensor, qmin: int, qmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Scalar, per-group reference for the MSE-optimal scale policy.

    Written independently of the vectorised implementation: plain Python
    loops, explicit dtype steps, and the same engine encoding contract
    (binary16 scale, binary32 reciprocal, round-to-nearest-even codes
    clamped to [qmin, qmax]).
    """
    group = 64
    values = weight.numpy().astype(np.float64)
    n, k = values.shape
    codes = np.zeros((n, k // group, group), dtype=np.int8)
    scales = np.zeros((n, k // group), dtype=np.float16)
    for row in range(n):
        for index in range(k // group):
            w = values[row, index * group : (index + 1) * group]
            max_abs = np.abs(w).max()

            def word_of(value: float) -> int:
                return int(_f16_words(np.array([value]))[0])

            s_hi = np.float16(np.float32(max_abs / qmax))
            if s_hi == 0 and max_abs > 0:
                s_hi = np.float16(np.ldexp(1.0, -24))
            s_ls = float(s_hi)
            segment = weight[row, index * group : (index + 1) * group].numpy()
            reciprocal_hi = np.float32(1.0 / float(s_hi)) if s_hi > 0 else np.float32(0.0)
            codes_hi = np.clip(np.rint(segment * reciprocal_hi), qmin, qmax)
            norm = float((codes_hi.astype(np.float64) ** 2).sum())
            if norm > 0:
                s_ls = float((w * codes_hi).sum() / norm)

            candidate_words = [word_of(float(s_hi))]
            for exponent in (-2, -1, 1, 2):
                candidate_words.append(
                    word_of(float(np.float16(float(s_hi) * (2.0**exponent))))
                )
            candidate_words.append(word_of(s_ls))
            for exponent in (-1, 1):
                candidate_words.append(
                    word_of(float(np.float16(s_ls * (2.0**exponent))))
                )
            for offset in (-2, -1, 1, 2):
                candidate_words.append((word_of(float(s_hi)) + offset) & 0xFFFF)
            for offset in (-2, 2):
                candidate_words.append((word_of(s_ls) + offset) & 0xFFFF)

            best_scale, best_error = float(s_hi), None
            for word in candidate_words:
                if not 0 <= word < 0x7C00:
                    continue
                scale = float(
                    np.frombuffer(
                        bytes((word & 0xFF, (word >> 8) & 0xFF)), dtype=np.float16
                    )[0]
                )
                reciprocal = np.float32(1.0 / scale) if scale > 0 else np.float32(0.0)
                encoded = np.clip(np.rint(segment * reciprocal), qmin, qmax)
                error = w - encoded.astype(np.float64) * scale
                group_error = float((error**2).sum())
                if best_error is None or group_error < best_error:
                    best_error, best_scale = group_error, scale

            chosen = np.float16(best_scale)
            scales[row, index] = chosen
            reciprocal = np.float32(1.0 / float(chosen)) if chosen > 0 else np.float32(0.0)
            codes[row, index] = np.clip(
                np.rint(segment * reciprocal), qmin, qmax
            ).astype(np.int8)
    return codes, scales


def test_mse_matches_independent_reference() -> None:
    torch.manual_seed(20260919)
    weight = torch.randn(3, 320, dtype=torch.float32)
    # Mix in adversarial shapes: a flat row, a near-zero row, an outlier row.
    weight[0] = 0.5
    weight[1] *= 1e-7
    weight[2, 5] = 40.0
    weight[2, 5::67] = -25.0

    quantized = quantize_matrix_mse(weight, "q6_g64_fp16", device="cpu")
    # The physical layout pads K up to the codec alignment; the reference must
    # see the same zero-filled padding groups.
    geometry = row_split_geometry("q6_g64_fp16", weight.shape)
    padded = weight
    if geometry.k_pad != geometry.k:
        padded = torch.zeros((geometry.n, geometry.k_pad), dtype=weight.dtype)
        padded[:, : geometry.k] = weight
    codes, scales = reference_mse_matrix(padded, -32, 31)
    assert np.array_equal(
        quantized.codes.cpu().numpy().astype(np.int8), codes
    ), "codes differ from the reference"
    assert np.array_equal(
        quantized.scales.cpu().numpy().astype(np.float16), scales
    ), "scales differ from the reference"


def test_mse_error_does_not_exceed_absmax() -> None:
    torch.manual_seed(7)
    weight = torch.randn(16, 512, dtype=torch.float32)
    weight[3] *= 0.01
    weight[9, ::11] *= 50.0

    mse = quantize_matrix_mse(weight, "q6_g64_fp16", device="cpu")
    absmax = quantize_matrix(weight, "q6_g64_fp16", device="cpu")

    for name, quantized in (("mse", mse), ("absmax", absmax)):
        codes = quantized.codes.cpu().numpy()
        assert (codes >= -32).all() and (codes <= 31).all(), (
            f"{name} codes out of range"
        )
        assert (quantized.scales.cpu() >= 0).all(), f"{name} has a non-positive scale"

    # Per-group encoded error, evaluated in binary64 from the stored integers
    # and scales: the MSE candidate set contains the max-abs scale, so it
    # cannot lose to the baseline on any group.
    values = weight.numpy().astype(np.float64).reshape(16, 8, 64)
    mse_error = (
        values
        - mse.codes.cpu().numpy().astype(np.float64)
        * mse.scales.cpu().numpy().astype(np.float64)[:, :, None]
    ) ** 2
    base_error = (
        values
        - absmax.codes.cpu().numpy().astype(np.float64)
        * absmax.scales.cpu().numpy().astype(np.float64)[:, :, None]
    ) ** 2
    assert (mse_error.sum(axis=2) <= base_error.sum(axis=2)).all(), (
        "MSE scale lost to max-abs on a group"
    )


def test_mse_uniform_group_matches_absmax() -> None:
    weight = torch.full((2, 128), 0.125, dtype=torch.float32)
    mse = quantize_matrix_mse(weight, "q6_g64_fp16", device="cpu")
    absmax = quantize_matrix(weight, "q6_g64_fp16", device="cpu")
    assert torch.equal(mse.codes, absmax.codes)
    assert torch.equal(mse.scales, absmax.scales)


def test_mse_zero_and_underflow_groups() -> None:
    weight = torch.zeros((4, 64), dtype=torch.float32)
    weight[1, 0] = torch.finfo(torch.float32).tiny
    weight[2, 0] = 1.0
    quantized = quantize_matrix_mse(weight, "q6_g64_fp16", device="cpu")
    assert quantized.scales[0, 0] == 0
    assert torch.count_nonzero(quantized.codes[0]) == 0
    assert int(quantized.scales[1, 0].view(torch.int16)) == 1
    # 2^-126 / 2^-24 = 2^-102, which rounds to the code 0 at every candidate
    # scale, so the index-0 tie-break keeps the canonical 2^-24 scale.
    assert quantized.codes[1, 0, 0].item() == 0
    assert int(quantized.scales[2, 0].view(torch.int16)) > 0
    assert quantized.codes[2, 0, 0].item() in (-32, 31)


def test_mse_outlier_group_stays_legal() -> None:
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, 0] = 1.0
    weight[0, 1:] = 0.008
    quantized = quantize_matrix_mse(weight, "q6_g64_fp16", device="cpu")
    scale = quantized.scales[0, 0]
    assert scale > 0 and torch.isfinite(scale.float())
    assert ((quantized.codes >= -32) & (quantized.codes <= 31)).all()
    values = weight.numpy().astype(np.float64).reshape(1, 64)
    dequantized = (
        quantized.codes[0].numpy().astype(np.float64)
        * float(scale)
    )
    mse_error = float(((values - dequantized) ** 2).sum())
    absmax = quantize_matrix(weight, "q6_g64_fp16", device="cpu")
    absmax_error = float(
        ((values - (absmax.codes[0].numpy().astype(np.float64) * float(absmax.scales[0, 0]))) ** 2).sum()
    )
    assert mse_error <= absmax_error


def test_mse_rejects_non_finite_source() -> None:
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, 0] = float("nan")
    try:
        quantize_matrix_mse(weight, "q6_g64_fp16", device="cpu")
    except ValueError:
        return
    raise AssertionError("NaN source must be rejected")


def test_mse_codes_roundtrip_through_row_split_codec() -> None:
    torch.manual_seed(11)
    weight = torch.randn(6, 128, dtype=torch.bfloat16)
    quantized = quantize_matrix_mse(weight, "q6_g64_fp16", device="cpu")
    payload = encode_row_split(
        quantized.codes, quantized.scales, "q6_g64_fp16", weight.shape
    )
    scales, codes = decode_row_split_codes(payload, "q6_g64_fp16", tuple(weight.shape))
    assert torch.equal(scales, quantized.scales)
    assert torch.equal(codes, quantized.codes)
    decoded = dequantize_row_split(
        payload, "q6_g64_fp16", tuple(weight.shape), dtype=torch.float32
    )
    # The non-bf16 dequantise path is exactly codes.float() * scales.float().
    expected = (
        quantized.codes.float() * quantized.scales.float().unsqueeze(-1)
    ).reshape(6, 128)
    assert torch.equal(decoded, expected)


def test_mse_cuda_matches_cpu() -> None:
    torch.manual_seed(3)
    weight = torch.randn(8, 256, dtype=torch.float32)
    weight[4, 17] = 30.0
    cpu = quantize_matrix_mse(weight, "q6_g64_fp16", device="cpu")
    if torch.cuda.is_available():
        cuda = quantize_matrix_mse(weight, "q6_g64_fp16", device="cuda")
        assert torch.equal(cuda.codes.cpu(), cpu.codes)
        assert torch.equal(cuda.scales.cpu(), cpu.scales)
