"""Minimal GGUF v3 reader and the PrismML ternary block decoders.

Only what the ternary Bonsai 2 converter needs: the key/value header, the
tensor directory, memory-mapped tensor bytes, exact decoders for the ``PQ2_0``
and ``PTQ1_0`` ternary block formats, and the direct F32/F16/BF16 words. The
reader has no gguf-py dependency and tolerates a truncated data section, so the
saved header sample of a large file can be inspected without the whole file.

Ternary blocks are 128 weights wide with one binary16 scale ``d``. ``PQ2_0``
stores two bits per weight (``00=-1, 01=0, 10=+1``; ``11`` is not ternary);
``PTQ1_0`` stores five base-3 trits per byte in staged chunks plus four trits
per byte for the tail, as in ``ggml-quants.c`` ``dequantize_row_ptq1_0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
import struct
from typing import Any, Sequence

import numpy as np

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

TYPE_F32 = 0
TYPE_F16 = 1
TYPE_BF16 = 30
TYPE_PQ2_0 = 142
TYPE_PTQ1_0 = 143

# ggml block types stored unchanged in a `gguf_blocks_v1` object, keyed by GGUF type id.
BLOCK_FORMATS = {
    fmt.ggml_type: fmt for fmt in GGUF_FORMATS_BY_TYPE.values()
}

TYPE_NAMES = {
    TYPE_F32: "F32",
    TYPE_F16: "F16",
    TYPE_BF16: "BF16",
    TYPE_PQ2_0: "PQ2_0",
    TYPE_PTQ1_0: "PTQ1_0",
    **{
        type_id: fmt.name.removeprefix("gguf_").upper()
        for type_id, fmt in BLOCK_FORMATS.items()
    },
}
TYPE_IDS = {name: type_id for type_id, name in TYPE_NAMES.items()}

# (elements per block, bytes per block)
BLOCK_GEOMETRY = {
    TYPE_F32: (1, 4),
    TYPE_F16: (1, 2),
    TYPE_BF16: (1, 2),
    TYPE_PQ2_0: (128, 34),
    TYPE_PTQ1_0: (128, 28),
    **{
        type_id: (fmt.block_elements, fmt.block_bytes)
        for type_id, fmt in BLOCK_FORMATS.items()
    },
}
TERNARY_GROUP = 128
TERNARY_TYPES = frozenset((TYPE_PQ2_0, TYPE_PTQ1_0))

_SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type_id, str(self.type_id))

    @property
    def elements(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        block_elements, block_bytes = BLOCK_GEOMETRY[self.type_id]
        if self.elements % block_elements:
            raise ValueError(
                f"{self.name}: {self.elements} elements are not whole blocks"
            )
        return self.elements // block_elements * block_bytes


@dataclass(frozen=True, slots=True)
class TernaryBlocks:
    """Decoded ternary rows: ``values`` in ``{-1, 0, +1}`` and one scale per 128."""

    values: np.ndarray  # int8 [rows, groups, 128]
    scales: np.ndarray  # float16 [rows, groups]

    @property
    def rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def k(self) -> int:
        return int(self.values.shape[1]) * TERNARY_GROUP


class _Cursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def unpack(self, format: str) -> tuple[Any, ...]:
        size = struct.calcsize(format)
        values = struct.unpack_from("<" + format, self.data, self.position)
        self.position += size
        return values

    def string(self) -> str:
        (length,) = self.unpack("Q")
        value = self.data[self.position : self.position + length]
        if len(value) != length:
            raise ValueError("GGUF string runs past the header")
        self.position += length
        return value.decode("utf-8")

    def value(self, type_id: int) -> Any:
        if type_id == _TYPE_STRING:
            return self.string()
        if type_id == _TYPE_ARRAY:
            (element_type,) = self.unpack("I")
            (count,) = self.unpack("Q")
            if element_type in _SCALAR_FORMATS:
                format = _SCALAR_FORMATS[element_type]
                values = self.unpack(format * count)
                return list(values)
            return [self.value(element_type) for _ in range(count)]
        if type_id not in _SCALAR_FORMATS:
            raise ValueError(f"unsupported GGUF value type {type_id}")
        return self.unpack(_SCALAR_FORMATS[type_id])[0]


class GGUFFile:
    """Parse the header of a GGUF v3 file and expose memory-mapped tensor bytes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.file_bytes = self.path.stat().st_size
        with self.path.open("rb") as handle:
            head = handle.read(64 * 1024 * 1024)
        cursor = _Cursor(head)
        magic = head[:4]
        cursor.position = 4
        if magic != GGUF_MAGIC:
            raise ValueError(f"{self.path}: not a GGUF file")
        (version,) = cursor.unpack("I")
        if version != GGUF_VERSION:
            raise ValueError(f"{self.path}: unsupported GGUF version {version}")
        (tensor_count,) = cursor.unpack("Q")
        (kv_count,) = cursor.unpack("Q")
        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = cursor.string()
            (type_id,) = cursor.unpack("I")
            kv[key] = cursor.value(type_id)
        tensors: dict[str, TensorInfo] = {}
        for _ in range(tensor_count):
            name = cursor.string()
            (rank,) = cursor.unpack("I")
            dims = cursor.unpack("Q" * rank)
            (type_id,) = cursor.unpack("I")
            (offset,) = cursor.unpack("Q")
            if name in tensors:
                raise ValueError(f"{self.path}: duplicate tensor {name}")
            # GGUF lists the fastest dimension first; NInfer shapes are row-major.
            tensors[name] = TensorInfo(name, tuple(reversed(dims)), type_id, offset)
        self.kv = kv
        self.tensors = tensors
        self.alignment = int(kv.get("general.alignment", DEFAULT_ALIGNMENT))
        self.header_bytes = cursor.position
        self.data_offset = align_up(self.header_bytes, self.alignment)
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def __enter__(self) -> GGUFFile:
        return self

    def __exit__(self, *_exc) -> None:
        self._map = None

    @property
    def data_bytes_available(self) -> int:
        return max(0, self.file_bytes - self.data_offset)

    def info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"{self.path}: tensor {name} is absent") from error

    def tensor_bytes(
        self, name: str, byte_begin: int = 0, byte_end: int | None = None
    ) -> np.ndarray:
        info = self.info(name)
        end = info.nbytes if byte_end is None else byte_end
        if not 0 <= byte_begin <= end <= info.nbytes:
            raise ValueError(
                f"{name}: byte range [{byte_begin},{end}) is outside {info.nbytes}"
            )
        start = self.data_offset + info.offset + byte_begin
        stop = self.data_offset + info.offset + end
        if stop > self.file_bytes:
            raise ValueError(
                f"{name}: bytes [{byte_begin},{end}) are past the end of the file "
                f"({self.file_bytes} bytes; the data section may be truncated)"
            )
        return self._map[start:stop]

    def _row_bytes(self, info: TensorInfo) -> int:
        if len(info.shape) != 2:
            raise ValueError(
                f"{info.name}: row access requires rank 2, got {info.shape}"
            )
        rows, columns = info.shape
        block_elements, block_bytes = BLOCK_GEOMETRY[info.type_id]
        if columns % block_elements:
            raise ValueError(f"{info.name}: {columns} columns are not whole blocks")
        return columns // block_elements * block_bytes

    def read_direct(self, name: str) -> np.ndarray:
        """Return F32/F16/BF16 words as float32 in the tensor's row-major shape."""

        info = self.info(name)
        raw = self.tensor_bytes(name)
        if info.type_id == TYPE_F32:
            values = np.frombuffer(raw, dtype="<f4")
        elif info.type_id == TYPE_F16:
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif info.type_id == TYPE_BF16:
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            values = words.view(np.float32)
        else:
            raise ValueError(f"{name}: {info.type_name} is not a direct format")
        return np.ascontiguousarray(values.reshape(info.shape))

    def read_bf16_words(self, name: str) -> np.ndarray:
        info = self.info(name)
        if info.type_id != TYPE_BF16:
            raise ValueError(f"{name}: expected BF16, got {info.type_name}")
        return np.ascontiguousarray(
            np.frombuffer(self.tensor_bytes(name), dtype="<u2").reshape(info.shape)
        )

    def read_blocks(
        self, name: str, row_begin: int = 0, row_end: int | None = None
    ) -> np.ndarray:
        """Rows ``[row_begin, row_end)`` of a ggml block matrix as ``uint8 [rows, row_bytes]``."""

        info = self.info(name)
        if info.type_id not in BLOCK_FORMATS:
            raise ValueError(f"{name}: {info.type_name} is not a stored block format")
        rows = info.shape[0]
        end = rows if row_end is None else row_end
        if not 0 <= row_begin <= end <= rows:
            raise ValueError(f"{name}: row range [{row_begin},{end}) is outside {rows}")
        row_bytes = self._row_bytes(info)
        raw = self.tensor_bytes(name, row_begin * row_bytes, end * row_bytes)
        return np.asarray(raw).reshape(end - row_begin, row_bytes)

    def read_ternary(
        self, name: str, row_begin: int = 0, row_end: int | None = None
    ) -> TernaryBlocks:
        """Decode rows ``[row_begin, row_end)`` of a PQ2_0 or PTQ1_0 matrix."""

        info = self.info(name)
        if info.type_id not in TERNARY_TYPES:
            raise ValueError(f"{name}: {info.type_name} is not a ternary format")
        rows, columns = info.shape
        end = rows if row_end is None else row_end
        if not 0 <= row_begin <= end <= rows:
            raise ValueError(f"{name}: row range [{row_begin},{end}) is outside {rows}")
        row_bytes = self._row_bytes(info)
        raw = self.tensor_bytes(name, row_begin * row_bytes, end * row_bytes)
        blocks = np.asarray(raw).reshape(end - row_begin, columns // TERNARY_GROUP, -1)
        if info.type_id == TYPE_PQ2_0:
            return decode_pq2_0(blocks)
        return decode_ptq1_0(blocks)


def decode_pq2_0(blocks: np.ndarray) -> TernaryBlocks:
    """Decode ``[rows, groups, 34]`` PQ2_0 blocks: binary16 ``d`` then 32 code bytes."""

    if blocks.ndim != 3 or blocks.shape[-1] != 34:
        raise ValueError(f"PQ2_0 blocks must be [rows, groups, 34], got {blocks.shape}")
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
    scales = (
        blocks[:, :, :2].copy().view("<f2").reshape(blocks.shape[0], blocks.shape[1])
    )
    packed = blocks[:, :, 2:]
    # element j lives in byte j // 4 at bits 2 * (j % 4)
    codes = np.stack([(packed >> (2 * bit)) & 3 for bit in range(4)], axis=-1)
    codes = codes.reshape(blocks.shape[0], blocks.shape[1], TERNARY_GROUP)
    if codes.max(initial=0) > 2:
        raise ValueError("PQ2_0 block holds code 3 (+2), which is not ternary")
    values = (codes.astype(np.int16) - 1).astype(np.int8)
    return TernaryBlocks(values, np.ascontiguousarray(scales))


_PTQ1_0_STAGES = (32, 16, 8)
_PTQ1_0_QS_BYTES = 24
_PTQ1_0_QH_BYTES = 2
_POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint16)


def _ptq1_0_chunks() -> list[tuple[int, int]]:
    """(byte offset, chunk width) pairs of the staged 5-trit bytes, in decode order."""

    chunks: list[tuple[int, int]] = []
    offset = 0
    for width in _PTQ1_0_STAGES:
        while offset + width <= _PTQ1_0_QS_BYTES:
            chunks.append((offset, width))
            offset += width
    return chunks


def decode_ptq1_0(blocks: np.ndarray) -> TernaryBlocks:
    """Decode ``[rows, groups, 28]`` PTQ1_0 blocks: 24 ``qs``, 2 ``qh``, binary16 ``d``."""

    if blocks.ndim != 3 or blocks.shape[-1] != 28:
        raise ValueError(
            f"PTQ1_0 blocks must be [rows, groups, 28], got {blocks.shape}"
        )
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
    rows, groups = blocks.shape[0], blocks.shape[1]
    qs = blocks[:, :, :_PTQ1_0_QS_BYTES].astype(np.uint16)
    qh = blocks[:, :, _PTQ1_0_QS_BYTES : _PTQ1_0_QS_BYTES + _PTQ1_0_QH_BYTES].astype(
        np.uint16
    )
    scales = blocks[:, :, 26:28].copy().view("<f2").reshape(rows, groups)
    parts: list[np.ndarray] = []
    for offset, width in _ptq1_0_chunks():
        chunk = qs[:, :, offset : offset + width]
        for trit in range(5):
            q = (chunk * _POW3[trit]) & 0xFF
            parts.append(((q * 3) >> 8).astype(np.int16))
    for trit in range(4):
        q = (qh * _POW3[trit]) & 0xFF
        parts.append(((q * 3) >> 8).astype(np.int16))
    codes = np.concatenate(parts, axis=-1)
    if codes.shape[-1] != TERNARY_GROUP:
        raise AssertionError("PTQ1_0 decode produced a wrong group width")
    values = (codes - 1).astype(np.int8)
    return TernaryBlocks(values, np.ascontiguousarray(scales))


def encode_pq2_0(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Pack ``{-1,0,+1}`` values ``[rows, groups, 128]`` and binary16 scales into PQ2_0 blocks."""

    values = np.asarray(values, dtype=np.int16)
    if values.ndim != 3 or values.shape[-1] != TERNARY_GROUP:
        raise ValueError("PQ2_0 values must be [rows, groups, 128]")
    if values.min(initial=0) < -1 or values.max(initial=0) > 1:
        raise ValueError("PQ2_0 values must be ternary")
    codes = (
        (values + 1).astype(np.uint8).reshape(*values.shape[:2], TERNARY_GROUP // 4, 4)
    )
    packed = np.zeros(codes.shape[:3], dtype=np.uint8)
    for bit in range(4):
        packed |= codes[..., bit] << (2 * bit)
    scale_bytes = (
        np.ascontiguousarray(scales, dtype="<f2")
        .view(np.uint8)
        .reshape(*values.shape[:2], 2)
    )
    return np.concatenate([scale_bytes, packed], axis=-1)


def encode_ptq1_0(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Pack ternary values into PTQ1_0 blocks, following ``quantize_row_ptq1_0_ref``."""

    values = np.asarray(values, dtype=np.int16)
    if values.ndim != 3 or values.shape[-1] != TERNARY_GROUP:
        raise ValueError("PTQ1_0 values must be [rows, groups, 128]")
    if values.min(initial=0) < -1 or values.max(initial=0) > 1:
        raise ValueError("PTQ1_0 values must be ternary")
    xi = (values + 1).astype(np.uint16)
    rows, groups = values.shape[:2]
    qs = np.zeros((rows, groups, _PTQ1_0_QS_BYTES), dtype=np.uint16)
    position = 0
    for offset, width in _ptq1_0_chunks():
        chunk = xi[:, :, position : position + 5 * width]
        q = np.zeros((rows, groups, width), dtype=np.uint16)
        for trit in range(5):
            q = q * 3 + chunk[:, :, trit * width : (trit + 1) * width]
        qs[:, :, offset : offset + width] = (q * 256 + 242) // 243
        position += 5 * width
    tail = xi[:, :, position:]
    qh = np.zeros((rows, groups, _PTQ1_0_QH_BYTES), dtype=np.uint16)
    for trit in range(4):
        qh = (
            qh * 3 + tail[:, :, trit * _PTQ1_0_QH_BYTES : (trit + 1) * _PTQ1_0_QH_BYTES]
        )
    qh = ((qh * 3) * 256 + 242) // 243
    scale_bytes = (
        np.ascontiguousarray(scales, dtype="<f2")
        .view(np.uint8)
        .reshape(rows, groups, 2)
    )
    return np.concatenate(
        [qs.astype(np.uint8), qh.astype(np.uint8), scale_bytes], axis=-1
    )


def write_gguf(
    path: str | Path,
    kv: dict[str, Any],
    tensors: Sequence[tuple[str, Sequence[int], int, bytes]],
    alignment: int = DEFAULT_ALIGNMENT,
) -> None:
    """Write a small GGUF v3 file (test fixtures and probes, not a general exporter).

    ``tensors`` holds ``(name, row-major shape, type id, raw block bytes)``. Scalar
    values are typed by Python type: ``bool``, ``int`` (int32 unless it does not fit,
    then uint64), ``float`` (float32), ``str``; lists must be homogeneous.
    """

    def encode_value(value: Any) -> bytes:
        if isinstance(value, bool):
            return struct.pack("<I?", _TYPE_BOOL, value)
        if isinstance(value, int):
            if -(2**31) <= value < 2**31:
                return struct.pack("<Ii", _TYPE_INT32, value)
            return struct.pack("<IQ", 10, value)
        if isinstance(value, float):
            return struct.pack("<If", _TYPE_FLOAT32, value)
        if isinstance(value, str):
            data = value.encode("utf-8")
            return struct.pack("<IQ", _TYPE_STRING, len(data)) + data
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("empty GGUF arrays are not written by this helper")
            first = value[0]
            if isinstance(first, bool):
                element = _TYPE_BOOL
                body = struct.pack("<" + "?" * len(value), *value)
            elif isinstance(first, int):
                element = _TYPE_INT32
                body = struct.pack("<" + "i" * len(value), *value)
            elif isinstance(first, float):
                element = _TYPE_FLOAT32
                body = struct.pack("<" + "f" * len(value), *value)
            elif isinstance(first, str):
                element = _TYPE_STRING
                body = b"".join(
                    struct.pack("<Q", len(item.encode("utf-8"))) + item.encode("utf-8")
                    for item in value
                )
            else:
                raise TypeError(f"unsupported GGUF array element {type(first)!r}")
            return struct.pack("<IIQ", _TYPE_ARRAY, element, len(value)) + body
        raise TypeError(f"unsupported GGUF value {type(value)!r}")

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kv))
    for key, value in kv.items():
        key_bytes = key.encode("utf-8")
        header += struct.pack("<Q", len(key_bytes)) + key_bytes
        header += encode_value(value)
    offset = 0
    payloads: list[bytes] = []
    for name, shape, type_id, raw in tensors:
        info = TensorInfo(name, tuple(shape), type_id, offset)
        if len(raw) != info.nbytes:
            raise ValueError(f"{name}: {len(raw)} bytes, expected {info.nbytes}")
        name_bytes = name.encode("utf-8")
        header += struct.pack("<Q", len(name_bytes)) + name_bytes
        header += struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", type_id, offset)
        payloads.append(raw)
        offset = align_up(offset + len(raw), alignment)
    data_offset = align_up(len(header), alignment)
    with Path(path).open("wb") as handle:
        handle.write(header)
        handle.write(b"\x00" * (data_offset - len(header)))
        cursor = 0
        for raw in payloads:
            handle.write(raw)
            cursor += len(raw)
            padded = align_up(cursor, alignment)
            handle.write(b"\x00" * (padded - cursor))
            cursor = padded


__all__ = [
    "BLOCK_GEOMETRY",
    "DEFAULT_ALIGNMENT",
    "GGUFFile",
    "TERNARY_GROUP",
    "TERNARY_TYPES",
    "TYPE_BF16",
    "TYPE_F16",
    "TYPE_F32",
    "TYPE_IDS",
    "TYPE_NAMES",
    "TYPE_PQ2_0",
    "TYPE_PTQ1_0",
    "TensorInfo",
    "TernaryBlocks",
    "align_up",
    "decode_pq2_0",
    "decode_ptq1_0",
    "encode_pq2_0",
    "encode_ptq1_0",
    "write_gguf",
]
