"""A Qwen3.5/3.8 dense GGUF kept in its own ggml block formats (GSQ-RCO mixed-precision releases).

Every text projection, the token table, the output head and the MTP head of an ``-mtp`` build are
stored as the exporter's own blocks, one ggml type per tensor, so no weight is re-quantised: a row
of the artifact is byte for byte a row of the GGUF. llama.cpp's Qwen3.5 exporter conventions are
undone where they touch rows or small tensors: value heads return to the grouped order,
zero-centred norms drop their stored ``1 + w`` and ``ssm_a`` becomes ``A_log`` again.

``ssm_out`` is the exception. Its tiled value heads are its input columns, which a row copy cannot
move without re-quantising, so its Use carries an ``input_columns`` auxiliary: column ``c`` of the
stored matrix multiplies element ``input_columns[c]`` of the grouped activation.

Vision comes from the release's ``mmproj`` GGUF (``--source vision=mmproj.gguf``) with the official
Vision formats, DFlash2 from ``--source dflash2`` and the frontend resources from ``--model``, which
needs only the base checkpoint's config and tokenizer files.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .methods import AuxiliaryValue, cast_direct, grouped_absmax, import_encoded
from .official_recipes import Q4, Q5, Q6, Q8
from .sources.gguf import GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source
from .ternary import (
    ATTENTION_HEAD_DIM,
    ATTENTION_KV_ROWS,
    ATTENTION_QUERY_ROWS,
    GDN_CHANNELS,
    GDN_HEAD_DIM,
    GDN_KEY_DIM,
    GDN_KEY_HEADS,
    GDN_TAPS,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    VOCABULARY,
    RowMap,
    _direct,
    _flat,
    _norm,
    attention_rows,
    full_attention,
    rows,
    untile,
    untiled_rows,
)

MTP_BLOCK = LAYERS

EXPECTED_HEADER = {
    "general.architecture": "qwen35",
    "qwen35.embedding_length": HIDDEN,
    "qwen35.feed_forward_length": INTERMEDIATE,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": ATTENTION_HEAD_DIM,
    "qwen35.attention.value_length": ATTENTION_HEAD_DIM,
    "qwen35.ssm.conv_kernel": GDN_TAPS,
    "qwen35.ssm.state_size": GDN_HEAD_DIM,
    "qwen35.ssm.group_count": GDN_KEY_HEADS,
    "qwen35.ssm.time_step_rank": GDN_VALUE_HEADS,
    "qwen35.ssm.inner_size": GDN_VALUE_DIM,
    "qwen35.full_attention_interval": 4,
}


def _attention_tensors(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        prefix + "attn_q.weight": ((2 * ATTENTION_QUERY_ROWS, HIDDEN), "blocks"),
        prefix + "attn_k.weight": ((ATTENTION_KV_ROWS, HIDDEN), "blocks"),
        prefix + "attn_v.weight": ((ATTENTION_KV_ROWS, HIDDEN), "blocks"),
        prefix + "attn_output.weight": ((HIDDEN, ATTENTION_QUERY_ROWS), "blocks"),
        prefix + "attn_q_norm.weight": ((ATTENTION_HEAD_DIM,), "F32"),
        prefix + "attn_k_norm.weight": ((ATTENTION_HEAD_DIM,), "F32"),
    }


def _layer_tensors(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        prefix + "attn_norm.weight": ((HIDDEN,), "F32"),
        prefix + "post_attention_norm.weight": ((HIDDEN,), "F32"),
        prefix + "ffn_gate.weight": ((INTERMEDIATE, HIDDEN), "blocks"),
        prefix + "ffn_up.weight": ((INTERMEDIATE, HIDDEN), "blocks"),
        prefix + "ffn_down.weight": ((HIDDEN, INTERMEDIATE), "blocks"),
    }


def expected_tensors(mtp: bool) -> dict[str, tuple[tuple[int, ...], str]]:
    """Row-major shape and kind ("blocks" = any stored ggml block type) of every tensor."""

    out = {
        "token_embd.weight": ((VOCABULARY, HIDDEN), "blocks"),
        "output.weight": ((VOCABULARY, HIDDEN), "blocks"),
        "output_norm.weight": ((HIDDEN,), "F32"),
    }
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        out |= _layer_tensors(p)
        if full_attention(layer):
            out |= _attention_tensors(p)
            continue
        out |= {
            p + "attn_qkv.weight": ((GDN_CHANNELS, HIDDEN), "blocks"),
            p + "attn_gate.weight": ((GDN_VALUE_DIM, HIDDEN), "blocks"),
            p + "ssm_out.weight": ((HIDDEN, GDN_VALUE_DIM), "blocks"),
            p + "ssm_alpha.weight": ((GDN_VALUE_HEADS, HIDDEN), "BF16"),
            p + "ssm_beta.weight": ((GDN_VALUE_HEADS, HIDDEN), "BF16"),
            p + "ssm_a": ((GDN_VALUE_HEADS,), "F32"),
            p + "ssm_dt.bias": ((GDN_VALUE_HEADS,), "F32"),
            p + "ssm_conv1d.weight": ((GDN_CHANNELS, GDN_TAPS), "F32"),
            p + "ssm_norm.weight": ((GDN_HEAD_DIM,), "F32"),
        }
    if mtp:
        p = f"blk.{MTP_BLOCK}."
        out |= _layer_tensors(p) | _attention_tensors(p)
        out |= {
            p + "nextn.eh_proj.weight": ((HIDDEN, 2 * HIDDEN), "blocks"),
            p + "nextn.enorm.weight": ((HIDDEN,), "F32"),
            p + "nextn.hnorm.weight": ((HIDDEN,), "F32"),
            p + "nextn.shared_head_norm.weight": ((HIDDEN,), "F32"),
        }
    return out


def has_mtp(gguf: GGUFFile) -> bool:
    return f"blk.{MTP_BLOCK}.nextn.eh_proj.weight" in gguf.tensors


def validate(gguf: GGUFFile) -> None:
    """Refuse any GGUF that is not a dense Qwen3.5/3.8-27B text tower of stored block types."""

    for key, expected in EXPECTED_HEADER.items():
        if gguf.kv.get(key) != expected:
            raise ValueError(
                f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}"
            )
    mtp = has_mtp(gguf)
    blocks = gguf.kv.get("qwen35.block_count")
    if blocks != LAYERS + (1 if mtp else 0):
        raise ValueError(f"{gguf.path}: {blocks} blocks, expected {LAYERS} (+1 for MTP)")
    expected = expected_tensors(mtp)
    if set(gguf.tensors) != set(expected):
        missing = sorted(set(expected) - set(gguf.tensors))[:5]
        extra = sorted(set(gguf.tensors) - set(expected))[:5]
        raise ValueError(
            f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})"
        )
    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        stored = info.type_id in GGUF_FORMATS_BY_TYPE
        if info.shape != shape or (kind == "blocks") != stored or (
            kind != "blocks" and info.type_name != kind
        ):
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")


def block_format(gguf: GGUFFile, tensor: str) -> str:
    return GGUF_FORMATS_BY_TYPE[gguf.info(tensor).type_id].name


def _dequantize(gguf: GGUFFile, tensor: str, first: int, last: int) -> torch.Tensor:
    try:
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
    except ImportError as error:
        raise ValueError(
            "reading the values of a GGUF block matrix needs the `gguf` package; "
            "storing its blocks does not"
        ) from error
    blocks = gguf.read_blocks(tensor, first, last)
    kind = GGMLQuantizationType(gguf.info(tensor).type_id)
    values = dequantize(np.ascontiguousarray(blocks), kind)
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).reshape(
        last - first, -1
    )


def block_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Rows of a stored block matrix as encoded ggml rows, and their values on request."""

    format = block_format(gguf, tensor)
    empty = torch.empty(0, dtype=torch.float16)

    def encoded(begin: int, end: int) -> EncodedRows:
        index = select(begin, end)
        low, high = int(index.min()), int(index.max()) + 1
        blocks = gguf.read_blocks(tensor, low, high)
        return EncodedRows(
            format, torch.from_numpy(np.ascontiguousarray(blocks[index - low])), empty
        )

    def values(first: int, last: int) -> torch.Tensor:
        index = select(first, last)
        low, high = int(index.min()), int(index.max()) + 1
        return _dequantize(gguf, tensor, low, high)[torch.from_numpy(index - low)]

    return LogicalSource(
        shape, f"{tensor}[{format}]{list(shape)}", _flat(values, shape[1]), encoded
    )


def tiled_input_columns() -> np.ndarray:
    """For each stored ``ssm_out`` column (tiled value heads), its grouped activation element."""

    columns = np.arange(GDN_VALUE_DIM, dtype=np.int64)
    tiled_head, lane = columns // GDN_HEAD_DIM, columns % GDN_HEAD_DIM
    per_key = GDN_VALUE_HEADS // GDN_KEY_HEADS
    grouped_head = (tiled_head % GDN_KEY_HEADS) * per_key + tiled_head // GDN_KEY_HEADS
    return (grouped_head * GDN_HEAD_DIM + lane).astype(np.int32)


def _attention(
    gguf: GGUFFile, g: str, a: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    q_shape, kv_shape = (ATTENTION_QUERY_ROWS, HIDDEN), (ATTENTION_KV_ROWS, HIDDEN)
    encoded = {
        a + "query": block_source(gguf, g + "attn_q.weight", q_shape, attention_rows(False)),
        a + "gate": block_source(gguf, g + "attn_q.weight", q_shape, attention_rows(True)),
        a + "key": block_source(gguf, g + "attn_k.weight", kv_shape, rows()),
        a + "value": block_source(gguf, g + "attn_v.weight", kv_shape, rows()),
        a + "output": block_source(
            gguf, g + "attn_output.weight", (HIDDEN, ATTENTION_QUERY_ROWS), rows()
        ),
    }
    direct = {
        a + "query_norm": _norm(gguf, g + "attn_q_norm.weight", True),
        a + "key_norm": _norm(gguf, g + "attn_k_norm.weight", True),
    }
    return encoded, direct


def _layer(
    gguf: GGUFFile, g: str, p: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    encoded = {
        p + "mlp/gate": block_source(gguf, g + "ffn_gate.weight", (INTERMEDIATE, HIDDEN), rows()),
        p + "mlp/up": block_source(gguf, g + "ffn_up.weight", (INTERMEDIATE, HIDDEN), rows()),
        p + "mlp/down": block_source(
            gguf, g + "ffn_down.weight", (HIDDEN, INTERMEDIATE), rows()
        ),
    }
    direct = {
        p + "input_norm": _norm(gguf, g + "attn_norm.weight", True),
        p + "post_attention_norm": _norm(gguf, g + "post_attention_norm.weight", True),
    }
    return encoded, direct


def text_sources(
    gguf: GGUFFile, *, mtp: bool
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Encoded block sources and direct sources of every GGUF-backed text (and MTP) parameter."""

    encoded = {
        "text/token_embedding": block_source(
            gguf, "token_embd.weight", (VOCABULARY, HIDDEN), rows()
        ),
        "text/output_head": block_source(gguf, "output.weight", (VOCABULARY, HIDDEN), rows()),
    }
    direct = {"text/final_norm": _norm(gguf, "output_norm.weight", True)}
    for layer in range(LAYERS):
        g, p = f"blk.{layer}.", f"text/layers/{layer}/"
        e, d = _layer(gguf, g, p)
        encoded |= e
        direct |= d
        if full_attention(layer):
            e, d = _attention(gguf, g, p + "attention/")
            encoded |= e
            direct |= d
            continue
        n = p + "gdn/"
        qkv = g + "attn_qkv.weight"
        encoded[n + "query"] = block_source(gguf, qkv, (GDN_KEY_DIM, HIDDEN), rows())
        encoded[n + "key"] = block_source(gguf, qkv, (GDN_KEY_DIM, HIDDEN), rows(GDN_KEY_DIM))
        encoded[n + "value"] = block_source(
            gguf, qkv, (GDN_VALUE_DIM, HIDDEN), untiled_rows(2 * GDN_KEY_DIM)
        )
        encoded[n + "z"] = block_source(
            gguf, g + "attn_gate.weight", (GDN_VALUE_DIM, HIDDEN), untiled_rows(0)
        )
        # Tiled input columns: the Use gathers the grouped activation (see tiled_input_columns).
        encoded[n + "output"] = block_source(
            gguf, g + "ssm_out.weight", (HIDDEN, GDN_VALUE_DIM), rows()
        )
        for role, tensor in (("a_projection", "ssm_alpha.weight"), ("b_projection", "ssm_beta.weight")):
            words = untile(gguf.read_bf16_words(g + tensor), 1)
            direct[n + role] = array_source(
                torch.from_numpy(np.ascontiguousarray(words.view(np.int16))).view(
                    torch.bfloat16
                ),
                g + tensor,
            )
        ssm_a = untile(gguf.read_direct(g + "ssm_a"), 1).astype(np.float64)
        if not np.all(ssm_a < 0):
            raise ValueError(f"{g}ssm_a must be strictly negative (-exp(A_log))")
        direct[n + "a_log"] = _direct(
            np.log(-ssm_a).astype(np.float32), torch.float32, g + "ssm_a"
        )
        direct[n + "dt_bias"] = _direct(
            untile(gguf.read_direct(g + "ssm_dt.bias"), 1), torch.float32, g + "ssm_dt.bias"
        )
        taps = gguf.read_direct(g + "ssm_conv1d.weight")
        channels = np.concatenate(
            [taps[: 2 * GDN_KEY_DIM], untile(taps[2 * GDN_KEY_DIM :], GDN_HEAD_DIM)]
        )
        direct[n + "convolution"] = _direct(
            channels.T, torch.bfloat16, g + "ssm_conv1d.weight"
        )
        direct[n + "norm"] = _norm(gguf, g + "ssm_norm.weight", False)
    if mtp:
        g, p = f"blk.{MTP_BLOCK}.", "mtp/layers/0/"
        e, d = _layer(gguf, g, p)
        encoded |= e
        direct |= d
        e, d = _attention(gguf, g, p + "attention/")
        encoded |= e
        direct |= d
        encoded["mtp/input_projection"] = block_source(
            gguf, g + "nextn.eh_proj.weight", (HIDDEN, 2 * HIDDEN), rows()
        )
        for role, tensor in (
            ("embedding_norm", "nextn.enorm.weight"),
            ("hidden_norm", "nextn.hnorm.weight"),
            ("final_norm", "nextn.shared_head_norm.weight"),
        ):
            direct["mtp/" + role] = _norm(gguf, g + tensor, True)
    return encoded, direct


def _vision_format(name: str) -> str:
    if name == "vision/patch_embedding":
        return Q6
    if name.startswith("vision/merger/"):
        return Q8
    if name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
        return Q4
    return Q5


def _vision_source(model, sources):
    from .sources.gguf_source import GGUFSource, HFAliasSource, qwen35_vision_name_map

    mmproj = sources["vision"]
    if not isinstance(mmproj, GGUFSource):
        return mmproj
    config = model.components["vision"]["config"]
    names, transforms, shapes, composites = qwen35_vision_name_map(
        config["depth"],
        hidden_size=config["hidden_size"],
        in_channels=config.get("in_channels", 3),
        temporal_patch_size=config["temporal_patch_size"],
        patch_size=config["patch_size"],
    )
    return HFAliasSource(
        mmproj, names, transforms=transforms, shapes=shapes, composites=composites
    )


def _companions(model, recipe, sources) -> None:
    """Vision from the mmproj GGUF in the official formats; DFlash2 as the official recipe."""

    vision = _vision_source(model, sources) if "vision" in model.components else None
    for name, parameter in model.parameters.items():
        if name.startswith("vision/"):
            source = model.source(name, vision)
            if parameter.projection:
                recipe.assign(
                    name, format=_vision_format(name), method=grouped_absmax, source=source
                )
            else:
                recipe.assign(
                    name,
                    format=parameter.direct_format,
                    method=cast_direct,
                    source=source,
                )
        elif name.startswith(("dflash/", "dflash2/")) and parameter.projection:
            if name.endswith(
                (
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            recipe.assign(name, format=Q8, method=grouped_absmax)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _group_same_format(recipe, names: list[str], formats: dict[str, str]) -> None:
    if len({formats[name] for name in names}) == 1:
        recipe.group(names)


def qwen3_8_27b_gguf(model, recipe, sources):
    """A Qwen3.8-27B GGUF in its own block formats from `--source gguf=MODEL.gguf`."""

    config = model.config
    if (
        "num_experts" in config
        or config["hidden_size"] != HIDDEN
        or config["num_hidden_layers"] != LAYERS
    ):
        raise ValueError("the GGUF block recipe requires the Qwen3.8-27B Dense geometry")
    gguf = sources["gguf"]
    if not isinstance(gguf, GGUFFile):
        raise ValueError("--source gguf must name the model's .gguf file")
    validate(gguf)
    mtp = "mtp" in model.components
    if mtp and not has_mtp(gguf):
        raise ValueError(f"{gguf.path}: the MTP component needs an -mtp GGUF (blk.64.nextn.*)")
    _companions(model, recipe, sources)
    encoded, direct = text_sources(gguf, mtp=mtp)
    formats = {}
    for name, source in encoded.items():
        format = source.read_encoded(0, 1).format
        formats[name] = format
        recipe.assign(
            name, format=format, method=import_encoded, source=source, activation_policy="AllowA8"
        )
    for name, source in direct.items():
        recipe.assign(
            name,
            format=model.parameters[name].direct_format,
            method=cast_direct,
            source=source,
        )
    prefixes = [f"text/layers/{layer}/" for layer in range(LAYERS)]
    if mtp:
        prefixes.append("mtp/layers/0/")
    for p in prefixes:
        if p.startswith("text/") and not full_attention(int(p.split("/")[2])):
            n = p + "gdn/"
            qkv = [n + "query", n + "key", n + "value"]
            recipe.group(qkv + [n + "z"] if formats[n + "z"] == formats[n + "query"] else qkv)
            recipe.separate([n + "a_projection", n + "b_projection"])
        else:
            a = p + "attention/"
            _group_same_format(
                recipe, [a + "query", a + "key", a + "gate", a + "value"], formats
            )
        _group_same_format(recipe, [p + "mlp/gate", p + "mlp/up"], formats)
    columns = AuxiliaryValue(
        "int32", (GDN_VALUE_DIM,), tiled_input_columns().astype("<i4").tobytes()
    )
    for layer in range(LAYERS):
        if full_attention(layer):
            continue
        name = f"text/layers/{layer}/gdn/output"
        for input_name in model.parameters[name].inputs:
            recipe.use(name, input_name, auxiliaries={"input_columns": columns})


RECIPES = {"qwen3_8_27b_gguf": qwen3_8_27b_gguf}

__all__ = [
    "EXPECTED_HEADER",
    "RECIPES",
    "block_format",
    "block_source",
    "expected_tensors",
    "has_mtp",
    "qwen3_8_27b_gguf",
    "text_sources",
    "tiled_input_columns",
    "validate",
]
