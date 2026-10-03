"""Select final resource bytes and derive the tokenizer's public token domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
VISION_RESOURCES = ("preprocessor_config.json", "video_preprocessor_config.json")

# Must stay byte-identical to kQwenSplitPattern in
# src/models/qwen3_5/frontend/tokenizer.cpp: the runtime tokenizer validates
# tokenizer.json's Split pattern for exact equality and rejects any artifact whose
# pattern differs.
STD_QWEN_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Qwen3.5 tokenizer_config.json invariants the runtime frontend hard-requires
# (src/models/qwen3_5/frontend/frontend.cpp validate_tokenizer_config): a missing
# field defaults to the failing value, so each must be present and set to the
# official Qwen3.5 value.
KQWEN35_PAD_TOKEN = "<|endoftext|>"


def token_domain(
    tokenizer: dict, config: dict, vocab_size: int
) -> tuple[int, tuple[int, ...]]:
    ids: dict[int, str] = {}
    special: dict[int, bool] = {}

    def add(index: object, content: object) -> None:
        if (
            type(index) is not int
            or not 0 <= index < vocab_size
            or not isinstance(content, str)
        ):
            raise ValueError(
                f"invalid tokenizer entry id={index!r}, content={content!r}"
            )
        if index in ids and ids[index] != content:
            raise ValueError(f"tokenizer resources disagree about token {index}")
        ids[index] = content

    def add_special(index: int, token: dict) -> None:
        flag = token.get("special", False)
        if type(flag) is not bool:
            raise ValueError(f"token {index}: special must be boolean")
        if index in special and special[index] != flag:
            raise ValueError(
                f"tokenizer resources disagree about token {index} special flag"
            )
        special[index] = flag

    model = tokenizer.get("model", {})
    vocabulary = model.get("vocab")
    if not isinstance(vocabulary, dict):
        raise ValueError("tokenizer model must provide a vocabulary mapping")
    for token, index in vocabulary.items():
        add(index, token)
    for token in tokenizer.get("added_tokens", []):
        add(token["id"], token["content"])
        add_special(token["id"], token)
    for raw_id, token in config.get("added_tokens_decoder", {}).items():
        index = int(raw_id)
        add(index, token["content"])
        add_special(index, token)
    if not ids or set(ids) != set(range(max(ids) + 1)):
        raise ValueError("this tokenizer requires a contiguous public token ID domain")
    return len(ids), tuple(sorted(index for index, flag in special.items() if flag))


def _dump_json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def _normalize_qwen_split(tokenizer: dict) -> bool:
    """Normalize a drifted pre-tokenizer split pattern to the standard Qwen one.

    The runtime tokenizer pre-tokenizes with its own hardcoded \\p{L}\\p{M} logic
    (not the JSON regex) but validates the stored Split pattern for exact equality
    against kQwenSplitPattern, so a drifted pattern (some third-party Qwen
    checkpoints drop \\p{M}) makes the artifact unloadable without changing
    tokenization. Rewriting it to the standard pattern is safe. Returns True if the
    pattern changed.
    """
    pre = tokenizer.get("pre_tokenizer")
    if not isinstance(pre, dict) or pre.get("type") != "Sequence":
        return False
    parts = pre.get("pretokenizers")
    if not isinstance(parts, list) or len(parts) != 2:
        return False
    split = parts[0]
    if not isinstance(split, dict) or split.get("type") != "Split":
        return False
    pattern = split.get("pattern")
    if not isinstance(pattern, dict):
        return False
    if pattern.get("Regex") == STD_QWEN_SPLIT_PATTERN:
        return False
    pattern["Regex"] = STD_QWEN_SPLIT_PATTERN
    split["behavior"] = "Isolated"
    split["invert"] = False
    return True


def _normalize_added_decoder(tokenizer: dict, config: dict) -> bool:
    """Ensure tokenizer_config.json carries an added_tokens_decoder object.

    The runtime tokenizer requires the field to exist as an object and
    cross-checks each entry against tokenizer.json's added_tokens for shared ids.
    Third-party Qwen checkpoints frequently omit the field; rebuild it from
    added_tokens (the canonical source) so the artifact loads and stays consistent.
    Returns True if the field was built.
    """
    if isinstance(config.get("added_tokens_decoder"), dict):
        return False
    decoder: dict[str, dict] = {}
    for token in tokenizer.get("added_tokens", []):
        raw_id = token.get("id")
        if type(raw_id) is not int:
            continue
        decoder[str(raw_id)] = {
            "content": token["content"],
            "single_word": token.get("single_word", False),
            "lstrip": token.get("lstrip", False),
            "rstrip": token.get("rstrip", False),
            "normalized": token.get("normalized", False),
            "special": token.get("special", False),
        }
    config["added_tokens_decoder"] = decoder
    return True


def _normalize_prefix_semantics(config: dict) -> bool:
    """Ensure the Qwen3.5 prefix-semantics invariants in tokenizer_config.json.

    The runtime frontend requires add_bos_token=false, add_prefix_space=false, and
    pad_token=<|endoftext|>; a missing field defaults to the failing value and a
    non-official pad token is rejected. Third-party Qwen checkpoints may omit or
    mis-set these. Returns True if any field was set.
    """
    changed = False
    if config.get("add_bos_token") is not False:
        config["add_bos_token"] = False
        changed = True
    if config.get("add_prefix_space") is not False:
        config["add_prefix_space"] = False
        changed = True
    if config.get("pad_token") != KQWEN35_PAD_TOKEN:
        config["pad_token"] = KQWEN35_PAD_TOKEN
        changed = True
    return changed


def load_resources(
    model_dir: Path,
    *,
    vocab_size: int,
    vision_config: Mapping[str, int] | None = None,
    overrides: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, bytes], int, tuple[int, ...]]:
    overrides = {} if overrides is None else dict(overrides)
    roles = {"text": TEXT_RESOURCES}
    if vision_config is not None:
        roles["vision"] = VISION_RESOURCES
    allowed = {role for names in roles.values() for role in names}
    if overrides.keys() - allowed:
        raise ValueError(
            f"resource overrides have no selected consumer: {sorted(overrides.keys()-allowed)}"
        )
    references: dict[str, dict[str, str]] = {}
    payloads: dict[str, bytes] = {}
    parsed: dict[str, dict] = {}
    for component, names in roles.items():
        references[component] = {}
        for role in names:
            path = Path(overrides[role]) if role in overrides else model_dir / role
            data = path.read_bytes()
            if not data:
                raise ValueError(f"{path}: resource is empty")
            text = data.decode("utf-8")
            if role.endswith(".json"):
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: resource must contain a JSON object")
                parsed[role] = value
                if component == "vision":
                    for field, config_field in (
                        ("patch_size", "patch_size"),
                        ("temporal_patch_size", "temporal_patch_size"),
                        ("merge_size", "spatial_merge_size"),
                    ):
                        if (
                            type(value.get(field)) is not int
                            or value[field] != vision_config[config_field]
                        ):
                            raise ValueError(
                                f"{path}: {field} differs from Vision config"
                            )
            object_id = f"resource/{component}/{role}"
            references[component][role] = object_id
            payloads[object_id] = data
    tokenizer_json = parsed["tokenizer.json"]
    tokenizer_config = parsed["tokenizer_config.json"]
    if _normalize_qwen_split(tokenizer_json):
        payloads["resource/text/tokenizer.json"] = _dump_json(tokenizer_json)
        print(
            "normalized tokenizer.json split pattern to the standard Qwen pattern",
            flush=True,
        )
    config_changed = False
    if _normalize_added_decoder(tokenizer_json, tokenizer_config):
        print(
            "built tokenizer_config.json added_tokens_decoder "
            f"({len(tokenizer_config['added_tokens_decoder'])} tokens)",
            flush=True,
        )
        config_changed = True
    if _normalize_prefix_semantics(tokenizer_config):
        print(
            "normalized tokenizer_config.json prefix semantics "
            "(add_bos_token/add_prefix_space=false, pad_token=<|endoftext|>)",
            flush=True,
        )
        config_changed = True
    if config_changed:
        payloads["resource/text/tokenizer_config.json"] = _dump_json(tokenizer_config)
    count, special = token_domain(tokenizer_json, tokenizer_config, vocab_size)
    return references, payloads, count, special
