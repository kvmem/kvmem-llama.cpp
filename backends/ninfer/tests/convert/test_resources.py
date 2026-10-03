from __future__ import annotations

import json

import pytest

from tools.convert.resources import STD_QWEN_SPLIT_PATTERN, load_resources, token_domain


def _write_text_resources(tmp_path, tokenizer, config):
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    (tmp_path / "generation_config.json").write_text("{}")


def test_final_resources_override_defaults_without_hash_pinning(tmp_path):
    (tmp_path / "tokenizer.json").write_text("invalid default")
    selected = tmp_path / "custom-tokenizer.json"
    selected.write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
        )
    )
    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id":2,"temperature":1.0}'
    )
    template = tmp_path / "custom.jinja"
    template.write_text("custom {{ messages }}")
    references, payloads, count, special = load_resources(
        tmp_path,
        vocab_size=4,
        overrides={"tokenizer.json": selected, "chat_template.jinja": template},
    )
    assert count == 3 and special == (2,)
    assert set(references) == {"text"}
    assert payloads[references["text"]["chat_template.jinja"]] == template.read_bytes()
    assert (
        payloads[references["text"]["generation_config.json"]]
        == (tmp_path / "generation_config.json").read_bytes()
    )


def test_special_tokens_merge_both_resources_with_consistent_flags():
    tokenizer = {
        "model": {"vocab": {"a": 0}},
        "added_tokens": [{"id": 1, "content": "<start>", "special": True}],
    }
    config = {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
    assert token_domain(tokenizer, config, 4) == (3, (1, 2))
    config["added_tokens_decoder"]["1"] = {"content": "<start>", "special": False}
    with pytest.raises(ValueError, match="special flag"):
        token_domain(tokenizer, config, 4)


def test_missing_added_tokens_decoder_is_built_from_added_tokens(tmp_path):
    tokenizer = {
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}},
        "added_tokens": [
            {
                "id": 2,
                "content": "<start>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
    }
    _write_text_resources(tmp_path, tokenizer, {"bos_token": "a"})
    references, payloads, count, special = load_resources(tmp_path, vocab_size=4)
    stored = json.loads(payloads[references["text"]["tokenizer_config.json"]])
    assert stored["added_tokens_decoder"] == {
        "2": {
            "content": "<start>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    }
    assert (count, special) == (3, (2,))


def test_drifted_split_pattern_is_normalized_to_standard(tmp_path):
    drifted = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}]+"
        r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    tokenizer = {
        "model": {"type": "BPE", "vocab": {"a": 0}},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": drifted},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
    }
    _write_text_resources(tmp_path, tokenizer, {})
    references, payloads, _, _ = load_resources(tmp_path, vocab_size=4)
    stored = json.loads(payloads[references["text"]["tokenizer.json"]])
    assert (
        stored["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        == STD_QWEN_SPLIT_PATTERN
    )


def test_standard_tokenizer_assets_are_not_reserialized(tmp_path):
    tokenizer = {
        "model": {"type": "BPE", "vocab": {"a": 0}},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": STD_QWEN_SPLIT_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
    }
    tok_bytes = json.dumps(tokenizer).encode("utf-8")
    cfg_bytes = json.dumps(
        {
            "add_bos_token": False,
            "add_prefix_space": False,
            "pad_token": "<|endoftext|>",
            "added_tokens_decoder": {},
        }
    ).encode("utf-8")
    (tmp_path / "tokenizer.json").write_bytes(tok_bytes)
    (tmp_path / "tokenizer_config.json").write_bytes(cfg_bytes)
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    (tmp_path / "generation_config.json").write_text("{}")
    references, payloads, _, _ = load_resources(tmp_path, vocab_size=4)
    assert payloads[references["text"]["tokenizer.json"]] == tok_bytes
    assert payloads[references["text"]["tokenizer_config.json"]] == cfg_bytes


def test_defective_prefix_semantics_are_normalized(tmp_path):
    # Mimics the NVIDIA NVFP4 export tokenizer_config.json: add_prefix_space is
    # correct, pad_token is the non-official <|im_end|>, and add_bos_token and
    # added_tokens_decoder are absent.
    tokenizer = {
        "model": {"type": "BPE", "vocab": {"a": 0}},
        "added_tokens": [
            {
                "id": 1,
                "content": "<start>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
    }
    config = {
        "add_prefix_space": False,
        "pad_token": "<|im_end|>",
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "tokenizer_class": "Qwen2Tokenizer",
    }
    _write_text_resources(tmp_path, tokenizer, config)
    references, payloads, _, _ = load_resources(tmp_path, vocab_size=4)
    stored = json.loads(payloads[references["text"]["tokenizer_config.json"]])
    # Every C++-required invariant now holds.
    assert stored["add_bos_token"] is False
    assert stored["add_prefix_space"] is False
    assert stored["pad_token"] == "<|endoftext|>"
    assert isinstance(stored["added_tokens_decoder"], dict)
    # Untouched source fields are preserved, not dropped.
    assert stored["eos_token"] == "<|im_end|>"
    assert stored["tokenizer_class"] == "Qwen2Tokenizer"
