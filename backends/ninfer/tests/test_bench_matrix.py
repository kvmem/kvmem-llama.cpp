from __future__ import annotations

import json
import pytest

from tools.bench.run_ninfer_bench_matrix import BenchCase, load_bench_report, report_rows


@pytest.mark.parametrize("ngram_enabled", [False, True])
def test_schema_v16_report_is_flattened_for_matrix_summary(tmp_path, ngram_enabled) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 16,
                "artifact_type": "ninfer_bench_report",
                "tool": "ninfer_bench",
                "artifact": {"path": "model.ninfer"},
                "environment": {"gpu_name": "RTX 5090"},
                "load": {
                    "architecture": "Qwen3_5ForCausalLM",
                    "name": "Qwen3.8-27B",
                    "formats": "nvfp4",
                    "prefill_signature": "qwen3_5-nvfp4",
                    "load_seconds": 2.5,
                    "upload_seconds": 2.0,
                    "artifact_bytes_read": 17_500_000_000,
                    "host_to_device_bytes": 17_400_000_000,
                    "peak_staging_bytes": 134_217_728,
                },
                "memory": {
                    "kv_capacity": 8192,
                    "kv_payload_bytes": 123_456,
                    "weights": {"capacity_bytes": 17_400_000_000},
                    "sequence": {"capacity_bytes": 2_000_000_000},
                    "workspace": {"capacity_bytes": 100_000_000},
                    "vision_workspace": {
                        "general_capacity_bytes": 75_000_000,
                        "handoff_capacity_bytes": 50_000_000,
                    },
                    "cuda_graph_allowance_bytes": 150_000_000,
                },
                "config": {
                    "max_context": 4096,
                    "prefill_chunk": 1024,
                    "kv_cache": "int8-group64",
                    "speculative_backend": "mtp",
                    "draft_tokens": 5,
                    "ngram_draft_tokens": 31 if ngram_enabled else 0,
                    "ngram_min_match": 12,
                    "proposal_head": "optimized",
                    "decode_path": "cuda-graph",
                    "decode_graph_prime": {"primed": True, "output_tokens": 65 if ngram_enabled else 13},
                    "repetitions": 2,
                    "warmup": 1,
                },
                "tests": [
                    {
                        "label": "tg3",
                        "kind": "tg",
                        "n_prompt": 0,
                        "n_gen": 3,
                        "requested_output_tokens": 4,
                        "workspace_peak_bytes": 1_048_576,
                        "workspace_allocator_peak_bytes": 524_288,
                        "decode_output_tok_s_mean": 4.5,
                        "decode_engine_tok_s_mean": 7.5,
                        "total_seconds_mean": 0.875,
                        "speculative": {
                            "acceptance_rate": 1.0,
                            "acceptance_length": 5.0,
                            "rounds": 1,
                            "drafted_tokens": 5,
                            "accepted_tokens": 5,
                            "ngram_rounds": 1 if ngram_enabled else 0,
                            "ngram_drafted_tokens": 5 if ngram_enabled else 0,
                            "ngram_accepted_tokens": 5 if ngram_enabled else 0,
                            "fallback_steps": 3,
                            "accepted_per_position": [1, 1, 1, 1, 1],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = report_rows(
        report_path,
        BenchCase("pure_decode", "tg3_k5_graph", (), repetitions=2, warmup=1),
    )

    assert len(rows) == 1
    row = rows[0]
    assert (row["suite"], row["case"], row["label"], row["kind"]) == (
        "pure_decode",
        "tg3_k5_graph",
        "tg3",
        "tg",
    )
    assert (row["architecture"], row["weight_formats"], row["artifact_path"], row["gpu_name"]) == (
        "Qwen3_5ForCausalLM",
        "nvfp4",
        "model.ninfer",
        "RTX 5090",
    )
    assert (row["decode_path"], row["decode_graph_primed"]) == (
        "cuda-graph",
        True,
    )
    assert (row["speculative_backend"], row["draft_tokens"]) == ("mtp", 5)
    assert (row["ngram_draft_tokens"], row["ngram_min_match"]) == (31 if ngram_enabled else 0, 12)
    assert (row["ngram_rounds"], row["ngram_drafted_tokens"], row["ngram_accepted_tokens"]) == (
        (1, 5, 5) if ngram_enabled else (0, 0, 0)
    )
    assert row["decode_graph_prime_output_tokens"] == (65 if ngram_enabled else 13)
    assert row["kv_capacity"] == 8192
    assert row["host_to_device_bytes"] == 17_400_000_000
    assert row["workspace_capacity_bytes"] == 100_000_000
    assert row["workspace_general_capacity_bytes"] == 75_000_000
    assert row["vision_handoff_capacity_bytes"] == 50_000_000
    assert row["cuda_graph_allowance_bytes"] == 150_000_000
    assert row["workspace_peak_bytes"] == 1_048_576
    assert row["workspace_allocator_peak_bytes"] == 524_288
    assert row["decode_output_tok_s_mean"] == 4.5
    assert row["decode_engine_tok_s_mean"] == 7.5
    assert row["spec_fallback_steps"] == 3
    assert row["spec_accepted_per_position"] == "[1,1,1,1,1]"


@pytest.mark.parametrize("version", [15, 17])
def test_matrix_rejects_different_report_schema(tmp_path, version) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"schema_version": version,
                                       "artifact_type": "ninfer_bench_report",
                                       "tool": "ninfer_bench"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported benchmark report identity"):
        load_bench_report(report_path)
