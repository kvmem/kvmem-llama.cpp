from __future__ import annotations

import struct

import numpy as np
import torch

from tools.convert.methods import AuxiliaryValue, import_encoded
from tools.convert.model import Model, Parameter
from tools.convert.proposal import add_official_proposal, add_proposal
from tools.convert.recipe import Recipe
from tools.convert.resources import token_domain
from tools.convert.sources.logical import EncodedRows, LogicalSource, array_source


def test_proposal_rows_ids_and_uses_follow_final_token_domain(tmp_path):
    head = torch.arange(8 * 4).reshape(8, 4).to(torch.bfloat16)
    count, special = token_domain(
        {
            "model": {"vocab": {str(i): i for i in range(5)}},
            "added_tokens": [{"id": 5, "content": "<eos>", "special": True}],
        },
        {"added_tokens_decoder": {}},
        vocab_size=8,
    )
    model = Model(
        {
            "text": {"config": {"vocab_size": 8}},
            "mtp": {"config": {}, "target": "text"},
        },
        token_count=count,
        special_token_ids=special,
    )
    model.add(Parameter("text/output_head", (8, 4), array_source(head, "head")))
    ranking = tmp_path / "ranking.i64"
    np.array([[9, 5, 9, 5, 0, 0, 999, 999], [999] * 8], dtype="<i8").tofile(ranking)
    recipe = Recipe(model)
    add_proposal(recipe, ranking=ranking, rows=3)
    ids = model.parameters["proposal/token_ids"].source.values()
    selected = model.parameters["proposal/head"].source.values().reshape(3, 4)
    assert ids.dtype == torch.int32 and ids.tolist() == [0, 2, 5]
    assert torch.equal(selected, head[[0, 2, 5]])
    assert model.parameters["proposal/head"].inputs == ("mtp/final_hidden",)
    assert model.components["text"]["proposal"] == {"domain": "indexed", "rows": 3}


def test_proposal_gathered_from_a_rotated_head_keeps_its_rotation(tmp_path):
    head = torch.arange(8 * 4).reshape(8, 4).to(torch.bfloat16)
    model = Model(
        {
            "text": {"config": {"vocab_size": 8}},
            "mtp": {"config": {}, "target": "text"},
        },
        token_count=8,
    )
    model.add(
        Parameter(
            "text/output_head",
            (8, 4),
            array_source(head, "head"),
            inputs=("text/final_hidden", "mtp/final_hidden"),
        )
    )
    ranking = tmp_path / "ranking.i64"
    np.array([[3, 2, 1, 0, 0, 0, 0, 0]], dtype="<i8").tofile(ranking)
    recipe = Recipe(model)
    signs = AuxiliaryValue(
        "bf16", (4,), struct.pack("<4H", 0x3F80, 0xBF80, 0x3F80, 0xBF80)
    )
    recipe.use(
        "text/output_head", "text/final_hidden", auxiliaries={"hadamard_signs": signs}
    )
    add_proposal(recipe, ranking=ranking, rows=2)
    assert (
        recipe.auxiliary_overrides[
            ("proposal/head", "mtp/final_hidden", "hadamard_signs")
        ]
        == signs
    )


def test_proposal_of_an_imported_head_keeps_its_exact_rows_and_encoding(tmp_path):
    rows, k = 8, 128
    codes = torch.arange(rows * k // 4, dtype=torch.int32).reshape(rows, k // 4).to(torch.uint8)
    scales = (torch.arange(rows, dtype=torch.float32) + 1).reshape(rows, 1).to(torch.float16)

    def encoded(begin, end):
        return EncodedRows("t2_g128_fp16", codes[begin:end], scales[begin:end])

    def values(begin, end):
        return torch.zeros(end - begin, dtype=torch.float32)

    ternary = LogicalSource((rows, k), "ternary head", values, encoded)
    model = Model(
        {
            "text": {"config": {"vocab_size": rows}},
            "mtp": {"config": {}, "target": "text"},
        },
        token_count=rows,
    )
    model.add(
        Parameter(
            "text/output_head",
            (rows, k),
            array_source(torch.zeros(rows, k, dtype=torch.bfloat16), "base head"),
        )
    )
    ranking = tmp_path / "ranking.i64"
    np.array([[1, 9, 1, 1, 1, 7, 1, 1]], dtype="<i8").tofile(ranking)
    recipe = Recipe(model)
    recipe.assign("text/output_head", format="t2_g128_fp16", method=import_encoded, source=ternary)
    add_official_proposal(recipe, ranking=ranking, rows=3)
    selection = recipe.selections["proposal/head"][0]
    assert selection.format == "t2_g128_fp16" and selection.method is import_encoded
    ids = model.parameters["proposal/token_ids"].source.values().tolist()
    gathered = model.parameters["proposal/head"].source.read_encoded(0, 3)
    assert torch.equal(gathered.codes, codes[ids]) and torch.equal(gathered.scales, scales[ids])
