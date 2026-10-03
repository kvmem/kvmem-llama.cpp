"""The existing frequency shortlist algorithm, using the final tokenizer resources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .methods import import_encoded
from .model import Parameter
from .recipe import Recipe
from .sources.logical import array_source, gather_source

DEFAULT_RANKING = (
    Path(__file__).resolve().parents[1]
    / "freq_corpus/fixtures/ranking/ranking.train.counts.i64"
)


def select_shortlist(total: np.ndarray, n: int, force_include=()) -> np.ndarray:
    counts = np.asarray(total, dtype=np.int64)
    if counts.ndim != 1 or not 0 < n <= counts.size:
        raise ValueError("shortlist size is outside the frequency domain")
    order = np.argsort(-counts, kind="stable")
    forced = np.array(
        sorted({int(i) for i in force_include if 0 <= int(i) < counts.size}),
        dtype=np.int64,
    )
    if forced.size >= n:
        return np.ascontiguousarray(
            forced[np.argsort(-counts[forced], kind="stable")][:n]
        )
    forced_set = set(forced.tolist())
    wanted = n - forced.size
    picked = []
    for token_id in order.tolist():
        if token_id not in forced_set:
            picked.append(token_id)
            if len(picked) == wanted:
                break
    selected = np.concatenate((np.asarray(picked, dtype=np.int64), forced))
    return np.ascontiguousarray(selected[np.argsort(-counts[selected], kind="stable")])


def add_proposal(
    recipe: Recipe,
    *,
    ranking: str | Path = DEFAULT_RANKING,
    rows: int = 131072,
    source=None,
    format="q4_g64_fp16",
) -> None:
    model = recipe.model
    if "proposal/head" in model.parameters:
        raise ValueError("proposal head already configured")
    vocab = model.config["vocab_size"]
    ranking = Path(ranking)
    if ranking.stat().st_size == 0 or ranking.stat().st_size % (vocab * 8):
        raise ValueError("ranking must contain complete int64 vocabulary rows")
    total = np.fromfile(ranking, dtype="<i8", count=vocab)
    selected = select_shortlist(
        total[: model.token_count], rows, model.special_token_ids
    )
    ids = torch.from_numpy(selected).to(torch.int32)
    derived = source is None
    encoded = None
    if source is None:
        selections = recipe.selections["text/output_head"]
        if len(selections) != 1:
            raise ValueError(
                "provide a proposal source when the main head uses split sources"
            )
        source = selections[0].source
        # A head imported in its source's own encoding (a ternary checkpoint's T2 rows) lends the
        # proposal those exact rows in the same encoding.
        if selections[0].method is import_encoded:
            encoded = selections[0].format
    head = gather_source(source, ids)
    inputs = tuple(
        component + "/final_hidden"
        for component in ("mtp", "dflash", "dflash2")
        if component in model.components
    )
    model.components["text"]["proposal"] = {"domain": "indexed", "rows": rows}
    model.add(
        Parameter(
            "proposal/head", head.shape, head, inputs=inputs, residency="proposal"
        )
    )
    model.add(
        Parameter(
            "proposal/token_ids",
            (rows,),
            array_source(ids, f"shortlist({ranking})"),
            direct_format="int32",
            residency="proposal",
        )
    )
    recipe.add_parameter("proposal/head")
    recipe.add_parameter("proposal/token_ids")
    if encoded is not None:
        recipe.assign("proposal/head", format=encoded, method=import_encoded)
    else:
        recipe.assign("proposal/head", format=format, method="grouped_absmax")
    if derived:
        # Rows gathered from a Hadamard-rotated head keep that head's input rotation.
        signs = next(
            (
                value
                for (parameter, _, role), value in recipe.auxiliary_overrides.items()
                if parameter == "text/output_head" and role == "hadamard_signs"
            ),
            None,
        )
        if signs is not None:
            for input_name in inputs:
                recipe.use(
                    "proposal/head", input_name, auxiliaries={"hadamard_signs": signs}
                )


def add_official_proposal(
    recipe: Recipe, *, ranking=DEFAULT_RANKING, rows=131072
) -> None:
    # The rows the output head is converted from: the recipe's own source when it selected one (a
    # ternary checkpoint's head), the base checkpoint's head otherwise.
    base = recipe.model.parameters["text/output_head"].source
    selections = recipe.selections["text/output_head"]
    if len(selections) == 1 and selections[0].source is not base:
        add_proposal(recipe, ranking=ranking, rows=rows)
        return
    add_proposal(recipe, ranking=ranking, rows=rows, source=base)
