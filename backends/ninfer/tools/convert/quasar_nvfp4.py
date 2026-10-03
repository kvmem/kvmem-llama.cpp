"""Convert the QUASAR Qwen3.8-27B NVFP4 checkpoint to a v3 artifact.

The public ``qwen3_8_27b_nvfp4`` recipe produces a mixed layout (NVFP4 MLP plus
FP8 attention) and does not match the QUASAR checkpoint, which quantises every
Linear layer to NVFP4 and keeps the embeddings, visual encoder, and MTP head in
BF16. This driver therefore uses a dedicated recipe that preserves the original
NVFP4 codes, group scales and weight divisor verbatim (``import_encoded``) for
every quantised Linear projection, keeps the BF16/FP32 parameters in their
source format, and grafts the DFlash2 draft verbatim from a reference v3
artifact (its draft weights keep the reference's exact storage, so nothing is
re-quantised there either).

One unavoidable representation change: the GDN ``a``/``b`` projections are
48-row NVFP4 tensors, but the v3 NVFP4 layout requires 128-row tiles, so they
cannot be imported as NVFP4. They are dequantised to BF16 (an upcast, not a
re-quantisation) and stored as a single BF16 object.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import time

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter, DEFAULT_MAX_FILE_BYTES
from tools.convert.methods import cast_direct, grouped_absmax, import_encoded
from tools.convert.pipeline import _json_default
from tools.convert.proposal import DEFAULT_RANKING, add_official_proposal
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource

# Text Linear projections whose row count is a multiple of the NVFP4 128-row tile.
_NVFP4_PROJECTIONS = (
    "/attention/query",
    "/attention/key",
    "/attention/value",
    "/attention/output",
    "/attention/gate",
    "/gdn/query",
    "/gdn/key",
    "/gdn/value",
    "/gdn/z",
    "/gdn/output",
    "/mlp/gate",
    "/mlp/up",
    "/mlp/down",
)
# GDN a/b are 48-row NVFP4 in the source; the v3 NVFP4 layout needs 128-row
# tiles, so dequantise them to BF16 instead of importing the codes.
_DEQUANTISE_TO_BF16 = ("/gdn/a_projection", "/gdn/b_projection")


# MTP K/V must be Q8: the fused linear_pair op hard-requires Q8 with n==1024.
# The MTP tower shares the main model's text/output_head, so re-quantising it to
# Q8 also switches the main model's logits to the fast n248320_k5120 kernel.
_MTP_KV = ("mtp/layers/0/attention/key", "mtp/layers/0/attention/value")
# The rest of the MTP transformer layer (q8_scope="full" also re-quantises these).
_MTP_LAYER = (
    "mtp/input_projection",
    "mtp/layers/0/attention/query",
    "mtp/layers/0/attention/gate",
    "mtp/layers/0/attention/output",
    "mtp/layers/0/mlp/gate",
    "mtp/layers/0/mlp/up",
    "mtp/layers/0/mlp/down",
)


def apply_quasar_recipe(model, recipe, sources, q8_scope="full") -> None:
    quantized = sources["quantized"]
    # q8_scope="kv": only MTP K/V + shared lm_head are Q8.
    # q8_scope="full": the whole MTP transformer layer + shared lm_head are Q8.
    q8_names = set(_MTP_KV) | (set(_MTP_LAYER) if q8_scope == "full" else set())
    q8_names.add("text/output_head")
    for name, parameter in model.parameters.items():
        if name in q8_names:
            recipe.assign(
                name,
                format="q8_g32_fp16",
                method=grouped_absmax,
                source=model.source(name, quantized),
            )
        elif name.startswith("text/"):
            if name.endswith(_NVFP4_PROJECTIONS):
                recipe.assign(
                    name,
                    format="nvfp4",
                    method=import_encoded,
                    source=model.source(name, quantized, "nvfp4"),
                    activation_policy="AllowA4",
                )
            elif name.endswith(_DEQUANTISE_TO_BF16):
                recipe.assign(
                    name,
                    method=cast_direct,
                    source=model.source(name, quantized, "nvfp4"),
                )
            else:
                recipe.assign(name, source=model.source(name, quantized))
        elif name.startswith(("vision/", "mtp/")):
            # QUASAR keeps the visual encoder and MTP head in BF16
            # (unless a name above is Q8-assigned).
            recipe.assign(name, source=model.source(name, quantized))
        # dflash2 is grafted after preparation, so it is not assigned here.


def _collect_dflash2(official: Artifact) -> dict:
    """Collect the dflash2 objects, bindings, uses and component for a byte-copy graft."""
    bindings: dict = {}
    object_ids: set[str] = set()
    for name, binding in official.directory.bindings.items():
        if not name.startswith("dflash2"):
            continue
        bindings[name] = deepcopy(binding)
        if isinstance(binding, dict):
            if "object" in binding:
                object_ids.add(binding["object"])
            for part in binding.get("parts", []):
                object_ids.add(part["object"])
    uses = [
        deepcopy(use)
        for use in official.directory.uses
        if use.get("parameter", "").startswith("dflash2")
    ]
    for use in uses:
        for ref in (use.get("auxiliaries") or {}).values():
            object_ids.add(ref["object"])
    component = deepcopy(official.directory.components["dflash2"])

    old_ids = sorted(object_ids)
    id_map: dict[str, str] = {}
    specs: list[TensorSpec] = []
    for index, old_id in enumerate(old_ids):
        obj = official.by_id[old_id]
        new_id = f"weight/{100000 + index:06d}"
        id_map[old_id] = new_id
        specs.append(TensorSpec(new_id, tuple(obj.shape), obj.format, obj.layout))

    def remap(binding):
        if not isinstance(binding, dict):
            return binding
        if "object" in binding:
            binding["object"] = id_map[binding["object"]]
        for part in binding.get("parts", []):
            part["object"] = id_map[part["object"]]
        return binding

    for use in uses:
        for role, ref in (use.get("auxiliaries") or {}).items():
            ref["object"] = id_map[ref["object"]]

    return {
        "specs": specs,
        "bindings": {name: remap(b) for name, b in bindings.items()},
        "uses": uses,
        "component": component,
        "id_map": id_map,
        "old_ids": old_ids,
    }


def _add_dflash2_head_uses(recipe, params) -> None:
    """Declare the grafted dflash2 component's uses of the shared heads.

    The dflash2 byte-clone above only carries uses for ``dflash2/*``
    parameters; its final projection consumes ``text/output_head`` (and,
    when a proposal head is present, ``proposal/head``) through the
    ``dflash2/final_hidden`` input. Without these Use entries the engine
    fails planning with ``missing Use text/output_head@dflash2/final_hidden``.
    The policies match the reference artifact's declarations.
    """
    for name, policy in (("text/output_head", "AllowA8"),
                         ("proposal/head", "A16Only")):
        if name not in params:
            continue
        param = params[name]
        if "dflash2/final_hidden" not in param.inputs:
            params[name] = replace(
                param, inputs=param.inputs + ("dflash2/final_hidden",))
            recipe.policies[(name, "dflash2/final_hidden")] = policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="QUASAR safetensors directory")
    parser.add_argument(
        "--dflash2",
        required=True,
        help="reference v3 artifact to graft the dflash2 draft from",
    )
    parser.add_argument("--out", required=True, help="output .ninfer path")
    parser.add_argument(
        "--q8-scope",
        choices=("kv", "full"),
        default="full",
        help=(
            "Q8 scope: 'kv' re-quantises MTP K/V + shared lm_head; "
            "'full' re-quantises the whole MTP transformer layer + shared lm_head "
            "(default: full)"
        ),
    )
    parser.add_argument("--name", default="qwen3.8-27b-quasar")
    parser.add_argument(
        "--proposal",
        action="store_true",
        help="include the indexed proposal head (--lm-head-draft support)",
    )
    parser.add_argument("--proposal-rows", type=int, default=131072)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument(
        "--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES
    )
    args = parser.parse_args()

    out = Path(args.out)
    report_path = Path(str(out) + ".conversion.json")
    if out.exists() or report_path.exists():
        raise FileExistsError(f"conversion output already exists: {out} or {report_path}")

    start = time.perf_counter()
    base = SafetensorsSource(args.model)
    model = build_model(base, components=("text", "vision", "mtp"))
    recipe = Recipe(model)
    apply_quasar_recipe(model, recipe, {"quantized": base}, args.q8_scope)
    if args.proposal:
        add_official_proposal(
            recipe, ranking=args.ranking, rows=args.proposal_rows)
    _add_dflash2_head_uses(recipe, model.parameters)
    prepared = recipe.prepare(device=args.device, rows_per_chunk=args.rows_per_chunk)

    # Read the grafted dflash2 payload while the reference is still open.
    dflash2_artifact_id = None
    with Artifact(args.dflash2) as official:
        dflash2_artifact_id = official.artifact_id.hex()
        graft = _collect_dflash2(official)
        dflash2_data = {
            graft["id_map"][old_id]: official.read_object(old_id)
            for old_id in graft["old_ids"]
        }

    resources = [
        ResourceSpec(key, len(value)) for key, value in model.resources.items()
    ]
    specs = (
        resources
        + [job.spec for job in prepared.weights]
        + [spec for spec, _ in prepared.auxiliaries]
        + graft["specs"]
    )
    components = dict(model.components)
    components["dflash2"] = graft["component"]
    bindings = dict(prepared.bindings)
    bindings.update(graft["bindings"])
    uses = list(prepared.uses) + graft["uses"]
    provenance = {
        "model": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 (compressed-tensors nvfp4-pack-quantized)",
        "dflash2_source": str(args.dflash2),
        "dflash2_artifact_id": dflash2_artifact_id,
        "q8_scope": args.q8_scope,
        "method": (
            "import_encoded (nvfp4, bit-exact) for quantised Linear projections; "
            "grouped_absmax q8_g32_fp16 for MTP head (scope=%s) + shared lm_head; "
            "cast_direct bf16 for GDN a/b (48-row, upcast); source-format otherwise; "
            "dflash2 grafted verbatim" % args.q8_scope
        ),
    }
    if args.proposal:
        provenance["proposal"] = {
            "rows": args.proposal_rows,
            "ranking": str(args.ranking),
            "method": ("indexed proposal head gathered from the QUASAR output "
                       "head, grouped_absmax; dflash2/final_hidden head uses "
                       "declared for the grafted dflash2 component"),
        }

    report = {
        "components": components,
        "name": args.name,
        "output": str(out),
        "device": args.device,
        "rows_per_chunk": args.rows_per_chunk,
        "provenance": provenance,
        "parameters": len(model.parameters),
        "uses": len(uses),
        "objects": len(specs),
        "formats": dict(Counter(job.spec.format for job in prepared.weights)),
        "methods": [
            {
                "object": job.spec.id,
                "parameters": job.parameters,
                "method": job.method_name,
                "format": job.spec.format,
                "shape": list(job.spec.shape),
            }
            for job in prepared.weights
        ],
    }

    with ArtifactWriter(
        out,
        specs,
        components=components,
        bindings=bindings,
        uses=uses,
        metadata={"name": args.name},
        provenance=provenance,
        max_file_bytes=args.max_file_bytes,
    ) as writer:
        for object_id, data in model.resources.items():
            writer.write_object(object_id, data)
        total = len(prepared.weights)
        for index, job in enumerate(prepared.weights):
            if index % 100 == 0:
                print(f"  weight {index}/{total}", flush=True)
            job.prepared.produce(TensorOutput(writer, job.spec.id))
        for spec, data in prepared.auxiliaries:
            writer.write_object(spec.id, data)
        for new_id, data in dflash2_data.items():
            writer.write_object(new_id, data)
        report["artifact_id"] = writer.artifact_id.hex()
        report["files"] = [
            {
                "path": str(out if i == 0 else out.parent / file.path),
                "payload_bytes": file.payload_bytes,
            }
            for i, file in enumerate(writer.directory.files)
        ]
        report["payload_bytes"] = writer.directory.payload_bytes

    report["seconds"] = time.perf_counter() - start
    temporary = Path(str(report_path) + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(
            report, stream, ensure_ascii=False, allow_nan=False,
            default=_json_default, indent=2,
        )
        stream.write("\n")
    temporary.replace(report_path)
    base.close()
    print(f"wrote {out} ({report['payload_bytes']} payload bytes, {len(specs)} objects) in {report['seconds']:.1f}s")
    print("formats:", report["formats"])


if __name__ == "__main__":
    main()
