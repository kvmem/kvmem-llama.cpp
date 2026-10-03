# Weight conversion

NInfer's converter creates `.ninfer` artifacts from local weights and a Python recipe. A recipe
can reuse an official conversion, change selected layers or projections, combine sources, or call
your own conversion method. The artifact contains the resulting configuration, encoded weights,
logical bindings and frontend resources.

Run the commands below from the repository root.

## Upgrade an existing v2 artifact

The offline upgrade tool supports the official Qwen3.6/3.8-27B groupwise-int and NVFP4 artifacts,
and Qwen3.6-35B-A3B groupwise-int. Update your checkout to the current `master` and
[rebuild NInfer](../README.md#quick-start), then run with Python 3.11:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  models/qwen3_8_27b_nvfp4.ninfer \
  models/qwen3_8_27b_nvfp4.v3.ninfer
```

The output must use a new path. After upgrading, use it directly or rename it to replace the
original file. Stored weight values and formats are preserved. The upgrade also installs the
matching template from `tools/chat_templates/`. Published SHA-256 checksums apply only to
downloaded files.

## Start with an official recipe

Source-weight conversion requires a Python 3.11 environment with PyTorch and NumPy. It uses CUDA
by default; `--device cpu` selects CPU conversion. The input paths below are placeholders for your
local checkpoint directories.

For Qwen3.6-27B floating-point source weights:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --name qwen3.6-27b \
  --out models/qwen3_6_27b.ninfer
```

`--components` defaults to `text`. Include only the optional components you want to distribute.
`--proposal` adds the indexed proposal head used by speculative decoding; it uses the repository's
token ranking and defaults to 131,072 rows. The ordinary full-vocabulary output head is retained.

The built-in recipes are ordinary Python functions in
[`official_recipes.py`](../tools/convert/official_recipes.py):

| Recipe | Main representation choices | Additional source |
|---|---|---|
| `qwen3_6_27b` | Q4/Q5 projections, Q6 vocabulary weights | None |
| `qwen3_8_27b` | Q4/Q5 projections, Q8 vocabulary weights | None |
| `qwen3_8_27b_q6` | Q4/Q5 projections, Q6 MLP gate/up, Q8 vocabulary weights | None |
| `qwen3_6_35b_a3b` | Q4 experts, Q5/Q6 expert down, Q8 shared/projection weights | None |
| `qwen3_6_35b_a3b_nvfp4` | Imported NVFP4 routed and shared experts, Q8 projection weights, Q8/Q6 vocabulary weights | `quantized` |
| `qwen3_6_27b_nvfp4` | Imported NVFP4, selected BF16 projections, Q8 vocabulary weights | `quantized` |
| `qwen3_8_27b_nvfp4` | Imported NVFP4/FP8, FP8 embedding generated from BF16 | `quantized` |
| `bonsai2_27b_ternary` | Imported ternary T2 text tower with Hadamard-rotated Uses, Q8 primal embedding | `ternary` (GGUF) |
| `qwen3_8_27b_gguf` | Every text, embedding, head and MTP tensor in its GGUF block format, byte for byte | `gguf` (GGUF), `vision` (mmproj GGUF) |

These names select conversion choices. Runtime execution is selected from the architecture,
configuration and actual bindings stored in the artifact. `--name` sets the public model name;
it does not select kernels.

`qwen3_8_27b_q6` differs from `qwen3_8_27b` in one place: the MLP gate and up projections carry Q6
instead of Q4. Those two parameters are 43% of the text weights, and at 4.25 bits per weight Q4
leaves precision unused there (11.5% relative weight error against 2.6% for Q6 at 6.25 bits per
weight). The vocabulary endpoints deliberately stay Q8, which at 8.5 bits per weight already
outranks Q6.

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_q6 \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b-q6 \
  --out models/qwen3_8_27b_q6.ninfer
```

Measured with `ninfer-perplexity` over a 47,917-token corpus at `--context 4096 --stride 2048`
with `--kv-dtype int8`, this artifact scores 1.520765 against 1.524538 for `qwen3_8_27b` and
1.530857 for `qwen3_8_27b_nvfp4`.

The FFN evaluates a Q6 gate/up pair through its existing materialized decomposition (the generic
`linear` op followed by `silu_mul`) rather than the fused `linear_swiglu` op, which has no Q6
variant. That decomposition costs nothing measurable: forcing the fused Q4 route through the same
decomposition moves the corpus perplexity from 1.524538 to 1.524618.

For a Qwen3.8-27B NVFP4/FP8 artifact with DFlash2:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_nvfp4.ninfer
```

Vision uses the main source, and so does MTP unless `--source mtp=PATH` supplies a separately trained
head under the checkpoint's `mtp.*` tensor names. DFlash and DFlash2 use the corresponding named
source, supplied as `--source dflash=PATH` or `--source dflash2=PATH`. An artifact may contain several optional
components; the Engine loads only the ones selected at startup, including at most one speculative
backend. Component availability and startup selection are independent.

### Ternary Bonsai 2 27B

`bonsai2_27b_ternary` ([`ternary.py`](../tools/convert/ternary.py)) builds a Qwen3.8-27B artifact
whose text tower comes from PrismML's `Ternary-Bonsai-2-27B-PQ2_0.gguf`. Every text projection
except the GDN A/B controls, the output head and the token-embedding table are stored as
`t2_g128_fp16`: the ternary codes and their per-128 scales are imported without rounding. Those
matrices are Hadamard-rotated in the checkpoint, so each projection's Uses carry the
`hadamard_signs` auxiliary of its input width (three shared vectors in total), and the runtime
rotates the matching activations. The token table has no Use of its own: the runtime restores each
gathered row to the primal basis with the hidden-width signs that the output head carries. The
recipe also undoes llama.cpp's exporter conventions: GDN value heads return from the tiled to the
grouped order, norms from `1 + w` to `w`, and `ssm_a` to `A_log`. Vision and the frontend
resources come from the Qwen3.8-27B checkpoint, which shares the geometry. The published artifact
takes its MTP head and DFlash2 adapter from ProCreations' heads trained on Bonsai 2 itself
([MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP),
[DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2)), which accept more drafts
than the vanilla ones. The recipe stores the adapter's feature, output and MLP projections in Q4,
since against a 2.125-bit target the drafter is a large share of every draft step's bytes; the
fused query/key/value projection stays Q8, which its three-output op requires.

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe bonsai2_27b_ternary \
  --source ternary=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --source mtp=/path/to/Ternary-Bonsai-2-27B-MTP/model_mtp.safetensors \
  --source dflash2=/path/to/Ternary-Bonsai-2-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name bonsai2-27b \
  --out models/bonsai2_27b.ninfer
```

Without `--source mtp` the MTP head comes from the checkpoint, and `--source
dflash2=/path/to/Qwen3.8-27B-DFlash2` selects z-lab's adapter. `--proposal` takes the proposal
head's rows from the ternary output head itself: PrismML's T2 rows of the 131,072 most frequent
tokens, in their own encoding and with the head's rotation, so a draft through `--lm-head-draft`
scores those tokens exactly as the full head does. It adds 170 MiB to the file and is loaded only
with `--lm-head-draft`.

A named source whose path ends in `.gguf` opens as a GGUF file; the recipe validates its header,
tensor set and Hadamard metadata before reading anything.

### A mixed-precision Qwen3.8-27B GGUF

`qwen3_8_27b_gguf` ([`gguf_blocks.py`](../tools/convert/gguf_blocks.py)) imports a Qwen3.8-27B GGUF
that assigns its own ggml block type to every tensor, such as ISTA-DASLab's
[GSQ-RCO releases](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF). The text
projections, token table, output head and, in an `-mtp` file, the MTP head keep their blocks: each
is stored as the matching `gguf_*` format in the `gguf_blocks_v1` layout, so a row of the artifact
is a row of the GGUF, byte for byte, and nothing is requantized. All fifteen block types llama.cpp
writes for dense models are supported: Q8_0, Q2_K to Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS. The recipe undoes llama.cpp's Qwen3.5 exporter conventions the
same way `bonsai2_27b_ternary` does, by row gathers and exact small-tensor transforms. The one
convention a row copy cannot undo is the tiled value-head order of the GDN output projection's
input columns: its Use carries an `input_columns` auxiliary, and the runtime reads the activation
through that permutation when it quantizes it. Vision comes from the release's `mmproj` file in the
official Vision formats, DFlash2 from `--source dflash2`, and `--proposal` gathers the proposal
head's rows from the output head in its own block format.

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_gguf \
  --source gguf=/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --source vision=/path/to/mmproj-Qwen3.8-27B-BF16.gguf \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_gsq_rco_iq3_s.ninfer
```

`--model` needs only the base checkpoint's configuration and tokenizer files. The products that
serve these formats are described in [GGUF block formats](gguf.md).

## Change part of a recipe

Save the following as `my_recipe.py`:

```python
from tools.convert.official_recipes import qwen3_6_27b


def configure(model, recipe, sources):
    qwen3_6_27b(model, recipe, sources)
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method="grouped_absmax",
    )
```

Run it with the same source and component selection:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe my_recipe.py \
  --components text,vision,mtp \
  --proposal \
  --out models/my_qwen.ninfer
```

The default entry function is `configure`; `--recipe my_recipe.py:customize` selects another
function. Alternatively, use an official `--recipe` and put only the changes in an `--override`
file. Overrides run after the base recipe and optional proposal-head setup.

`model.parameters` maps logical names to their shape, source and mathematical inputs. To see the
available names for your selected components, a recipe can print them:

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```

`recipe.assign` accepts one name, a list of names, or shell-style patterns such as
`text/layers/*/mlp/down`. A selector that matches nothing fails. Assignments run in Python order;
later assignments replace only the explicitly supplied choices. Changing `format` does not
automatically change `method`. Use `layout="auto"` to select the registered layout for the format
when overriding an earlier explicit layout choice.

The principal choices are:

| Argument | Meaning |
|---|---|
| `format` | Persistent numeric format |
| `layout` | Physical encoding; inferred from format unless explicitly set |
| `method` | Built-in method name or Python callable |
| `source` | Logical values or encoded rows from the selected source |
| `parameters` | JSON-serializable numerical parameters passed to the method |
| `rows=(begin, end)` | Override complete leading-axis rows in a half-open range |
| `activation_policy` | Permission for activation precision at the parameter's mathematical inputs |

For example, `rows=(0, 128)` can give the first 128 rows a different format. This creates multiple
physical parts when necessary. The container can represent that result; the intended Op must also
support consuming those parts. Current native projections generally require a contiguous parent
region, so arbitrary splits of one projection are not automatically executable.

## Formats, methods and activation precision

The converter currently writes these formats:

| Format | Built-in method for floating-point input | Import of already encoded input |
|---|---|---|
| `bf16`, `fp32`, `int32` | `cast_direct` | Direct words through the source reader |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | `grouped_absmax` | Supply a custom method/source if needed |
| `fp8_e4m3fn_row_bf16` | `fp8_row_maxabs` | `import_encoded` |
| `nvfp4` | Supply a custom quantizer | `import_encoded` |

`grouped_absmax` stores one FP16 scale per group and signed integer codes. `fp8_row_maxabs` first
rounds input values to BF16, then produces E4M3FN codes and one BF16 multiplier per row.
`import_encoded` preserves compatible code and scale words, including NVFP4's matrix weight divisor.
It does not dequantize and requantize them.

The exact numeric and packing rules are in [numeric formats](maintainer/tensor-formats.md) and
[storage layouts](maintainer/storage-layouts.md). Source format names alone do not establish
compatibility: scale direction, granularity, code meaning and axis order must also match.

Activation permissions are independent of the stored weight format:

| Policy | Permitted activation paths |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16, A8 |
| `AllowA4` | A16, A8, A4 |

They permit choices; they do not force a kernel to use the lowest precision. A fused operation that
shares one activation across several projections must respect the intersection of their
permissions. `recipe.use(parameter, input_name, ...)` can set one mathematical input independently;
the names are available in `parameter.inputs`.

An NVFP4 A4 input requires a positive finite activation divisor. `import_encoded` obtains it from
the selected source, or a recipe supplies it through
`recipe.use(..., auxiliaries={"activation_input_divisor": value})`. Shared weights retain separate
Use records; sharing weights does not share calibration implicitly.

## Fused parents and logical projections

The Qwen adapter exposes Q, K, gate and V separately, even when they came from fused source tensors.
It also supplies finite packing groups for attention, GDN, MLP and MoE. Compatible selections are
packed into a shared parent automatically by the built-in methods.

For the Dense groupwise recipe, attention Q/K form one Q4 parent and gate/V form one Q5 parent.
The native fused Op receives two weights. To use the supported single-parent FP8 form, an override
can assign all four projections together:

```python
def configure(model, recipe, sources):
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind != "full_attention":
            continue
        prefix = f"text/layers/{layer}/attention/"
        recipe.assign(
            [prefix + role for role in ("query", "key", "gate", "value")],
            format="fp8_e4m3fn_row_bf16",
            layout="auto",
            method="fp8_row_maxabs",
            activation_policy="AllowA8",
        )
```

Use this file as `--override` after `qwen3_6_27b` or `qwen3_8_27b`. It leaves the other parameters
under that recipe. The adapter handles source Q/gate row order; the override works with logical
projections. Changing their representation can change numerical results and the physical kernels.

For explicit organization, `recipe.group([names...])` concatenates compatible selections in the
given order, `recipe.separate(names)` disables automatic grouping for those parameters, and
`recipe.share(parameter, target)` binds equal-shaped parameters to the same physical data.
Explicit groups must be disjoint and use unsplit selections with matching format, layout, method
and method parameters. NVFP4 parents also require a common weight divisor. Automatic grouping is
limited to built-in methods; custom methods can request explicit groups.

Grouping chooses storage. Model execution code chooses the supported fused implementation. The
loader uploads the stored representation, without repacking an inconvenient arrangement.

## Read another source

`--model` supplies the main config, default resources and the source named `base`. Add other
Safetensors sources with repeated `--source NAME=PATH`; they are opened when used. Single-file
Safetensors and indexed shards are supported. Additional tensor-only sources can omit model config;
sources carrying config are checked against the relevant model geometry.

To replace a logical parameter from another compatible checkpoint in a recipe:

```python
name = "text/layers/0/mlp/down"
recipe.assign(
    name,
    source=model.source(name, sources["alternate"]),
    format="q6_g64_fp16",
    method="grouped_absmax",
)
```

Supply `--source alternate=/path/to/alternate-checkpoint`. `model.source` applies the architecture's
source-name and axis mapping, including Q/gate extraction. The built-in compressed-tensors reader
understands the implemented per-row FP8 and NVFP4 code/scale conventions. It can expose decoded
values for another quantizer or encoded rows for exact import.

For another file format or quantization convention, provide a `LogicalSource`. Its value reader
accepts flat C-order element bounds and returns exactly that range. For example, a recipe can read
a logical matrix from a NumPy file stored beside the recipe:

```python
from pathlib import Path
import numpy as np
import torch
from tools.convert.sources.logical import LogicalSource


def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    path = Path(__file__).with_name("mlp-down.npy")
    data = np.load(path, mmap_mode="r")
    if tuple(data.shape) != model.parameters[name].shape or not data.flags.c_contiguous:
        raise ValueError("mlp-down.npy must have the logical shape and C-order storage")

    def read_values(begin, end):
        values = data.reshape(-1)[begin:end].astype(np.float32, copy=True)
        return torch.from_numpy(values)

    source = LogicalSource(tuple(data.shape), str(path), read_values)
    recipe.assign(name, source=source, format="q6_g64_fp16", method="grouped_absmax")
```

Use this as an override. The NumPy file must already follow the logical row/column order. For an
unfamiliar quantized source, its reader performs the corresponding decoding before returning
values. To preserve existing compatible encoded words, also provide `read_encoded` returning
`EncodedRows`, and the format's required divisor accessors. Their definitions are in
[`sources/logical.py`](../tools/convert/sources/logical.py).

## Convert from a GGUF quant instead of the full checkpoint

The base `--model` checkpoint only needs to supply `config.json` and the frontend
resources (tokenizer, chat template, generation config); it does not need every
weight tensor present. If you already have a GGUF export of the model -- F16, or a
ggml quant level such as `Q8_0` or `Q4_K_M` -- an override recipe can read the actual
tensor values from that file instead of downloading the multi-gigabyte Safetensors
checkpoint again. Keep only the checkpoint's small config/tokenizer files locally, and
pass the GGUF file as a named `--source`:

```bash
python3 -m tools.convert \
  --model /path/to/checkpoint-config-only \
  --recipe qwen3_6_27b \
  --source quantized=/path/to/model-Q4_K_M.gguf \
  --override use_gguf_source.py \
  --out models/my_qwen.ninfer
```

`tools/convert/sources/gguf_source.py` reads the GGUF file (via the `gguf` package --
`pip install gguf`; it performs the actual block dequantization, since GGUF's quant
byte layouts are intricate enough that reusing the maintained ggml-org reader beats
reimplementing them) and exposes its tensors as FP32 values. Because GGUF tensor
names never match the HF Safetensors names a recipe's `source_name`s are written
against (`model.layers.0.self_attn.q_proj.weight` vs. `blk.0.attn_q.weight`), and
some architectures reorder tensors on export, `HFAliasSource` wraps a `GGUFSource`
with an explicit `{hf_name: gguf_name}` map (plus optional per-tensor transforms)
so it satisfies the same interface a recipe already expects from `sources["..."]`.

`use_gguf_source.py`:

```python
from tools.convert.sources.gguf_source import HFAliasSource, standard_dense_name_map


def configure(model, recipe, sources):
    gguf_source = sources["quantized"]  # opened as a GGUFSource because the path ends in .gguf
    name_map, transforms, shapes = standard_dense_name_map(
        model.config["num_hidden_layers"],
        tie_word_embeddings=model.config["tie_word_embeddings"],
        text_prefix="model.language_model.",  # "model." for a text-only (non-vision) checkpoint
    )
    quantized = HFAliasSource(gguf_source, name_map, transforms=transforms, shapes=shapes)
    for name, parameter in model.parameters.items():
        if parameter.source_factory is None:
            continue
        recipe.assign(name, source=model.source(name, quantized))
```

Parameters whose HF source name isn't in `name_map` (mixture-of-experts routing,
MTP, vision, ...) fail with a clear "no GGUF tensor mapped for ..." error rather
than silently reading the wrong bytes; extend `name_map`/`transforms` for those
before assigning them too, or leave them assigned to the base recipe's original
source by only looping over the names you've actually mapped.

### MTP, vision, and DFlash2 from GGUF

`--components text,vision,mtp,dflash2` works with GGUF sources too, matching a
normal multi-component conversion, though each component needs its own wiring:

- **MTP** is usually exported as a *separate* GGUF file (llama.cpp names these
  `mtp-*.gguf`), not fused into the main model's file. It appears as one more
  transformer layer appended after the base model's real layers (layer 64 for a
  64-layer model), reusing the same attention/MLP/norm tensor roles
  `standard_dense_name_map` already covers, plus four MTP-specific tensors.
  `qwen35_mtp_name_map(gguf_layer_index, hf_prefix=...)` builds that map --
  `hf_prefix="mtp."` when the MTP tensors are fused into the main checkpoint's
  state dict, or `hf_prefix=""` for a standalone MTP-only checkpoint whose
  tensors have no prefix. Pass it its own `--source mtp_quantized=mtp-*.gguf`
  and wrap that in its own `HFAliasSource`.

- **Vision** is exported as a separate `mmproj-*.gguf` file (GGUF architecture
  `clip`, not the text model's `qwen35`/`qwen35moe`) covering the ViT tower and
  the projector MLP. `qwen35_vision_name_map(depth, hidden_size=..., patch_size=...,
  temporal_patch_size=..., ...)` builds its name map from the checkpoint's own
  `vision_config`. It also needs a `--source vision_quantized=mmproj-*.gguf`.
  One tensor needs more than a name/value transform: the patch-embed Conv3d
  kernel's temporal dimension is split across two separate GGUF tensors
  (`v.patch_embd.weight` / `.weight.1`), which `HFAliasSource`'s `composites`
  argument reassembles by reading both from the underlying `GGUFSource` and
  stacking them -- see its docstring if you need the same pattern elsewhere.

- **DFlash2** needs no GGUF-specific code at all in the cases seen so far: it's
  a small enough draft head that a plain Safetensors checkpoint for it is the
  normal way to get one, and `tools.convert` already reads that natively via
  `--source dflash2=PATH` plus `--components ...,dflash2` -- the same mechanism
  `qwen3_8_27b_nvfp4`'s example in this doc already uses. An override recipe
  only needs to touch `dflash2/*` parameters if you actually have a GGUF export
  of the draft head to source them from instead.

All three name maps above were checked against a real checkpoint/GGUF pair on
this machine the same way `qwen35_linear_attention_name_map` was (see its
docstring): every tensor's dequantized GGUF value was compared back to the
original Safetensors value. MTP's own norm tensors turned out to need **no**
+1 shift (unlike the base model's layers -- verified, not assumed), vision
needs no shift at all (it's LayerNorm, not RMSNorm), and the patch-embed
temporal split reassembles to the exact original kernel. None of the three
cover mixture-of-experts tensors, and the vision map doesn't cover deepstack
layers (no checkpoint with those to verify against was available).

This keeps every format/method choice the base recipe already made (`qwen3_6_27b`
here); only the byte source for each parameter's values changes. Formats needing
higher source fidelity than the GGUF file provides will simply re-quantize its
already-lossy values -- e.g. re-deriving `q4_g64_fp16` from a `Q4_K_M` source
compounds two rounds of quantization error. Prefer the highest-fidelity GGUF you
have (F16 or `Q8_0`/`Q6_K`) as the source when the target format is more precise
than the GGUF's own quant level.

`standard_dense_name_map` only covers ordinary attention/MLP/norm tensors. Run
`python3 -m tools.convert.sources.gguf_source /path/to/model.gguf` to list every tensor
name, shape and ggml type in your file, and extend the map for anything it doesn't
cover -- mixture-of-experts routing/shared-expert tensors, MTP/nextn tensors, or
(for Qwen3.5's hybrid linear-attention layers) the GDN tensors handled by
`qwen35_linear_attention_name_map` and `reorder_linear_attention_v_heads` in the
same module. Verify a full conversion's outputs (`tools.artifact.inspect`, then the
normal CLI/serving smoke test) before relying on it -- name-mapping mistakes for a
component this map doesn't cover fail as a missing-tensor error, but a wrong
element ordering for a component it does claim to cover would not.

## Write a conversion method

A method receives a `PrepareRequest` and returns `request.job(produce=...)`. Preparation validates
the target and determines auxiliary values. The `produce` function reads bounded source regions
and writes values or codes/scales through `TensorOutput`; the writer owns placement and file I/O.

This example adds explicit clipping before the existing grouped quantizer. It demonstrates the
method interface; the clipping threshold is a numerical choice made by the recipe author.

```python
import math
import torch
from tools.artifact.formats import QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix


def clipped_grouped(request):
    if len(request.target.shape) != 2 or not isinstance(
        get_format(request.target.format), QuantFormat
    ):
        raise ValueError("clipped_grouped requires a grouped-integer matrix")
    limit = float(request.parameters["clip"])
    if not math.isfinite(limit) or limit <= 0 or request.rows_per_chunk <= 0:
        raise ValueError("clip and rows_per_chunk must be positive")
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("source contains non-finite values")
            encoded = quantize_matrix(
                values.clamp(-limit, limit),
                request.target.format,
                device=request.device,
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method=clipped_grouped,
        parameters={"clip": 1.0},
    )
```

Use this as an override. `request.values` traverses the prepared logical inputs in parent order,
including explicit groups. `output.write_codes` performs the registered packing and validates
codes/scales; the method should not duplicate that byte-layout logic. Direct output uses
`output.write_values`. Keep source blocks and temporary device tensors bounded to the method's
working set. `--rows-per-chunk` defaults to 512; custom methods own how they use it.

## Resources, files and inspection

Text includes `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json`. Vision adds its image and video processor configs. Resources come from
`--model`; `--resource ROLE=PATH` replaces a selected resource:

```text
--resource chat_template.jinja=/path/to/chat_template.jinja
```

The official conversion examples select these maintained templates:

| Model | Template | Defaults |
|---|---|---|
| Qwen3.6 Dense/MoE | [qwen3_6.jinja](../tools/chat_templates/qwen3_6.jinja) | thinking on; closed-turn reasoning omitted |
| Qwen3.8 | [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja) | thinking on; effort `xhigh`; closed-turn reasoning retained |

[froggeric_v22_5.jinja](../tools/chat_templates/froggeric_v22_5.jinja) is an unmodified copy of
the v22.5 universal Qwen3.5/3.6/3.8 template from
[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
(Apache-2.0), offered for `--chat-template`: it maps the whole effort vocabulary itself and renders
identically in NInfer's Jinja engine and in Python Jinja2. It is not one of the maintained
templates above, whose behavior the frontend tests pin.

Use your own Jinja file to change the artifact's default template. A startup
[`--chat-template FILE`](cli.md#text-input) overrides the stored template.
`generation_config.json` is preserved; sampling presets remain determined by the architecture
and explicit application/request settings.

The default maximum file size is 32,000,000,000 bytes, including framing. Smaller artifacts remain
one file. Larger artifacts use an entry such as `models/my_qwen.ninfer` plus
`my_qwen.ninfer.part-0001`, `my_qwen.ninfer.part-0002`, and so on in the same directory. Pass only the
entry path to NInfer and keep all its recorded parts together. `--max-file-bytes` changes the limit.

Conversion writes `models/my_qwen.ninfer.conversion.json` alongside the artifact, recording sources,
methods, formats, component configs, files and timing. Existing output files are not overwritten.
The report is useful for reproducing a recipe; the Engine reads the artifact itself.

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninfer --json
```

Inspection reads directory facts without running inference. Conversion rejects missing logical
coverage, invalid source geometry, unsupported encodings and invalid method output. Actual Op
support is checked by consumers during preparation, resource queries, warmup or execution. A
valid file may need additional Op support before its chosen combination can run. Exercise the
phases and optional components you intend to use through the normal [CLI](cli.md) or
[serving](serving.md) route.
