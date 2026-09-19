#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-torch==0.4.1",
#     "coreai-core==1.0.0b2",
#     "coreai-opt==0.2.1",
#     "coremltools==9.0",
#     "torchao==0.17.0",
#     "safetensors==0.7.0",
#     "numpy==2.2.6",
#     "scikit-learn",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Stage 3b (optional): the W8 / fp32-table variant — 48 linear weights as 8-bit scalar k-means palettes.

    python3 export_granite_embedding_w8.py --seq-len 512 --target macos
    python3 export_granite_embedding_w8.py --seq-len 512 --target ios --aot h18p --palettes <macos run>/provenance/palettes.safetensors

What it does, and what it deliberately does not: `KMeansPalettizerConfig.presets.w8()` (per-tensor,
256-entry fp32 LUT, cluster_dim 1, seed 0, one worker) on the 48 attention/MLP linears only.
The 180,000 x 384 embedding table (276 MB, 71% of the bundle) stays fp32, and so does compute,
which is why the bundle is only ~22% smaller (305 vs 390 MB). The same HF gate runs at three
stages (prepared, finalized, torch-decomposed) and every LUT/index tensor is hashed so the iOS
export can reuse the Mac palettes byte for byte (`--palettes`) instead of re-clustering.

This needs its own environment: coreai-opt 0.2.1 pins safetensors <= 0.7.0, which conflicts
with the fp32 exporter's pins. `uv run` this file and the pins above resolve it.
"""
import argparse
import importlib.metadata as metadata
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.utils import parametrize as P

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (MODEL_ID, MODEL_SHA, export_root, file_inventory, fixtures_dir, golden, hashes,  # noqa: E402
                     results_dir, sha256_of, verify_source, write_json)
from _gate_metrics import gate_vectors, tensor_metrics  # noqa: E402
from _granite_model import PARAMETER_COUNT, load_granite  # noqa: E402
from export_granite_embedding import TOKENIZER_FILES, aot_compile, convert, inputs_of  # noqa: E402

EXPECTED_SHAPES = {f"layers.{layer}.{name}": shape for layer in range(12) for name, shape in {
    "attn.Wqkv": (1152, 384), "attn.Wo": (384, 384), "mlp.Wi": (3072, 384), "mlp.Wo": (384, 1536)}.items()}
PINNED = {"coreai-core": "1.0.0b2", "coreai-torch": "0.4.1", "coreai-opt": "0.2.1", "coremltools": "9.0",
          "torchao": "0.17.0", "torch": "2.9.0", "numpy": "2.2.6", "safetensors": "0.7.0"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_sha(value):
    return __import__("hashlib").sha256(memoryview(value.detach().contiguous().numpy())).hexdigest()


def inspect_palettes(model, finalized: bool) -> tuple[dict, dict]:
    linears = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    require(set(linears) == set(EXPECTED_SHAPES), "linear target set changed")
    parametrized = {(name, attr) for name, module in model.named_modules() if P.is_parametrized(module) for attr in module.parametrizations}
    require(parametrized == {(name, "weight") for name in EXPECTED_SHAPES}, f"unexpected palettized targets: {parametrized}")
    audit, tensors = {}, {}
    for name, shape in EXPECTED_SHAPES.items():
        chain = linears[name].parametrizations.weight
        require(len(chain) == 1, f"unexpected parametrization chain: {name}")
        palette = chain[0]
        lut, indices = palette.lut, palette.indices
        require(tuple(lut.shape) == (1, 1, 256, 1) and lut.dtype == torch.float32, f"not a scalar w8 LUT: {name}")
        require(tuple(indices.shape) == shape and indices.dtype == torch.uint8, f"invalid w8 indices: {name}")
        require(bool(torch.isfinite(lut).all()), f"non-finite palette: {name}")
        effective = linears[name].weight
        require(tuple(effective.shape) == shape and effective.dtype == torch.float32, f"invalid reconstructed weight: {name}")
        audit[name] = {"lut_sha256": tensor_sha(lut), "indices_sha256": tensor_sha(indices), "effective_weight_sha256": tensor_sha(effective)}
        if not finalized:
            tensors[name + ".lut"] = lut.detach().contiguous()
            tensors[name + ".indices"] = indices.detach().contiguous()
    require(len(audit) == 48, "expected exactly 48 linear palettes")
    return audit, tensors


def evaluate(model, fixtures, hidden_path=None):
    """The HF gate on the palettized model; hidden states are checked finite/shaped (diagnostic, no fp32 bar)."""
    vectors, diagnostics = {}, {}
    hidden = np.load(hidden_path) if hidden_path else None
    try:
        with torch.inference_mode():
            for spec in fixtures["fixtures"]:
                inputs = inputs_of(spec)
                if hidden is None:
                    embedding = model(**inputs)
                else:
                    output = model.forward_intermediates(**inputs)
                    embedding = output["embedding"]
                    diagnostics[spec["id"]] = [{"name": f"hidden_{i}", **tensor_metrics(hidden[f"{spec['id']}__hidden_{i}"], v.numpy())}
                                               for i, v in enumerate(output["hidden_states"])]
                vectors[spec["id"]] = embedding[0].numpy().tolist()
    finally:
        if hidden is not None:
            hidden.close()
    result = gate_vectors(fixtures, vectors)
    keys = list(vectors)
    wrong = {k: vectors[keys[(i + 1) % len(keys)]] for i, k in enumerate(keys)}
    negative = gate_vectors(fixtures, wrong)
    bad_layers = [[k, m["name"]] for k, ms in diagnostics.items() for m in ms if not m["finite"] or not m["shape_match"]]
    if negative["status"] != "FAIL" or bad_layers:
        result["status"] = "FAIL"
        result["failures"].append("non-finite hidden state or ineffective wrong-pairing control")
    result.update(layer_metrics=diagnostics, wrong_pairing_control={"status": negative["status"], "min_cosine": negative["min_cosine"]})
    return result, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-len", type=int, choices=[128, 512], required=True)
    parser.add_argument("--target", choices=["macos", "ios"], default="macos")
    parser.add_argument("--aot", metavar="ARCH")
    parser.add_argument("--palettes", type=Path, help="reuse the LUT/index tensors of an earlier run instead of clustering again")
    parser.add_argument("--artifact-tag", default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-other-versions", action="store_true", help="skip the exact dependency-pin check")
    args = parser.parse_args()
    if args.aot and args.target != "ios":
        parser.error("--aot applies to --target ios")
    installed = {name: metadata.version(name) for name in PINNED}
    if installed != PINNED and not args.allow_other_versions:
        raise SystemExit(f"dependency pins differ from the qualified set: {installed}\n(expected {PINNED}; --allow-other-versions to proceed)")
    torch.set_num_threads(4)
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    source = verify_source()
    fixtures = golden(args.seq_len)
    hidden_path = fixtures_dir() / f"intermediates_s{args.seq_len}.npz"
    name = f"granite97m_w8_fp32table_s{args.seq_len}" + (f"_{args.artifact_tag}" if args.artifact_tag else "")
    require(not args.artifact_tag or re.fullmatch(r"[A-Za-z0-9_-]+", args.artifact_tag), "invalid artifact tag")
    out_dir = args.output_dir or export_root() / args.target / f"w8-fp32table-s{args.seq_len}"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "provenance").mkdir()
    started = time.perf_counter()

    model = load_granite(source, args.seq_len, dtype=torch.float32)
    require(len(model.state_dict()) == 74 and sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT, "strict load changed")
    linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    require(set(linears) == set(EXPECTED_SHAPES), "unexpected linear set")
    require(sum(m.weight.numel() for m in linears.values()) == 28_311_552, "unexpected palettized weight count")
    untouched_names = set(model.state_dict()) - {n + ".weight" for n in EXPECTED_SHAPES}
    untouched = {n: tensor_sha(model.state_dict()[n]) for n in untouched_names}
    stages = {}

    if args.palettes:
        # Reuse: register the saved palettes as parametrizations; no clustering.
        from coreai_opt._utils.export_utils import clear_parametrization_original
        from coreai_torch._compression.custom_layers import PalettizeModule
        from coreai_torch._compression.utils import wrap_for_parametrization
        palettes = load_file(str(args.palettes), device="cpu")
        Wrapper = wrap_for_parametrization(PalettizeModule)
        for k, v in linears.items():
            P.register_parametrization(v, "weight", Wrapper(indices=palettes[k + ".indices"], lut=palettes[k + ".lut"], vector_axis=None))
            clear_parametrization_original(v, "weight")
        with torch.inference_mode(), P.cached():
            audit, _ = inspect_palettes(model, True)
        stages["reused_palettes"] = str(args.palettes)
        shutil.copy2(args.palettes, out_dir / "provenance" / "palettes.safetensors")
    else:
        from coreai_opt.common import ExportBackend
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
        cfg = KMeansPalettizerConfig.presets.w8()
        require(not cfg.module_name_configs and not cfg.module_type_configs, "unexpected preset overrides")
        g = cfg.global_config
        require(not g.op_input_spec and not g.op_output_spec and g.enable_fast_kmeans_mode is True and g.rounding_precision == 4,
                "unexpected activation compression or clustering settings")
        for spec in g.op_state_spec.values():
            require(spec.n_bits == 8 and spec.cluster_dim == 1 and spec.lut_qspec is None and not spec.enable_per_channel_scale
                    and spec.granularity.__class__.__name__ == "PerTensorGranularity", "refusing non-scalar / non-per-tensor clustering")
        example = inputs_of(fixtures["fixtures"][6])
        palettizer = KMeansPalettizer(model, cfg)
        model = palettizer.prepare((example["input_ids"], example["attention_mask"]), num_workers=1)
        with torch.inference_mode(), P.cached():
            prepared_audit, tensors = inspect_palettes(model, False)
            save_file(tensors, str(out_dir / "provenance" / "palettes.safetensors"), metadata={"model_sha": MODEL_SHA, "preset": "scalar per-tensor w8"})
            result, prepared_vectors = evaluate(model, fixtures, hidden_path)
        stages["prepared"] = result["status"]
        require(result["status"] == "PASS", f"prepared gate failed: {result['failures']}")
        model = palettizer.finalize(backend=ExportBackend.CoreAI)
        with torch.inference_mode(), P.cached():
            audit, _ = inspect_palettes(model, True)
            require(audit == prepared_audit, "finalization changed LUTs / indices / effective weights")
            result, _ = evaluate(model, fixtures, hidden_path)
        stages["finalized"] = result["status"]
        require(result["status"] == "PASS", f"finalized gate failed: {result['failures']}")
    require({n: tensor_sha(model.state_dict()[n]) for n in untouched_names} == untouched, "a non-linear tensor changed")

    from coreai_torch import get_decomp_table
    with torch.no_grad():
        exported = torch.export.export(model, args=(), kwargs=inputs_of(fixtures["fixtures"][6]))
        decomposed = exported.run_decompositions(get_decomp_table())
    lut_ops = sum(n.op == "call_function" and "coreai.lut_to_dense" in str(n.target) for n in decomposed.graph.nodes)
    require(lut_ops == 48, f"expected 48 exported LUT ops, found {lut_ops}")
    result, vectors = evaluate(decomposed.module(), fixtures)
    stages["decomposed"] = result["status"]
    require(result["status"] == "PASS", f"decomposed gate failed: {result['failures']}")
    write_json(out_dir / "provenance" / "torch-gate.json", {k: result[k] for k in ("status", "thresholds", "failures", "min_cosine", "max_abs", "retrieval", "wrong_pairing_control")})

    source_bundle = (out_dir if not args.aot else out_dir / "_source") / (name + ".aimodel")
    source_bundle.parent.mkdir(parents=True, exist_ok=True)
    description = (f"Granite-Embedding-97M-Multilingual-R2 CLS embedding; w8 scalar palettes on 48 linears, fp32 table + compute; "
                   f"S={args.seq_len}; source {MODEL_ID}@{MODEL_SHA}; intended {args.target}")
    op_counts = convert(decomposed, description, source_bundle)
    bundle = source_bundle
    if args.aot:
        bundle = aot_compile(source_bundle, out_dir, args.aot)
        shutil.rmtree(source_bundle.parent)
    (out_dir / "tokenizer").mkdir(exist_ok=True)
    for file in TOKENIZER_FILES:
        shutil.copy2(source / file, out_dir / "tokenizer" / file)
    shutil.copy2(fixtures_dir() / f"golden_s{args.seq_len}.json", out_dir / "reference.json")
    record = {"status": "PASS", "model": MODEL_ID, "model_sha": MODEL_SHA, "bundle": bundle.name, "intended_runtime": args.target,
              "format": f"AOT .aimodelc ({args.aot})" if args.aot else "JIT .aimodel", "aot": bool(args.aot),
              "precision": "w8scalar-fp32compute-fp32table", "sequence_length": args.seq_len, "stages": stages,
              "settings": {"preset": "KMeansPalettizerConfig.presets.w8()", "n_bits": 8, "cluster_dim": 1, "granularity": "per_tensor",
                           "seed": 0, "num_workers": 1, "selected_linear_count": 48, "selected_weight_count": 28_311_552},
              "palettes": audit, "files": file_inventory(bundle), "op_counts": op_counts, "versions": installed,
              "seconds": time.perf_counter() - started,
              "runtime_gate": "NOT RUN — gate_granite_embedding.py (Mac) / the device gate (iPhone)",
              "input_hashes": hashes([Path(__file__), Path(__file__).parent / "_granite_model.py", fixtures_dir() / f"golden_s{args.seq_len}.json"])}
    record["bytes"] = sum(f["bytes"] for f in record["files"])
    record["palettes_sha256"] = sha256_of(out_dir / "provenance" / "palettes.safetensors")
    write_json(out_dir / "provenance" / "export-manifest.json", record)
    print(json.dumps({k: record[k] for k in ("status", "bundle", "format", "precision", "sequence_length", "bytes", "stages", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
