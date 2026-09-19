#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-torch==0.4.1",
#     "coreai-core==1.0.0b2",
#     "safetensors>=0.7.0",
#     "numpy==2.2.6",
#     "tokenizers==0.22.2",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Stage 3: export ibm-granite/granite-embedding-97m-multilingual-r2 (ModernBERT, 97M) as a static Core AI graph.

    (input_ids [1,S] int32, attention_mask [1,S] int32) -> embedding [1,384] fp32, CLS-pooled, L2-normalized

    python3 export_granite_embedding.py --seq-len 512 --target macos                  # JIT .aimodel (Mac)
    python3 export_granite_embedding.py --seq-len 512 --target ios --aot h18p         # + xcrun coreai-build -> .aimodelc (iPhone 17 Pro)

Prerequisites, in order: oracle_granite_embedding.py, then gate_granite_authoring.py for the same
grid (this script refuses to run without a PASS record). The graph is re-authored from the raw
safetensors (`_granite_model.py`), never from transformers. The torch-exported and decomposed
graph is gated against the HF fixtures BEFORE conversion; the runtime is gated after
(gate_granite_embedding.py on Mac, the device gate on iPhone).

Output layout mirrors the published repo: <exports>/granite-embedding-97m/<target>/<variant>-s<S>/
with the bundle, tokenizer/, reference.json (the HF fixtures) and provenance/export-manifest.json.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (MODEL_ID, MODEL_SHA, export_root, file_inventory, fixtures_dir, golden, hashes,  # noqa: E402
                     results_dir, verify_hashes, verify_source, write_json)
from _gate_metrics import gate_vectors  # noqa: E402
from _granite_model import load_granite  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def inputs_of(spec):
    return {"input_ids": torch.tensor([spec["input_ids"]], dtype=torch.int32),
            "attention_mask": torch.tensor([spec["attention_mask"]], dtype=torch.int32)}


def torch_export_and_gate(model, fixtures):
    """torch.export -> coreai decompositions -> the same HF gate, before any conversion."""
    with torch.no_grad():
        exported = torch.export.export(model, args=(), kwargs=inputs_of(fixtures["fixtures"][6]))
        decomposed = exported.run_decompositions(get_decomp_table())
        candidate = decomposed.module()
        vectors = {r["id"]: candidate(**inputs_of(r))[0].numpy().tolist() for r in fixtures["fixtures"]}
    gate = gate_vectors(fixtures, vectors)
    assert gate["status"] == "PASS", gate["failures"]
    return decomposed, gate, vectors


def convert(decomposed, description: str, out: Path) -> dict:
    program = TorchConverter().add_exported_program(
        exported_program=decomposed, input_names=["input_ids", "attention_mask"], output_names=["embedding"]).to_coreai()
    program.optimize()
    module = program._mlir_module
    assert module.operation.verify()
    counts: Counter = Counter()

    def inspect(operation):
        counts[operation.name] += 1
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    inspect(child.operation)

    inspect(module.operation)
    metadata = AIModelAssetMetadata()
    metadata.author = "mlboydaisuke (coreai-model-zoo); weights IBM Granite"
    metadata.license = "Apache-2.0"
    metadata.model_description = description
    program.save_asset(out, metadata)
    return dict(counts)


def aot_compile(source_bundle: Path, out_dir: Path, architecture: str) -> Path:
    """The recorded iPhone compile: GPU-preferred, one device architecture, iOS 27 minimum."""
    argv = ["xcrun", "coreai-build", "compile", str(source_bundle), "--output", str(out_dir), "--platform", "iOS",
            "--min-deployment-version", "27.0", "--preferred-compute", "gpu", "--architecture", architecture]
    print("$", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)
    compiled = sorted(out_dir.glob("*.aimodelc"))
    assert len(compiled) == 1, compiled
    assert (compiled[0] / f"main-{architecture}.mlirb").exists() and not (compiled[0] / "main.mlirb").exists(), \
        "expected a compiled-only bundle (main-<arch>.mlirb, no portable main.mlirb)"
    return compiled[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-len", required=True, type=int, choices=[128, 512])
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                        help="fp16 fails the authoring layer gate on this checkpoint; fp32 ships")
    parser.add_argument("--target", choices=["macos", "ios"], default="macos")
    parser.add_argument("--aot", metavar="ARCH", help="iOS only: compile with coreai-build for one architecture (h18p = iPhone 17 Pro)")
    parser.add_argument("--artifact-tag", default="", help="suffix on the bundle name (the published fp32 bundles carry `bound`)")
    parser.add_argument("--output-dir", type=Path, help="default: <exports>/granite-embedding-97m/<target>/<dtype>-s<S>")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.aot and args.target != "ios":
        parser.error("--aot applies to --target ios")
    torch.set_num_threads(4)
    source = verify_source()
    authoring_path = results_dir() / f"authoring_{args.dtype}_s{args.seq_len}.json"
    authoring = json.loads(authoring_path.read_text())
    assert authoring["status"] == "PASS", "the authoring gate must pass before conversion"
    verify_hashes(authoring["input_hashes"])

    name = f"granite97m_{args.dtype}_s{args.seq_len}"
    if args.artifact_tag:
        assert re.fullmatch(r"[A-Za-z0-9_-]+", args.artifact_tag)
        name += "_" + args.artifact_tag
    out_dir = args.output_dir or export_root() / args.target / f"{args.dtype}-s{args.seq_len}"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    source_bundle = (out_dir if not args.aot else out_dir / "_source") / (name + ".aimodel")
    source_bundle.parent.mkdir(parents=True, exist_ok=True)

    fixtures = golden(args.seq_len)
    model = load_granite(source, args.seq_len, dtype=torch.float32 if args.dtype == "fp32" else torch.float16)
    start = time.perf_counter()
    decomposed, export_gate, _ = torch_export_and_gate(model, fixtures)
    print("torch export + decomposition parity PASS", flush=True)
    description = (f"Granite-Embedding-97M-Multilingual-R2 CLS embedding; {args.dtype}; S={args.seq_len}; "
                   f"source {MODEL_ID}@{MODEL_SHA}; intended {args.target}")
    op_counts = convert(decomposed, description, source_bundle)
    print("saved", source_bundle, flush=True)

    bundle = source_bundle
    if args.aot:
        bundle = aot_compile(source_bundle, out_dir, args.aot)
        shutil.rmtree(source_bundle.parent)  # the compiled bundle is the artifact; the IR is reproducible
    for file in TOKENIZER_FILES:
        (out_dir / "tokenizer").mkdir(exist_ok=True)
        shutil.copy2(source / file, out_dir / "tokenizer" / file)
    shutil.copy2(fixtures_dir() / f"golden_s{args.seq_len}.json", out_dir / "reference.json")

    record = {"status": "PASS", "model": MODEL_ID, "model_sha": MODEL_SHA, "bundle": bundle.name,
              "intended_runtime": args.target, "format": f"AOT .aimodelc ({args.aot})" if args.aot else "JIT .aimodel",
              "aot": bool(args.aot), "precision": args.dtype, "sequence_length": args.seq_len,
              "inputs": {"input_ids": {"dtype": "int32", "shape": [1, args.seq_len]},
                         "attention_mask": {"dtype": "int32", "shape": [1, args.seq_len]}},
              "outputs": {"embedding": {"dtype": "float32", "shape": [1, 384]}},
              "files": file_inventory(bundle), "op_counts": op_counts, "seconds": time.perf_counter() - start,
              "torch_export_gate": {k: export_gate[k] for k in ("status", "min_cosine", "max_abs", "thresholds")},
              "runtime_gate": "NOT RUN — gate_granite_embedding.py (Mac) / the device gate (iPhone)",
              "input_hashes": hashes([authoring_path, Path(__file__), Path(__file__).parent / "_granite_model.py"])}
    record["bytes"] = sum(f["bytes"] for f in record["files"])
    write_json(out_dir / "provenance" / "export-manifest.json", record)
    print(json.dumps({k: record[k] for k in ("status", "bundle", "format", "precision", "sequence_length", "bytes", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
