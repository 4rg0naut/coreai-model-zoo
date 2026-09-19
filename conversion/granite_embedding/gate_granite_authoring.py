#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.9.0", "safetensors>=0.7.0", "numpy==2.2.6"]
# ///
"""Stage 1: the re-authored graph vs the oracle, at every residual layer, with negative controls.

    python3 gate_granite_authoring.py --seq-len 128 --dtype fp32 --negative-controls

The embedding gate (cos >= 0.999, |err| <= 0.02, retrieval order) is the same one the runtime
and the phone are held to. The LAYER gate is stricter (max |err| <= 1e-4 per hidden state at
fp32) and is the one that matters: a local-attention radius of 63 instead of 64 passes the
embedding gate and fails only here. `--negative-controls` proves the gate can go red: four
architecture mutations (all-global, all-local, ignore padding, mean pooling) must fail the
embedding gate, and `window63` must fail the layer gate.

Whole-model fp16 fails this layer gate on both grids (measured 2026-09-19); fp32 is the ship dtype.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_SHA, fixtures_dir, golden, hashes, results_dir, verify_source, write_json  # noqa: E402
from _gate_metrics import gate_vectors, tensor_metrics  # noqa: E402
from _granite_model import MUTATIONS, load_granite  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-len", type=int, required=True, choices=[128, 512])
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = verify_source()
    oracle = json.loads((results_dir() / "oracle.json").read_text())
    assert oracle["status"] == "PASS"
    fixtures = golden(args.seq_len)
    intermediates = np.load(fixtures_dir() / f"intermediates_s{args.seq_len}.npz")
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    model = load_granite(source, args.seq_len, dtype=dtype)
    vectors, layers = {}, {}
    start = time.perf_counter()
    for spec in fixtures["fixtures"]:
        ids = torch.tensor([spec["input_ids"]], dtype=torch.int32)
        mask = torch.tensor([spec["attention_mask"]], dtype=torch.int32)
        with torch.inference_mode():
            output = model.forward_intermediates(ids, mask)
        vectors[spec["id"]] = output["embedding"][0].numpy().tolist()
        metrics = [{"name": f"hidden_{n}", **tensor_metrics(intermediates[f"{spec['id']}__hidden_{n}"], hidden.float().numpy())}
                   for n, hidden in enumerate(output["hidden_states"])]
        metrics.append({"name": "last_hidden_state", **tensor_metrics(intermediates[f"{spec['id']}__last_hidden_state"],
                                                                     output["last_hidden_state"].float().numpy())})
        layers[spec["id"]] = metrics
        print(spec["id"], "max layer abs", max(m.get("max_abs", 1e30) for m in metrics), flush=True)
    result = gate_vectors(fixtures, vectors)
    layer_tolerance = 0.0001 if args.dtype == "fp32" else 0.5
    layer_failures = [(k, m["name"]) for k, ms in layers.items() for m in ms if not m["finite"] or m.get("max_abs", 1e30) > layer_tolerance]
    result.update(layer_metrics=layers, layer_tolerance=layer_tolerance, layer_failures=layer_failures,
                  seconds=time.perf_counter() - start, dtype=args.dtype, sequence_length=args.seq_len)
    if layer_failures:
        result["status"] = "FAIL"
    result["negative_controls"] = {}
    if args.negative_controls:
        for mutation in MUTATIONS[1:]:
            mutated = load_granite(source, args.seq_len, mutation=mutation)
            wrong, mutation_layer_max_abs = {}, 0.0
            for spec in fixtures["fixtures"]:
                inputs = (torch.tensor([spec["input_ids"]], dtype=torch.int32), torch.tensor([spec["attention_mask"]], dtype=torch.int32))
                with torch.inference_mode():
                    if mutation == "window63":
                        intermediate = mutated.forward_intermediates(*inputs)
                        embedding = intermediate["embedding"]
                        for n, hidden in enumerate(intermediate["hidden_states"]):
                            error = float(np.max(np.abs(intermediates[f"{spec['id']}__hidden_{n}"] - hidden.numpy())))
                            mutation_layer_max_abs = max(mutation_layer_max_abs, error)
                    else:
                        embedding = mutated(*inputs)
                wrong[spec["id"]] = embedding[0].numpy().tolist()
            gate = gate_vectors(fixtures, wrong)
            if mutation == "window63":
                gate["max_layer_abs"] = mutation_layer_max_abs
                gate["authoring_layer_gate"] = "FAIL" if mutation_layer_max_abs > layer_tolerance else "PASS"
            result["negative_controls"][mutation] = gate
            print("NEGATIVE", mutation, gate["status"], gate["min_cosine"], flush=True)
            del mutated
        required = ["all_global", "all_local", "ignore_padding", "wrong_pooling"]
        if any(result["negative_controls"][m]["status"] != "FAIL" for m in required):
            result["status"] = "FAIL"
            result["failures"].append("negative control failed to detect substantial architectural corruption")
        if result["negative_controls"]["window63"]["authoring_layer_gate"] != "FAIL":
            result["status"] = "FAIL"
            result["failures"].append("exact layer gate failed to detect off-by-one local attention window")
    result["model_sha"] = MODEL_SHA
    result["input_hashes"] = hashes([source / "model.safetensors", source / "config.json",
                                     fixtures_dir() / f"golden_s{args.seq_len}.json", Path(__file__).parent / "_granite_model.py"])
    write_json(results_dir() / f"authoring_{args.dtype}_s{args.seq_len}.json", result)
    (results_dir() / f"authoring_vectors_{args.dtype}_s{args.seq_len}.json").write_text(json.dumps(vectors) + "\n")
    print(result["status"], "cosine", result["min_cosine"], "max_abs", result["max_abs"], "layer failures", len(layer_failures))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
