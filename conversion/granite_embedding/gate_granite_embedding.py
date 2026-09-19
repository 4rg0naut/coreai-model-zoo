#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["coreai-core==1.0.0b2", "numpy==2.2.6"]
# ///
"""Stage 4 (Mac): the exported bundle on the Core AI runtime vs the same HF fixtures, plus timing.

    python3 gate_granite_embedding.py <exports>/granite-embedding-97m/macos/fp32-s512 --compute gpu
    python3 gate_granite_embedding.py <dir> --compute cpu_only      # the parity option, not a speed option

Refuses an iOS bundle: a compiled iPhone `.aimodelc` must never be loaded on a Mac (it wedges the
GPU stack). The iPhone gate is the device runner described on the card, fed the same
reference.json; its results are `provenance/runtime-gate.json` in the published `ios/` folders.

Per fixture: `--repeats` extra runs (drift must be <= 1e-6, i.e. deterministic), warm timings,
and a wrong-pairing control (every vector matched to the wrong text must FAIL the gate).
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import coreai.runtime as rt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_SHA, sha256_of, write_json  # noqa: E402
from _gate_metrics import gate_vectors  # noqa: E402


async def run(variant_dir: Path, compute: str, repeats: int):
    manifest = json.loads((variant_dir / "provenance" / "export-manifest.json").read_text())
    assert manifest["intended_runtime"] == "macos" and not manifest["aot"], "Mac gate only: never load an iOS bundle here"
    bundle = variant_dir / manifest["bundle"]
    assert bundle.suffix == ".aimodel", bundle
    for item in manifest["files"]:
        file = bundle / item["path"]
        assert file.stat().st_size == item["bytes"] and sha256_of(file) == item["sha256"], f"artifact changed: {item['path']}"
    fixtures = json.loads((variant_dir / "reference.json").read_text())
    assert fixtures["model_sha"] == manifest["model_sha"] == MODEL_SHA
    options = rt.SpecializationOptions.cpu_only() if compute == "cpu_only" else rt.SpecializationOptions.default()
    start = time.perf_counter()
    model = await rt.AIModel.load(bundle, options)
    load_ms = (time.perf_counter() - start) * 1000
    assert model.function_names == ["main"]
    fn = model.load_function("main")
    vectors, timings, drift = {}, {}, {}
    for spec in fixtures["fixtures"]:
        inputs = {"input_ids": rt.NDArray(np.asarray([spec["input_ids"]], dtype=np.int32)),
                  "attention_mask": rt.NDArray(np.asarray([spec["attention_mask"]], dtype=np.int32))}
        samples, first, max_drift = [], None, 0.0
        for _ in range(repeats + 1):
            t = time.perf_counter()
            out = await fn(inputs)
            array = np.array(out["embedding"].numpy(), copy=True)
            samples.append((time.perf_counter() - t) * 1000)
            assert array.shape == (1, 384) and np.isfinite(array).all(), spec["id"]
            if first is None:
                first = array[0]
            else:
                max_drift = max(max_drift, float(np.max(np.abs(array[0] - first))))
        vectors[spec["id"]] = first.tolist()
        timings[spec["id"]] = {"first_ms": samples[0], "warm_median_ms": float(np.median(samples[1:]))}
        drift[spec["id"]] = max_drift
        print(spec["id"], f"first {samples[0]:.2f} ms  warm {timings[spec['id']]['warm_median_ms']:.2f} ms  drift {max_drift:.2e}", flush=True)
    assert model is not None  # keep the model alive until every output is copied
    result = gate_vectors(fixtures, vectors)
    if max(drift.values()) > 1e-6:
        result["status"] = "FAIL"
        result["failures"].append("non-deterministic output")
    keys = list(vectors)
    control = gate_vectors(fixtures, {k: vectors[keys[(i + 1) % len(keys)]] for i, k in enumerate(keys)})
    if control["status"] != "FAIL":
        result["status"] = "FAIL"
        result["failures"].append("wrong-pairing control passed")
    warm = [t["warm_median_ms"] for t in timings.values()]
    result.update(compute=compute, specialization_options=str(options), load_ms=load_ms, timings=timings, repeat_max_abs=drift,
                  warm_median_ms=float(np.median(warm)), warm_min_ms=min(warm), warm_max_ms=max(warm),
                  wrong_pairing_control={"status": control["status"], "min_cosine": control["min_cosine"]},
                  model_sha=MODEL_SHA, precision=manifest["precision"], sequence_length=manifest["sequence_length"],
                  bundle=manifest["bundle"], process_pid=os.getpid())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant_dir", type=Path, help="an exported <target>/<variant>-s<S> directory (macos only)")
    parser.add_argument("--compute", choices=["gpu", "cpu_only"], default="gpu")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, help="default: <variant_dir>/provenance/runtime-gate.json")
    args = parser.parse_args()
    result = asyncio.run(run(args.variant_dir, args.compute, args.repeats))
    out = args.output or args.variant_dir / "provenance" / "runtime-gate.json"
    write_json(out, result)
    print(result["status"], "min cosine", result["min_cosine"], "max abs", result["max_abs"],
          f"warm median {result['warm_median_ms']:.2f} ms", "failures", result["failures"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
