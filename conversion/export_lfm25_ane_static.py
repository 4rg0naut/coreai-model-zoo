#!/usr/bin/env python3
"""LFM2.5 (LFM2 conv + attention hybrid) on Apple's STATIC iOS path -> Neural Engine: the zoo builder
`conversion/overlay/files/python/src/coreai_models/models/ios/lfm2.py` registered at runtime, then the
same steps as export_ane_stock.py (stock `coreai.llm.export --platform iOS`, AOT h18p for the ANE with the
region count, a loadable device bundle, HF staging).

    python conversion/export_lfm25_ane_static.py [LiquidAI/LFM2.5-1.2B-Instruct] \\
        [--qconfig conversion/lfm25_pal8_g32.yaml] [--max-context-length 4096] \\
        [--devbundle lfm25_1_2b_ane_pal8] [--stage LFM2.5-1.2B-CoreAI]

Why a separate entry point: Apple's registry has no `ios_class` for `lfm2`, and the checkout's
registry.py stays Apple's file. The builder module exposes `register()`; this script starts the stock
CLI in-process after calling it (`python -c "...register(); main()"`), so the export is Apple's pipeline
end to end (k-means palettizer, 4-entrypoint trace, static shape ladder) with one extra model class.

What is different inside the bundle (see the builder's header): every layer's key/value cache row is
hidden/2 = 1024 channels wide; the 10 conv layers keep their previous two gated-input columns in their
own rows (position-indexed, read back through a one-hot built from the runner's causal_mask); the 6
attention layers use the first 512 channels. The runner (Apple's StaticShapeEngine, unmodified) sees
an ordinary two-state static bundle.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("export_ane_stock", HERE / "export_ane_stock.py")
stock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stock)

BUILDER_MODULE = "coreai_models.models.ios.lfm2"
BOOT = (f"import sys; from {BUILDER_MODULE} import register; register(); "
        "from coreai_models.llm.export import main; main()")
DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
DEFAULT_QCONFIG = HERE / "lfm25_pal8_g32.yaml"
DEFAULT_AUTHOR = "LiquidAI"
DEFAULT_LICENSE = "LFM Open License v1.0"


def export_with_builder(model: str, out_dir: Path, checkout: Path, compression: str | None, qconfig: Path | None,
                        max_context: int, overwrite: bool, extra: list[str]) -> Path:
    """Apple's stock static iOS export with the LFM2 builder registered in-process."""
    py = checkout / ".venv/bin/python"
    builder = checkout / "python/src/coreai_models/models/ios/lfm2.py"
    if not builder.exists():
        raise SystemExit(f"{builder} missing: apply the zoo overlay (conversion/overlay/apply.py) or symlink "
                         f"conversion/overlay/files/python/src/coreai_models/models/ios/lfm2.py there")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(py), "-c", BOOT, model, "--platform", "iOS", "--experimental", "--compute-precision", "float16",
           "--max-context-length", str(max_context), "--output-dir", str(out_dir)]
    if qconfig is not None:
        cmd += ["--compression-config", str(qconfig.resolve())]
    elif compression is not None:
        cmd += ["--compression", compression]
    if overwrite:
        cmd.append("--overwrite")
    cmd += extra
    log = out_dir / "_export.log"
    with log.open("a") as f:
        f.write(f"export start {time.strftime('%Y-%m-%d %H:%M:%S')} checkout={checkout} builder={BUILDER_MODULE}\n")
    t0 = time.time()
    stock.sh(cmd, cwd=checkout, log=log, env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    with log.open("a") as f:
        f.write(f"export end {time.strftime('%Y-%m-%d %H:%M:%S')} ({time.time() - t0:.0f} s)\n")
    aimodels = sorted(p for p in out_dir.rglob("*.aimodel") if p.is_dir())
    if not aimodels:
        raise SystemExit(f"no *.aimodel under {out_dir} (see {log})")
    return aimodels[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("--compression", default=None, help="Apple iOS palettization preset (instead of --qconfig)")
    ap.add_argument("--qconfig", default=str(DEFAULT_QCONFIG), help="kmeans_palettization_config yaml (default: 8-bit g32)")
    ap.add_argument("--max-context-length", type=int, default=stock.IOS_MAX_CONTEXT)
    ap.add_argument("--num-layers", type=int, default=None, help="truncate (probe / bisect runs)")
    ap.add_argument("--output-dir", default=None, help="default: exports/<name>_ios_ane_<qconfig stem>")
    ap.add_argument("--checkout", default=str(stock.DEFAULT_CHECKOUT))
    ap.add_argument("--arch", default=stock.AOT_ARCH)
    ap.add_argument("--author", default=DEFAULT_AUTHOR); ap.add_argument("--license", default=DEFAULT_LICENSE)
    ap.add_argument("--description", default=None)
    ap.add_argument("--devbundle", default=None, metavar="NAME", help="clone the device bundle to exports/apple_bench/devbundles/NAME")
    ap.add_argument("--stage", default=None, metavar="REPO", help="clone ios-ane-<arch>/ + ios-static/ to exports/hf_stage/REPO/")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--export-extra", default="")
    a = ap.parse_args()

    checkout = Path(a.checkout).resolve()
    qconfig = Path(a.qconfig).resolve() if a.qconfig and a.compression is None else None
    short = a.model.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    tag = qconfig.stem if qconfig else (a.compression or stock.IOS_COMPRESSION)
    out_dir = Path(a.output_dir).resolve() if a.output_dir else stock.EXPORTS / f"{short}_ios_ane_{tag}"
    extra = a.export_extra.split()
    if a.num_layers:
        extra += ["--num-layers", str(a.num_layers)]

    if a.skip_export:
        aimodel = sorted(p for p in out_dir.rglob("*.aimodel") if p.is_dir())[-1]
    else:
        aimodel = export_with_builder(a.model, out_dir, checkout, a.compression, qconfig, a.max_context_length,
                                      a.overwrite, extra)
    ir_dir = aimodel.parent
    desc = a.description or (f"{a.model} — Apple static iOS export for the Neural Engine ({tag}); LFM2 conv "
                             "history kept in the KV cache rows (zoo builder models/ios/lfm2.py)")
    stock.patch_metadata(aimodel, a.author, a.license, desc)

    aot_dir = ir_dir.parent / f"aot_{a.arch}_ane"
    aimodelc, regions = stock.aot(aimodel, aot_dir, a.arch)
    stock.make_device_bundle(ir_dir, aimodel, aimodelc, aot_dir, a.arch)
    summary = {
        "model": a.model, "builder": BUILDER_MODULE, "checkout": checkout.name,
        "checkout_sha": subprocess.run(["git", "-C", str(checkout), "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "compression": tag, "max_context_length": a.max_context_length, "num_layers": a.num_layers,
        "aimodel": str(aimodel), "aimodel_bytes": stock.du_bytes(aimodel),
        "aimodelc": str(aimodelc), "aimodelc_bytes": stock.du_bytes(aimodelc),
        "resources_bin_bytes": sum(p.stat().st_size for p in aimodelc.rglob("resources.bin")),
        "ane_regions": regions, "standard_regions": stock.STANDARD_REGIONS,
    }
    if regions == 0:
        print("0 ANE regions: the ANE compiler rejected a graph and coreai-build fell back to the GPU — NOT an ANE bundle")
    if a.devbundle:
        dst = stock.DEVBUNDLES / a.devbundle
        stock.clone_tree(aot_dir, dst)
        summary["devbundle"] = str(dst)
        print(f"device bundle: {dst}")
    if a.stage:
        summary["hf_stage"] = stock.stage_hf(a.stage, ir_dir, aimodel, aot_dir, a.arch)
        print(f"staged: {summary['hf_stage']}")
    (ir_dir.parent / "_lfm25_ane.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if regions else 3)


if __name__ == "__main__":
    main()
