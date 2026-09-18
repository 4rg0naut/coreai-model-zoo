#!/usr/bin/env python3
"""S6 export driver (session-local): LFM2.5 on Apple's static iOS contract via the zoo overlay builder
`models/ios/lfm2.py` (registered at runtime — Apple's registry.py stays unmodified), then AOT for the
Neural Engine (h18p) with the ANE region count and the ANEC diagnostics, then a loadable devbundle
under coreai-models/exports/apple_bench/devbundles/<name>/ for the gate/bench apps.

  _ane_export_s6.py --hf-id LiquidAI/LFM2.5-1.2B-Instruct --out lfm25_1_2b_ios_probe_L3 \
      --compression-config ../../coreai-models-community/conversion/lfm25_pal8_g32.yaml \
      --num-layers 3 --max-context-length 256 --devbundle lfm25_probe_L3
  _ane_export_s6.py --hf-id LiquidAI/LFM2.5-1.2B-Instruct --out lfm25_1_2b_ios_pal8_g32 \
      --compression-config ... --max-context-length 4096 --devbundle lfm25_1_2b_ane_pal8

--compression none = fp16 dense control (S3's arm E). --skip-export reuses the .aimodel and only AOTs.
"""
import argparse, datetime, json, os, re, shutil, subprocess, sys
from pathlib import Path

COREAI = Path.home() / "code/coreai"
REBASE = COREAI / "coreai-models-rebase"
EXPORTS = COREAI / "coreai-models/exports"
DEVB = EXPORTS / "apple_bench/devbundles"
AOT_XCODE = "/Applications/Xcode-27.0.0-RC.app/Contents/Developer"
PY = REBASE / ".venv/bin/python"
BOOT = ("import sys; from coreai_models.models.ios.lfm2 import register; register(); "
        "from coreai_models.llm.export import main; main()")

ap = argparse.ArgumentParser()
ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-1.2B-Instruct")
ap.add_argument("--out", required=True, help="dir name under coreai-models/exports/")
ap.add_argument("--compression", default=None, help="Apple preset name or 'none'")
ap.add_argument("--compression-config", default=None, help="kmeans yaml")
ap.add_argument("--max-context-length", type=int, default=4096)
ap.add_argument("--num-layers", type=int, default=None)
ap.add_argument("--devbundle", default=None, help="devbundles/<name> (omit = no devbundle)")
ap.add_argument("--skip-export", action="store_true")
ap.add_argument("--skip-aot", action="store_true")
ap.add_argument("--extra-args", default="")
a = ap.parse_args()

out = EXPORTS / a.out
out.mkdir(parents=True, exist_ok=True)
log = out / "_export.log"
sha = subprocess.run(["git", "-C", str(REBASE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ver = subprocess.run([str(PY), "-c", "import coreai_torch;print(coreai_torch.__version__)"], capture_output=True, text=True).stdout.strip()


def run(cmd, env=None, cwd=None):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(str(c) for c in cmd)}\n"); f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"FAILED ({r.returncode}): {' '.join(str(c) for c in cmd)}\nsee {log}"); sys.exit(r.returncode)


t_start = datetime.datetime.now()
if not a.skip_export:
    with open(log, "a") as f:
        f.write(f"export start {t_start:%c} checkout={sha} coreai-torch={ver} compression={a.compression or a.compression_config} "
                f"layers={a.num_layers} ctx={a.max_context_length}\n")
    cmd = [PY, "-c", BOOT, a.hf_id, "--experimental", "--compute-precision", "float16", "--platform", "iOS",
           "--max-context-length", str(a.max_context_length), "--output-dir", str(out), "--overwrite"]
    if a.compression_config:
        cmd += ["--compression-config", str(Path(a.compression_config).resolve())]
    else:
        cmd += ["--compression", a.compression or "4bit_weight_palettized_group32"]
    if a.num_layers:
        cmd += ["--num-layers", str(a.num_layers)]
    cmd += a.extra_args.split()
    run(cmd, cwd=REBASE, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    with open(log, "a") as f:
        f.write(f"export done {datetime.datetime.now():%c} ({(datetime.datetime.now() - t_start).total_seconds():.0f} s)\n")

aimodel = next(p for p in out.rglob("*.aimodel") if p.is_dir())
bundle_dir = aimodel.parent
meta = json.loads((bundle_dir / "metadata.json").read_text())
size = subprocess.run(["du", "-sh", str(aimodel)], capture_output=True, text=True).stdout.split()[0]
print(f"export {aimodel.name} ({size}) in {(datetime.datetime.now() - t_start).total_seconds():.0f} s")

summary = {"aimodel": str(aimodel), "aimodel_size": size, "checkout": sha, "coreai_torch": ver,
           "compression": a.compression or a.compression_config, "num_layers": a.num_layers, "max_context_length": a.max_context_length}

if not a.skip_aot:
    aot = out / "aot_h18p_ane"
    if aot.exists(): shutil.rmtree(aot)
    aot.mkdir()
    aot_log = out / "_aot.log"
    t0 = datetime.datetime.now()
    with open(log, "a") as f: f.write(f"aot start {t0:%c}\n")
    with open(aot_log, "w") as f:
        r = subprocess.run(["xcrun", "coreai-build", "compile", str(aimodel), "--output", str(aot), "--platform", "iOS",
                            "--preferred-compute", "neural-engine", "--architecture", "h18p"],
                           env={**os.environ, "DEVELOPER_DIR": AOT_XCODE}, stdout=f, stderr=subprocess.STDOUT)
    aot_s = (datetime.datetime.now() - t0).total_seconds()
    txt = aot_log.read_text(errors="replace")
    anec_fail = len(re.findall(r"ANECCompile(?:Offline)?\(\) failed", txt))
    msgs = re.findall(r'ane_validation_message[^\n]{0,300}', txt)
    kinds = {}
    for m in msgs:
        k = re.sub(r'.*?"(?:ane_validation_message)"?\s*[:=]?\s*', "", m)[:160]
        kinds[k] = kinds.get(k, 0) + 1
    if r.returncode != 0:
        print(f"AOT FAILED exit {r.returncode}; see {aot_log}"); sys.exit(r.returncode)
    aimodelc = next(aot.glob("*.aimodelc"))
    regions = sum(1 for _ in aimodelc.rglob("*ANE_region*"))
    fns = sorted(set(re.findall(r"(?:extend|prompt_opt)_\d+_\d+", txt)))
    csize = subprocess.run(["du", "-sh", str(aimodelc)], capture_output=True, text=True).stdout.split()[0]
    with open(log, "a") as f:
        f.write(f"aot done {datetime.datetime.now():%c} ({aot_s:.0f} s) {aimodelc.name} {csize} ANE_regions={regions} anec_fail={anec_fail} "
                f"validation_msgs={len(msgs)}\n")
    print(f"AOT {aimodelc.name} ({csize}) in {aot_s:.0f} s: {regions} ANE regions, ANEC failures {anec_fail}, "
          f"ane_validation_message {len(msgs)}")
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:4d}x {k}")
    summary.update({"aimodelc": str(aimodelc), "aimodelc_size": csize, "ane_regions": regions, "anec_failures": anec_fail,
                    "validation_messages": len(msgs), "validation_kinds": kinds, "aot_s": aot_s})
    (out / "_s6_summary.json").write_text(json.dumps(summary, indent=2))
    if regions == 0:
        print("0 ANE regions: silent GPU fallback — not an ANE arm")
        if a.devbundle is None: sys.exit(3)
    if a.devbundle:
        dev = DEVB / a.devbundle
        if dev.exists(): shutil.rmtree(dev)
        dev.mkdir(parents=True)
        meta["assets"]["main"] = aimodelc.name
        meta["compilation"]["targets"] = ["h18p neural-engine (xcrun coreai-build compile --platform iOS "
                                          "--preferred-compute neural-engine --architecture h18p)"]
        subprocess.run(["cp", "-Rc", str(aimodelc), str(dev / aimodelc.name)], check=True)
        (dev / "metadata.json").write_text(json.dumps(meta, indent=2))
        shutil.copytree(bundle_dir / "tokenizer", dev / "tokenizer")
        dsize = subprocess.run(["du", "-sh", str(dev)], capture_output=True, text=True).stdout.split()[0]
        rsize = sum(p.stat().st_size for p in aimodelc.rglob("resources.bin")) / 1e9
        print(f"devbundle {dev} ({dsize}, resources.bin {rsize:.2f} GB) main={meta['assets']['main']}")
        with open(log, "a") as f: f.write(f"devbundle {dev} {dsize} resources.bin {rsize:.2f} GB\n")
        summary.update({"devbundle": str(dev), "devbundle_size": dsize, "resources_bin_gb": round(rsize, 3)})
        (out / "_s6_summary.json").write_text(json.dumps(summary, indent=2))
else:
    (out / "_s6_summary.json").write_text(json.dumps(summary, indent=2))
