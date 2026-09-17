#!/usr/bin/env python3
"""S1 export driver (session-local). Runs Apple's stock `coreai.llm.export` in the coreai-models-rebase
checkout (zoo-0.4, coreai-torch 0.4.2 — the same toolchain as today's gated MiniCPM5 ANE bundles),
optionally AOT-compiles for the Neural Engine (h18p) and counts ANE regions, and lays a loadable
devbundle dir under coreai-models/exports/apple_bench/devbundles/<name>/ for the bench/gate apps.

  ane_export.py --hf-id openbmb/MiniCPM5-2B --out minicpm5_2b_ios_pal8_g32 --mode ios-ane \
      --compression-config <yaml> --devbundle minicpm5_2b_ane_pal8 --eos '<|im_end|>'
  ane_export.py --hf-id openbmb/MiniCPM5-2B --out minicpm5_2b_int4lin --mode macos-dyn \
      --compression 4bit --devbundle minicpm5_2b_gpu_int4lin --eos '<|im_end|>'
"""
import argparse, datetime, json, os, shutil, subprocess, sys
from pathlib import Path

COREAI = Path.home() / "code/coreai"
REBASE = COREAI / "coreai-models-rebase"
EXPORTS = COREAI / "coreai-models/exports"
DEVB = EXPORTS / "apple_bench/devbundles"
AOT_XCODE = "/Applications/Xcode-27.0.0-RC.app/Contents/Developer"

ap = argparse.ArgumentParser()
ap.add_argument("--hf-id", required=True)
ap.add_argument("--out", required=True, help="dir name under coreai-models/exports/")
ap.add_argument("--mode", choices=["ios-ane", "macos-dyn"], required=True)
ap.add_argument("--compression", default=None)
ap.add_argument("--compression-config", default=None)
ap.add_argument("--max-context-length", type=int, default=4096, help="iOS static graph cap (ignored for macos-dyn)")
ap.add_argument("--devbundle", required=True, help="devbundles/<name>")
ap.add_argument("--eos", default=None, help="rewrite tokenizer eos (MiniCPM5 chat: <|im_end|>)")
ap.add_argument("--skip-export", action="store_true", help="reuse an existing export, only AOT/devbundle")
ap.add_argument("--extra-args", default="", help="extra coreai.llm.export flags, space separated (e.g. --disable-embedding-quantization-ios)")
a = ap.parse_args()

out = EXPORTS / a.out
out.mkdir(parents=True, exist_ok=True)
log = out / "_export.log"
sha = subprocess.run(["git", "-C", str(REBASE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ver = subprocess.run([str(REBASE / ".venv/bin/python"), "-c", "import coreai_torch;print(coreai_torch.__version__)"],
                     capture_output=True, text=True).stdout.strip()

def run(cmd, env=None, cwd=None):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n"); f.flush()
        r = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"FAILED ({r.returncode}): {' '.join(cmd)}\nsee {log}"); sys.exit(r.returncode)

if not a.skip_export:
    with open(log, "a") as f:
        f.write(f"export start {datetime.datetime.now():%c} checkout={sha} coreai-torch={ver} mode={a.mode} "
                f"compression={a.compression or a.compression_config}\n")
    cmd = ["uv", "run", "coreai.llm.export", a.hf_id, "--experimental", "--compute-precision", "float16",
           "--output-dir", str(out)]
    if a.mode == "ios-ane":
        cmd += ["--platform", "iOS", "--max-context-length", str(a.max_context_length)]
    else:
        cmd += ["--platform", "macOS"]
    if a.compression_config:
        cmd += ["--compression-config", str(Path(a.compression_config).resolve())]
    else:
        cmd += ["--compression", a.compression or ("4bit_weight_palettized_group32" if a.mode == "ios-ane" else "4bit")]
    cmd += a.extra_args.split()
    run(cmd, cwd=REBASE, env={**os.environ, "HF_HUB_OFFLINE": "1"})

aimodel = next(p for p in out.rglob("*.aimodel") if p.is_dir())
bundle_dir = aimodel.parent
meta = json.loads((bundle_dir / "metadata.json").read_text())

def fix_eos(tok_dir: Path, eos: str):
    cfg = tok_dir / "tokenizer_config.json"
    d = json.loads(cfg.read_text()); d["eos_token"] = eos
    cfg.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    stm = tok_dir / "special_tokens_map.json"
    if stm.exists():
        m = json.loads(stm.read_text()); e = m.get("eos_token")
        if isinstance(e, dict): e["content"] = eos
        else: m["eos_token"] = eos
        stm.write_text(json.dumps(m, ensure_ascii=False, indent=2))

dev = DEVB / a.devbundle
if dev.exists(): shutil.rmtree(dev)
dev.mkdir(parents=True)

if a.mode == "ios-ane":
    aot = out / "aot_h18p_ane"
    if aot.exists(): shutil.rmtree(aot)
    aot.mkdir()
    with open(log, "a") as f: f.write(f"aot start {datetime.datetime.now():%c}\n")
    run(["xcrun", "coreai-build", "compile", str(aimodel), "--output", str(aot), "--platform", "iOS",
         "--preferred-compute", "neural-engine", "--architecture", "h18p"],
        env={**os.environ, "DEVELOPER_DIR": AOT_XCODE})
    aimodelc = next(aot.glob("*.aimodelc"))
    regions = sum(1 for _ in aimodelc.rglob("*ANE_region*"))
    with open(log, "a") as f: f.write(f"aot done {datetime.datetime.now():%c} {aimodelc.name} ANE_regions={regions}\n")
    print(f"AOT {aimodelc.name}: {regions} ANE regions")
    if regions == 0:
        print("0 ANE regions: silent GPU fallback — not an ANE arm"); sys.exit(3)
    meta["assets"]["main"] = aimodelc.name
    meta["compilation"]["targets"] = ["h18p neural-engine (xcrun coreai-build compile --platform iOS "
                                      "--preferred-compute neural-engine --architecture h18p)"]
    subprocess.run(["cp", "-Rc", str(aimodelc), str(dev / aimodelc.name)], check=True)
else:
    subprocess.run(["cp", "-Rc", str(aimodel), str(dev / aimodel.name)], check=True)
(dev / "metadata.json").write_text(json.dumps(meta, indent=2))
shutil.copytree(bundle_dir / "tokenizer", dev / "tokenizer")
if a.eos:
    fix_eos(dev / "tokenizer", a.eos)
    fix_eos(bundle_dir / "tokenizer", a.eos)
size = subprocess.run(["du", "-sh", str(dev)], capture_output=True, text=True).stdout.split()[0]
print(f"devbundle {dev} ({size}) main={meta['assets']['main']} compression={meta.get('compression')}")
with open(log, "a") as f: f.write(f"devbundle {dev} {size}\n")
