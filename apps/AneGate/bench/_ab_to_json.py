#!/usr/bin/env python3
"""Turn an `_ab_<tag>.log` / `_abn_<tag>.log` (same-day interleaved A/B, AppleBenchRunnerGA) into the zoo's
`iphone-ane-vs-gpu-ab/1` record.  Usage:
  _ab_to_json.py <log> <out.json> --device "iPhone 17 Pro, iOS 27.0 (24A437)" --arm <name>="<bundle description>" ...
Every STATS line becomes one run (round, prompt/gen tok/s, load_s, footprint) with its five trials."""
import argparse, json, re

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("out")
ap.add_argument("--device", required=True)
ap.add_argument("--arm", action="append", default=[], help='name="description"')
ap.add_argument("--note", default="")
a = ap.parse_args()

desc = dict(x.split("=", 1) for x in a.arm)
runs, trials = {}, {}
proto = {}
for line in open(a.log):
    m = re.match(r"^(AB|ABN) (\S+) .*?p=(\d+) g=(\d+) n=(\d+)", line)
    if m:
        proto = {"p": int(m.group(3)), "g": int(m.group(4)), "n": int(m.group(5))}
    m = re.match(r"^r(\d+) (\S+): (.*?)\| STATS .*?prompt_tps=([\d.]+) gen_tps=([\d.]+) load_s=([\d.]+) footprint_gb=([\d.]+)", line)
    if m:
        r, arm, _, pt, gt, ls, fp = m.groups()
        runs.setdefault(arm, []).append({"round": int(r), "prompt_tps": float(pt), "gen_tps": float(gt),
                                         "load_s": float(ls), "footprint_gb": float(fp),
                                         "trials_prompt_tps": [], "trials_gen_tps": []})
        continue
    m = re.match(r"^\s+r(\d+) (\S+) trial \d+: prompt ([\d.]+) tok/s, gen ([\d.]+) tok/s", line)
    if m:
        r, arm, pt, gt = m.groups()
        run = next(x for x in runs[arm] if x["round"] == int(r))
        run["trials_prompt_tps"].append(float(pt)); run["trials_gen_tps"].append(float(gt))

order = list(runs.keys())
rec = {
    "schema": "iphone-ane-vs-gpu-ab/1",
    "date": "2026-09-15",
    "device": a.device,
    "app": "AppleBenchRunnerGA (Apple llm-benchmark method: splitmix64 random prompt, greedy, 1 warmup + n timed trials, "
           "genTps excludes the first token) on the unmodified Apple main 7359dbc package; all bundles embedded in ONE app, "
           "picked per launch by AB_MODEL, order " + " ".join(order) + " repeated per round, 20 s idle between launches",
    "protocol": proto,
    "note": a.note,
    "arms": {arm: {"bundle": desc.get(arm, arm), "runs": rs} for arm, rs in runs.items()},
}
json.dump(rec, open(a.out, "w"), indent=1)
print(f"wrote {a.out}: " + "; ".join(f"{arm} gen " + " / ".join(f"{x['gen_tps']:.1f}" for x in rs) for arm, rs in runs.items()))
