#!/usr/bin/env python3
"""S7: one row per prompt from a s6_lfm2_check.py --device-log record (_s7_fx*_<tag>.json): device-vs-twin top-1
agreement, |gap diff| mean / median / p90 on jointly-correct steps, signed median (dev - twin) and the median
gap RATIO dev/twin (S6: the phone compresses short-prompt gaps to ~0.78x of the twin). Usage: _s7_gap_report.py a.json [b.json ...]"""
import json, statistics as st, sys

def rows(path):
    d = json.load(open(path)); out = []
    for name, r in d["results"].items():
        cmp = r.get("device_compare") or []
        if not cmp: continue
        agree = sum(c["mac_got"] == c["dev_got"] for c in cmp)
        jc = [c for c in cmp if c["mac_got"] == c["dev_got"] == c["exp"]]
        diff = [abs(c["mac_gap"] - c["dev_gap"]) for c in jc]
        sgn = [c["dev_gap"] - c["mac_gap"] for c in jc]
        ratio = [c["dev_gap"] / c["mac_gap"] for c in jc if c["mac_gap"] > 0.5]
        q = lambda v, p: (sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan"))
        out.append((name, f"{agree}/{len(cmp)}", len(jc), st.mean(diff) if diff else float("nan"), st.median(diff) if diff else float("nan"),
                    q(diff, 0.9), st.median(sgn) if sgn else float("nan"), st.median(ratio) if ratio else float("nan"), r["verdict"], len(r["fails"])))
    return out

print(f"{'record':34s} {'prompt':8s} {'top1':>7s} {'n':>3s} {'mean':>6s} {'med':>6s} {'p90':>6s} {'sgn':>6s} {'ratio':>6s}  twin-verdict")
for p in sys.argv[1:]:
    for name, agree, n, mean, med, p90, sgn, ratio, v, nf in rows(p):
        print(f"{p[:34]:34s} {name:8s} {agree:>7s} {n:3d} {mean:6.2f} {med:6.2f} {p90:6.2f} {sgn:+6.2f} {ratio:6.2f}  {v} fails={nf}")
