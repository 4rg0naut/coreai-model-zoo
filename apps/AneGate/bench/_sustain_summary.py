#!/usr/bin/env python3
"""Summarize Sustain.swift device logs: per arm, the SUSTAIN_STATS line plus the decode/prefill/thermal/battery
trajectory at fixed elapsed marks, the first `serious` time and the battery delta per 1k tokens (only meaningful
when bstate is `unplugged`).  Usage: _sustain_summary.py _device_sustain_<tag>.log [...]"""
import re, sys

MARKS = [0, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
TRIAL = re.compile(r"SUSTAIN arm=(\S+) phase=trial i=(\d+) elapsed_s=([\d.]+) prompt_tps=([\d.]+) gen_tps=([\d.]+) "
                   r"tokens=(\d+) battery=(-?\d+) bstate=(\S+) thermal=(\S+)")
PHASE = re.compile(r"SUSTAIN arm=(\S+) phase=(idle_start|idle_end|load|warmup)(.*)")
STATS = re.compile(r"SUSTAIN_STATS arm=(\S+) (.*)")

for path in sys.argv[1:]:
    print(f"== {path}")
    trials, phases, stats = {}, {}, {}
    start_line = ""
    for line in open(path):
        if line.startswith("SUSTAIN_START"):
            start_line = line.strip()
        m = TRIAL.search(line)
        if m:
            arm, i, el, pt, gt, tok, bat, bs, th = m.groups()
            trials.setdefault(arm, []).append((float(el), float(pt), float(gt), int(tok), int(bat), bs, th))
            continue
        m = PHASE.search(line)
        if m:
            phases.setdefault(m.group(1), []).append(m.group(2) + m.group(3).strip())
            continue
        m = STATS.search(line)
        if m:
            stats[m.group(1)] = m.group(2)
    if start_line:
        print("  " + start_line[:200])
    for arm, r in trials.items():
        print(f"  arm {arm}: n={len(r)}")
        for ph in phases.get(arm, []):
            print("    " + ph[:230])
        pick = lambda mk: min(r, key=lambda x: abs(x[0] - mk))
        print("    decode@t : " + " ".join(f"{int(round(pick(mk)[0]))}s:{pick(mk)[2]:.1f}({pick(mk)[6][:3]})" for mk in MARKS))
        print("    prefill@t: " + " ".join(f"{int(round(pick(mk)[0]))}s:{int(pick(mk)[1])}" for mk in MARKS))
        print("    battery@t: " + " ".join(f"{int(round(pick(mk)[0]))}s:{pick(mk)[4]}%({pick(mk)[5][:4]})" for mk in MARKS))
        first_serious = next((x[0] for x in r if x[6] == "serious"), None)
        tokens = sum(x[3] for x in r)
        b0, b1 = r[0][4], r[-1][4]
        states = {x[5] for x in r}
        print(f"    first serious at {first_serious}s; tokens={tokens}; battery {b0}%→{b1}% ({','.join(sorted(states))})"
              + (f"; {tokens / max(1, b0 - b1):.0f} tokens per battery-% ({(b0 - b1) * 1000 / tokens:.2f} % per 1k tokens)" if b0 > b1 else "; no battery drop"))
        if arm in stats:
            print("    STATS " + stats[arm][:400])
