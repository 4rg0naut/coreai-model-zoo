"""Decode the FR/TF token ids in a device log with the model's tokenizer (human-readable report).
Usage: python decode_log.py --hf-id <id> <device log> [--fixture fixtures/<m>/fixture.json]"""
import argparse, json, os, re
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--hf-id", required=True); ap.add_argument("--fixture")
a = ap.parse_args(); tok = AutoTokenizer.from_pretrained(a.hf_id)
fx = {p["name"]: p for p in json.load(open(a.fixture))["prompts"]} if a.fixture else {}
for line in open(a.log):
    m = re.match(r"FR (\w+) got\((\d+)\)=\[([\d, ]*)\]", line)
    if m:
        ids = [int(x) for x in m.group(3).split(",") if x.strip()]
        print(f"[{m.group(1)}] device FR : {tok.decode(ids)!r}")
        if m.group(1) in fx: print(f"[{m.group(1)}] oracle    : {fx[m.group(1)]['expected_text']!r}")
    m = re.match(r"TF (\w+) k=(\d+) exp=(\d+) got=(\d+) second=(\d+) oracle_margin=([\d.]+) dev_gap=([\d.-]+) (FAIL|KNIFE)", line)
    if m:
        print(f"[{m.group(1)}] TF k={m.group(2)} {m.group(8)}: exp {tok.decode([int(m.group(3))])!r} got {tok.decode([int(m.group(4))])!r} (2nd {tok.decode([int(m.group(5))])!r}) oracle_margin={m.group(6)} dev_gap={m.group(7)}")
