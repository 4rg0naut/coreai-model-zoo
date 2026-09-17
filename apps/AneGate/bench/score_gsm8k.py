"""Score TaskEval answer files (Documents/sustain/<tag>_answers.log, one JSON per line) with the same extraction and
normalization as ~/code/litertlm-convert/minicpm5_work/eval_gsm8k_*.py (prefer '#### N', then \\boxed{N}, then the last
number; numeric normalization '26.00' == '26'), so an accuracy from here is comparable to the litertlm-convert card rows.

    python3 score_gsm8k.py --arm 4bit=_device_task_4bit_answers.log --arm 6bit=… --arm int8=… [--json out.json]
Prints per-arm accuracy, EOS/cap rates, speed, and pairwise agreement (same extracted answer) between arms.
"""
import argparse, json, re, itertools


def norm(x):
    if x is None:
        return None
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return x.strip()


def extract(text):
    if not text:
        return None
    t = text.replace(",", "")
    m = re.findall(r"####\s*\$?(-?\d+(?:\.\d+)?)", t)
    if m:
        return m[-1]
    m = re.findall(r"\\boxed\{\s*\$?(-?\d+(?:\.\d+)?)", t)
    if m:
        return m[-1]
    m = re.findall(r"(-?\d+(?:\.\d+)?)", t)
    return m[-1] if m else None


def load(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line); rows[d["i"]] = d
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=path")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    arms = {}
    for spec in a.arm:
        name, path = spec.split("=", 1); arms[name] = load(path)
    common = sorted(set.intersection(*[set(r) for r in arms.values()]))
    out = {"n_common": len(common), "arms": {}, "agreement": {}}
    for name, rows in arms.items():
        ok = sum(1 for i in common if norm(extract(rows[i]["text"])) == norm(rows[i]["gold"]))
        eos = sum(1 for i in common if rows[i]["eos"]); capped = sum(1 for i in common if rows[i]["capped"])
        toks = sum(rows[i]["tokens"] for i in common); gen_s = sum(rows[i]["gen_s"] for i in common)
        ttft = sum(rows[i]["ttft_s"] for i in common) / max(1, len(common))
        out["arms"][name] = {"n": len(common), "correct": ok, "acc": round(ok / max(1, len(common)), 4), "eos": eos, "capped": capped,
                             "mean_tokens": round(toks / max(1, len(common)), 1), "gen_tps": round(toks / gen_s, 2) if gen_s else None,
                             "mean_ttft_s": round(ttft, 2), "n_rows": len(rows)}
        print(f"{name:8s} acc {ok}/{len(common)} = {ok / max(1, len(common)):.3f}  eos {eos}  capped {capped}  mean tokens {toks / max(1, len(common)):.0f}  "
              f"gen {toks / gen_s if gen_s else 0:.1f} tok/s  ttft {ttft:.2f} s")
    for x, y in itertools.combinations(arms, 2):
        same = sum(1 for i in common if norm(extract(arms[x][i]["text"])) == norm(extract(arms[y][i]["text"])))
        both_ok = sum(1 for i in common if norm(extract(arms[x][i]["text"])) == norm(arms[x][i]["gold"]) == norm(extract(arms[y][i]["text"])))
        out["agreement"][f"{x}~{y}"] = {"same_answer": same, "both_correct": both_ok, "n": len(common)}
        print(f"{x} ~ {y}: same extracted answer {same}/{len(common)}, both correct {both_ok}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1); print("wrote", a.json)


if __name__ == "__main__":
    main()
