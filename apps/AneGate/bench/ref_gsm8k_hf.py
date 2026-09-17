"""Reference GSM8K run of the HF checkpoint on the Mac (transformers, greedy, no-think chat template, the same CoT suffix
and 640-token cap as the phone's TaskEval), written in TaskEval's answer JSONL so score_gsm8k.py treats it as an arm.
bf16 on MPS by default (what litertlm-convert's parity_gsm8k.py uses as its baseline); --dtype float32 --device cpu for fp32.

    ~/.venv-coreml-llm-py312/bin/python ref_gsm8k_hf.py --task tasks/gsm8k_200.jsonl --out _ref_hf_bf16_answers.log
"""
import argparse, json, os, time, warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1"); warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COT = "\n\nSolve this step by step. After your reasoning, write the final answer on its own line in the exact form:\n#### <number>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="openbmb/MiniCPM5-2B"); ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--recipe", default=None, help="palettization yaml (or preset name) applied to the fp32 weights with coreai-opt's "
                    "k-means before the run (../simulate_recipe.py; run in the coreai-models-rebase venv) = the Mac proxy of an ANE bundle")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.hf_id)
    model = AutoModelForCausalLM.from_pretrained(a.hf_id, dtype=torch.float32).eval()
    if a.recipe:
        import importlib.util, pathlib, time as _t
        spec_ = importlib.util.spec_from_file_location("sim", pathlib.Path(__file__).resolve().parent.parent / "simulate_recipe.py")
        sim = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(sim)
        if a.recipe == "int8_per_channel_linear":   # the 1B's shipped GPU recipe (conversion/minicpm5_int8sym.yaml): symmetric int8, one scale per output channel
            r = {"name": a.recipe, "global": {"linear_int8_per_channel": True}, "by_name": []}
        else:
            r = sim.load_recipe(None, a.recipe) if a.recipe in sim.PRESETS or a.recipe == "int8_block32_linear" else sim.load_recipe(a.recipe, None)
        t0 = _t.time(); n8 = 0
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                spec = sim.spec_for(r, "extend." + n)
                if spec.get("linear_int8_per_channel"):
                    w = m.weight.data.float(); sc = w.abs().amax(dim=1, keepdim=True) / 127.0; sc[sc == 0] = 1.0
                    m.weight.data = (torch.round(w / sc).clamp(-127, 127) * sc).to(m.weight.dtype); n8 += 1; continue
                if spec.get("linear_int8_block32"):
                    m.weight.data = sim.int8_block32(m.weight.data); n8 += 1; continue
                m.weight.data = sim.palettize(m.weight.data, spec); n8 += int(spec["n_bits"]) == 8
        print(f"recipe {r['name']} applied ({n8} modules at 8-bit) in {_t.time() - t0:.0f} s", flush=True)
    model = model.to(getattr(torch, a.dtype)).to(a.device)
    # stop on the chat turn end (<|im_end|>, 130073) as the phone bundles do, not only on the base eos (</s>)
    eos_ids = set()
    for e in (tok.eos_token_id, getattr(model.config, "eos_token_id", None), tok.convert_tokens_to_ids("<|im_end|>")):
        if isinstance(e, int) and e >= 0: eos_ids.add(e)
        elif isinstance(e, (list, tuple)): eos_ids.update(int(x) for x in e)
    eos = sorted(eos_ids); pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    print("stop ids:", eos, flush=True)
    items = [json.loads(l) for l in open(a.task) if l.strip()][a.start:a.start + a.n]
    done = {json.loads(l)["i"] for l in open(a.out)} if os.path.exists(a.out) else set()
    with open(a.out, "a") as f:
        for it in items:
            if it["i"] in done: continue
            ids = tok.apply_chat_template([{"role": "user", "content": it["question"] + COT}], add_generation_prompt=True,
                                          return_tensors="pt", enable_thinking=False)
            if not hasattr(ids, "shape"): ids = ids["input_ids"]
            ids = ids.to(a.device); t0 = time.time()
            with torch.no_grad():
                out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.max_tokens, do_sample=False,
                                     eos_token_id=eos, pad_token_id=pad)
            gen = out[0, ids.shape[1]:].tolist(); stopped = bool(gen) and gen[-1] in eos_ids
            if stopped: gen = gen[:-1]
            dt = time.time() - t0
            rec = {"i": it["i"], "prompt_ids": int(ids.shape[1]), "tokens": len(gen), "eos": stopped, "capped": (not stopped) and len(gen) >= a.max_tokens,
                   "ttft_s": 0.0, "gen_s": round(dt, 3), "gen_tps": round(len(gen) / dt, 2) if dt else 0, "gold": it["gold"],
                   "text": tok.decode(gen, skip_special_tokens=True)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{it['i']}] tokens={len(gen)} eos={stopped} {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
