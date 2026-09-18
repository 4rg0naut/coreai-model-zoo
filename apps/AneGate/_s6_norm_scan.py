"""Mac verification of RMSNormK's scale choice: over GSM8K chat prompts, record per norm kind the max |input|,
the max of (K*x)^2 (fp16 overflow at 65504), and the min of mean((K*x)^2) + K^2*eps (fp16 subnormal below 6.1e-5)."""
import json, os, sys, time, importlib.util, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
spec = importlib.util.spec_from_file_location("h", "s6_lfm2_check.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
from transformers import AutoTokenizer
from coreai_models.models.ios.lfm2 import Lfm2ForCausalLMForiOS, RMSNormK
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
m = Lfm2ForCausalLMForiOS.from_hf("LiquidAI/LFM2.5-1.2B-Instruct", max_context_length=512, target_dtype=torch.float16).eval()
m.load_state_dict(torch.load("/private/tmp/ane_gate_dd/s6_pal8_fp16_sd.pt"))
st = {}
def mk(kind):
    def f(mod, inp, out):
        x = inp[0].float(); k = float(mod._k); y = x * k
        ms = (y * y).mean(-1) + float(mod._eps)
        d = st.setdefault(kind, {"in_max": 0.0, "y2_max": 0.0, "ms_min": 1e9, "n_tokens": 0})
        d["in_max"] = max(d["in_max"], float(x.abs().max())); d["y2_max"] = max(d["y2_max"], float((y * y).max()))
        d["ms_min"] = min(d["ms_min"], float(ms.min())); d["n_tokens"] += x.shape[1] if x.dim() == 4 else x.shape[-2]
    return f
for name, mod in m.named_modules():
    if isinstance(mod, RMSNormK):
        kind = name.split(".")[-1]; mod.register_forward_hook(mk(kind))
COT = "\n\nSolve this step by step. After your reasoning, write the final answer on its own line in the exact form:\n#### <number>"
items = [json.loads(l) for l in open("bench/tasks/gsm8k_200.jsonl") if l.strip()][:N]
t0 = time.time()
for it in items:
    ids = tok.apply_chat_template([{"role": "user", "content": it["question"] + COT}], add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True, return_dict=True)["input_ids"]
    seq = list(ids)
    logits, walk, run_until = h.emulate(m, seq, 512, torch.float16, print)
    run_until(len(seq), 0)
print(f"{len(items)} prompts ({time.time() - t0:.0f} s)")
for k, d in st.items():
    print(f"{k:16s} K-scaled: |x| max {d['in_max']:7.3f}  (Kx)^2 max {d['y2_max']:9.1f} (overflow 65504, margin {65504 / max(d['y2_max'], 1e-9):5.1f}x)  "
          f"mean+eps min {d['ms_min']:.2e} (subnormal < 6.1e-05, margin {d['ms_min'] / 6.1e-5:5.1f}x)")
