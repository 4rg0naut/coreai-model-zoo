"""Try candidate no-think chat prompts on an HF checkpoint (fp32) and print, per prompt, the
rollout length, whether it stopped on EOS, the minimum per-step top-2 softmax margin and the
margin at the EOS step — to pick a `chat` prompt for make_fixture.py that reaches EOS on a
margin-clear step. Usage: python explore_chat.py --hf-id <id> [--n 24] [--msg ... --msg ...]"""
import argparse, os, warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1"); warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MSGS = ["1+1=?", "What is the capital of France?", "Say hello.", "Reply with only the word yes.",
                "What is 2+2? Answer with just the number.", "How many days are in a week?",
                "Translate 'thank you' to French.", "Is water wet? Answer yes or no.",
                "Name the first three letters of the alphabet.", "What is 3 times 4?"]

ap = argparse.ArgumentParser()
ap.add_argument("--hf-id", required=True); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--msg", action="append")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.hf_id)
model = AutoModelForCausalLM.from_pretrained(a.hf_id, dtype=torch.float32).eval()
eos = set()
for e in (getattr(tok, "eos_token_id", None), getattr(model.config, "eos_token_id", None)):
    if isinstance(e, int): eos.add(e)
    elif isinstance(e, (list, tuple)): eos.update(int(x) for x in e)
print("eos ids:", sorted(eos))
for m in (a.msg or DEFAULT_MSGS):
    ids = tok.apply_chat_template([{"role": "user", "content": m}], add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=False)
    if not hasattr(ids, "shape"): ids = ids["input_ids"]
    gen, mp = [], []
    with torch.no_grad():
        for _ in range(a.n):
            row = model(ids).logits[0, -1].float(); p = torch.softmax(row, -1); t2 = torch.topk(p, 2).values
            nxt = int(row.argmax()); mp.append(float(t2[0] - t2[1])); gen.append(nxt)
            if nxt in eos: break
            ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
    stopped = gen[-1] in eos
    print(f"{m!r:52} n={len(gen):2d} stop={stopped!s:5} min={min(mp):.3f} eos_m={(mp[-1] if stopped else float('nan')):.3f} "
          f"text={tok.decode(gen)!r}")
