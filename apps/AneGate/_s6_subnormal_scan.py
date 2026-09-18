"""Mac verification: how much of each intermediate tensor sits in the fp16 subnormal range (|v| < 6.1e-5) — the
exposure of the k8 bundle to an accelerator that flushes subnormals. Fraction of elements and fraction of energy
(sum of squares) per tensor kind, over the fixture prompts (prefill + decode walk)."""
import json, os, importlib.util, torch, torch.nn.functional as F
os.environ.setdefault("HF_HUB_OFFLINE", "1")
spec = importlib.util.spec_from_file_location("h", "s6_lfm2_check.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
from coreai_models.models.ios.lfm2 import Lfm2ForCausalLMForiOS, RMSNormK, ShortConv, Attention
from coreai_models.primitives.ios.mlp import MLP
TH = 2.0 ** -14
m = Lfm2ForCausalLMForiOS.from_hf("LiquidAI/LFM2.5-1.2B-Instruct", max_context_length=512, target_dtype=torch.float16).eval()
m.load_state_dict(torch.load("/private/tmp/ane_gate_dd/s6_pal8_fp16_sd.pt"))
acc = {}
def add(kind, t):
    t = t.float(); a = t.abs(); sub = (a < TH) & (a > 0); zero = a == 0
    d = acc.setdefault(kind, {"n": 0, "sub": 0, "zero": 0, "e": 0.0, "e_sub": 0.0, "min_nonzero": 1.0, "max": 0.0})
    d["n"] += t.numel(); d["sub"] += int(sub.sum()); d["zero"] += int(zero.sum()); d["e"] += float((t * t).sum()); d["e_sub"] += float((t[sub] ** 2).sum())
    nz = a[a > 0]; d["min_nonzero"] = min(d["min_nonzero"], float(nz.min()) if nz.numel() else 1.0); d["max"] = max(d["max"], float(a.max()))
def conv_hook(mod, inp, out):
    x = inp[0].transpose(-3, -1); bcx = mod.in_proj(x); b, c, xv = torch.split(bcx, mod.dim, dim=1)
    add("conv.B", b); add("conv.x", xv); add("conv.Bx", b * xv); add("conv.C", c); add("conv.out(residual add)", out)
def mlp_hook(mod, inp, out):
    x = inp[0]; b, q, _, d = x.shape; xr = x.reshape(b * q, d, 1, 1)
    g = mod.gate_proj(xr); u = mod.up_proj(xr); add("mlp.gate", g); add("mlp.up", u); add("mlp.silu(g)*u", F.silu(g) * u); add("mlp.out(residual add)", out)
def attn_hook(mod, inp, out):
    add("attn.in(normed)", inp[0]); add("attn.out(residual add)", out)
def norm_hook(kind):
    def f(mod, inp, out):
        add(f"norm.{kind}.in", inp[0]); add(f"norm.{kind}.y*y (K-scaled squares)", (inp[0] * mod._k) ** 2); add(f"norm.{kind}.out", out)
    return f
def layer_hook(mod, inp, out): add("residual stream", out)
for name, mod in m.named_modules():
    if isinstance(mod, ShortConv): mod.register_forward_hook(conv_hook)
    elif isinstance(mod, MLP): mod.register_forward_hook(mlp_hook)
    elif isinstance(mod, Attention): mod.register_forward_hook(attn_hook)
    elif isinstance(mod, RMSNormK): mod.register_forward_hook(norm_hook("qk" if "layernorm" in name else ("final" if "embedding_norm" in name else "stream")))
    elif name.startswith("extend.model.layers.") and name.count(".") == 3: mod.register_forward_hook(layer_hook)
fx = json.load(open("fixtures/lfm25_1_2b/fixture.json"))
for p in fx["prompts"]:
    seq = list(p["prompt_ids"]) + list(p["expected_ids"])[:12]; P = len(p["prompt_ids"])
    logits, walk, run_until = h.emulate(m, seq, 512, torch.float16, print)
    cur = run_until(P, 0)
    for i in range(P, len(seq)): cur = run_until(i + 1, cur)
print(f"{'tensor':40s} {'subnormal %':>11s} {'zero %':>7s} {'energy in subnormals':>20s} {'min |v|>0':>10s} {'max |v|':>9s}")
for k, d in acc.items():
    print(f"{k:40s} {100 * d['sub'] / d['n']:10.3f}% {100 * d['zero'] / d['n']:6.2f}% {100 * d['e_sub'] / max(d['e'], 1e-30):19.5f}% {d['min_nonzero']:10.2e} {d['max']:9.2f}")
