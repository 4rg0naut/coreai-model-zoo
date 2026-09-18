#!/usr/bin/env python3
"""Mac-side numerics check of the LFM2 static iOS builder (models/ios/lfm2.py) against the HF fp32
checkpoint, driving the eager module EXACTLY as Apple's StaticShapeEngine drives the graphs:
static query lengths {8, 16, 64}, the context ladder {256, 512, ...}, aligned batches
(batchStart = position // q * q, re-run on every decode step), position_ids / in_step /
causal_mask filled as InputHandler+StaticBucket.swift fills them (0 for p <= in_step + t,
-40000 elsewhere, columns beyond tokensInBatch fully masked), padding tokens = 0.

Teacher-forced on [prompt + the HF greedy continuation]: per position, argmax of the emulated
engine's logits vs the HF argmax, plus the max |logit| deviation. The prompt is > 64 tokens so
the walk covers prompt_opt(q=64) -> extend(q=64) -> decode(q=8) — the transition where a
"last written" conv state would break.

    ~/code/coreai/coreai-models-rebase/.venv/bin/python s6_lfm2_check.py [--num-layers 3] [--steps 24]
"""
import argparse, json, os, time, warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1"); warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STATIC_Q = (8, 16, 64)
MIN_CTX = 256
SENTINEL = -40000.0

PROMPT = ("Here is a short passage. The Neural Engine on Apple silicon is a block of matrix multiplication "
          "hardware that runs fixed-shape graphs at low power. Language models with a short-convolution mixer "
          "keep only two previous columns of state per layer, which is far less than a full attention cache. "
          "Question: in one paragraph, why does a hybrid model with few attention layers need less memory "
          "during decoding than a dense transformer of the same size?")


def ladder_ctx(position: int, max_ctx: int) -> int:
    ctx = MIN_CTX
    while ctx <= position:
        ctx *= 2
    assert ctx <= max_ctx, (position, max_ctx)
    return ctx


def pick_q(remaining: int) -> int:
    for q in STATIC_Q:
        if q >= remaining:
            return q
    return STATIC_Q[-1]


@torch.no_grad()
def emulate(model, seq: list[int], max_ctx: int, dtype, log):
    """Run the engine's batch walk over `seq` as it grows one token at a time after `prompt_len`
    (teacher-forced): returns logits[position] for every position (last run wins, as on device)."""
    n_layers = len(model.extend.model.layers)
    W = model.cache_width
    k_cache = torch.zeros(n_layers, 1, W, 1, max_ctx, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    logits = {}
    walk = []

    def run_until(total: int, current: int) -> int:
        while current < total:
            remaining = total - current
            use_prefill = remaining > STATIC_Q[-1]
            q = pick_q(remaining)
            ctx = ladder_ctx(current, max_ctx)
            batch_start = (current // q) * q
            batch_end = min(batch_start + q - 1, total - 1)
            n_tok = batch_end - batch_start + 1
            toks = seq[batch_start:batch_end + 1] + [0] * (q - n_tok)
            ids = torch.tensor([toks], dtype=torch.int32)
            pos = torch.arange(batch_start, batch_start + q).to(torch.uint16).unsqueeze(0)
            in_step = torch.tensor([batch_start], dtype=torch.int32)
            mask = torch.full((1, ctx, 1, q), SENTINEL, dtype=dtype)
            for t in range(n_tok):
                mask[0, : min(batch_start + t, ctx - 1) + 1, 0, t] = 0.0
            kc = k_cache.narrow(-1, 0, ctx)
            vc = v_cache.narrow(-1, 0, ctx)
            out = model(ids, pos, in_step, mask, kc, vc)  # [1, 1, q, vocab]
            out = out.reshape(q, -1).float()
            for t in range(n_tok):
                logits[batch_start + t] = out[t].clone()
            walk.append(("prompt_opt" if use_prefill else "extend", ctx, q, batch_start, n_tok))
            current = batch_end + 1
        return current

    return logits, walk, run_until


WEIGHT_KINDS: dict = {}


def tag_weight_kinds(model):
    """Map every Conv2d weight id -> 'attn' | 'conv' | 'mlp' for ANE_EMU_SUBSET."""
    layers = {int(x) for x in os.environ.get("ANE_EMU_LAYERS", "").split(",") if x.strip().isdigit()}
    # ANE_EMU_FP16 = "mlp:1,7;attn:*;conv:*" -> those (kind, layer) matmuls stay exact (a candidate mixed recipe's fp16 parts)
    fp16 = {}
    for part in os.environ.get("ANE_EMU_FP16", "").split(";"):
        if ":" in part:
            k, ls = part.split(":", 1); fp16[k.strip()] = None if ls.strip() == "*" else {int(x) for x in ls.split(",") if x.strip().isdigit()}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            kind = "attn" if ".self_attn." in name else ("conv" if ".conv." in name else "mlp")
            li = int(name.split(".layers.")[1].split(".")[0]) if ".layers." in name else -1
            keep = kind in fp16 and (fp16[kind] is None or li in fp16[kind])
            WEIGHT_KINDS[id(mod.weight)] = "skip" if keep else (kind if (not layers or li in layers) else "skip")


class _FTZMode(torch.overrides.TorchFunctionMode):
    """Flush fp16 subnormals to zero after every torch function (what a flush-to-zero accelerator does)."""
    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        with torch._C.DisableTorchFunction():
            return _ftz(out)


def _ftz(t):
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.float16 and t.numel() > 0:
            small = t.abs() < 2.0 ** -14
            if bool(small.any()):
                t = t.clone(); t[small] = 0
        return t
    if isinstance(t, (tuple, list)):
        return type(t)(_ftz(x) for x in t)
    return t


FTZ_MODE = _FTZMode()


def install_ane_emulation(mode: str):
    """Monkeypatch F.conv2d (1x1) and Tensor.__matmul__ with a reduced-precision accumulator model."""
    import torch.nn.functional as F
    kind = mode.split(":")[0]
    tile = int(mode.split(":")[1]) if ":" in mode and mode.split(":")[1].isdigit() else 32

    qmax = (2 ** (int(kind[1:]) - 1) - 1) if kind[0] == "a" and kind[1:].isdigit() else 127   # aN = N-bit symmetric

    def q8(x, per_token):  # symmetric N-bit fake-quant of an fp16 activation matrix [..., M, K]
        xf = x.float()
        amax = xf.abs().amax(dim=-1, keepdim=True) if per_token else xf.abs().amax()
        sc = torch.clamp(amax, min=1e-8) / qmax
        return (torch.round(xf / sc).clamp(-qmax, qmax) * sc).to(x.dtype)

    def mm_emul(A, B):  # A [..., M, K] @ B [..., K, N], both fp16
        if kind[0] == "a" and kind[1:].isdigit():   # aN-bit activations (A side), exact accumulate: 'a8:tensor' or 'a8:token'
            per_token = mode.endswith("token")
            return (q8(A, per_token).float() @ B.float()).to(A.dtype)
        if kind == "bf16":
            return (A.to(torch.bfloat16) @ B.to(torch.bfloat16)).to(A.dtype)
        if kind == "fp16out":
            return (A.float() @ B.float()).to(A.dtype)
        K = A.shape[-1]
        acc = None
        for k0 in range(0, K, tile):  # fp32 inside a tile, fp16 running sum across tiles
            part = (A[..., k0:k0 + tile].float() @ B[..., k0:k0 + tile, :].float()).to(torch.float16)
            acc = part if acc is None else (acc + part)  # fp16 add
        return acc.to(A.dtype)

    orig_conv2d = F.conv2d
    orig_matmul = torch.Tensor.__matmul__

    subset = os.environ.get("ANE_EMU_SUBSET", "")   # a8 only where the weight matches: attn (dim 2048/512 q,k,v,out), conv (in/out_proj), mlp

    def conv2d_emul(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if (subset or WEIGHT_KINDS) and weight.dim() == 4:
            o, i = weight.shape[0], weight.shape[1]
            is_mlp = (o == 8192 and i == 2048) or (o == 2048 and i == 8192)
            is_conv = (o == 6144 and i == 2048)
            is_attn = (o in (2048, 512) and i == 2048 and not is_conv) or (o == 2048 and i == 2048)
            # conv.out_proj and attn.out_proj are both 2048x2048: tell them apart by the caller's module (weight identity)
            wid = id(weight)
            kind_w = WEIGHT_KINDS.get(wid, "mlp" if is_mlp else ("conv" if is_conv else "attn"))
            if kind_w == "skip" or (subset and kind_w not in subset.split(",")):
                return orig_conv2d(x, weight, bias, stride, padding, dilation, groups)
        if weight.dim() == 4 and weight.shape[2] == 1 and weight.shape[3] == 1 and groups == 1 and x.dtype == torch.float16:
            b, c, h, w = x.shape
            xm = x.permute(0, 2, 3, 1).reshape(b * h * w, c)               # [N, K]
            y = mm_emul(xm, weight.reshape(weight.shape[0], c).t())         # [N, O]
            y = y.reshape(b, h, w, weight.shape[0]).permute(0, 3, 1, 2)
            if bias is not None:
                y = y + bias.reshape(1, -1, 1, 1)
            return y
        return orig_conv2d(x, weight, bias, stride, padding, dilation, groups)

    def matmul_emul(self, other):
        if self.dtype == torch.float16 and other.dtype == torch.float16 and self.dim() >= 2 and other.dim() >= 2:
            if kind[0] == "a" and kind[1:].isdigit() and self.shape[-2] > 4096:   # the tied head: table [vocab, hidden] @ out [hidden, q] -> quantize `out`
                per_token = mode.endswith("token")
                o = q8(other.transpose(-2, -1), per_token).transpose(-2, -1)
                return (self.float() @ o.float()).to(self.dtype)
            return mm_emul(self, other)
        return orig_matmul(self, other)

    F.conv2d = conv2d_emul
    torch.Tensor.__matmul__ = matmul_emul
    print(f"ANE emulation installed: {mode} (tile {tile})", flush=True)


def install_stats(model):
    """Hooks: per attention layer max |score| before the mask, per layer residual RMS, conv Bx max."""
    from coreai_models.models.ios import lfm2 as builder
    stats = {}

    def hook_layer(idx):
        def f(mod, inp, out):
            x = out.float()
            stats.setdefault(idx, {})["resid_rms"] = round(float(x.pow(2).mean().sqrt()), 2)
            stats[idx]["resid_max"] = round(float(x.abs().max()), 1)
        return f

    for i, layer in enumerate(model.extend.model.layers):
        layer.register_forward_hook(hook_layer(i))
        if layer.is_attention:
            def sdpa_pre(mod, args, i=i):
                q, k, v, mask = args
                sc = (q.float().transpose(-3, -1) @ (k.float().transpose(-3, -1) * (mod._scale_factor)).transpose(-2, -1)) if False else None
                # per head: q [1, H*D, 1, q], k [1, KV*D, 1, L] -> heads
                H = q.shape[1] // mod.head_dim; KV = k.shape[1] // mod.head_dim
                qh = q.float().reshape(H, mod.head_dim, -1); kh = k.float().reshape(KV, mod.head_dim, -1)
                g = H // KV
                smax = 0.0
                for h in range(H):
                    s = (qh[h].t() @ kh[h // g]) * float(mod._scale_factor)   # [q, L]
                    valid = s[:, : (mask[0, :, 0, 0] == 0).sum()]
                    smax = max(smax, float(valid.abs().max()))
                stats.setdefault(i, {})["score_max"] = round(smax, 1)
                stats[i]["q_max"] = round(float(q.float().abs().max()), 1); stats[i]["k_max"] = round(float(k.float().abs().max()), 1)
            layer.self_attn.sdpa.register_forward_pre_hook(sdpa_pre)
        else:
            def conv_hook(mod, inp, out, i=i):
                stats.setdefault(i, {})["conv_out_max"] = round(float(out.float().abs().max()), 1)
            layer.conv.register_forward_hook(conv_hook)
    model._s6_stats = stats
    print("stats hooks installed", flush=True)


@torch.no_grad()
def run_fixture(a):
    """Teacher-forced sweep of every fixture prompt through the engine walk; a mismatch on a margin-clear step
    (oracle top-2 softmax gap >= floor) = FAIL, exactly AneGateRunner's TF rule."""
    from coreai_models.models.ios import lfm2 as builder
    from coreai_models.models.ios.lfm2 import Lfm2ForCausalLMForiOS
    if a.fp32_parts:
        builder.FP32_PARTS.update(p for p in a.fp32_parts.split(",") if p)
    dtype = getattr(torch, a.dtype)
    fx = json.load(open(a.fixture)); floor = fx["floor"]
    t0 = time.time()
    model = Lfm2ForCausalLMForiOS.from_hf(a.hf_id, max_context_length=a.max_ctx, target_dtype=dtype, num_layers=a.num_layers,
                                          disable_embedding_quantization=a.no_emb_quant).eval()
    print(f"ios module {a.dtype} emb_quant={not a.no_emb_quant} fp32_parts={sorted(builder.FP32_PARTS)} loaded in {time.time() - t0:.1f} s", flush=True)
    if a.load_palettized:
        model.load_state_dict(torch.load(a.load_palettized), strict=True)
        print(f"loaded palettized weights from {a.load_palettized}", flush=True)
    elif a.palettize:
        import importlib.util, pathlib
        spec_ = importlib.util.spec_from_file_location("sim", pathlib.Path(__file__).resolve().parent / "simulate_recipe.py")
        sim = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(sim)
        recipe = sim.load_recipe(a.palettize, None)
        t0 = time.time(); n = 0; kept = 0
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                w = m.weight.data
                spec = sim.spec_for(recipe, name)   # the iOS module already names its matmuls "extend.model.layers.N..." = the recipes' regex targets
                if spec is None:   # a `null` module_name_config: this matmul stays fp16 on the device
                    kept += 1; continue
                m.weight.data = sim.palettize(w.float().reshape(w.shape[0], -1), spec).reshape(w.shape).to(w.dtype)
                n += 1
        print(f"palettized {n} Conv2d weights ({kept} kept fp16) with {recipe['name']} in {time.time() - t0:.0f} s", flush=True)
        if a.save_palettized:
            torch.save(model.state_dict(), a.save_palettized); print(f"saved palettized weights to {a.save_palettized}", flush=True)
    if a.ane_emu:
        tag_weight_kinds(model)
        install_ane_emulation(a.ane_emu)
    if a.ftz_squares:
        from coreai_models.primitives.ios.rms_norm import RMSNorm
        from coreai_models.models.ios.lfm2 import RMSNormK

        def ftz_sq_forward(self, input):
            y = input * self._k if hasattr(self, "_k") else input
            square = y * y
            square = torch.where(square.abs() < 2.0 ** -14, torch.zeros_like(square), square)
            inv_rms = torch.rsqrt(square.mean(-1, keepdim=True) + self._eps)
            return y * inv_rms * self.weight
        RMSNorm.forward = ftz_sq_forward; RMSNormK.forward = ftz_sq_forward
        print("FTZ of the squares only (eps kept) installed", flush=True)
    if a.ftz == "norms":
        from coreai_models.primitives.ios.rms_norm import RMSNorm
        from coreai_models.models.ios.lfm2 import RMSNormK

        def ftz_norm_forward(self, input):
            y = input * self._k if hasattr(self, "_k") else input
            square = y * y
            square = torch.where(square.abs() < 2.0 ** -14, torch.zeros_like(square), square)
            mean_square = square.mean(-1, keepdim=True)
            eps = self._eps if float(self._eps) >= 2.0 ** -14 else torch.zeros_like(self._eps)
            inv_rms = torch.rsqrt(mean_square + eps)
            return y * inv_rms * self.weight
        RMSNorm.forward = ftz_norm_forward; RMSNormK.forward = ftz_norm_forward
        print("FTZ inside RMSNorm installed", flush=True)
    elif a.ftz == "all":
        FTZ_MODE.__enter__()
        print("FTZ TorchFunctionMode installed (every fp16 op output)", flush=True)
    if a.eps_zero or a.norm_noise:
        from coreai_models.primitives.ios.rms_norm import RMSNorm as _AppleRMSNorm
        from coreai_models.models.ios.lfm2 import RMSNormK
        RMSNorm = (_AppleRMSNorm, RMSNormK)
        n = 0
        kinds = {k for k in a.eps_zero_kinds.split(",") if k}
        def kind_of(name):
            if name.endswith("q_layernorm") or name.endswith("k_layernorm"): return "qk"
            if name.endswith("embedding_norm"): return "final"
            return "stream"
        for name, m in model.named_modules():
            if isinstance(m, RMSNorm):
                if a.eps_zero and (not kinds or kind_of(name) in kinds):
                    m._eps.zero_(); n += 1
        if a.norm_noise:
            sigma = a.norm_noise
            g = torch.Generator().manual_seed(0)

            def noisy_forward(self, input):
                square = input * input
                mean_square = square.mean(-1, keepdim=True)
                inv_rms = torch.rsqrt(mean_square + self._eps)
                inv_rms = inv_rms * (1 + sigma * torch.randn(inv_rms.shape, generator=g)).to(inv_rms.dtype)
                return input * inv_rms * self.weight
            for cls in RMSNorm: cls.forward = noisy_forward
        print(f"RMSNorm knobs: eps_zero={a.eps_zero} norm_noise={a.norm_noise} on {n} norms", flush=True)
    if a.stats:
        install_stats(model)
    dev = {}
    if a.device_log:
        import re
        for line in open(a.device_log):
            m = re.match(r"TF (\w+) k=(\d+) exp=(\d+) got=(\d+) second=(\d+) oracle_margin=([\d.]+) dev_gap=([\d.-]+) (ok|KNIFE|FAIL)", line)
            if m:
                dev.setdefault(m.group(1), {})[int(m.group(2))] = {"exp": int(m.group(3)), "got": int(m.group(4)), "second": int(m.group(5)),
                                                                   "margin": float(m.group(6)), "dev_gap": float(m.group(7)), "status": m.group(8)}
        print(f"device log: {sum(len(v) for v in dev.values())} TF steps over {list(dev)}", flush=True)
    if a.rollout_fixture:
        out = dict(fx); out["oracle"] = {"kind": "mac-proxy-rollout", "dtype": a.dtype, "emb_quant": not a.no_emb_quant,
                                          "palettize": a.palettize, "fp32_parts": sorted(builder.FP32_PARTS), "engine_walk": True}
        out["prompts"] = []
        for p in fx["prompts"]:
            P = len(p["prompt_ids"]); n_max = len(p["expected_ids"]); eos = set(p.get("eos_ids") or [])
            seq = list(p["prompt_ids"])
            logits, walk, run_until = emulate(model, seq, a.max_ctx, dtype, print)
            t0 = time.time(); cur = run_until(P, 0)
            gen, margins, margins_logit, second = [], [], [], []
            for k in range(n_max):
                row = logits[len(seq) - 1].float()
                pr = torch.softmax(row, -1); t2p = torch.topk(pr, 2); t2l = torch.topk(row, 2)
                nxt = int(row.argmax()); gen.append(nxt)
                margins.append(round(float(t2p.values[0] - t2p.values[1]), 4)); margins_logit.append(round(float(t2l.values[0] - t2l.values[1]), 4))
                second.append(int(t2l.indices[1]))
                if nxt in eos: break
                seq.append(nxt); cur = run_until(len(seq), cur)
            q = dict(p); q.update({"expected_ids": gen, "margins": margins, "margins_logit": margins_logit, "second_best_ids": second,
                                   "stopped_on_eos": bool(gen and gen[-1] in eos), "expected_text": None,
                                   "knife_ok": True if p["name"] == "sky" else p.get("knife_ok", False)})
            out["prompts"].append(q)
            same = sum(int(x == y) for x, y in zip(gen, p["expected_ids"]))
            first = next((i for i, (x, y) in enumerate(zip(gen, p["expected_ids"])) if x != y), None)
            print(f"[{p['name']}] proxy rollout {len(gen)} steps (eos={q['stopped_on_eos']}) vs fp32 oracle: same prefix until {first} "
                  f"({same} equal of {min(len(gen), n_max)}), min margin {min(margins):.3f} ({time.time() - t0:.0f} s)", flush=True)
        json.dump(out, open(a.rollout_fixture, "w"), indent=1)
        print(f"wrote {a.rollout_fixture}")
        return
    results = {}
    for p in fx["prompts"]:
        seq = list(p["prompt_ids"]) + list(p["expected_ids"]); P = len(p["prompt_ids"]); n = len(p["expected_ids"])
        logits, walk, run_until = emulate(model, seq, a.max_ctx, dtype, print)
        t0 = time.time()
        cur = run_until(P, 0)
        for i in range(P, len(seq)):
            cur = run_until(i + 1, cur)
        fails, knife, ok, steps, cmp = [], [], 0, [], []
        for k in range(n):
            row = logits[P - 1 + k]; got = int(row.argmax()); exp = int(p["expected_ids"][k]); m = float(p["margins"][k])
            t2 = torch.topk(row, 2); gap = float(t2.values[0] - t2.values[1])
            steps.append({"k": k, "exp": exp, "got": got, "gap": round(gap, 3)})
            if got == exp:
                ok += 1
            elif m >= floor:
                fails.append((k, exp, got, round(m, 3)))
            else:
                knife.append(k)
            d = dev.get(p["name"], {}).get(k)
            if d is not None:
                cmp.append({"k": k, "exp": exp, "mac_got": got, "dev_got": d["got"], "mac_gap": round(gap, 3), "dev_gap": d["dev_gap"],
                            "mac_logit_exp_minus_devgot": round(float(row[exp] - row[d["got"]]), 3), "dev_status": d["status"], "margin": d["margin"]})
        verdict = "PASS" if not fails else "FAIL"
        results[p["name"]] = {"ok": ok, "n": n, "fails": fails, "knife": knife, "verdict": verdict, "steps": steps, "device_compare": cmp}
        if a.stats:
            st = getattr(model, "_s6_stats", {})
            print("   layer stats (last run): " + "; ".join(f"L{i}:" + ",".join(f"{k}={v}" for k, v in st[i].items()) for i in sorted(st)), flush=True)
        if cmp:
            agree = sum(int(c["mac_got"] == c["dev_got"]) for c in cmp)
            gap_diff = [abs(c["mac_gap"] - c["dev_gap"]) for c in cmp if c["mac_got"] == c["dev_got"] == c["exp"]]
            print(f"   device vs this run: same top-1 on {agree}/{len(cmp)} steps; |gap diff| on jointly-correct steps: "
                  f"mean {sum(gap_diff) / max(1, len(gap_diff)):.3f} max {max(gap_diff) if gap_diff else 0:.3f}", flush=True)
            for c in cmp:
                if c["dev_status"] == "FAIL" or c["mac_got"] != c["dev_got"]:
                    print(f"   k={c['k']:3d} exp={c['exp']} dev_got={c['dev_got']} ({c['dev_status']}, dev_gap {c['dev_gap']}) mac_got={c['mac_got']} "
                          f"mac_gap {c['mac_gap']} mac: logit[exp]-logit[dev_got]={c['mac_logit_exp_minus_devgot']}", flush=True)
        print(f"[{p['name']}] tf {ok}/{n} knife={len(knife)} fail={len(fails)} {verdict} ({time.time() - t0:.0f} s) fails={fails[:12]}", flush=True)
    summary = {"dtype": a.dtype, "emb_quant": not a.no_emb_quant, "fp32_parts": sorted(builder.FP32_PARTS), "num_layers": a.num_layers,
               "results": results, "verdict": "PASS" if all(r["verdict"] == "PASS" for r in results.values()) else "FAIL"}
    print("VERDICT=" + summary["verdict"])
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--max-ctx", type=int, default=512)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", default="float32", help="float32 | float16 (the device runs fp16 activations everywhere)")
    ap.add_argument("--no-emb-quant", action="store_true", help="keep the embedding table in the model dtype (device: int8 per-tensor)")
    ap.add_argument("--fixture", default=None, help="AneGate fixture json: teacher-force its prompts (the device gate's TF rule) instead of a free prompt")
    ap.add_argument("--fp32-parts", default="", help="comma list of parts computed in fp32 inside an fp16 run (bisect): attn_proj, scores, norms, rope, conv, mlp")
    ap.add_argument("--rollout-fixture", default=None, metavar="OUT.json",
                    help="with --fixture: instead of judging, FREE-RUN the (proxy) module on each prompt through the engine walk and "
                         "write a fixture whose oracle is this run (expected ids + margins) — the device is then gated against its Mac twin")
    ap.add_argument("--device-log", default=None, help="AneGateRunner device log: per TF step, compare the device's choice/gap with this run's")
    ap.add_argument("--save-palettized", default=None, help="torch.save the palettized state_dict here (reuse with --load-palettized)")
    ap.add_argument("--load-palettized", default=None, help="load a palettized state_dict saved by --save-palettized (skips the k-means)")
    ap.add_argument("--ane-emu", default=None, metavar="MODE",
                    help="emulate a less precise accumulator in EVERY matmul (Conv2d 1x1, q@k, p@v, head): 'fp16acc:<tile>' = fp32 within "
                         "K-tiles of <tile>, fp16 running sum across tiles; 'fp16out' = exact accumulate, fp16 output (= plain fp16); "
                         "'bf16' = matmuls in bfloat16")
    ap.add_argument("--eps-zero", action="store_true", help="RMSNorm eps -> 0 (a device that flushes the fp16-subnormal eps=1e-5 to zero)")
    ap.add_argument("--eps-zero-kinds", default="", help="with --eps-zero: comma list of norm kinds to zero (stream = operator/ffn, qk, final); default all")
    ap.add_argument("--ftz-squares", action="store_true", help="flush only the subnormal squares inside RMSNorm (eps kept)")
    ap.add_argument("--norm-noise", type=float, default=0.0, help="multiply every RMSNorm inv_rms by (1 + N(0, sigma)): sensitivity to an approximate rsqrt")
    ap.add_argument("--ftz", default=None, choices=["all", "norms"],
                    help="flush fp16 subnormals (|x| < 2^-14) to zero: 'all' = after every torch op (TorchFunctionMode), "
                         "'norms' = only inside RMSNorm (the squares and eps)")
    ap.add_argument("--stats", action="store_true", help="print per-layer activation statistics (attention score max, residual RMS, Bx max)")
    ap.add_argument("--palettize", default=None, help="kmeans yaml: fake-palettize every Conv2d weight of the iOS module with coreai-opt's k-means "
                    "(simulate_recipe.palettize) = the full Mac proxy of the device bundle together with --dtype float16")
    a = ap.parse_args()
    if a.fixture:
        return run_fixture(a)

    from coreai_models.models.ios.lfm2 import Lfm2ForCausalLMForiOS

    tok = AutoTokenizer.from_pretrained(a.hf_id)
    kw = {"num_hidden_layers": a.num_layers} if a.num_layers else {}
    hf = AutoModelForCausalLM.from_pretrained(a.hf_id, dtype=torch.float32, **kw).eval()
    prompt = a.prompt or PROMPT
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
    if not hasattr(ids, "shape"):
        ids = ids["input_ids"]
    prompt_ids = ids[0].tolist()
    print(f"prompt {len(prompt_ids)} ids; hf layers {hf.config.num_hidden_layers}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        gen = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.steps, do_sample=False)
    seq = gen[0].tolist()
    with torch.no_grad():
        ref = hf(torch.tensor([seq])).logits[0].float()  # [T, vocab]
    print(f"hf greedy {len(seq) - len(prompt_ids)} steps in {time.time() - t0:.1f} s: {tok.decode(seq[len(prompt_ids):])!r}", flush=True)
    del hf

    t0 = time.time()
    model = Lfm2ForCausalLMForiOS.from_hf(a.hf_id, max_context_length=a.max_ctx, target_dtype=torch.float32,
                                          num_layers=a.num_layers, disable_embedding_quantization=True).eval()
    print(f"ios module loaded in {time.time() - t0:.1f} s (cache_width {model.cache_width})", flush=True)

    logits, walk, run_until = emulate(model, seq, a.max_ctx, torch.float32, print)
    t0 = time.time()
    current = run_until(len(prompt_ids), 0)
    for i in range(len(prompt_ids), len(seq)):
        current = run_until(i + 1, current)  # the oracle token i appended -> one more decode run
    print(f"engine walk: {len(walk)} runs in {time.time() - t0:.1f} s", flush=True)
    kinds = {}
    for g, ctx, q, start, n in walk:
        key = f"{g}_{ctx}_{q}"
        kinds[key] = kinds.get(key, 0) + 1
    print("graphs used:", kinds)

    T = len(seq)
    ok = 0; worst = 0.0; worst_pos = -1; mism = []
    for p in range(T):
        d = (logits[p] - ref[p]).abs().max().item()
        if d > worst: worst, worst_pos = d, p
        eq = int(logits[p].argmax()) == int(ref[p].argmax())
        ok += eq
        if not eq:
            mism.append((p, int(ref[p].argmax()), int(logits[p].argmax()), round(float(ref[p].softmax(-1).max()), 3)))
    gen_ok = sum(int(logits[p].argmax()) == seq[p + 1] for p in range(len(prompt_ids) - 1, T - 1))
    n_gen = T - len(prompt_ids)
    print(f"TF argmax match: {ok}/{T} positions (prompt+continuation); continuation next-token match {gen_ok}/{n_gen}")
    print(f"max |dlogit| {worst:.4f} at position {worst_pos}; mismatches {mism[:10]}")
    verdict = "PASS" if gen_ok == n_gen and ok >= T - 2 else "FAIL"
    print(f"VERDICT={verdict}")
    if a.out:
        json.dump({"hf_id": a.hf_id, "num_layers": a.num_layers, "prompt_ids": prompt_ids, "seq": seq, "tf_match": ok,
                   "positions": T, "continuation_match": gen_ok, "continuation": n_gen, "max_abs_dlogit": worst,
                   "worst_pos": worst_pos, "mismatches": mism, "graphs": kinds, "verdict": verdict},
                  open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
