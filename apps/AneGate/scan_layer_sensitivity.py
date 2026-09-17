"""Which MiniCPM5-2B layers are 4-bit-palettization-sensitive? (S5, 2026-09-16)

Picks the 8-bit layers for Apple's mixed 4/8 iOS recipe shape (models/qwen3/qwen3_0_6b_mixed_4bit_8bit.yaml:
4-bit k-means g8 everywhere, a few layers 8-bit per-tensor). Apple's per-model layer lists (0.6B: 0,2,4,7,15,16;
4B: 6,8,11,33,34) are evidently data-driven, so this measures rather than scales them.

Per layer (and the untied lm_head): palettize ONLY that layer's 7 projections with a 16-centroid Lloyd k-means
per group of 8 output channels (the preset's shape; a proxy for coreai-opt's k-means, same n_bits/granularity),
run the fp32 model teacher-forced on the fixture's `sky` prompt + oracle answer, and record against the fp32
baseline: mean KL over the answer steps, the number of argmax flips on margin-clear steps, and the step-0
softmax margin. Higher = more sensitive. Cost: ~1-2 s k-means + one 134-token forward per layer on CPU.

    ~/.venv-coreml-llm-py312/bin/python scan_layer_sensitivity.py --fixture fixtures/minicpm5_2b/fixture.json \
        --out fixtures/minicpm5_2b/layer_sensitivity_4bit_g8.json
"""
import argparse, json, os, time, warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1"); warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def kmeans_palettize(w: torch.Tensor, n_bits: int = 4, group: int = 8, iters: int = 12) -> torch.Tensor:
    """Per-grouped-channel (axis 0, group rows) k-means with 2**n_bits centroids, Lloyd iterations from a
    uniform-quantile init. Returns the reconstructed weight (same shape/dtype)."""
    out_ch, in_ch = w.shape
    k = 2 ** n_bits
    x = w.reshape(out_ch // group, group * in_ch).float()          # [G, N]
    G = x.shape[0]
    # init: quantiles of each group
    q = torch.linspace(0, 1, k + 2)[1:-1]
    c = torch.quantile(x, q, dim=1).T.contiguous()                    # [G, k]
    chunk = max(1, int(2e8 // (x.shape[1] * k)))                      # keep the distance tensor ~< 1 GB
    for _ in range(iters):
        new_c = torch.zeros_like(c); cnt = torch.zeros(G, k)
        for g0 in range(0, G, chunk):
            xs = x[g0:g0 + chunk]; cs = c[g0:g0 + chunk]
            a = (xs.unsqueeze(2) - cs.unsqueeze(1)).abs().argmin(dim=2)   # [g, N]
            new_c[g0:g0 + chunk].scatter_add_(1, a, xs)
            cnt[g0:g0 + chunk].scatter_add_(1, a, torch.ones_like(xs))
        moved = cnt > 0
        c = torch.where(moved, new_c / cnt.clamp(min=1), c)
    rec = torch.empty_like(x)
    for g0 in range(0, G, chunk):
        xs = x[g0:g0 + chunk]; cs = c[g0:g0 + chunk]
        a = (xs.unsqueeze(2) - cs.unsqueeze(1)).abs().argmin(dim=2)
        rec[g0:g0 + chunk] = torch.gather(cs, 1, a)
    return rec.reshape(out_ch, in_ch).to(w.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--fixture", required=True); ap.add_argument("--prompt", default="sky")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bits", type=int, default=4); ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--layers", default="", help="comma list to restrict (default all + lm_head)")
    ap.add_argument("--engine", choices=["lloyd", "coreai-opt"], default="lloyd",
                    help="coreai-opt = the exporter's own optimal 1-D k-means (run in the coreai-models-rebase venv; "
                         "the Lloyd proxy is ~1.5x worse on these heavy-tailed tensors)")
    a = ap.parse_args()
    if a.engine == "coreai-opt":
        import importlib.util, pathlib
        spec_ = importlib.util.spec_from_file_location("sim", pathlib.Path(__file__).with_name("simulate_recipe.py"))
        sim = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(sim)
        def pal(w, n_bits, group):
            return sim.palettize(w, {"n_bits": n_bits, "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": group}})
    else:
        pal = kmeans_palettize
    fx = json.load(open(a.fixture)); p = next(p for p in fx["prompts"] if p["name"] == a.prompt)
    ids = torch.tensor([p["prompt_ids"] + p["expected_ids"]])
    P = len(p["prompt_ids"]); n = len(p["expected_ids"])
    exp = torch.tensor(p["expected_ids"]); margins = torch.tensor(p["margins"]); clear = margins >= fx["floor"]

    model = AutoModelForCausalLM.from_pretrained(a.hf_id, dtype=torch.float32).eval()
    layers = model.model.layers; L = len(layers)

    @torch.no_grad()
    def answer_logits():
        return model(ids).logits[0, P - 1:P - 1 + n].float()          # logits predicting each answer token
    base = answer_logits(); base_lp = torch.log_softmax(base, -1)
    base_top = base.argmax(-1)
    assert (base_top == exp).all(), f"fp32 baseline does not reproduce the oracle: {(base_top != exp).sum()} steps differ"
    print(f"baseline ok: {n} answer steps, {int(clear.sum())} margin-clear; layers={L}")

    targets = [int(x) for x in a.layers.split(",") if x] if a.layers else list(range(L)) + ["lm_head"]
    results = []
    for t in targets:
        t0 = time.time()
        if t == "lm_head":
            mods = {"lm_head": model.lm_head}
        else:
            mods = {f"layers.{t}.{name}": layers[t].get_submodule(name) for name in PROJ}
        saved = {k: m.weight.data.clone() for k, m in mods.items()}
        for k, m in mods.items():
            m.weight.data = pal(saved[k], a.n_bits, a.group)
        lg = answer_logits()
        for k, m in mods.items():
            m.weight.data = saved[k]
        lp = torch.log_softmax(lg, -1)
        kl = (base_lp.exp() * (base_lp - lp)).sum(-1)                  # KL(fp32 || palettized) per step
        top = lg.argmax(-1)
        flips_clear = int(((top != exp) & clear).sum()); flips_all = int((top != exp).sum())
        sm = torch.softmax(lg[0], -1).topk(2).values
        first_flip = next((i for i in range(n) if top[i] != exp[i] and clear[i]), None)
        r = {"layer": t, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "flips_margin_clear": flips_clear,
             "flips_all": flips_all, "first_clear_flip_step": first_flip, "step0_margin": float(sm[0] - sm[1]),
             "step0_top1_ok": bool(top[0] == exp[0]), "seconds": round(time.time() - t0, 1)}
        results.append(r)
        print(f"{str(t):>7}: kl_mean={r['kl_mean']:.4f} kl_max={r['kl_max']:.3f} flips_clear={flips_clear:2d} "
              f"flips_all={flips_all:2d} first_clear_flip={first_flip} step0_margin={r['step0_margin']:.3f} "
              f"step0_ok={r['step0_top1_ok']} ({r['seconds']} s)", flush=True)
    ranked = sorted(results, key=lambda r: (-r["flips_margin_clear"], -r["kl_mean"]))
    json.dump({"hf_id": a.hf_id, "prompt": a.prompt, "n_bits": a.n_bits, "group": a.group, "engine": a.engine,
               "method": f"per-layer {a.engine} k-means ({2 ** a.n_bits} centroids per {a.group} output channels) on the 7 "
                         "projections, teacher-forced on prompt+oracle answer, KL(fp32||palettized) per answer step",
               "results": results, "ranked": [r["layer"] for r in ranked]}, open(a.out, "w"), indent=1)
    print("ranked (most sensitive first):", [r["layer"] for r in ranked][:16])


if __name__ == "__main__":
    main()
