# granite_embedding/ — Granite-Embedding-97M-Multilingual-R2 → Core AI

The recipe behind [`models/granite-embedding-97m/`](../../models/granite-embedding-97m/README.md)
and [mlboydaisuke/Granite-Embedding-97M-Multilingual-R2-CoreAI](https://huggingface.co/mlboydaisuke/Granite-Embedding-97M-Multilingual-R2-CoreAI).
Five stages, each a script, each gated against the one before; run them in this order from the
repo root (paths resolve through [`../_paths.py`](../_paths.py)):

| Stage | Script | Gate |
|---|---|---|
| 0 oracle | `oracle_granite_embedding.py` | official HF eager CPU fp32, 35 texts × S=128/512, every hidden state saved |
| 1 authoring | `gate_granite_authoring.py --seq-len S --negative-controls` | re-authored graph vs oracle: embedding cos ≥ 0.999 **and** every residual layer ≤ 1e-4; 5 mutations must fail |
| 2 host | `gate_granite_tokenizer.py [--swift-bin]` | independent tokenizer vs `AutoTokenizer`, 681 texts × 2 grids, ids + masks exact; 4 mutations must be caught |
| 3 export | `export_granite_embedding.py --seq-len S --target macos\|ios [--aot h18p]` | torch-export + decomposition vs oracle before conversion; `coreai-build` for the phone |
| 3b W8 | `export_granite_embedding_w8.py …` | the same gate at prepared / finalized / decomposed; 48 LUT ops; palettes hashed and reusable |
| 4 runtime | `gate_granite_embedding.py <variant dir> --compute gpu\|cpu_only` | the `.aimodel` on the Core AI runtime: gate + determinism + wrong-pairing control + warm timings |

Stage 4 for the iPhone is the device gate: the same `reference.json`, the compiled `.aimodelc`,
a headless runner that hashes every transferred file — its results ship as
`ios/<variant>/provenance/runtime-gate.json` in the HF repo.

Environments: stages 0–2 need transformers (`oracle`/`tokenizer` pins in the script headers);
stages 3–4 need `coreai-torch 0.4.1` + `coreai-core 1.0.0b2` + torch 2.9.0; stage 3b needs
`coreai-opt 0.2.1`, whose `safetensors <= 0.7.0` pin conflicts with stage 3 — a separate env
(`uv run` on the script resolves each header on its own).

`_granite_model.py` is the graph (plain PyTorch from the raw safetensors, strict load, five
negative-control mutations), `_granite_tokenizer.py` the host recipe (no HF import),
`_gate_metrics.py` the one gate every stage shares. What the port taught:
[`knowledge/granite-embedding-97m-port.md`](../../knowledge/granite-embedding-97m-port.md).
