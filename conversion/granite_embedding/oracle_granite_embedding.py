#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==4.57.6",
#     "sentence-transformers==5.1.2",
#     "safetensors>=0.7.0",
#     "numpy==2.2.6",
#     "huggingface_hub",
# ]
# ///
"""Stage 0: the independent oracle. Official HF eager CPU fp32 fixtures; never imports the re-authored graph.

Writes, under the work dir (`python3 conversion/_paths.py` prints it):
  fixtures/texts.json                 the 35 texts (JA / EN / mixed / edge / boundary)
  fixtures/golden_s{128,512}.json     exact HF token ids + masks + the normalized CLS embedding per text
  fixtures/intermediates_s{S}.npz     every hidden state per text (the layer gate reads these)
  results/oracle.json                 what was checked, versions, output hashes

The golden path is the upstream README's own example: raw AutoTokenizer -> ModernBertModel ->
CLS -> L2 normalize. sentence-transformers is checked separately because it STRIPS the text
before tokenizing (documented in results/oracle.json); the golden fixtures keep raw whitespace.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_SHA, SEQ_LENS, download_source, fixtures_dir, hashes, results_dir, write_json  # noqa: E402

PAD, CLS, SEP = 179935, 179934, 179938


def cases():
    texts = [
        ("empty", ""), ("short", "猫"), ("whitespace", " \t\n "),
        ("whitespace_edges", "  東京駅から train を探す。\n"),
        ("unicode", "ＡＢＣ abc Café Café 👨‍👩‍👧‍👦 ﾄｳｷｮｳ 東京　大阪"),
        ("english", "A small offline application can search documents without sending their contents to a server."),
        ("japanese", "端末内の文書を日本語で検索します。入力した個人情報を外部のサーバーへ送信せずに処理できます。"),
        ("mixed", "Core AIでembeddingを計算し、Tokyoのメモと大阪のdocumentsを検索する。version 2.0 / 2026-09-16"),
        ("q_train", "東京から新大阪へ、のぞみで何時間かかりますか？"),
        ("q_refund", "How do I get a refund for a cancelled train ticket?"),
        ("q_backup", "ネット接続なしでメモのバックアップを保存する方法"),
        ("q_garden", "How should I water basil growing in a pot?"),
        ("d_train", "東海道新幹線のぞみは、東京駅から新大阪駅まで通常およそ2時間30分で結びます。"),
        ("d_train_negative1", "東京駅から名古屋駅まで、のぞみの所要時間はおよそ1時間40分です。"),
        ("d_train_negative2", "東京と大阪の間を夜行バスで移動すると、所要時間はおよそ8時間です。"),
        ("d_refund", "To refund an unused train ticket, submit a cancellation request before departure. The refund goes to the original payment method, minus any cancellation fee."),
        ("d_refund_negative1", "To change the date on a train ticket, select a new departure time in your booking. A price difference may apply, but this does not cancel the ticket."),
        ("d_refund_negative2", "Cancelled concert tickets can be refunded through the event organizer. Train tickets are handled separately by the railway operator."),
        ("d_backup", "インターネットに接続しないバックアップは、メモを端末内のファイルへ書き出し、USBストレージへコピーして保存します。"),
        ("d_backup_negative1", "クラウド同期を有効にするとメモをサーバーに保存できます。この操作にはインターネット接続が必要です。"),
        ("d_backup_negative2", "端末内のメモを削除するには一覧から選択して削除ボタンを押します。バックアップの作成とは異なる操作です。"),
        ("d_garden", "Water potted basil when the top layer of soil feels dry. Water the soil thoroughly and let excess water drain out of the pot."),
        ("d_garden_negative1", "Freshly cut basil stems keep well in a glass of water on a kitchen counter. This advice concerns harvested stems rather than basil growing in soil."),
        ("d_garden_negative2", "Rosemary prefers its soil to dry between waterings. Avoid keeping the roots constantly wet."),
    ]
    # Token counts that cross the local-attention radius (64) and the truncation boundaries.
    words = ["apple", "bridge", "river", "garden", "train", "paper", "music", "window"]
    for body_length in [62, 63, 64, 126, 127, 128, 510, 511, 512, 700]:
        texts.append((f"boundary_{body_length}", " ".join(words[i % len(words)] for i in range(body_length))))
    texts.append(("long_japanese", "会議の記録には日時、場所、参加者、決定事項を残します。次の会議では前回の進捗を確認します。" * 45))
    return [{"id": key, "text": text,
             "kind": "query" if key.startswith("q_") else "document" if key.startswith("d_") else "edge"}
            for key, text in texts]


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    source = download_source()
    fixtures_dir().mkdir(parents=True, exist_ok=True)
    results_dir().mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    assert (tok.pad_token_id, tok.cls_token_id, tok.sep_token_id) == (PAD, CLS, SEP)
    assert tok.padding_side == tok.truncation_side == "right"
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config.reference_compile = False
    model, loading = AutoModel.from_pretrained(source, config=config, local_files_only=True, dtype=torch.float32,
                                               attn_implementation="eager", output_loading_info=True)
    model.eval()
    assert not any(loading[k] for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]), loading
    weights = load_file(source / "model.safetensors")
    assert set(weights) == set(model.state_dict())
    assert all(torch.equal(model.state_dict()[k], v.float()) for k, v in weights.items())
    parameter_count = sum(v.numel() for v in weights.values())
    assert len(weights) == 74 and parameter_count == 97_441_152
    del weights
    specimens = cases()
    write_json(fixtures_dir() / "texts.json", specimens)
    report = {"status": "PASS", "model_sha": MODEL_SHA, "backend": "CPU", "precision": "fp32",
              "attention_implementation": "eager", "parameter_count": parameter_count, "tensor_count": 74,
              "preprocessing": "official raw AutoTokenizer path; whitespace preserved; no task prefix",
              "grids": {}, "sentence_transformer_checks": []}
    for length in SEQ_LENS:
        rows, arrays = [], {}
        start = time.perf_counter()
        for index, spec in enumerate(specimens):
            tokens = tok(spec["text"], padding="max_length", truncation=True, max_length=length, return_tensors="pt")
            active = int(tokens["attention_mask"].sum())
            original_length = len(tok(spec["text"])["input_ids"])
            assert tokens["input_ids"][0, 0] == CLS and tokens["input_ids"][0, active - 1] == SEP
            assert torch.all(tokens["input_ids"][0, active:] == PAD)
            with torch.inference_mode():
                out = model(**tokens, output_hidden_states=True)
                pooled = out.last_hidden_state[:, 0]
                embedding = F.normalize(pooled, p=2, dim=1)
                short = {k: v[:, :active] for k, v in tokens.items()}
                unpadded = F.normalize(model(**short).last_hidden_state[:, 0], dim=1)
            e = embedding[0].numpy().copy()
            assert np.isfinite(e).all() and abs(np.linalg.norm(e) - 1) < 1e-5
            pad_cos = cosine(e, unpadded.numpy())
            assert pad_cos >= 0.999999, (spec["id"], pad_cos)  # right padding is invisible to the CLS vector
            for n, hidden in enumerate(out.hidden_states):
                arrays[f"{spec['id']}__hidden_{n}"] = hidden.numpy().copy()
            arrays[f"{spec['id']}__last_hidden_state"] = out.last_hidden_state.numpy().copy()
            rows.append(dict(spec, input_ids=tokens["input_ids"][0].tolist(), attention_mask=tokens["attention_mask"][0].tolist(),
                             embedding=e.tolist(), active_tokens=active, original_tokens=original_length,
                             truncated=original_length > length, padding_cosine=pad_cos))
            print(f"HF S={length} {index + 1}/{len(specimens)} {spec['id']} active={active} original={original_length}", flush=True)
        (fixtures_dir() / f"golden_s{length}.json").write_text(
            json.dumps({"model_sha": MODEL_SHA, "sequence_length": length, "fixtures": rows}, ensure_ascii=False) + "\n")
        np.savez(fixtures_dir() / f"intermediates_s{length}.npz", **arrays)
        report["grids"][str(length)] = {"count": len(rows), "min_padding_cosine": min(r["padding_cosine"] for r in rows),
                                        "truncated_count": sum(r["truncated"] for r in rows), "seconds": time.perf_counter() - start}
        del arrays
    # The high-level recipe (sentence-transformers) agrees with the raw path only on STRIPPED text.
    st = SentenceTransformer(str(source), device="cpu", local_files_only=True,
                             model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"}).eval()
    assert st.prompts == {"query": "", "document": ""}
    for length in SEQ_LENS:
        st.max_seq_length = length
        for spec in specimens[:10]:
            st_vec = st.encode([spec["text"]], normalize_embeddings=True, show_progress_bar=False)[0]
            tokens = tok(spec["text"].strip(), padding="max_length", truncation=True, max_length=length, return_tensors="pt")
            with torch.inference_mode():
                direct = F.normalize(model(**tokens).last_hidden_state[:, 0], dim=1).numpy()[0]
            c = cosine(st_vec, direct)
            assert c >= 0.999999, (spec["id"], c)
            report["sentence_transformer_checks"].append({"sequence_length": length, "id": spec["id"], "cosine": c,
                                                          "strip_changes_text": spec["text"].strip() != spec["text"]})
    report["sentence_transformer_note"] = ("sentence-transformers' Transformer.tokenize strips the text; the golden fixtures "
                                           "use the upstream README's raw AutoTokenizer path and keep whitespace.")
    report["output_hashes"] = hashes([fixtures_dir() / f"{name}_s{s}.{ext}" for s in SEQ_LENS
                                      for name, ext in [("golden", "json"), ("intermediates", "npz")]])
    write_json(results_dir() / "oracle.json", report)
    print(json.dumps({k: report[k] for k in ("status", "grids")}, indent=2))


if __name__ == "__main__":
    main()
