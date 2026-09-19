#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers==4.57.6", "tokenizers==0.22.2", "regex", "numpy==2.2.6"]
# ///
"""Stage 2: the host tokenizer, gated in Python (and optionally Swift) against HF, ids and masks exact.

    python3 gate_granite_tokenizer.py                       # Python recipe vs HF AutoTokenizer
    python3 gate_granite_tokenizer.py --swift-bin <exe>     # + the Swift port (see the card)

`_granite_tokenizer.py` never imports HF. The contract it implements: regex Split(Isolated) ->
ByteLevel (no prefix space) -> byte BPE with `ignore_merges=true` -> CLS / body[:S-2] / SEP ->
right-pad with 179935 and mask 0. No stripping, no normalization, no task prefix. The gate also
proves it can fail: four mutations (pad with 0 / truncate after adding specials, losing SEP /
strip the text / ignore_merges=false) must each be caught by at least one case.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_SHA, SEQ_LENS, fixtures_dir, golden, results_dir, sha256_of, verify_source, write_json  # noqa: E402
from _granite_tokenizer import GraniteTokenizer  # noqa: E402


def make_cases(model_dir: Path) -> list[dict]:
    cases = json.loads((fixtures_dir() / "texts.json").read_text())
    cases += [{"id": "extra_" + str(i), "text": text} for i, text in enumerate([
        "hello", "Hello WORLD testCase", "I'm YOU'RE it's isn't DON'T we'd I'LL",
        "\x00\x01\x1f\x7f\r\n\t", "  \t\r\n東京\n\n  ",
        "é é Å Å Å ﬁ fi", "👩🏽‍💻🏳️‍🌈 𠮷野家 🇯🇵🇺🇸",
        "가나다 한글 가나다", "العَرَبِيَّة हिंदी ไทย עברית",
        "ß ẞ ſ i İ ı I", "​‌‍⁠﻿   ",
        # Token 2999 must stay whole: the one case that separates ignore_merges=true from false.
        " ક",
        "1234567890１２３４５６٧٨٩①②③ⅣⅤⅥ", "https://example.com/a//b?q=x&lang=ja\r\n",
        " ".join(["x"] * 700), "猫" * 700,
    ])]
    data = json.loads((model_dir / "tokenizer.json").read_text())
    for token in data["added_tokens"]:
        for index, (left, right) in enumerate([("", ""), ("a", "b"), ("猫", "犬"), (" ", " "), ("́", "‍")]):
            cases.append({"id": f"added_{token['id']}_{index}", "text": left + token["content"] + right})
    rng = random.Random(20260916)
    alphabet = list("abcABCxyzXYZ0123 \n\r\t.,!?'/_日本語東京都大阪") + [
        "é", "́", "Å", "Å", "ß", "İ", "ı", "Ａ", "ﾄ", "ッ", "ก", "م", "ا",
        "ह", "ि", "𠮷", "🌸", "👩", "‍", "‌", " ", " ",
    ]
    for i in range(300):
        cases.append({"id": f"random_{i:03}", "text": "".join(rng.choices(alphabet, k=rng.randrange(1, 121)))})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--swift-bin", type=Path, help="tokenizer CLI built from the Swift port; args: tokenizer.json tokenizer_config.json requests.json")
    args = parser.parse_args()
    model_dir = verify_source()
    candidate = GraniteTokenizer(model_dir)
    oracle = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    cases = make_cases(model_dir)
    requests, expected, comparisons, failures = [], [], [], []
    negative_controls = {"wrong_pad_zero": False, "truncate_after_specials": False, "strip_whitespace": False, "ignore_merges_disabled": False}
    witnesses: dict[str, str] = {}
    from tokenizers import Tokenizer
    mutated_data = json.loads((model_dir / "tokenizer.json").read_text())
    mutated_data["model"]["ignore_merges"] = False
    mutated = Tokenizer.from_str(json.dumps(mutated_data, ensure_ascii=False))
    mutated.no_padding()
    mutated.no_truncation()
    for length in SEQ_LENS:
        saved = {row["id"]: row for row in golden(length)["fixtures"]}
        for case in cases:
            text, case_id = case["text"], f"{case['id']}_s{length}"
            ref = oracle(text, max_length=length, padding="max_length", truncation=True)
            ids, mask = candidate.tokenize(text, length)
            actual_ids, actual_mask = ids[0].tolist(), mask[0].tolist()
            passed = actual_ids == ref["input_ids"] and actual_mask == ref["attention_mask"]
            saved_pass = None
            if case["id"] in saved:
                saved_pass = ref["input_ids"] == saved[case["id"]]["input_ids"] and ref["attention_mask"] == saved[case["id"]]["attention_mask"]
                passed = passed and saved_pass
            row = {"id": case_id, "pass": passed, "saved_golden_match": saved_pass, "active_tokens": sum(ref["attention_mask"])}
            comparisons.append(row)
            if not passed:
                failures.append({**row, "text": text, "expected_ids": ref["input_ids"], "actual_ids": actual_ids, "actual_mask": actual_mask})
            requests.append({"id": case_id, "text": text, "sequence_length": length})
            expected.append({"id": case_id, "input_ids": ref["input_ids"], "attention_mask": ref["attention_mask"]})
            wrong_pad = [value if active else 0 for value, active in zip(ref["input_ids"], ref["attention_mask"])]
            raw = oracle(text, add_special_tokens=True)["input_ids"]
            stripped = oracle(text.strip(), max_length=length, padding="max_length", truncation=True)
            mutations = {
                "wrong_pad_zero": wrong_pad != ref["input_ids"],
                "truncate_after_specials": len(raw) > length and raw[:length] != ref["input_ids"],
                "strip_whitespace": stripped["input_ids"] != ref["input_ids"],
                "ignore_merges_disabled": mutated.encode(text).ids != raw,
            }
            for mutation, detected in mutations.items():
                negative_controls[mutation] |= detected
                if detected and mutation not in witnesses:
                    witnesses[mutation] = case_id
    requests_path, expected_path = fixtures_dir() / "tokenizer_requests.json", fixtures_dir() / "tokenizer_expected.json"
    write_json(requests_path, requests)
    write_json(expected_path, expected)
    swift_result: dict = {"status": "NOT RUN"}
    if args.swift_bin:
        command = [str(args.swift_bin.resolve()), str(model_dir / "tokenizer.json"), str(model_dir / "tokenizer_config.json"), str(requests_path)]
        completed = subprocess.run(command, capture_output=True, check=False, text=True)
        if completed.returncode:
            swift_result = {"status": "FAIL", "command": command, "returncode": completed.returncode, "stderr": completed.stderr[-4000:]}
        else:
            actual = json.loads(completed.stdout)
            mismatches = [{"expected": want, "actual": have} for want, have in zip(expected, actual) if want != have]
            swift_result = {"status": "PASS" if len(actual) == len(expected) and not mismatches else "FAIL",
                            "command": command, "count": len(actual), "expected_count": len(expected), "mismatches": mismatches[:20]}
    ok = not failures and all(negative_controls.values()) and swift_result["status"] in ("PASS", "NOT RUN")
    report = {"status": "PASS" if ok else "FAIL", "python_status": "PASS" if not failures else "FAIL", "model_sha": MODEL_SHA,
              "contract": "raw AutoTokenizer, no prompts or stripping, S=128/512, B=1, ids/masks exact",
              "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "regex", "numpy")},
              "case_count": len(cases), "comparison_count": len(comparisons),
              "negative_controls_detected": negative_controls, "negative_control_witnesses": witnesses,
              "python_failures": failures[:20], "swift": swift_result,
              "hashes": {p.name: sha256_of(p) for p in [model_dir / "tokenizer.json", model_dir / "tokenizer_config.json",
                                                         Path(__file__).parent / "_granite_tokenizer.py"]}}
    write_json(results_dir() / "tokenizer-gate.json", report)
    print(json.dumps({k: report[k] for k in ["status", "python_status", "case_count", "comparison_count", "negative_controls_detected"]}, indent=2))
    print("Swift", swift_result["status"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
