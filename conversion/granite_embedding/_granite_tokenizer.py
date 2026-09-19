#!/usr/bin/env python3
"""Independent raw-text Granite 97M R2 tokenizer; no HF tokenizer in this path.

The supported contract is the pinned checkpoint's Regex Split -> ByteLevel -> BPE
with ignore_merges=True -> CLS/body/SEP template, with right truncation/padding.
Metadata mismatches fail rather than silently falling back to a generic tokenizer.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import regex


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import source_dir  # noqa: E402


def _byte_alphabet() -> dict[int, str]:
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {value: chr(value) for value in visible}
    extra = 0
    for value in range(256):
        if value not in mapping:
            mapping[value] = chr(256 + extra)
            extra += 1
    return mapping


class GraniteTokenizer:
    """Only a single raw Unicode string; no prompts, normalization or stripping."""

    def __init__(self, model_dir: str | Path | None = None):
        model_dir = Path(model_dir) if model_dir else source_dir()
        data = json.loads((model_dir / "tokenizer.json").read_text())
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
        model = data["model"]
        if not (model["type"] == "BPE" and model["ignore_merges"] is True
                and model["dropout"] is None and model["unk_token"] is None
                and model["byte_fallback"] is False
                and model["continuing_subword_prefix"] is None
                and model["end_of_word_suffix"] is None
                and data["normalizer"] is None):
            raise ValueError("Unsupported BPE/normalizer configuration")
        pre = data["pre_tokenizer"]
        if pre["type"] != "Sequence" or len(pre["pretokenizers"]) != 2:
            raise ValueError("Expected Split + ByteLevel pretokenizer")
        split, bytelevel = pre["pretokenizers"]
        if not (split["type"] == "Split" and split["behavior"] == "Isolated"
                and split["invert"] is False and set(split["pattern"]) == {"Regex"}
                and bytelevel["type"] == "ByteLevel"
                and bytelevel["add_prefix_space"] is False
                and bytelevel["use_regex"] is False):
            raise ValueError("Unsupported pretokenizer configuration")
        self.pattern = regex.compile(split["pattern"]["Regex"])
        self.vocab: dict[str, int] = model["vocab"]
        self.ranks = {tuple(pair): rank for rank, pair in enumerate(model["merges"])}
        if any(len(pair) != 2 for pair in self.ranks):
            raise ValueError("Expected two-string merge pairs")
        self.byte_alphabet = _byte_alphabet()
        self.added: dict[str, int] = {}
        self.single_word: set[str] = set()
        for token in data["added_tokens"]:
            if any(token[key] for key in ("lstrip", "rstrip", "normalized")):
                raise ValueError("Unsupported added-token matching flags")
            if not token["content"]:
                raise ValueError("Empty added token")
            self.added[token["content"]] = token["id"]
            if token["single_word"]:
                self.single_word.add(token["content"])
        self.added_pattern = regex.compile("|".join(regex.escape(token) for token in
                                                  sorted(self.added, key=len, reverse=True)))
        self.cls_id, self.sep_id, self.pad_id = 179934, 179938, 179935
        post = data["post_processor"]
        expected_single = [
            {"SpecialToken": {"id": "<|startoftext|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<|return|>", "type_id": 0}},
        ]
        if post["type"] != "TemplateProcessing" or post["single"] != expected_single:
            raise ValueError("Unsupported single-text postprocessor")
        for name, expected_id in (("cls_token", self.cls_id), ("sep_token", self.sep_id),
                                  ("pad_token", self.pad_id)):
            token = config[name]
            token = token["content"] if isinstance(token, dict) else token
            if self.added.get(token) != expected_id:
                raise ValueError(f"Unexpected {name}")
        if config.get("padding_side", "right") != "right" or config.get("truncation_side", "right") != "right":
            raise ValueError("Expected right padding and truncation")

    @lru_cache(maxsize=32768)
    def _bpe(self, token: str) -> tuple[int, ...]:
        # ignore_merges does not disable BPE: a whole pretoken vocabulary match wins.
        if token in self.vocab:
            return (self.vocab[token],)
        # HF BPE merge_word omits missing initial symbols when both UNK and byte
        # fallback are absent. The checkpoint lacks e.g. mapped NUL (U+0100).
        symbols = [symbol for symbol in token if symbol in self.vocab]
        while len(symbols) > 1:
            best = min(((self.ranks.get((left, right), float("inf")), index)
                        for index, (left, right) in enumerate(zip(symbols, symbols[1:]))),
                       default=(float("inf"), 0))
            if best[0] == float("inf"):
                break
            index = best[1]
            symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
        # Every surviving symbol/merge must be in the checkpoint vocabulary.
        return tuple(self.vocab[symbol] for symbol in symbols)

    def _ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        cursor = 0
        for match in self.pattern.finditer(text):
            # Split(Isolated) retains nonmatching gaps as individual pieces too.
            if match.start() > cursor:
                piece = text[cursor:match.start()]
                ids.extend(self._bpe("".join(self.byte_alphabet[b] for b in piece.encode("utf-8"))))
            piece = match.group(0)
            if piece:
                ids.extend(self._bpe("".join(self.byte_alphabet[b] for b in piece.encode("utf-8"))))
            cursor = match.end()
        if cursor < len(text):
            ids.extend(self._bpe("".join(self.byte_alphabet[b] for b in text[cursor:].encode("utf-8"))))
        return ids

    def encode_body(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Expected one Unicode string")
        ids: list[int] = []
        cursor = 0
        for match in self.added_pattern.finditer(text):
            if match.group(0) in self.single_word:
                left_word = match.start() > 0 and regex.fullmatch(r"\w", text[match.start() - 1])
                right_word = match.end() < len(text) and regex.fullmatch(r"\w", text[match.end()])
                if left_word or right_word:
                    continue
            ids.extend(self._ordinary(text[cursor:match.start()]))
            ids.append(self.added[match.group(0)])
            cursor = match.end()
        ids.extend(self._ordinary(text[cursor:]))
        return ids

    def tokenize(self, text: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 2:
            raise ValueError("seq_len must be an integer >=2")
        active = [self.cls_id, *self.encode_body(text)[:seq_len - 2], self.sep_id]
        ids = np.full((1, seq_len), self.pad_id, dtype=np.int32)
        mask = np.zeros((1, seq_len), dtype=np.int32)
        ids[0, :len(active)] = active
        mask[0, :len(active)] = 1
        return ids, mask


@lru_cache(maxsize=1)
def _default_tokenizer() -> GraniteTokenizer:
    return GraniteTokenizer()


def tokenize(text: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return int32 IDs and binary mask, both shaped [1, seq_len]."""
    return _default_tokenizer().tokenize(text, seq_len)
