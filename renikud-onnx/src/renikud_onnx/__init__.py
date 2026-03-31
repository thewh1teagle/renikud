"""renikud-onnx: G2P inference via ONNX (structured upsample CTC model)."""

from __future__ import annotations

import json
import unicodedata

import numpy as np
import onnxruntime as ort

UPSAMPLE_FACTOR = 3


def _normalize(text: str, strip_accents: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    if strip_accents:
        text = "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    return text


class G2P:
    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(model_path)
        meta = self._session.get_modelmeta().custom_metadata_map
        self._cls_id = int(meta["cls_token_id"])
        self._sep_id = int(meta["sep_token_id"])
        self._strip_accents: bool = meta.get("strip_accents", "true") == "true"
        self._input_chars: frozenset[str] = frozenset(json.loads(meta["input_chars"]))
        output_tokens: list[str] = json.loads(meta["output_tokens"])
        # id_to_token: 1-indexed (0 = CTC blank)
        self._id_to_token: dict[int, str] = {i: t for i, t in enumerate(output_tokens)}

    def _tokenize(self, norm: str) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        ids = [self._cls_id]
        offsets: list[tuple[int, int]] = [(0, 0)]  # CLS
        for i, c in enumerate(norm):
            ids.append(ord(c))
            offsets.append((i, i + 1))
        ids.append(self._sep_id)
        offsets.append((0, 0))  # SEP
        mask = [1] * len(ids)
        return ids, mask, offsets

    def phonemize(self, text: str) -> str:
        norm = _normalize(text, strip_accents=self._strip_accents)
        ids, mask, offsets = self._tokenize(norm)

        logits, = self._session.run(
            ["logits"],
            {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64),
            },
        )
        # logits: [1, S*K, vocab_size]
        preds = logits[0].argmax(axis=-1)  # [S*K]

        result = []
        prev_end = 0
        K = UPSAMPLE_FACTOR
        blank_id = 0

        for tok_idx, (start, end) in enumerate(offsets):
            if start > prev_end:
                result.append(norm[prev_end:start])

            if end - start != 1:
                if end > start:
                    prev_end = end
                continue

            char = norm[start:end]
            prev_end = end

            if char not in self._input_chars:
                result.append(char)
                continue

            # Greedy CTC decode over K slots for this char
            slot_start = tok_idx * K
            slot_preds = preds[slot_start:slot_start + K].tolist()

            tokens = []
            prev = blank_id
            for p in slot_preds:
                if p != blank_id and p != prev:
                    tok = self._id_to_token.get(p - 1, "")
                    if tok and tok != "∅":
                        tokens.append(tok)
                prev = p
            result.extend(tokens)

        if prev_end < len(norm):
            result.append(norm[prev_end:])

        return "".join(result)
