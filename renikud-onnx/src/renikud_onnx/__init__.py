"""renikud-onnx: Hebrew grapheme-to-phoneme inference via ONNX."""

from __future__ import annotations

import json
import unicodedata

import numpy as np
import onnxruntime as ort


class G2P:
    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(model_path)
        meta = self._session.get_modelmeta().custom_metadata_map
        self._vocab: dict[str, int] = json.loads(meta["vocab"])
        self._ipa_vocab: dict[int, str] = {int(k): v for k, v in json.loads(meta["ipa_vocab"]).items()}
        self._cls_id = int(meta["cls_token_id"])
        self._sep_id = int(meta["sep_token_id"])
        self._blank_id = 0
        _ipa_token_to_id = {v: k for k, v in self._ipa_vocab.items()}
        self._stress_id: int = _ipa_token_to_id.get("ˈ", -1)

    def _tokenize(self, text: str) -> tuple[list[int], list[int]]:
        text = unicodedata.normalize("NFD", text)
        unk_id = self._vocab.get("[UNK]", 0)
        ids = [self._cls_id] + [self._vocab.get(c, unk_id) for c in text] + [self._sep_id]
        mask = [1] * len(ids)
        return ids, mask

    def _decode(self, token_ids: list[int]) -> str:
        result = []
        prev = None
        for t in token_ids:
            if t == self._blank_id:
                prev = None
                continue
            if t != prev:
                token = self._ipa_vocab.get(t, "")
                if token not in ("<pad>", "<unk>", "<blank>"):
                    result.append(token)
            prev = t
        return "".join(result)

    def _run_onnx(self, ids: list[int], mask: list[int]) -> np.ndarray:
        logits, input_lengths = self._session.run(
            ["logits", "input_lengths"],
            {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64),
            },
        )
        length = int(input_lengths[0])
        return logits[0, :length].copy()  # [T, vocab]

    def _apply_constraints(self, logits: np.ndarray, text: str) -> np.ndarray:
        if self._stress_id == -1:
            return logits

        word_has_stress = False
        is_word_start = True

        for i, char in enumerate(text):
            # each input character maps to 2 frames due to upsample_factor=2
            # [CLS] token occupies no extra frames — character i maps to frames i*2, i*2+1
            frame_start = i * 2
            frame_end = frame_start + 2
            if frame_end > len(logits):
                break

            if char == " ":
                # reset per-word state on word boundary
                word_has_stress = False
                is_word_start = True
                continue

            if word_has_stress:
                # only one stress allowed per word
                logits[frame_start:frame_end, self._stress_id] = -1e9
            else:
                if is_word_start:
                    # stress must come after at least one phoneme — block it on the
                    # first frame of the word so it can't lead before any consonant
                    logits[frame_start, self._stress_id] = -1e9
                if logits[frame_start:frame_end].argmax() == self._stress_id:
                    word_has_stress = True

            is_word_start = False

        return logits

    def phonemize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        ids, mask = self._tokenize(text)
        logits = self._run_onnx(ids, mask)
        logits = self._apply_constraints(logits, normalized)
        pred_ids = logits.argmax(axis=-1).tolist()
        return self._decode(pred_ids)
