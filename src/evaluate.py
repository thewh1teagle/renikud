"""Evaluation helpers for Hebrew G2P training."""

from __future__ import annotations

import numpy as np
from jiwer import cer, wer

from tokenization import decode_ctc, decode_ipa


def build_compute_metrics():
    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred

        input_lengths = None
        if isinstance(predictions, tuple):
            # Trainer returns (logits, input_lengths) when model outputs both
            if len(predictions) == 2:
                predictions, input_lengths = predictions[0], predictions[1]
            else:
                predictions = predictions[0]

        logits = np.asarray(predictions)
        label_ids = np.asarray(labels)

        pred_texts = []
        for i, row in enumerate(logits.argmax(axis=-1)):
            length = int(input_lengths[i]) if input_lengths is not None else len(row)
            pred_texts.append(decode_ctc(row[:length].tolist()))

        label_texts = [
            decode_ipa([int(token_id) for token_id in row if int(token_id) != -100])
            for row in label_ids
        ]

        return {
            "cer": cer(label_texts, pred_texts),
            "wer": wer(label_texts, pred_texts),
        }

    return compute_metrics
