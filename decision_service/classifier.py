"""Score one-token answer labels with a local GGUF model."""

from __future__ import annotations

import threading

import numpy as np

REPO_ID = "Qwen/Qwen3-0.6B-GGUF"
FILENAME = "Qwen3-0.6B-Q8_0.gguf"
CONTEXT_SIZE = 2048
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Classifier:
    def __init__(self, model=None):
        self._model = model
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            from llama_cpp import Llama

            self._model = Llama.from_pretrained(
                repo_id=REPO_ID,
                filename=FILENAME,
                n_ctx=CONTEXT_SIZE,
                logits_all=True,
                verbose=False,
            )
        return self._model

    def pick(self, context: str, question: str, options: list[str]):
        labels = LABELS[: len(options)]
        choices = "\n".join(
            f"{label}. {option}" for label, option in zip(labels, options, strict=True)
        )
        prompt = (
            "<|im_start|>system\nChoose one option. Reply with its letter."
            "<|im_end|>\n<|im_start|>user\n"
            f"{choices}\n\nContext: {context}\nQuestion: {question}"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

        # llama.cpp mutates its context during eval, so keep the whole call locked.
        with self._lock:
            model = self._get_model()
            tokens = model.tokenize(prompt.encode(), add_bos=False, special=True)
            if len(tokens) >= CONTEXT_SIZE:
                raise ValueError("The request exceeds the model context window")
            token_ids = []
            for label in labels:
                ids = model.tokenize(label.encode(), add_bos=False)
                if len(ids) != 1:
                    raise RuntimeError(f"Option label {label} is not one token")
                token_ids.append(ids[0])
            model.eval(tokens=tokens)
            logits = np.asarray(
                [model.scores[model.n_tokens - 1][token_id] for token_id in token_ids],
                dtype=np.float64,
            )

        log_probabilities = logits - np.logaddexp.reduce(logits)
        confidence = np.exp(log_probabilities)
        return (
            dict(zip(options, logits.astype(float).tolist(), strict=True)),
            dict(zip(options, log_probabilities.astype(float).tolist(), strict=True)),
            dict(zip(options, confidence.astype(float).tolist(), strict=True)),
        )
