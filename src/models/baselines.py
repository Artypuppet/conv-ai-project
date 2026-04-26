"""Cheap text-only baselines."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def perplexity_baseline(
    lm,
    dataset: List[dict],
) -> np.ndarray:
    """Perplexity of the answer tokens, conditioned on the question."""
    model = lm._model
    tokenizer = lm.tokenizer
    model.eval()
    device = next(model.parameters()).device
    scores: List[float] = []
    with torch.no_grad():
        for ex in dataset:
            question = ex["question"]
            answer = ex["answer"]
            full = f"Question: {question}\nAnswer: {answer}"
            prefix = f"Question: {question}\nAnswer: "
            enc = tokenizer(full, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
            start = int(prefix_ids.shape[1])
            if start >= int(input_ids.shape[1]) or input_ids.shape[1] < 2:
                scores.append(float("nan"))
                continue
            logits = model(input_ids).logits[0]
            # logits at i predict token i+1; answer token slice
            start_logits = start - 1
            answer_logits = logits[start_logits:-1]
            target = input_ids[0, start:]
            if answer_logits.shape[0] == 0 or target.shape[0] == 0:
                scores.append(float("nan"))
                continue
            nll = F.cross_entropy(answer_logits, target, reduction="mean")
            scores.append(float(torch.exp(nll).item()))
    return np.array(scores, dtype=np.float64)


def length_baseline(dataset: List[dict]) -> np.ndarray:
    """Whitespace-token count of the answer."""
    return np.array(
        [len(str(ex.get("answer", "")).split()) for ex in dataset],
        dtype=np.float64,
    )


def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
    step: float = 0.1,
    high_score_is_hallucination: bool = True,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Grid search score thresholds and keep the best F1."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    curve: List[Tuple[float, float]] = []
    # 0.05, 0.15, ... 0.95 for default low/step/high
    thresholds = np.arange(low, high + 1e-9, step)
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        if high_score_is_hallucination:
            preds = (scores > threshold).astype(np.int64)
        else:
            preds = (scores < threshold).astype(np.int64)
        f1 = float(
            f1_score(labels, preds, average="binary", zero_division=0),
        )
        curve.append((float(threshold), f1))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, curve


def plot_threshold_sweep(
    threshold_results: Sequence[Tuple[float, float]],
    title: str = "",
) -> plt.Figure:
    """F1 vs threshold; returns the figure (call ``show()`` in notebooks)."""
    if not threshold_results:
        fig, ax = plt.subplots()
        ax.set_title(title or "threshold sweep (empty)")
        return fig
    xs = [a[0] for a in threshold_results]
    ys = [a[1] for a in threshold_results]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys, marker="o", ms=3)
    ax.set_xlabel("threshold")
    ax.set_ylabel("F1")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
