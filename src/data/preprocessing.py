"""Tokenization and cheap string scores for dataset labeling."""

import torch
from transformers import PreTrainedTokenizer


def tokenize_qa(
    question: str,
    answer: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int = 512,
) -> torch.Tensor:
    """Tokenize the Question / Answer text used by extraction."""
    text = f"Question: {question}\nAnswer: {answer}"
    enc = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"]


def compute_f1(prediction: str, reference: str) -> float:
    """Token-set F1 using lowercased whitespace tokens."""
    if not prediction.strip() or not reference.strip():
        return 0.0
    pr = set(prediction.lower().split())
    re = set(reference.lower().split())
    if not pr and not re:
        return 1.0
    if not pr or not re:
        return 0.0
    inter = pr & re
    if not inter:
        return 0.0
    p = len(inter) / len(pr)
    r = len(inter) / len(re)
    return 2 * p * r / (p + r)
