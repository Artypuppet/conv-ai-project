"""Spectral features from attention maps."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nnsight import LanguageModel

from src.utils.config import Config
from src.utils.reproducibility import set_seed


def saved_value(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    v = getattr(x, "value", x)
    if not isinstance(v, torch.Tensor):
        raise TypeError("expected a tensor from nnsight save()")
    return v


def compute_laplacian_features(
    attn_matrix: np.ndarray,
    top_k: int = 5,
) -> np.ndarray:
    """Top diagonal Laplacian values for one attention head."""
    t = int(attn_matrix.shape[0])
    if t == 0:
        return np.zeros((0,), dtype=np.float64)
    a = np.asarray(attn_matrix, dtype=np.float64)
    d = np.zeros(t, dtype=np.float64)
    for i in range(t):
        col_sum = a[:, i].sum()
        d[i] = col_sum / max(t - i, 1)
    d_mat = np.diag(d)
    l_ap = d_mat - a
    diag = np.diag(l_ap)
    srt = np.sort(diag)  # ascending
    k = min(top_k, t)
    # largest k: last k in ascending order, then reverse
    return srt[-k:][::-1].astype(np.float32)


def extract_attention_features(
    lm: LanguageModel,
    text: str,
    cfg: Config,
    top_k: int = 5,
) -> Dict[int, np.ndarray]:
    """Per-layer, per-head diagonal Laplacian features."""
    _ = cfg
    n = len(lm.model.layers)
    saved: Dict[int, Any] = {}
    with lm.trace(text):
        for li in range(n):
            w = lm.model.layers[li].self_attn.output[1]
            saved[li] = w.save()

    out: Dict[int, np.ndarray] = {}
    for li in range(n):
        aw = saved_value(saved[li]).float()  # (B, H, S, S)
        aw = aw.squeeze(0)  # (H, S, S)
        head_feats = []
        for hi in range(int(aw.shape[0])):
            am = aw[hi].detach().cpu().numpy()
            head_feats.append(compute_laplacian_features(am, top_k=top_k))
        out[li] = np.stack(head_feats, axis=0).astype(np.float32)
    return out


def extract_all_attention_features(
    lm: LanguageModel,
    dataset: List[dict],
    cfg: Config,
    split_name: str,
) -> Tuple[Dict[int, np.ndarray], torch.Tensor]:
    """Run attention feature extraction for one dataset split."""
    set_seed(cfg.seed)
    out_path = cfg.output_dir / f"attention_features_{split_name}.pt"
    partial_name = f"attention_features_{split_name}_partial.pt"
    partial_path = cfg.output_dir / partial_name
    top_k = 5  # same default as single-example helper

    if len(dataset) == 0:
        empty = {}
        lab = torch.tensor([], dtype=torch.long)
        torch.save({"by_layer": empty, "labels": lab}, out_path)
        return empty, lab

    rows: List[Dict[int, np.ndarray]] = []
    label_rows: List[int] = []
    start_at = 0
    n_layers: Optional[int] = None
    feature_k = top_k

    if partial_path.is_file():
        b = torch.load(partial_path, map_location="cpu", weights_only=False)
        rows = b["rows"]
        label_rows = b["labels"]
        start_at = len(label_rows)
        n_layers = b.get("n_layers")
        feature_k = b.get("top_k", top_k)
        if n_layers is None and rows:
            n_layers = len(rows[0])
        print(f"Resuming from checkpoint: {start_at} examples already done")

    for i in range(start_at, len(dataset)):
        ex = dataset[i]
        s = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        row = extract_attention_features(lm, s, cfg, top_k=feature_k)
        if n_layers is None:
            n_layers = len(row)
        rows.append(row)
        label_rows.append(int(ex["label"]))

        if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
            torch.save(
                {
                    "rows": rows,
                    "labels": label_rows,
                    "n_layers": n_layers,
                    "top_k": feature_k,
                },
                partial_path,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  attn {split_name}: {i + 1}/{len(dataset)}")

    assert n_layers is not None
    by_layer: Dict[int, np.ndarray] = {}
    for li in range(n_layers):
        by_layer[li] = np.stack([r[li] for r in rows], axis=0)

    labels = torch.tensor(label_rows, dtype=torch.long)
    torch.save(
        {"by_layer": by_layer, "labels": labels, "top_k": feature_k},
        out_path,
    )
    if partial_path.is_file():
        partial_path.unlink()
    return by_layer, labels
