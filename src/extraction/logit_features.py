"""Last-token logit KL and entropy between early and late layers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from torch.distributions import Categorical

from src.utils.config import Config
from src.utils.reproducibility import set_seed


def saved_value(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    v = getattr(x, "value", x)
    if not isinstance(v, torch.Tensor):
        raise TypeError("expected a tensor from nnsight save()")
    return v


def last_token_logits(x: torch.Tensor) -> torch.Tensor:
    # logits: (1,S,V) or (S,V) depending on nnsight (cf. hidden state shapes)
    if x.dim() == 3:
        t = x[:, -1, :]
        if t.size(0) == 1:
            t = t.squeeze(0)
    elif x.dim() == 2:
        t = x[-1, :]
    else:
        shape = tuple(x.shape)
        raise ValueError(f"expected 2D or 3D logits, got shape {shape}")
    return t


def extract_logit_dynamics(
    lm: LanguageModel,
    text: str,
    cfg: Config,
    early_layer: int = 4,
    late_layer: int = 14,
) -> Dict[str, float]:
    """KL(late || early) plus both last-token entropies."""
    _ = cfg
    saved_e: Any
    saved_l: Any
    with lm.trace(text):
        hs_e = lm.model.layers[early_layer].output[0]
        hs_l = lm.model.layers[late_layer].output[0]
        le = lm.lm_head(lm.model.norm(hs_e))
        ll = lm.lm_head(lm.model.norm(hs_l))
        saved_e = le.save()
        saved_l = ll.save()

    te = last_token_logits(saved_value(saved_e).float())
    tl = last_token_logits(saved_value(saved_l).float())
    l_early = F.log_softmax(te, dim=-1)
    p_late = F.softmax(tl, dim=-1)
    # (1, V) for batchmean; both last-token rows are 1D (V,) here
    kl = F.kl_div(
        l_early.unsqueeze(0),
        p_late.unsqueeze(0),
        reduction="batchmean",
    )
    p_early = F.softmax(te, dim=-1)
    h_early = Categorical(probs=p_early).entropy()
    h_late = Categorical(probs=p_late).entropy()
    return {
        "kl_divergence": float(kl.item()),
        "early_entropy": float(h_early.item()),
        "late_entropy": float(h_late.item()),
    }


def extract_all_logit_dynamics(
    lm: LanguageModel,
    dataset: List[dict],
    cfg: Config,
    split_name: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract logit dynamics for one split, with periodic saves."""
    set_seed(cfg.seed)
    out_path = cfg.output_dir / f"logit_features_{split_name}.pt"
    partial_path = cfg.output_dir / f"logit_features_{split_name}_partial.pt"

    if len(dataset) == 0:
        z = torch.tensor([], dtype=torch.float32)
        empty_l = torch.tensor([], dtype=torch.long)
        d = {
            "kl_divergence": z,
            "early_entropy": z,
            "late_entropy": z,
            "labels": empty_l,
        }
        torch.save(d, out_path)
        return {k: v for k, v in d.items() if k != "labels"}, empty_l

    kl_rows: List[float] = []
    early_entropy_rows: List[float] = []
    late_entropy_rows: List[float] = []
    label_rows: List[int] = []
    start_at = 0

    if partial_path.is_file():
        b = torch.load(partial_path, map_location="cpu", weights_only=False)
        kl_rows = b["kl_divergence"]
        early_entropy_rows = b["early_entropy"]
        late_entropy_rows = b["late_entropy"]
        label_rows = b["labels"]
        start_at = len(label_rows)
        print(f"Resuming from checkpoint: {start_at} examples already done")

    for i in range(start_at, len(dataset)):
        ex = dataset[i]
        s = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        o = extract_logit_dynamics(lm, s, cfg)
        kl_rows.append(o["kl_divergence"])
        early_entropy_rows.append(o["early_entropy"])
        late_entropy_rows.append(o["late_entropy"])
        label_rows.append(int(ex["label"]))

        if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
            torch.save(
                {
                    "kl_divergence": kl_rows,
                    "early_entropy": early_entropy_rows,
                    "late_entropy": late_entropy_rows,
                    "labels": label_rows,
                },
                partial_path,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  logits {split_name}: {i + 1}/{len(dataset)}")

    t_kl = torch.tensor(kl_rows, dtype=torch.float32)
    t_e = torch.tensor(early_entropy_rows, dtype=torch.float32)
    t_l = torch.tensor(late_entropy_rows, dtype=torch.float32)
    lab = torch.tensor(label_rows, dtype=torch.long)
    out = {
        "kl_divergence": t_kl,
        "early_entropy": t_e,
        "late_entropy": t_l,
        "labels": lab,
    }
    torch.save(out, out_path)
    if partial_path.is_file():
        partial_path.unlink()
    return {k: v for k, v in out.items() if k != "labels"}, lab
