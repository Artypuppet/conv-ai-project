"""Layer-wise mean-pooled hidden states from a forward trace."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from nnsight import LanguageModel

from src.utils.config import Config
from src.utils.reproducibility import set_seed


def saved_value(x: Any) -> torch.Tensor:
    # nnsight .save() gives a torch.Tensor (or a thin subclass) after the trace
    if isinstance(x, torch.Tensor):
        return x
    v = getattr(x, "value", x)
    if not isinstance(v, torch.Tensor):
        raise TypeError("expected a tensor from nnsight save()")
    return v


def extract_hidden_states(
    lm: LanguageModel,
    text: str,
    cfg: Config,
) -> Dict[int, torch.Tensor]:
    """Mean-pool each layer's hidden state into one vector."""
    _ = cfg  # API matches plan; extraction uses the model's tokenizer in trace
    n_layers = len(lm.model.layers)
    # capture each layer's *decoder block* output; [0] is the hidden tensor
    saved: Dict[int, Any] = {}
    with lm.trace(text):
        for li in range(n_layers):
            h = lm.model.layers[li].output[0]
            saved[li] = h.save()

    out: Dict[int, torch.Tensor] = {}
    for li in range(n_layers):
        t = saved_value(saved[li]).float()
        # nnsight can return 2D or 3D tensors here; pool over token dims.
        d = t.size(-1)
        v = t.reshape(-1, d).mean(dim=0)
        out[li] = v.detach().cpu()
    return out


def extract_all_hidden_states(
    lm: LanguageModel,
    dataset: List[dict],
    cfg: Config,
    split_name: str,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Extract hidden states for one split, with periodic saves."""
    set_seed(cfg.seed)
    out_path = cfg.output_dir / f"hidden_states_{split_name}.pt"
    partial_path = cfg.output_dir / f"hidden_states_{split_name}_partial.pt"

    if len(dataset) == 0:
        by_layer, labels = {}, torch.tensor([], dtype=torch.long)
        torch.save({"by_layer": by_layer, "labels": labels}, out_path)
        return by_layer, labels

    rows: List[Dict[int, torch.Tensor]] = []
    label_rows: List[int] = []
    start_at = 0
    n_layers: Optional[int] = None

    if partial_path.is_file():
        bundle = torch.load(
            partial_path,
            map_location="cpu",
            weights_only=False,
        )
        rows = bundle["rows"]
        label_rows = bundle["labels"]
        start_at = len(label_rows)
        n_layers = bundle.get("n_layers")
        if n_layers is None and rows:
            n_layers = len(rows[0])
        print(f"Resuming from checkpoint: {start_at} examples already done")

    for i in range(start_at, len(dataset)):
        ex = dataset[i]
        s = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        feats = extract_hidden_states(lm, s, cfg)
        if n_layers is None:
            n_layers = len(feats)
        rows.append(feats)
        label_rows.append(int(ex["label"]))

        if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
            torch.save(
                {"rows": rows, "labels": label_rows, "n_layers": n_layers},
                partial_path,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  {split_name}: {i + 1}/{len(dataset)}")

    # stack to tensors
    assert n_layers is not None
    by_layer = {
        li: torch.stack([r[li] for r in rows], dim=0)
        for li in range(n_layers)
    }
    labels = torch.tensor(label_rows, dtype=torch.long)

    torch.save({"by_layer": by_layer, "labels": labels}, out_path)
    if partial_path.is_file():
        partial_path.unlink()
    return by_layer, labels
