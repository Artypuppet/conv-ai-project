"""Small helpers for the cached feature files used in the notebook."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from src.utils.config import Config


def as_numpy(value: Any) -> np.ndarray:
    """Detach torch tensors before handing them to numpy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_feature_bundle(cfg: Config, kind: str, split: str) -> dict:
    """Load ``outputs/{kind}_{split}.pt`` on CPU."""
    path = cfg.output_dir / f"{kind}_{split}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def flatten_attention(
    bundle: Mapping[str, Any],
    layers: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Flatten all chosen attention layers into one table."""
    by_layer = bundle["by_layer"]
    if layers is None:
        layers = sorted(int(k) for k in by_layer.keys())

    parts = []
    for layer in layers:
        values = as_numpy(by_layer[layer])
        if values.ndim >= 3:
            values = values.reshape(values.shape[0], -1)
        parts.append(values)
    return np.concatenate(parts, axis=1).astype(np.float32)


def stack_logit_features(bundle: Mapping[str, Any]) -> np.ndarray:
    """Columns are KL, early entropy, late entropy."""
    return np.stack(
        [
            as_numpy(bundle["kl_divergence"]),
            as_numpy(bundle["early_entropy"]),
            as_numpy(bundle["late_entropy"]),
        ],
        axis=1,
    ).astype(np.float32)
