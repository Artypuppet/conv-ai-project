"""A small gated-fusion sweep used by the notebook."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.models.gated_fusion import (
    GatedFusionDataset,
    GatedFusionProbe,
    prepare_hidden_features,
    train_gated_fusion,
)
from src.utils.config import Config
from src.utils.features import (
    flatten_attention,
    load_feature_bundle,
    stack_logit_features,
)
from src.utils.reproducibility import set_seed


def make_loader(
    hidden: torch.Tensor,
    attn: np.ndarray,
    logits: np.ndarray,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        GatedFusionDataset(
            hidden,
            torch.from_numpy(attn).float(),
            torch.from_numpy(logits).float(),
            labels.float(),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_gated_fusion(
    model: GatedFusionProbe,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    labels: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for hidden, attn, logit, batch_labels in loader:
        prob, _ = model(
            hidden.to(device),
            attn.to(device),
            logit.to(device),
        )
        labels.append(batch_labels.numpy())
        probs.append(prob.cpu().numpy())
    y_true = np.concatenate(labels).astype(np.int64)
    y_prob = np.concatenate(probs)
    preds = (y_prob >= 0.5).astype(np.int64)
    return compute_metrics(y_true, preds, y_prob)


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sweep_gated_fusion(
    cfg: Optional[Config] = None,
    *,
    epochs: int = 50,
    patience: int = 5,
    best_patience: int = 10,
    batch_size: Optional[int] = None,
    enc_hidden: int = 128,
    dropouts: Optional[Iterable[float]] = None,
    weight_decays: Optional[Iterable[float]] = None,
    lrs: Optional[Iterable[float]] = None,
    output_path: Optional[Path] = None,
    save_state: bool = True,
) -> Dict[str, Any]:
    """Try a few gated-fusion settings and keep the best validation AUC."""
    cfg = cfg or Config()
    set_seed(cfg.seed)
    batch = int(batch_size or cfg.batch_size)
    output_path = output_path or (cfg.output_dir / "gated_sweep_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hidden_train_bundle = load_feature_bundle(cfg, "hidden_states", "train")
    hidden_val_bundle = load_feature_bundle(cfg, "hidden_states", "val")
    hidden_test_bundle = load_feature_bundle(cfg, "hidden_states", "test")
    attn_train_bundle = load_feature_bundle(cfg, "attention_features", "train")
    attn_val_bundle = load_feature_bundle(cfg, "attention_features", "val")
    attn_test_bundle = load_feature_bundle(cfg, "attention_features", "test")
    logit_train_bundle = load_feature_bundle(cfg, "logit_features", "train")
    logit_val_bundle = load_feature_bundle(cfg, "logit_features", "val")
    logit_test_bundle = load_feature_bundle(cfg, "logit_features", "test")

    hidden_train = prepare_hidden_features(hidden_train_bundle["by_layer"])
    hidden_val = prepare_hidden_features(hidden_val_bundle["by_layer"])
    hidden_test = prepare_hidden_features(hidden_test_bundle["by_layer"])
    attn_train = flatten_attention(attn_train_bundle)
    attn_val = flatten_attention(attn_val_bundle)
    attn_test = flatten_attention(attn_test_bundle)
    logit_train = stack_logit_features(logit_train_bundle)
    logit_val = stack_logit_features(logit_val_bundle)
    logit_test = stack_logit_features(logit_test_bundle)

    device = cfg.device
    hidden_dim = int(hidden_train.shape[1])
    attn_dim = int(attn_train.shape[1])
    logit_dim = int(logit_train.shape[1])

    train_loader = make_loader(
        hidden_train,
        attn_train,
        logit_train,
        hidden_train_bundle["labels"],
        batch,
        shuffle=True,
    )
    val_loader = make_loader(
        hidden_val,
        attn_val,
        logit_val,
        hidden_val_bundle["labels"],
        batch,
        shuffle=False,
    )
    test_loader = make_loader(
        hidden_test,
        attn_test,
        logit_test,
        hidden_test_bundle["labels"],
        batch,
        shuffle=False,
    )

    if dropouts is None:
        dropout_grid = [0.0, 0.2, 0.3, 0.5]
    else:
        dropout_grid = list(dropouts)
    if weight_decays is None:
        weight_decay_grid = [0.0, 1e-4, 1e-3]
    else:
        weight_decay_grid = list(weight_decays)
    lr_grid = list(lrs) if lrs is not None else [5e-4, 1e-3, 3e-3]
    configs = list(
        itertools.product(
            dropout_grid,
            weight_decay_grid,
            lr_grid,
        )
    )

    results: List[Dict[str, Any]] = []
    for idx, (dropout, weight_decay, lr) in enumerate(configs, start=1):
        print(
            f"[{idx}/{len(configs)}] dropout={dropout} "
            f"weight_decay={weight_decay} lr={lr}",
            flush=True,
        )
        val_loss_history: List[float] = []
        probe = GatedFusionProbe(
            hidden_dim,
            attn_dim,
            logit_dim,
            dropout=float(dropout),
            enc_hidden=enc_hidden,
        )
        probe = train_gated_fusion(
            probe,
            train_loader,
            val_loader,
            n_epochs=epochs,
            lr=float(lr),
            patience=patience,
            device=device,
            seed=cfg.seed,
            val_loss_history=val_loss_history,
            weight_decay=float(weight_decay),
        )
        val_metrics = evaluate_gated_fusion(probe, val_loader, device)
        results.append(
            {
                "dropout": float(dropout),
                "weight_decay": float(weight_decay),
                "lr": float(lr),
                "enc_hidden": enc_hidden,
                "batch_size": batch,
                "epochs": epochs,
                "patience": patience,
                "val_loss_history": val_loss_history,
                "val_metrics": val_metrics,
                "val_roc_auc": val_metrics.get("roc_auc"),
                "val_f1": val_metrics.get("f1"),
            }
        )
        print(
            f"  val_roc_auc={val_metrics.get('roc_auc', float('nan')):.4f} "
            f"val_f1={val_metrics.get('f1', float('nan')):.4f}",
            flush=True,
        )

    results.sort(
        key=lambda row: float(row.get("val_roc_auc") or -1.0),
        reverse=True,
    )
    best = results[0]
    best_json = json.dumps(make_json_safe(best), indent=2)
    print("Best config:", best_json, flush=True)

    best_val_loss_history: List[float] = []
    best_probe = GatedFusionProbe(
        hidden_dim,
        attn_dim,
        logit_dim,
        dropout=float(best["dropout"]),
        enc_hidden=int(best["enc_hidden"]),
    )
    best_probe = train_gated_fusion(
        best_probe,
        train_loader,
        val_loader,
        n_epochs=epochs,
        lr=float(best["lr"]),
        patience=best_patience,
        device=device,
        seed=cfg.seed,
        val_loss_history=best_val_loss_history,
        weight_decay=float(best["weight_decay"]),
    )
    best_val_metrics = evaluate_gated_fusion(best_probe, val_loader, device)
    best_test_metrics = evaluate_gated_fusion(best_probe, test_loader, device)

    state_path: Optional[Path] = None
    if save_state:
        state_path = cfg.output_dir / "gated_fusion_tuned_state.pt"
        torch.save(best_probe.state_dict(), state_path)

    payload = {
        "feature_dims": {
            "hidden": hidden_dim,
            "attention": attn_dim,
            "logit": logit_dim,
        },
        "all_configs": results,
        "best_config": {
            "dropout": best["dropout"],
            "weight_decay": best["weight_decay"],
            "lr": best["lr"],
            "enc_hidden": best["enc_hidden"],
            "batch_size": best["batch_size"],
        },
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "best_val_loss_history": best_val_loss_history,
        "state_path": str(state_path) if state_path is not None else None,
    }
    safe_payload = make_json_safe(payload)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(safe_payload, handle, indent=2)
    print("Wrote", output_path, flush=True)
    if state_path is not None:
        print("Saved tuned state", state_path, flush=True)
    return safe_payload


def top_gated_configs(
    sweep_results: Dict[str, Any],
    n: int = 5,
) -> List[Dict[str, Any]]:
    """Convenience for the notebook table."""
    rows = list(sweep_results.get("all_configs", []))
    rows.sort(
        key=lambda row: float(row.get("val_roc_auc") or -1.0),
        reverse=True,
    )
    return rows[:n]
