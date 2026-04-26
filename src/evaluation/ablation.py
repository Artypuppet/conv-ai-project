"""Ablation studies used in the notebook."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import compute_metrics
from src.models.probe_xgboost import predict_xgboost, train_xgboost


def as_numpy(
    a: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """Convert a tensor/array to a numpy array."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def sort_layer_dict(
    m: Mapping[int, Union[np.ndarray, torch.Tensor]],
) -> Dict[int, np.ndarray]:
    """Sort a layer -> matrix mapping by layer number."""
    items = sorted(m.items(), key=lambda x: int(x[0]))
    return {int(k): as_numpy(v) for k, v in items}


def fit_xgb_and_score(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_te: np.ndarray,
    name: str,
    random_state: int = 42,
) -> Dict[str, float]:
    """Train one XGB probe and print its test AUC."""
    clf = train_xgboost(X_tr, y_tr, X_va, y_va, random_state=random_state)
    preds, probs = predict_xgboost(clf, X_te)
    metrics = compute_metrics(y_te, preds, probs)
    print(
        f"{name}: test roc_auc={metrics.get('roc_auc', float('nan')):.4f}",
        flush=True,
    )
    return metrics


def layer_isolation_study(
    by_layer_train: Mapping[int, Union[np.ndarray, torch.Tensor]],
    by_layer_val: Mapping[int, Union[np.ndarray, torch.Tensor]],
    by_layer_test: Mapping[int, Union[np.ndarray, torch.Tensor]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[int, Dict[str, float]]:
    """Train one XGBoost probe per hidden layer."""
    tr = sort_layer_dict(by_layer_train)
    va = sort_layer_dict(by_layer_val)
    te = sort_layer_dict(by_layer_test)
    y_tr = as_numpy(y_train).ravel().astype(np.int64)
    y_va = as_numpy(y_val).ravel().astype(np.int64)
    y_te = as_numpy(y_test).ravel().astype(np.int64)
    out: Dict[int, Dict[str, float]] = {}
    layers = sorted(set(tr) & set(va) & set(te))
    for li in layers:
        print(f"layer {li}: train xgb...", flush=True)
        clf = train_xgboost(
            tr[li],
            y_tr,
            va[li],
            y_va,
            random_state=random_state,
        )
        preds, probs = predict_xgboost(clf, te[li])
        m = compute_metrics(y_te, preds, probs)
        out[li] = m
        print(
            f"  layer {li} test  "
            f"roc_auc={m.get('roc_auc', float('nan')):.4f}  "
            f"f1={m['f1']:.4f}",
            flush=True,
        )
    return out


def signal_modality_study(
    hidden_train: np.ndarray,
    hidden_val: np.ndarray,
    hidden_test: np.ndarray,
    attn_train: np.ndarray,
    attn_val: np.ndarray,
    attn_test: np.ndarray,
    logit_train: np.ndarray,
    logit_val: np.ndarray,
    logit_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Try hidden-only, attention-only, logits-only, and all together."""
    y_tr = as_numpy(y_train).ravel().astype(np.int64)
    y_va = as_numpy(y_val).ravel().astype(np.int64)
    y_te = as_numpy(y_test).ravel().astype(np.int64)

    h_tr, h_va, h_te = map(as_numpy, (hidden_train, hidden_val, hidden_test))
    a_tr, a_va, a_te = map(as_numpy, (attn_train, attn_val, attn_test))
    z_tr, z_va, z_te = map(as_numpy, (logit_train, logit_val, logit_test))

    res: Dict[str, Dict[str, float]] = {}
    res["hidden_only"] = fit_xgb_and_score(
        h_tr, h_va, h_te, y_tr, y_va, y_te, "hidden_only", random_state
    )
    res["attention_only"] = fit_xgb_and_score(
        a_tr, a_va, a_te, y_tr, y_va, y_te, "attention_only", random_state
    )
    res["logit_only"] = fit_xgb_and_score(
        z_tr, z_va, z_te, y_tr, y_va, y_te, "logit_only", random_state
    )
    all_tr = np.hstack([h_tr, a_tr, z_tr])
    all_va = np.hstack([h_va, a_va, z_va])
    all_te = np.hstack([h_te, a_te, z_te])
    res["all_concatenated"] = fit_xgb_and_score(
        all_tr,
        all_va,
        all_te,
        y_tr,
        y_va,
        y_te,
        "all_concatenated",
        random_state,
    )
    return res


def internal_vs_hybrid_study(
    X_int_tr: np.ndarray,
    X_int_va: np.ndarray,
    X_int_te: np.ndarray,
    text_tr: np.ndarray,
    text_va: np.ndarray,
    text_te: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compare internal features with internal + text features."""
    y_tr = as_numpy(y_train).ravel().astype(np.int64)
    y_va = as_numpy(y_val).ravel().astype(np.int64)
    y_te = as_numpy(y_test).ravel().astype(np.int64)
    h_tr, h_va, h_te = map(as_numpy, (X_int_tr, X_int_va, X_int_te))
    t_tr, t_va, t_te = map(as_numpy, (text_tr, text_va, text_te))

    clf_i = train_xgboost(h_tr, y_tr, h_va, y_va, random_state=random_state)
    preds, probs = predict_xgboost(clf_i, h_te)
    m_int = compute_metrics(y_te, preds, probs)
    print("internal only", m_int, flush=True)

    hy_tr = np.hstack([h_tr, t_tr])
    hy_va = np.hstack([h_va, t_va])
    hy_te = np.hstack([h_te, t_te])
    clf_h = train_xgboost(hy_tr, y_tr, hy_va, y_va, random_state=random_state)
    preds, probs = predict_xgboost(clf_h, hy_te)
    m_hy = compute_metrics(y_te, preds, probs)
    print("hybrid (internal+text)", m_hy, flush=True)
    return {"internal_only": m_int, "hybrid": m_hy}


def cross_dataset_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train on one feature table, score on another."""
    X_tr = as_numpy(X_train)
    y_tr = as_numpy(y_train).ravel().astype(np.int64)
    X_te = as_numpy(X_test)
    y_te = as_numpy(y_test).ravel().astype(np.int64)
    if X_val is None or y_val is None:
        X_a, X_va, y_a, y_va = train_test_split(
            X_tr,
            y_tr,
            test_size=0.15,
            random_state=random_state,
            stratify=y_tr,
        )
    else:
        X_a, y_a, X_va, y_va = (
            X_tr,
            y_tr,
            as_numpy(X_val),
            as_numpy(y_val).ravel().astype(np.int64),
        )
    clf = train_xgboost(X_a, y_a, X_va, y_va, random_state=random_state)
    preds, probs = predict_xgboost(clf, X_te)
    m = compute_metrics(y_te, preds, probs)
    print("cross-dataset / OOD test:", m, flush=True)
    return {"test_metrics": m, "clf": clf}


def plot_layer_auc_heatmap(
    layer_results: Mapping[int, Mapping[str, float]],
    ax: Optional[Axes] = None,
) -> Axes:
    """Single-row heatmap of layer ROC-AUC values."""
    layers = sorted(int(k) for k in layer_results.keys())
    row = []
    for i in layers:
        row.append(float(layer_results[i].get("roc_auc", float("nan"))))
    mat = np.array([row], dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.35 * len(layers)), 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks([0])
    ax.set_yticklabels(["ROC-AUC"])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(
        [str(x) for x in layers],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_xlabel("layer")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return ax
