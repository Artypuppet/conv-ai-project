"""A few metrics and plots used in the report notebook."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Standard binary classification metrics."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, average="binary", zero_division=0)
        ),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        prob = np.asarray(y_prob, dtype=np.float64).ravel()
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    return out


def print_metrics(m: Dict[str, Any], name: str = "") -> None:
    """Print metrics in a notebook-friendly way."""
    prefix = f"[{name}] " if name else ""
    for k, v in m.items():
        if isinstance(v, (float, np.floating)):
            print(f"{prefix}{k}: {float(v):.4f}")
        elif isinstance(v, (int, np.integer)):
            print(f"{prefix}{k}: {int(v)}")
        else:
            print(f"{prefix}{k}: {v}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    ax: Optional[Axes] = None,
) -> Axes:
    """ROC curve with a chance line."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=label or "model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    ax: Optional[Axes] = None,
) -> Axes:
    """Precision-recall curve."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    p, r, _ = precision_recall_curve(y_true, y_prob)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, label=label or "model")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
    ax: Optional[Axes] = None,
) -> Axes:
    """Simple confusion-matrix heatmap."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def results_table(
    all_results: Dict[str, Union[Dict[str, Any], Any]],
) -> pd.DataFrame:
    """Turn a dict of metric dicts into a small dataframe."""
    rows: List[Dict[str, Any]] = []
    for name, d in all_results.items():
        if not isinstance(d, dict):
            continue
        r = {"method": name}
        for k, v in d.items():
            if k != "method" and isinstance(
                v, (float, int, np.floating, np.integer, str, type(None))
            ):
                r[k] = v
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    extra = [c for c in df.columns if c != "method"]
    extra.sort()
    return df[["method"] + extra]
