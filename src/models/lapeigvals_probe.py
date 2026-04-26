"""Logistic regression on PCA of stacked attention spectrum features."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def build_lapeigvals_features(
    attn_features: Union[Dict[int, np.ndarray], Any],
    layers: Optional[List[int]] = None,
) -> np.ndarray:
    """Flatten per-layer attention features into one design matrix."""
    by_layer: Dict[int, np.ndarray]
    if isinstance(attn_features, dict) and "by_layer" in attn_features:
        by_layer = attn_features["by_layer"]
    else:
        by_layer = attn_features  # type: ignore[assignment]
    if layers is None:
        layers = sorted(by_layer.keys())
    parts = []
    for i in layers:
        arr = np.asarray(by_layer[i])
        if arr.ndim >= 3:
            arr = arr.reshape(arr.shape[0], -1)
        parts.append(arr)
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_lapeigvals_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_components: int = 512,
    random_state: int = 42,
) -> tuple[Pipeline, Dict[str, float]]:
    """Fit PCA plus balanced logistic regression."""
    n_features = int(X_train.shape[1])
    n_samples = int(X_train.shape[0])
    pca_dims = int(min(n_components, n_samples, n_features))
    steps = [
        (
            "pca",
            PCA(
                n_components=pca_dims,
                svd_solver="auto",
                random_state=random_state,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    ]
    pipe: Pipeline = Pipeline(steps)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    proba = pipe.predict_proba(X_val)[:, 1]
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "precision": float(
            precision_score(y_val, pred, average="binary", zero_division=0),
        ),
        "recall": float(
            recall_score(y_val, pred, average="binary", zero_division=0),
        ),
        "f1": float(f1_score(y_val, pred, average="binary", zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_val, proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return pipe, metrics
