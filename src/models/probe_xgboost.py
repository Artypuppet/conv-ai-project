"""XGBoost on stacked internal features."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> XGBClassifier:
    """Fit XGBoost with validation-based early stopping."""
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return clf


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Try a small grid and keep the best validation ROC-AUC."""
    depth_grid = [3, 6, 9]
    tree_grid = [50, 100, 200]
    lr_grid = [0.01, 0.1, 0.3]
    best: Optional[XGBClassifier] = None
    best_auc = -1.0
    best_params: Dict[str, Any] = {}
    for depth in depth_grid:
        for n_trees in tree_grid:
            for lr in lr_grid:
                clf = XGBClassifier(
                    n_estimators=n_trees,
                    max_depth=depth,
                    learning_rate=lr,
                    eval_metric="logloss",
                    early_stopping_rounds=20,
                    random_state=random_state,
                    n_jobs=-1,
                )
                clf.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                proba = clf.predict_proba(X_val)[:, 1]
                try:
                    auc = float(roc_auc_score(y_val, proba))
                except ValueError:
                    auc = 0.5
                if auc > best_auc:
                    best_auc = auc
                    best = clf
                    best_params = {
                        "max_depth": depth,
                        "n_estimators": n_trees,
                        "learning_rate": lr,
                        "val_roc_auc": auc,
                    }
    assert best is not None
    return best, best_params


def predict_xgboost(
    model: XGBClassifier,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Class predictions and probability for class 1."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds.astype(np.int64), probs
