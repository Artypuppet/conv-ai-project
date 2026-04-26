"""Save the summary figures used by the report notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


def val_test_colors() -> Tuple[Any, ...]:
    """First two swatches of the muted palette (validation vs test bars)."""
    return tuple(sns.color_palette("muted", 2))


def apply_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="muted",
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Helvetica Neue",
                "Arial",
                "sans-serif",
            ],
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linestyle": "-",
        }
    )


def add_metric_row(rows: List[dict], method: str, split: str, d: dict) -> None:
    row = d.copy()
    if "best_threshold" in row:
        del row["best_threshold"]
    row["method"] = method
    row["split"] = split
    rows.append(row)


def collect_rows(m: dict) -> pd.DataFrame:
    rows: List[dict] = []

    for sp in ("val", "test"):
        k = f"xgboost_{sp}"
        if k in m and isinstance(m[k], dict):
            add_metric_row(rows, "XGBoost", sp, m[k])
    if "lapeigvals_val_fit" in m:
        add_metric_row(rows, "LapEigvals + LR", "val", m["lapeigvals_val_fit"])
    if "lapeigvals_test" in m:
        add_metric_row(rows, "LapEigvals + LR", "test", m["lapeigvals_test"])
    for sp in ("val", "test"):
        k = f"gated_{sp}"
        if k in m and isinstance(m[k], dict):
            add_metric_row(rows, "Gated fusion", sp, m[k])
    if "length_baseline_val" in m:
        d = m["length_baseline_val"].copy()
        t = d.get("best_threshold")
        if t is None:
            label = "Length (word count)"
        else:
            label = f"Length (th={t:.2f})"
        add_metric_row(rows, label, "val", d)
    if "perplexity_baseline_val" in m:
        d = m["perplexity_baseline_val"].copy()
        t = d.get("best_threshold")
        plab = f"Perplexity (th={t:.2f})" if t is not None else "Perplexity"
        add_metric_row(rows, plab, "val", d)

    return pd.DataFrame(rows)


def main_models(df: pd.DataFrame) -> pd.DataFrame:
    keep = ("XGBoost", "LapEigvals + LR", "Gated fusion")
    return df[df["method"].isin(keep)].copy()


def melt_metrics(
    df: pd.DataFrame,
    metrics: List[str],
) -> pd.DataFrame:
    out: List[dict] = []
    for _, row in df.iterrows():
        for name in metrics:
            if name not in row or pd.isna(row[name]):
                continue
            out.append(
                {
                    "method": row["method"],
                    "split": row["split"],
                    "metric": name.replace("_", " ").title(),
                    "value": float(row[name]),
                }
            )
    return pd.DataFrame(out)


def save_report_figures(
    metrics: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Union[Path, str]] = None,
    json_path: Optional[Union[Path, str]] = None,
) -> List[Path]:
    """Build the validation/test plots and write PNG files."""
    if metrics is None:
        if json_path is None:
            raise ValueError("metrics or json_path required")
        p = Path(json_path)
        with open(p, encoding="utf-8") as f:
            metrics = json.load(f)

    out = Path(out_dir) if out_dir is not None else default_figures_dir()
    out.mkdir(parents=True, exist_ok=True)

    apply_theme()
    df = collect_rows(metrics)
    written: List[Path] = []
    if df.empty:
        return written

    main = main_models(df)
    metric_keys = [
        "accuracy",
        "f1",
        "roc_auc",
        "precision",
        "recall",
    ]
    long_main = melt_metrics(main, metric_keys)
    if not long_main.empty:
        pref = [
            "Accuracy",
            "F1",
            "Roc Auc",
            "Precision",
            "Recall",
        ]
        u = set(long_main["metric"].unique())
        col_order = [x for x in pref if x in u]
        col_order += [x for x in u if x not in pref]
        method_order = ["XGBoost", "LapEigvals + LR", "Gated fusion"]
        split_map = {"val": "Validation", "test": "Test"}
        lm = long_main.copy()
        lm["data_split"] = lm["split"].map(lambda s: split_map.get(s, s))
        n_m = len(col_order)
        ncols = 2
        nrows = (n_m + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.4 * ncols, 3.7 * nrows),
            facecolor="white",
            squeeze=False,
        )
        vt = val_test_colors()
        for idx, met in enumerate(col_order):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            subm = lm[lm["metric"] == met]
            if subm.empty:
                ax.set_visible(False)
                continue
            sns.barplot(
                data=subm,
                x="method",
                y="value",
                hue="data_split",
                order=method_order,
                hue_order=["Validation", "Test"],
                palette=list(vt),
                ax=ax,
            )
            ax.set_title(met, fontweight="600", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("Score" if c == 0 else "")
            ax.set_ylim(0, 1.02)
            for tick in ax.get_xticklabels():
                tick.set_rotation(18)
                tick.set_ha("right")
            leg_ax = ax.get_legend()
            if leg_ax is not None:
                leg_ax.remove()
        for j in range(n_m, nrows * ncols):
            r, c_ = divmod(j, ncols)
            axes[r][c_].set_visible(False)
        fig.legend(
            handles=[
                Patch(
                    facecolor=vt[0],
                    edgecolor="0.35",
                    label="Validation",
                ),
                Patch(
                    facecolor=vt[1],
                    edgecolor="0.35",
                    label="Test",
                ),
            ],
            loc="lower center",
            ncol=2,
            frameon=True,
            title="Data split",
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.suptitle(
            "Classifiers: validation vs test",
            y=0.995,
            fontweight="600",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 0.98))
        path = out / "classifier_metrics_validation_test_subplots"
        fig.savefig(path.with_suffix(".png"), facecolor="white")
        plt.close(fig)
        written.append(path.with_suffix(".png"))

    split_map = {"val": "Validation", "test": "Test"}
    vt2 = val_test_colors()
    for key, title in [
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]:
        sub = melt_metrics(main, [key])
        if sub.empty:
            continue
        sub = sub.copy()
        sub["data_split"] = sub["split"].map(lambda s: split_map.get(s, s))
        fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
        sns.barplot(
            data=sub,
            x="method",
            y="value",
            hue="data_split",
            order=["XGBoost", "LapEigvals + LR", "Gated fusion"],
            hue_order=["Validation", "Test"],
            palette=list(vt2),
            ax=ax,
        )
        ax.set_ylabel(title)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.02)
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
            tick.set_ha("right")
        ax.set_title(
            f"{title} — validation vs test",
            fontweight="600",
            pad=10,
        )
        h = ax.get_legend()
        if h is not None:
            h.set_title("Data split")
        path = out / f"metric_{key}_validation_test"
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"), facecolor="white")
        plt.close(fig)
        written.append(path.with_suffix(".png"))

    # test-set heatmap (main three)
    test = main[main["split"] == "test"]
    if not test.empty:
        mcols = []
        for c in metric_keys:
            if c in test.columns and test[c].notna().any():
                mcols.append(c)
        pvt: Optional[pd.DataFrame] = None
        if mcols:
            pvt = test.set_index("method")[mcols]
            pvt = pvt.reindex(
                [
                    x
                    for x in ["XGBoost", "LapEigvals + LR", "Gated fusion"]
                    if x in pvt.index
                ]
            )
            pvt = pvt.dropna(axis=1, how="all")
            pvt.columns = [c.replace("_", " ").title() for c in pvt.columns]
        if pvt is not None and not pvt.empty and not pvt.isna().all().all():
            h = max(2.2 + 0.45 * len(pvt), 2.5)
            w = max(0.9 * len(pvt.columns) + 3, 5.0)
            fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
            sns.heatmap(
                pvt,
                annot=True,
                fmt=".2f",
                cmap="crest",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Score", "shrink": 0.75},
                linewidths=0.5,
                linecolor="white",
                ax=ax,
            )
            ax.set_title(
                "Test split: models x metrics",
                fontweight="600",
                pad=10,
            )
            path = out / "test_split_metrics_heatmap"
            fig.tight_layout()
            fig.savefig(path.with_suffix(".png"), facecolor="white")
            plt.close(fig)
            written.append(path.with_suffix(".png"))

    # keep text baselines separate from the main classifier plot
    main_names = ("XGBoost", "LapEigvals + LR", "Gated fusion")
    bl = df[~df["method"].isin(main_names)]
    if not bl.empty:
        sub = melt_metrics(bl, metric_keys)
        if not sub.empty:
            n_hue = int(sub["method"].nunique())
            fig, ax = plt.subplots(figsize=(7.0, 3.6), facecolor="white")
            pal = sns.color_palette("muted", n_colors=max(n_hue, 3))[:n_hue]
            sns.barplot(
                data=sub,
                x="metric",
                y="value",
                hue="method",
                palette=pal,
                ax=ax,
            )
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("")
            ax.set_title("Text-only baselines (validation split)")
            hleg = ax.get_legend()
            if hleg is not None:
                hleg.set_frame_on(True)
            path = out / "text_baselines_validation_split"
            fig.tight_layout()
            fig.savefig(path.with_suffix(".png"), facecolor="white")
            plt.close(fig)
            written.append(path.with_suffix(".png"))

    return written


def default_figures_dir() -> Path:
    from src.utils.config import Config

    c = Config()
    d = c.output_dir / "figures" / "validation_test"
    return d


def load_metrics_json(path: Path) -> dict:
    """Load a metrics JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    import sys

    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    a = argparse.ArgumentParser(
        description="Build validation/test report figures from metrics JSON",
    )
    a.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=None,
        help="path to classifier_metrics.json",
    )
    a.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output directory (default: outputs/figures/validation_test)",
    )
    args = a.parse_args()
    p = args.json_path
    if p is None:
        p = default_figures_dir().parent.parent / "classifier_metrics.json"
    paths = save_report_figures(json_path=p, out_dir=args.out)
    for q in paths:
        print(q)
