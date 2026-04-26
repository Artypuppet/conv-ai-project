"""TruthfulQA / TriviaQA loading, model labeling, and split I/O."""

from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from nnsight import LanguageModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.preprocessing import compute_f1
from src.extraction.model_loader import generate_text_batch
from src.utils.config import Config
from src.utils.reproducibility import set_seed


def load_truthfulqa() -> Dataset:
    """TruthfulQA generation split used for the main run."""
    return load_dataset("truthful_qa", "generation", split="validation")


def load_triviaqa(n_samples: int = 500) -> Dataset:
    """Small TriviaQA slice, mostly for possible OOD checks."""
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=42)
    n = min(n_samples, len(ds))
    return ds.select(range(n))


def label_output(
    generated_answer: str,
    reference_answers: list,
    threshold: float = 0.3,
) -> int:
    """1 if the generated answer has weak F1 overlap with all refs."""
    if not reference_answers:
        return 1
    best = 0.0
    for ref in reference_answers:
        s = str(ref) if ref is not None else ""
        f = compute_f1(generated_answer, s)
        if f > best:
            best = f
    return 1 if best < threshold else 0


def _prompt_for_question(q: str) -> str:
    return f"Question: {q}\nAnswer:"


def _refs_from_row(ex: dict) -> list[str]:
    if "correct_answers" in ex and ex["correct_answers"] is not None:
        return [str(x) for x in ex["correct_answers"]]
    a = ex.get("answer")
    if isinstance(a, dict):
        val = a.get("value", "")
        al = a.get("aliases") or []
        return [str(val)] + [str(x) for x in al]
    return []


def build_dataset(
    lm: LanguageModel, qa_data, cfg: Config, resume: bool = True
) -> list[dict]:
    """Generate answers and attach the simple F1-overlap label.

    `qa_data` can be TruthfulQA or TriviaQA-style rows. We save a partial file
    every few examples because this step is slow on the laptop GPU.
    """
    set_seed(cfg.seed)
    n = len(qa_data)
    data_dir = cfg.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    partial_path = data_dir / "dataset_partial.pt"
    rows: list[dict] = []
    if resume and partial_path.exists():
        try:
            loaded = torch.load(
                partial_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception:
            # stale partial from an interrupted run; easiest to restart
            loaded = None
        if isinstance(loaded, list) and loaded:
            rows = loaded[:n]
            if len(rows) == n:
                print(
                    f"resume: {partial_path.name} already has {n} "
                    f"rows — nothing to do"
                )
                return rows
            if len(rows) < n:
                print(
                    f"resume: loaded {len(rows)}/{n} from "
                    f"{partial_path.name}, continuing"
                )
    start = len(rows)
    if start >= n:
        return rows
    gen_bs = 2
    pbar = tqdm(total=n, initial=start, desc="build_dataset", unit="ex")
    i = start
    while i < n:
        end = min(i + gen_bs, n)
        chunk = [qa_data[j] for j in range(i, end)]
        prompts = [_prompt_for_question(ex["question"]) for ex in chunk]
        anss = generate_text_batch(lm, prompts, max_new_tokens=128)
        for ex, ans in zip(chunk, anss):
            q = ex["question"]
            refs = _refs_from_row(ex)
            lab = label_output(ans, refs)
            rows.append(
                {
                    "question": q,
                    "answer": ans,
                    "references": refs,
                    "label": lab,
                }
            )
        pbar.update(len(chunk))
        if len(rows) % 10 == 0 and len(rows) > 0:
            torch.save(rows, partial_path)
            print(f"  saved partial: {len(rows)}/{n} -> {partial_path.name}")
        i = end
    pbar.close()
    torch.save(rows, partial_path)
    if n and len(rows) % 10 != 0:
        print(f"  saved final partial: {len(rows)}/{n} -> {partial_path.name}")
    return rows


def split_dataset(
    data: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """70/15/15 split; stratify unless a class is too tiny."""
    labels = [d["label"] for d in data]
    c0, c1 = sum(1 for y in labels if y == 0), sum(1 for y in labels if y == 1)
    strat1 = labels if c0 >= 2 and c1 >= 2 else None
    train, rest = train_test_split(
        data, test_size=0.3, random_state=seed, stratify=strat1
    )
    y_r = [d["label"] for d in rest]
    c0, c1 = sum(1 for y in y_r if y == 0), sum(1 for y in y_r if y == 1)
    strat2 = y_r if c0 >= 2 and c1 >= 2 else None
    val, test = train_test_split(
        rest, test_size=0.5, random_state=seed, stratify=strat2
    )
    return train, val, test


def save_splits(
    train: list[dict], val: list[dict], test: list[dict], data_dir: Path
) -> None:
    """Write train/val/test as .pt files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train, data_dir / "train.pt")
    torch.save(val, data_dir / "val.pt")
    torch.save(test, data_dir / "test.pt")


def load_splits(
    data_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load the three saved split files."""
    data_dir = Path(data_dir)
    _kw = {"map_location": "cpu", "weights_only": False}
    train = torch.load(data_dir / "train.pt", **_kw)
    val = torch.load(data_dir / "val.pt", **_kw)
    test = torch.load(data_dir / "test.pt", **_kw)
    return train, val, test


def print_label_stats(data: list[dict], name: str = "dataset") -> None:
    """Print class counts and percentages for a split."""
    n = len(data)
    if n == 0:
        print(f"{name}: empty")
        return
    n1 = sum(1 for d in data if d["label"] == 1)
    n0 = n - n1
    p0, p1 = 100.0 * n0 / n, 100.0 * n1 / n
    print(f"{name}: n={n} | 0: {n0} ({p0:.1f}%) | 1: {n1} ({p1:.1f}%)")
    if n0 / n > 0.8 or n1 / n > 0.8:
        print("  (note: strong class skew — may want class weighting later)")
