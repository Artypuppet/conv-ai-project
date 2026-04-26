# Internal Representations for Hallucination Detection in LLMs

Conversational AI course project (Winter 2026). Explores whether hidden states, attention patterns, and token-level confidence from an LLM's internal layers can detect hallucinations before they reach the user.

## Objective

Design, implement, and evaluate a lightweight hallucination detector that uses internal representations of an open-source LLM. Specifically:

1. Generate/collect a dataset of hallucinated and non-hallucinated model outputs
2. Extract internal representations (hidden states, attention maps, logit dynamics) from an open-source LLM
3. Train lightweight classifiers (XGBoost, LapEigvals + LR, gated fusion) to detect hallucinations from these signals
4. Analyze which internal signals are most informative via ablation studies
5. Compare against text-only baselines

## Models


| Model                      | Params | Use case                                                           |
| -------------------------- | ------ | ------------------------------------------------------------------ |
| **Llama-3.2-1B** (primary) | 1.06B  | Main experiments. Fits on RTX 3050 4GB with 4-bit NF4 quantization |


## Datasets

- **TruthfulQA** -- questions designed to elicit common misconceptions

The generated split files `data/train.pt`, `data/val.pt`, and `data/test.pt` are committed. The final submission also includes the cached feature tensors and trained probe outputs under `outputs/`, so the notebook can render the main tables and plots without rerunning the slow extraction and training steps.

## Repository Structure

```
conv-ai-project/
├── README.md
├── requirements.txt
├── requirements-colab.txt
├── setup.sh
├── notebooks/
│   └── main.ipynb              # Primary deliverable (Colab notebook)
├── src/
│   ├── data/                   # Dataset construction and preprocessing
│   ├── extraction/             # Hidden state, attention, logit extraction
│   ├── models/                 # Probe classifiers and tuning utilities
│   ├── evaluation/             # Metrics, ablation framework
│   └── utils/                  # Config, reproducibility, feature helpers
├── data/                       # Reproducible train/val/test split files
├── outputs/                    # Cached features, checkpoints (gitignored)
├── references/                 # Research report and paper notes
└── literature_review.md        # Summary of related work
```

## Getting Started

### On Google Colab (for submission)

Open `notebooks/main.ipynb` in Colab. The first cell clones the repo and installs dependencies:

```python
git clone https://github.com/YOUR_USERNAME/conv-ai-project.git /content/conv-ai-project
%cd /content/conv-ai-project
%pip install -r requirements-colab.txt -q
```

Select a GPU runtime: **Runtime > Change runtime type > T4 GPU**.

The notebook starts with two flags:

```python
RUN_FROM_SCRATCH = False
SKIP_MODEL_REQUIRED_CELLS = True
```

For quick grading, keep these defaults. The notebook uses committed caches and skips the cells that require loading Llama. To fully reproduce the dataset generation, feature extraction, and live attention-map trace, set `RUN_FROM_SCRATCH = True` and `SKIP_MODEL_REQUIRED_CELLS = False`.

Full reproduction requires a Hugging Face token with access to `meta-llama/Llama-3.2-1B`:

```python
from huggingface_hub import login
login()
```

### Local development (for me it was RTX 3050 4GB on a laptop)

```bash
git clone https://github.com/YOUR_USERNAME/conversational-ai.git
cd conversational-ai/conv-ai-project
bash setup.sh
```

The code automatically detects the available GPU and applies 4-bit quantization. All extraction pipelines are designed to fit within 4GB VRAM.

## Hardware Requirements


| Environment | GPU             | VRAM | Notes                            |
| ----------- | --------------- | ---- | -------------------------------- |
| Local dev   | NVIDIA RTX 3050 | 4GB  | Uses 4-bit NF4 quantization      |
| Colab free  | NVIDIA T4       | 16GB | More headroom for larger batches |


## Reproducibility

All experiments use a fixed random seed (default: 42) pinned across `torch`, `numpy`, `random`, and `transformers`. Results are reproducible by running `notebooks/main.ipynb` end-to-end. By default, the notebook loads committed cached artifacts and prints when it is using caches. If `RUN_FROM_SCRATCH = True`, it regenerates the dataset, features, and classifier outputs through the project modules.

## Attribution

This project was completed as part of the Conversational AI course (Winter 2026). Code attribution is documented in function docstrings via `Origin:` tags -- each function is marked as either `Original` or `Adapted from [source]`.

## Project Guidelines

See [Project_Guidelines.docx](Project_Guidelines.docx) for full course requirements and [Project_Description.docx](Project_Description.docx) for the original project proposal.