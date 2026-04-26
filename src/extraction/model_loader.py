"""Load the LLM for nnsight and simple generation helpers."""

import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig

from src.utils.config import Config


def load_model(cfg: Config) -> LanguageModel:
    """Wrap the configured model with nnsight."""
    # "auto" often parks part of the model on CPU when VRAM is small, which
    # eats system RAM and looks like low GPU usage. The 1B model fits in 4GB.
    kwargs: dict = {
        "torch_dtype": torch.float16,
        "dispatch": True,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = "auto"
    if cfg.use_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.use_4bit:
        # bnb 4-bit wants a real GPU; fp16 on CPU only
        print("warning: use_4bit but no CUDA; using fp16 on CPU (no 4bit)")
    lm = LanguageModel(cfg.model_name, **kwargs)
    lm._model.eval()
    return lm


def generate_text(
    lm: LanguageModel,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """Generate only the continuation, stripping the prompt."""
    tok = lm.tokenizer
    m = lm._model
    dev = next(m.parameters()).device
    enc = tok(prompt, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(dev)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(dev)
    with torch.no_grad():
        out = m.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=False,
        )
    w = input_ids.shape[1]
    new_tok = out[:, w:]
    text = tok.decode(new_tok[0], skip_special_tokens=True)
    return text.strip()


def generate_text_batch(
    lm: LanguageModel,
    prompts: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    """Batched greedy decoding, one stripped completion per prompt."""
    if not prompts:
        return []
    tok = lm.tokenizer
    m = lm._model
    dev = next(m.parameters()).device
    enc = tok(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(dev)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(dev)
    w = input_ids.shape[1]
    with torch.no_grad():
        out = m.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=False,
        )
    new_tok = out[:, w:]
    return [
        tok.decode(new_tok[b], skip_special_tokens=True).strip()
        for b in range(new_tok.shape[0])
    ]
