import gc
import sys

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME


def _stderr(msg):
    print(msg, file=sys.stderr, flush=True)


def load_model(model_name: str = MODEL_NAME, device: str = None) -> HookedTransformer:
    """Load a HookedTransformer.

    The naive `HookedTransformer.from_pretrained(model_name, dtype=fp16)` path
    OOMs on Kaggle 2x T4 (~33 GB system RAM) for Qwen2.5-7B because TL holds
    the full bf16 state_dict from disk AND a fp16 copy in memory during the
    conversion — peak ~30 GB CPU RAM, plus baseline Python state. We work
    around this by pre-loading the HF model with `torch_dtype=fp16` and
    `low_cpu_mem_usage=True` so weights stream from disk directly into fp16
    tensors (no transient bf16 + fp16 coexistence), then handing the result
    to TL via the `hf_model=` parameter so TL skips its own load entirely.
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"Loading {model_name} on {device}...")

    n_devices = torch.cuda.device_count() if device == "cuda" else 1
    if n_devices > 1:
        print(f"Detected {n_devices} GPUs! Splitting the model across devices to save VRAM.")

    # Free any leftover state from prior phases (HF models from Phase 1/4,
    # tokenizers, datasets) so TL's load has maximum headroom.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False)

    # Pre-load HF in fp16 with low_cpu_mem_usage. This avoids TL's CPU RAM
    # blow-up: HF streams each tensor from disk straight into fp16, never
    # holding the full bf16 state_dict simultaneously.
    _stderr("[loader] Pre-loading HF backbone (fp16, low_cpu_mem_usage)...")
    try:
        hf_backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        _stderr("[loader] HF backbone loaded; handing off to TransformerLens.")
        # TL builds its wrapper from the pre-loaded HF model. It still
        # allocates TL-format weights, but it does so layer-by-layer from
        # the already-fp16 HF tensors instead of materializing a fresh
        # state_dict from disk.
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_backbone,
            device=device,
            n_devices=n_devices,
            tokenizer=tokenizer,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            default_prepend_bos=False,
            dtype=torch.float16,
        )
        # Drop our reference to the HF backbone; TL has copied what it needs.
        del hf_backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except TypeError as e:
        # Older TL versions may not accept hf_model= kwarg. Fall back to the
        # plain path with the dtype kwarg forwarded to HF underneath.
        _stderr(f"[loader] hf_model= path failed ({e}); falling back to plain TL load.")
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            n_devices=n_devices,
            tokenizer=tokenizer,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            default_prepend_bos=False,
            dtype=torch.float16,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    return model
