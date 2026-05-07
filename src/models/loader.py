import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from src.config import MODEL_NAME

def load_model(model_name: str = MODEL_NAME, device: str = None) -> HookedTransformer:
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Loading {model_name} on {device}...")
    
    # Initialize tokenizer explicitly to avoid the missing BOS token error in Qwen models
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False)
    
    # Using bfloat16 to save memory, default for many Qwen models
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        tokenizer=tokenizer,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        default_prepend_bos=False,
        dtype=torch.float16 if device == "mps" else torch.bfloat16
    )
    return model
