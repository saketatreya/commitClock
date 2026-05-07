import torch
from transformer_lens import HookedTransformer
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
    # Using bfloat16 to save memory, default for many Qwen models
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=torch.float16 if device == "mps" else torch.bfloat16
    )
    return model
