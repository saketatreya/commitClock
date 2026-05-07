import pickle
import torch
import numpy as np
from tqdm import tqdm

from src.config import PHASE3_OUT_DIR, PHASE4_OUT_DIR
from src.models.loader import load_model

def load_forced_branches():
    file_path = PHASE3_OUT_DIR / "forced_branches.pkl"
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def extract_pre_answer_activations(model, text: str, num_layers: int):
    """
    Extracts activations at the token right before the answer token.
    We assume the text ends with "So the answer is Yes." or "So the answer is No."
    We truncate before the "Yes" or "No".
    """
    idx = text.lower().rfind("so the answer is")
    if idx == -1:
        return None
        
    # Keep everything up to and including "so the answer is "
    text_before_ans = text[:idx + len("so the answer is ")]
    
    tokens = model.to_tokens(text_before_ans, prepend_bos=False)
    
    # We want the activation at the LAST token of this sequence.
    last_token_idx = tokens.shape[1] - 1
    
    layer_names = [f"blocks.{l}.hook_resid_post" for l in range(num_layers)]
    
    activations = torch.zeros(
        (num_layers, model.cfg.d_model),
        dtype=torch.float16,
        device='cpu'
    )
    
    def cache_hook(resid_post, hook):
        layer_idx = int(hook.name.split('.')[1])
        activations[layer_idx, :] = resid_post[0, last_token_idx, :].cpu().to(torch.float16)
        return resid_post

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            fwd_hooks=[(name, cache_hook) for name in layer_names]
        )
        
    return activations.numpy()

def run_phase4(limit=None):
    data = load_forced_branches()
    if not data:
        return
        
    if limit:
        data = data[:limit]
        
    model = load_model()
    num_layers = model.cfg.n_layers
    
    results = []
    checkpoint_path = PHASE4_OUT_DIR / "forced_activations.pkl"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"Resuming with {len(results)} previously processed examples.")
    processed_qids = {r['question_id'] for r in results}
    
    for item in tqdm(data, desc="Extracting Forced Branch Activations"):
        q_id = item['question_id']
        if q_id in processed_qids:
            continue
            
        correct_label = item['correct_label']
        
        act1 = extract_pre_answer_activations(model, item['cond1'], num_layers)
        act2 = extract_pre_answer_activations(model, item['cond2'], num_layers)
        act3 = extract_pre_answer_activations(model, item['cond3'], num_layers)
        act4 = extract_pre_answer_activations(model, item['cond4'], num_layers)
        
        if any(a is None for a in [act1, act2, act3, act4]):
            continue
            
        record = {
            "question_id": q_id,
            "correct_label": correct_label,
            "cond1_act": act1,
            "cond2_act": act2,
            "cond3_act": act3,
            "cond4_act": act4
        }
        results.append(record)
        
        if len(results) % 50 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)
        
    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Phase 4 complete. Extracted for {len(results)} questions.")

if __name__ == "__main__":
    run_phase4()
