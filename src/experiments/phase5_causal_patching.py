import pickle
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import PHASE3_OUT_DIR, PHASE5_OUT_DIR
from src.models.loader import load_model

def load_forced_branches():
    file_path = PHASE3_OUT_DIR / "forced_branches.pkl"
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def get_pre_answer_tokens(model, text: str):
    idx = text.lower().rfind("so the answer is")
    if idx == -1:
        return None
    text_before_ans = text[:idx + len("so the answer is ")]
    return model.to_tokens(text_before_ans, prepend_bos=False)

def run_phase5(limit=None):
    data = load_forced_branches()
    if not data:
        return
        
    if limit:
        data = data[:limit]
        
    model = load_model()
    num_layers = model.cfg.n_layers
    
    # We will compute the causal effect per layer across all valid questions
    # Causal effect: log_prob(correct_answer | patched) - log_prob(correct_answer | clean cond1)
    
    results = []
    checkpoint_path = PHASE5_OUT_DIR / "causal_patching.pkl"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"Resuming with {len(results)} previously processed examples.")
    processed_qids = {r['question_id'] for r in results}
    
    for item in tqdm(data, desc="Causal Patching"):
        q_id = item['question_id']
        if q_id in processed_qids:
            continue
            
        correct_label = item['correct_label']
        
        cond1_tokens = get_pre_answer_tokens(model, item['cond1'])
        cond3_tokens = get_pre_answer_tokens(model, item['cond3'])
        
        if cond1_tokens is None or cond3_tokens is None:
            continue
            
        last_idx_1 = cond1_tokens.shape[1] - 1
        last_idx_3 = cond3_tokens.shape[1] - 1
        
        # Get target token ID
        ans_str = " Yes" if correct_label == 1 else " No"
        target_token_id = model.to_single_token(ans_str)
        if target_token_id is None: # fallback
            # Try without space
            ans_str = "Yes" if correct_label == 1 else "No"
            target_token_id = model.to_single_token(ans_str)
            
        if target_token_id is None:
            continue
            
        # Clean run Cond1
        with torch.no_grad():
            clean_logits = model(cond1_tokens)
            clean_log_probs = torch.nn.functional.log_softmax(clean_logits[0, last_idx_1, :], dim=0)
            clean_target_lp = clean_log_probs[target_token_id].item()
            
        # We need h_C3 for all layers. Let's cache them.
        _, cache_c3 = model.run_with_cache(cond3_tokens, names_filter=lambda n: "resid_post" in n)
        
        layer_effects = []
        
        for l in range(num_layers):
            h_C3_l = cache_c3[f"blocks.{l}.hook_resid_post"][0, last_idx_3, :].clone()
            
            def patch_hook(resid_post, hook):
                resid_post[0, last_idx_1, :] = h_C3_l
                return resid_post
                
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    cond1_tokens,
                    fwd_hooks=[(f"blocks.{l}.hook_resid_post", patch_hook)]
                )
                patched_log_probs = torch.nn.functional.log_softmax(patched_logits[0, last_idx_1, :], dim=0)
                patched_target_lp = patched_log_probs[target_token_id].item()
                
            effect = patched_target_lp - clean_target_lp
            layer_effects.append(effect)
            
        results.append({
            "question_id": q_id,
            "effects": layer_effects
        })
        
        if len(results) % 50 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)
        
    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)
        
    # Plot average effect
    if results:
        avg_effects = np.mean([r['effects'] for r in results], axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_layers), avg_effects, marker='o')
        plt.title("Causal Patching: Effect of Wrong Reasoning (Condition 3 -> Condition 1)")
        plt.xlabel("Layer")
        plt.ylabel("Change in Correct Answer Log-Probability")
        plt.axhline(0, color='r', linestyle='--')
        plt.grid(True)
        plt.savefig(PHASE5_OUT_DIR / "causal_effect.png")
        plt.close()
        
    print(f"Phase 5 complete. Evaluated {len(results)} questions.")

if __name__ == "__main__":
    run_phase5()
