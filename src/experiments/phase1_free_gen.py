import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
from typing import List, Dict, Any

from src.config import MODEL_NAME, PHASE1_OUT_DIR, MIN_CHAIN_LENGTH, MAX_CHAIN_LENGTH, NUM_FRACTIONAL_POSITIONS
from src.data.loader import load_strategyqa, format_strategyqa_prompt, parse_strategyqa_answer
from src.models.loader import load_model

def get_fractional_positions(chain_length: int, num_positions: int = NUM_FRACTIONAL_POSITIONS) -> List[int]:
    """Returns exactly `num_positions` indices spread evenly from 0 to chain_length - 1."""
    if chain_length == 0:
        return []
    # e.g., chain_length=100 -> 0, 11, 22, 33, 44, 55, 66, 77, 88, 99
    return [int(round(i * (chain_length - 1) / (num_positions - 1))) for i in range(num_positions)]

def extract_activations_for_prompt(model, prompt_tokens, answer_token_idx, num_layers):
    """
    Runs forward pass and extracts activations at 10 fractional positions
    between the end of the prompt and the answer token.
    """
    # Number of reasoning tokens
    prompt_len = prompt_tokens.shape[1]
    reasoning_length = answer_token_idx - prompt_len
    
    if reasoning_length < MIN_CHAIN_LENGTH or reasoning_length > MAX_CHAIN_LENGTH:
        return None, None
        
    fractional_offsets = get_fractional_positions(reasoning_length, NUM_FRACTIONAL_POSITIONS)
    absolute_positions = [prompt_len + offset for offset in fractional_offsets]
    
    # We want resid_post for all layers
    layer_names = [f"blocks.{l}.hook_resid_post" for l in range(num_layers)]
    
    activations = torch.zeros(
        (NUM_FRACTIONAL_POSITIONS, num_layers, model.cfg.d_model),
        dtype=torch.float16,
        device='cpu' # store on CPU to save VRAM
    )
    
    def cache_hook(resid_post, hook):
        layer_idx = int(hook.name.split('.')[1])
        # resid_post is [batch, pos, d_model]
        for i, pos in enumerate(absolute_positions):
            activations[i, layer_idx, :] = resid_post[0, pos, :].cpu().to(torch.float16)
        return resid_post

    with torch.no_grad():
        model.run_with_hooks(
            prompt_tokens,
            fwd_hooks=[(name, cache_hook) for name in layer_names]
        )
        
    return absolute_positions, activations

def run_phase1_strategyqa(limit=None):
    dataset = load_strategyqa()
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        
    model = load_model()
    num_layers = model.cfg.n_layers
    
    results = []
    
    # For StrategyQA, ground truth is boolean
    for idx, item in enumerate(tqdm(dataset, desc="Phase 1: Free Generation")):
        question = item['question']
        correct_answer = 1 if item['answer'] else 0
        
        prompt = format_strategyqa_prompt(question)
        
        # Free generation
        # We need to find "So the answer is Yes." or similar.
        # Generate with a reasonable max_new_tokens
        with torch.no_grad():
            output_text = model.generate(
                prompt,
                max_new_tokens=300,
                temperature=0.0,
                verbose=False,
                stop_at_eos=True
            )
            
        model_answer = parse_strategyqa_answer(output_text)
        
        if model_answer == -1:
            # Model didn't answer cleanly
            continue
            
        # Find the token index where the answer starts.
        # This is a bit tricky: we look for "So the answer is" in the tokenized output
        # Let's just find the string index and map it to token index.
        # A simpler way: parse_strategyqa_answer already matched. Let's find the position.
        
        # We will tokenize "So the answer is Yes" or "So the answer is No" and find it in the sequence.
        # Or better, just find the string offset and use `model.to_tokens(..., return_offsets_mapping=True)`
        # But we can approximate by finding the token sequence for "So the answer is"
        
        # A robust way:
        ans_str = "Yes" if model_answer == 1 else "No"
        trigger_str = f"so the answer is {ans_str}"
        
        # Simple substring search in text
        match_idx = output_text.lower().rfind(trigger_str)
        if match_idx == -1:
            continue
            
        # Convert string index to token index
        # To do this safely, we tokenize the substring up to the answer.
        text_before_answer = output_text[:match_idx + len("so the answer is ")]
        tokens_before = model.to_tokens(text_before_answer, prepend_bos=False)[0]
        answer_token_idx = len(tokens_before)
        
        # Now we extract activations
        prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
        # Re-run forward pass with the *full* generated sequence up to the answer token
        
        full_tokens = model.to_tokens(text_before_answer + (" Yes" if model_answer == 1 else " No"), prepend_bos=True)
        
        positions, activations = extract_activations_for_prompt(
            model, full_tokens, answer_token_idx, num_layers
        )
        
        if positions is None:
            # Filtered out by chain length
            continue
            
        record = {
            "question_id": item['qid'],
            "correct_label": correct_answer,
            "model_answer": model_answer,
            "chain_length": answer_token_idx - prompt_tokens.shape[1],
            "position_indices": positions,
            "activations": activations.numpy(), # Convert to numpy for saving
            "generated_text": output_text,
            "prompt": prompt
        }
        results.append(record)
        
        # Save checkpoints
        if len(results) % 50 == 0:
            with open(PHASE1_OUT_DIR / "strategyqa_activations.pkl", "wb") as f:
                pickle.dump(results, f)
                
    # Final save
    with open(PHASE1_OUT_DIR / "strategyqa_activations.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"Phase 1 complete. Saved {len(results)} valid examples.")

if __name__ == "__main__":
    # Run a small test first
    run_phase1_strategyqa(limit=10)
