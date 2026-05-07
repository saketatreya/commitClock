import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
import gc
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME, PHASE1_OUT_DIR, MIN_CHAIN_LENGTH, MAX_CHAIN_LENGTH, NUM_FRACTIONAL_POSITIONS
from src.data.loader import load_strategyqa, format_strategyqa_prompt, parse_strategyqa_answer
from src.models.loader import load_model

def get_fractional_positions(chain_length: int, num_positions: int = NUM_FRACTIONAL_POSITIONS) -> List[int]:
    """Returns exactly `num_positions` indices spread evenly from 0 to chain_length - 1."""
    if chain_length == 0:
        return []
    return [int(round(i * (chain_length - 1) / (num_positions - 1))) for i in range(num_positions)]

def extract_activations_for_prompt(model, prompt_tokens, answer_token_idx, num_layers):
    prompt_len = prompt_tokens.shape[1]
    reasoning_length = answer_token_idx - prompt_len
    
    if reasoning_length < MIN_CHAIN_LENGTH or reasoning_length > MAX_CHAIN_LENGTH:
        return None, None
        
    fractional_offsets = get_fractional_positions(reasoning_length, NUM_FRACTIONAL_POSITIONS)
    absolute_positions = [prompt_len + offset for offset in fractional_offsets]
    
    layer_names = [f"blocks.{l}.hook_resid_post" for l in range(num_layers)]
    
    activations = torch.zeros(
        (NUM_FRACTIONAL_POSITIONS, num_layers, model.cfg.d_model),
        dtype=torch.float16,
        device='cpu'
    )
    
    def cache_hook(resid_post, hook):
        layer_idx = int(hook.name.split('.')[1])
        for i, pos in enumerate(absolute_positions):
            activations[i, layer_idx, :] = resid_post[0, pos, :].cpu().to(torch.float16)
        return resid_post

    with torch.no_grad():
        model.run_with_hooks(
            prompt_tokens,
            fwd_hooks=[(name, cache_hook) for name in layer_names]
        )
        
    return absolute_positions, activations

def generate_texts_fast(dataset, limit=None):
    """Uses native HuggingFace for much faster multi-GPU autoregressive generation."""
    print("\n--- Step 1: Fast Generation using Native HuggingFace ---")
    generated_path = PHASE1_OUT_DIR / "generated_texts.pkl"
    if generated_path.exists():
        with open(generated_path, "rb") as f:
            generated_data = pickle.load(f)
        print(f"Loaded {len(generated_data)} previously generated texts.")
        return generated_data
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos_token=False)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    generated_data = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Generating")):
        question = item['question']
        correct_answer = 1 if item['answer'] else 0
        prompt = format_strategyqa_prompt(question)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = parse_strategyqa_answer(output_text)
        
        if model_answer != -1:
            generated_data.append({
                "question_id": item['qid'],
                "correct_label": correct_answer,
                "model_answer": model_answer,
                "generated_text": output_text,
                "prompt": prompt
            })
            
    with open(generated_path, "wb") as f:
        pickle.dump(generated_data, f)
        
    # Free up memory before loading HookedTransformer
    del hf_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return generated_data

def run_phase1_strategyqa(limit=None):
    dataset = load_strategyqa()
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        
    generated_data = generate_texts_fast(dataset, limit=limit)
    
    print("\n--- Step 2: Activation Extraction using TransformerLens ---")
    model = load_model()
    num_layers = model.cfg.n_layers
    
    results = []
    checkpoint_path = PHASE1_OUT_DIR / "strategyqa_activations.pkl"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"Resuming with {len(results)} previously processed examples.")
        
    processed_qids = {r['question_id'] for r in results}
    
    for item in tqdm(generated_data, desc="Extracting Activations"):
        if item['question_id'] in processed_qids:
            continue
            
        output_text = item['generated_text']
        model_answer = item['model_answer']
        prompt = item['prompt']
        
        ans_str = "Yes" if model_answer == 1 else "No"
        trigger_str = f"so the answer is {ans_str}"
        
        match_idx = output_text.lower().rfind(trigger_str)
        if match_idx == -1:
            continue
            
        text_before_answer = output_text[:match_idx + len("so the answer is ")]
        tokens_before = model.to_tokens(text_before_answer, prepend_bos=False)[0]
        answer_token_idx = len(tokens_before)
        
        prompt_tokens = model.to_tokens(prompt, prepend_bos=False)
        full_tokens = model.to_tokens(text_before_answer + (" Yes" if model_answer == 1 else " No"), prepend_bos=False)
        
        positions, activations = extract_activations_for_prompt(
            model, full_tokens, answer_token_idx, num_layers
        )
        
        if positions is None:
            continue
            
        record = {
            "question_id": item['question_id'],
            "correct_label": item['correct_label'],
            "model_answer": model_answer,
            "chain_length": answer_token_idx - prompt_tokens.shape[1],
            "position_indices": positions,
            "activations": activations.numpy(),
            "generated_text": output_text,
            "prompt": prompt
        }
        results.append(record)
        
        if len(results) % 50 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)
                
    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Phase 1 complete. Saved {len(results)} valid examples.")

if __name__ == "__main__":
    run_phase1_strategyqa(limit=10)
