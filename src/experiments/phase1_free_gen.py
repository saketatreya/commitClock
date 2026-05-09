import torch
import numpy as np
from tqdm import tqdm
import pickle
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME, PHASE1_OUT_DIR, MIN_CHAIN_LENGTH, MAX_CHAIN_LENGTH, NUM_FRACTIONAL_POSITIONS
from src.data.loader import load_strategyqa, format_strategyqa_prompt, parse_strategyqa_answer


def get_fractional_positions(chain_length: int, num_positions: int = NUM_FRACTIONAL_POSITIONS) -> List[int]:
    """Returns exactly `num_positions` indices spread evenly from 0 to chain_length - 1."""
    if chain_length == 0:
        return []
    return [int(round(i * (chain_length - 1) / (num_positions - 1))) for i in range(num_positions)]


def _generate(hf_model, tokenizer, dataset, batch_size: int = 16):
    """Step 1: batched HF generation with early stopping. Returns list of dicts."""
    generated_path = PHASE1_OUT_DIR / "generated_texts.pkl"
    if generated_path.exists():
        with open(generated_path, "rb") as f:
            generated_data = pickle.load(f)
        print(f"Loaded {len(generated_data)} previously generated texts.")
        return generated_data

    stop_strings = [
        "so the answer is Yes", "so the answer is No",
        "so the answer is yes", "so the answer is no",
    ]

    generated_data = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating in Batches"):
        batch = dataset[i:i + batch_size]
        prompts = [format_strategyqa_prompt(q) for q in batch['question']]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(hf_model.device)
        prompt_lens = inputs["attention_mask"].sum(dim=-1).tolist()  # un-padded prompt lens

        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=stop_strings,
                tokenizer=tokenizer,
                use_cache=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences  # [B, prompt_padded + new]
        # Decode only the newly generated portion (skipping the left-padded prompt) for clean text
        # but keep the full sequence for tokenization downstream.
        output_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for j in range(len(prompts)):
            model_answer = parse_strategyqa_answer(output_texts[j])
            correct_answer = 1 if batch['answer'][j] else 0
            # Save every record, even parse failures (model_answer == -1), so we can
            # inspect raw outputs if the regex misses. Extraction filters them out
            # later via the "so the answer is" rfind check.
            generated_data.append({
                "question_id": batch['qid'][j],
                "correct_label": correct_answer,
                "model_answer": model_answer,
                "generated_text": output_texts[j],
                "prompt": prompts[j],
                "prompt_len": int(prompt_lens[j]),
                "token_ids": generated_ids[j].cpu(),
            })

    n_parsed = sum(1 for g in generated_data if g['model_answer'] != -1)
    print(f"  Parse success: {n_parsed}/{len(generated_data)} "
          f"({n_parsed / max(1, len(generated_data)):.0%})")
    if n_parsed == 0 and generated_data:
        print("  WARNING: 0 outputs matched 'so the answer is yes/no'. Sample outputs:")
        for g in generated_data[:3]:
            print(f"  --- qid={g['question_id']} ---")
            print(f"    prompt    : {g['prompt']!r}")
            print(f"    generated : {g['generated_text'][:600]!r}")

    with open(generated_path, "wb") as f:
        pickle.dump(generated_data, f)
    return generated_data


def _prepare_extraction_items(generated_data, tokenizer):
    """For each generated record, locate the answer-token position in tokens, filter by chain length,
    and return a list of dicts ready for batched extraction."""
    items = []
    trigger_str = "so the answer is "
    for item in generated_data:
        text = item['generated_text']
        match_idx = text.lower().rfind(trigger_str)
        if match_idx == -1:
            continue
        # Tokenize text up to and including "so the answer is " — this is the prefix whose
        # final token position is the "answer-decision" point.
        text_before_answer = text[:match_idx + len(trigger_str)]
        # Tokenize the trimmed prefix exactly once.
        prefix_ids = tokenizer(text_before_answer, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        answer_token_idx = int(prefix_ids.shape[0])  # number of tokens in the prefix

        # Append the answer token (" Yes" or " No") so the forward pass sees the same context as before.
        ans_str = " Yes" if item['model_answer'] == 1 else " No"
        ans_ids = tokenizer(ans_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_ids = torch.cat([prefix_ids, ans_ids], dim=0)

        prompt_len = item['prompt_len']
        reasoning_length = answer_token_idx - prompt_len
        if reasoning_length < MIN_CHAIN_LENGTH or reasoning_length > MAX_CHAIN_LENGTH:
            continue

        fractional_offsets = get_fractional_positions(reasoning_length, NUM_FRACTIONAL_POSITIONS)
        unpadded_positions = [prompt_len + offset for offset in fractional_offsets]

        items.append({
            "item": item,
            "full_ids": full_ids,
            "unpadded_positions": unpadded_positions,
            "reasoning_length": reasoning_length,
        })
    return items


def _extract_activations(hf_model, tokenizer, generated_data, batch_size: int = 16):
    """Step 2: batched HF forward passes with output_hidden_states to extract residual-stream
    activations at the requested fractional positions."""
    checkpoint_path = PHASE1_OUT_DIR / "strategyqa_activations.pkl"
    results = []
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"Resuming with {len(results)} previously processed examples.")
    processed_qids = {r['question_id'] for r in results}

    pending = [g for g in generated_data if g['question_id'] not in processed_qids]
    prepared = _prepare_extraction_items(pending, tokenizer)
    print(f"Filtered to {len(prepared)} items in chain-length range "
          f"[{MIN_CHAIN_LENGTH}, {MAX_CHAIN_LENGTH}]; running batched extraction.")

    num_layers = hf_model.config.num_hidden_layers
    pad_id = tokenizer.pad_token_id
    device = hf_model.device

    for start in tqdm(range(0, len(prepared), batch_size), desc="Extracting Activations"):
        batch = prepared[start:start + batch_size]
        max_len = max(p['full_ids'].shape[0] for p in batch)
        B = len(batch)

        padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long)
        # Per-row padded positions to read activations at.
        padded_positions = []
        for b, p in enumerate(batch):
            seq = p['full_ids']
            seq_len = seq.shape[0]
            pad_offset = max_len - seq_len  # left-padding offset
            padded_ids[b, pad_offset:] = seq
            attn_mask[b, pad_offset:] = 1
            padded_positions.append([pos + pad_offset for pos in p['unpadded_positions']])

        padded_ids = padded_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            out = hf_model(
                input_ids=padded_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # hidden_states is a tuple of length num_layers + 1.
        # Index [l + 1] corresponds to TransformerLens's blocks.{l}.hook_resid_post.
        hs = out.hidden_states  # tuple of [B, T, D] tensors

        # Stack the layers we care about into a single tensor [num_layers, B, T, D] is too big;
        # instead, gather per-row to keep memory bounded.
        for b, p in enumerate(batch):
            positions_b = torch.tensor(padded_positions[b], device=device, dtype=torch.long)
            # acts_layers: list of [num_pos, D] per layer
            acts_layers = [hs[l + 1][b].index_select(0, positions_b) for l in range(num_layers)]
            acts = torch.stack(acts_layers, dim=1).contiguous()  # [num_pos, num_layers, D]
            activations_np = acts.to(torch.float16).cpu().numpy()

            item = p['item']
            results.append({
                "question_id": item['question_id'],
                "correct_label": item['correct_label'],
                "model_answer": item['model_answer'],
                "chain_length": p['reasoning_length'],
                "position_indices": p['unpadded_positions'],
                "activations": activations_np,
                "generated_text": item['generated_text'],
                "prompt": item['prompt'],
            })

        # Free per-batch GPU tensors before next iter
        del out, hs, padded_ids, attn_mask

        if len(results) % 500 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Phase 1 complete. Saved {len(results)} valid examples.")
    return results


def run_phase1_strategyqa(limit=None):
    dataset = load_strategyqa()
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print("\n--- Loading HF model (fp16, sdpa) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos_token=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).eval()

    print("\n--- Step 1: Fast Generation using Native HuggingFace ---")
    generated_data = _generate(hf_model, tokenizer, dataset)

    print("\n--- Step 2: Batched Activation Extraction (HF output_hidden_states) ---")
    _extract_activations(hf_model, tokenizer, generated_data)


if __name__ == "__main__":
    run_phase1_strategyqa(limit=10)
