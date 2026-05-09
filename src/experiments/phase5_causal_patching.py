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


def _pre_answer_text(text: str):
    idx = text.lower().rfind("so the answer is")
    if idx == -1:
        return None
    return text[:idx + len("so the answer is ")]


def _resolve_target_token_id(model, correct_label: int):
    ans_str = " Yes" if correct_label == 1 else " No"
    tid = model.to_single_token(ans_str)
    if tid is None:
        ans_str = "Yes" if correct_label == 1 else "No"
        tid = model.to_single_token(ans_str)
    return tid


def _left_pad_batch(tokenizer, texts, device):
    """Tokenize a list of strings with left-padding. Returns (input_ids[B,T], last_idx[B])."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    T = input_ids.shape[1]
    # With left-padding, the last non-pad position is always T - 1.
    last_idx = torch.full((input_ids.shape[0],), T - 1, dtype=torch.long, device=device)
    return input_ids, attn, last_idx


def run_phase5(limit=None, batch_size: int = 8):
    data = load_forced_branches()
    if not data:
        return

    if limit:
        data = data[:limit]

    model = load_model()
    num_layers = model.cfg.n_layers
    device = model.cfg.device

    # Use the model's tokenizer with left-padding for batched extraction.
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    checkpoint_path = PHASE5_OUT_DIR / "causal_patching.pkl"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"Resuming with {len(results)} previously processed examples.")
    processed_qids = {r['question_id'] for r in results}

    # Pre-filter: must have valid trimmed cond1, cond3, and a resolvable target token.
    pending = []
    for item in data:
        q_id = item['question_id']
        if q_id in processed_qids:
            continue
        c1 = _pre_answer_text(item['cond1'])
        c3 = _pre_answer_text(item['cond3'])
        if c1 is None or c3 is None:
            continue
        tgt = _resolve_target_token_id(model, item['correct_label'])
        if tgt is None:
            continue
        pending.append({
            "q_id": q_id,
            "correct_label": item['correct_label'],
            "cond1_text": c1,
            "cond3_text": c3,
            "target_token_id": tgt,
        })

    print(f"Phase 5: {len(pending)} questions × {num_layers} layers; running in batches of {batch_size}.")

    for start in tqdm(range(0, len(pending), batch_size), desc="Causal Patching"):
        batch = pending[start:start + batch_size]
        B = len(batch)

        cond1_ids, cond1_attn, last_idx_1 = _left_pad_batch(
            tokenizer, [b['cond1_text'] for b in batch], device
        )
        cond3_ids, cond3_attn, last_idx_3 = _left_pad_batch(
            tokenizer, [b['cond3_text'] for b in batch], device
        )
        target_ids = torch.tensor([b['target_token_id'] for b in batch], device=device, dtype=torch.long)
        row_idx = torch.arange(B, device=device)

        # 1) Clean run on cond1 — need log-prob of target at last_idx_1 per row.
        with torch.no_grad():
            clean_logits = model(cond1_ids, attention_mask=cond1_attn)
            clean_lp_full = torch.nn.functional.log_softmax(
                clean_logits[row_idx, last_idx_1, :], dim=-1
            )
            clean_target_lp = clean_lp_full[row_idx, target_ids]  # [B]

        # 2) Cache cond3 across all resid_post layers in one batched forward.
        with torch.no_grad():
            _, cache_c3 = model.run_with_cache(
                cond3_ids,
                attention_mask=cond3_attn,
                names_filter=lambda n: "resid_post" in n,
            )

        # 3) For each layer, batched patched forward.
        per_layer_effect = torch.zeros((B, num_layers), dtype=torch.float32)
        for l in range(num_layers):
            h_C3_l = cache_c3[f"blocks.{l}.hook_resid_post"][row_idx, last_idx_3, :].detach()  # [B, D]

            def patch_hook(resid_post, hook, _h=h_C3_l, _idx=last_idx_1):
                resid_post[row_idx, _idx, :] = _h
                return resid_post

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    cond1_ids,
                    attention_mask=cond1_attn,
                    fwd_hooks=[(f"blocks.{l}.hook_resid_post", patch_hook)],
                )
                patched_lp_full = torch.nn.functional.log_softmax(
                    patched_logits[row_idx, last_idx_1, :], dim=-1
                )
                patched_target_lp = patched_lp_full[row_idx, target_ids]  # [B]
            per_layer_effect[:, l] = (patched_target_lp - clean_target_lp).cpu()

        del cache_c3, clean_logits

        for b, item in enumerate(batch):
            results.append({
                "question_id": item['q_id'],
                "effects": per_layer_effect[b].tolist(),
            })

        if len(results) % 50 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)

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
