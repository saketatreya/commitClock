import gc
import pickle
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import PHASE3_OUT_DIR, PHASE5_OUT_DIR
from src.models.loader import load_model
from src.data.loader import ANSWER_TRIGGER_RE


def load_forced_branches():
    file_path = PHASE3_OUT_DIR / "forced_branches.pkl"
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def _pre_answer_text(text: str):
    matches = list(ANSWER_TRIGGER_RE.finditer(text))
    if not matches:
        return None
    return text[:matches[-1].start(1)]


def _resolve_target_token_id(model, correct_label: int):
    ans_str = " Yes" if correct_label == 1 else " No"
    tid = model.to_single_token(ans_str)
    if tid is None:
        ans_str = "Yes" if correct_label == 1 else "No"
        tid = model.to_single_token(ans_str)
    return tid


def _flush(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def run_phase5(limit=None):
    """Causal patching, one question at a time. Original Phase 5 was OOM-killed
    on Kaggle 2x T4 even at batch_size=4 — the killer was system RAM during TL's
    forward path with padding + attention_mask, not GPU memory. With B=1 there
    is no padding and no attention_mask handling, which matches the original
    pre-rewrite code's memory profile while still benefiting from fp16 (1.5-2x
    over bf16 emulated on Turing). The cache-cond3-once optimization across
    layers is preserved (1 + L forwards per question, not N*L)."""
    data = load_forced_branches()
    if not data:
        return

    if limit:
        data = data[:limit]

    _flush("Phase 5: loading TransformerLens model...")
    model = load_model()
    num_layers = model.cfg.n_layers
    _flush(f"Phase 5: TL loaded. n_layers={num_layers} d_model={model.cfg.d_model}")

    results = []
    checkpoint_path = PHASE5_OUT_DIR / "causal_patching.pkl"
    if checkpoint_path.exists():
        _flush(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        _flush(f"Resuming with {len(results)} previously processed examples.")
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

    _flush(f"Phase 5: {len(pending)} questions x {num_layers} layers, B=1 (one question per iter)")

    for q_idx, item in enumerate(tqdm(pending, desc="Causal Patching")):
        # Tokenize each condition independently — no padding, no attention_mask drama.
        cond1_tokens = model.to_tokens(item['cond1_text'], prepend_bos=False)  # [1, T1]
        cond3_tokens = model.to_tokens(item['cond3_text'], prepend_bos=False)  # [1, T3]
        last_idx_1 = cond1_tokens.shape[1] - 1
        last_idx_3 = cond3_tokens.shape[1] - 1
        target_token_id = item['target_token_id']

        # 1) Clean run on cond1 to get baseline log-prob of correct answer.
        with torch.no_grad():
            clean_logits = model(cond1_tokens)
            clean_log_probs = torch.nn.functional.log_softmax(
                clean_logits[0, last_idx_1, :], dim=0
            )
            clean_target_lp = clean_log_probs[target_token_id].item()
        del clean_logits, clean_log_probs

        # 2) Cache cond3 resid_post across all layers in one forward.
        with torch.no_grad():
            _, cache_c3 = model.run_with_cache(
                cond3_tokens,
                names_filter=lambda n: "resid_post" in n,
            )

        # 3) Per layer: patch h_C1[last_idx_1] := h_C3[last_idx_3] and re-run.
        layer_effects = []
        for l in range(num_layers):
            h_C3_l = cache_c3[f"blocks.{l}.hook_resid_post"][0, last_idx_3, :].clone()

            def patch_hook(resid_post, hook, _h=h_C3_l):
                resid_post[0, last_idx_1, :] = _h
                return resid_post

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    cond1_tokens,
                    fwd_hooks=[(f"blocks.{l}.hook_resid_post", patch_hook)],
                )
                patched_log_probs = torch.nn.functional.log_softmax(
                    patched_logits[0, last_idx_1, :], dim=0
                )
                patched_target_lp = patched_log_probs[target_token_id].item()
            layer_effects.append(patched_target_lp - clean_target_lp)
            del patched_logits, patched_log_probs

        del cache_c3, cond1_tokens, cond3_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        results.append({
            "question_id": item['q_id'],
            "effects": layer_effects,
        })

        # Save checkpoint frequently so an OOM kill doesn't lose all progress.
        if (q_idx + 1) % 25 == 0 or q_idx == len(pending) - 1:
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

    _flush(f"Phase 5 complete. Evaluated {len(results)} questions.")


if __name__ == "__main__":
    run_phase5()
