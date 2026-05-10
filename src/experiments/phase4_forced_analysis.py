import gc
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME, PHASE3_OUT_DIR, PHASE4_OUT_DIR
from src.data.loader import ANSWER_TRIGGER_RE


def _free_model_and_report(model, tag=""):
    """See phase1_free_gen for rationale."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  [free/{tag}] before  cuda:{i} alloc="
                  f"{torch.cuda.memory_allocated(i)/1e9:.2f}GB  "
                  f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            print(f"  [free/{tag}] after   cuda:{i} alloc="
                  f"{torch.cuda.memory_allocated(i)/1e9:.2f}GB  "
                  f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB", flush=True)


def load_forced_branches():
    file_path = PHASE3_OUT_DIR / "forced_branches.pkl"
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def _trim_to_pre_answer(text: str):
    """Returns the prefix up to (but not including) the 'yes'/'no' answer token,
    matching any accepted trigger variant. Returns None if no trigger found."""
    matches = list(ANSWER_TRIGGER_RE.finditer(text))
    if not matches:
        return None
    return text[:matches[-1].start(1)]


def _extract_last_token_acts_batched(hf_model, tokenizer, texts, num_layers):
    """Tokenize `texts` (list[str]) with left padding, do one forward pass with output_hidden_states,
    and return a list of [num_layers, d_model] fp16 numpy arrays — one per text — taken at each
    sequence's final non-pad token."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(hf_model.device)
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]
    # last non-pad position per row (works for both left- and right-padding because we sum 1s)
    # for left-padding the last non-pad is simply T - 1.
    T = input_ids.shape[1]
    last_idx = torch.full((input_ids.shape[0],), T - 1, dtype=torch.long, device=hf_model.device)

    with torch.no_grad():
        out = hf_model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hs = out.hidden_states  # tuple of length num_layers + 1; [l+1] = resid_post of layer l

    results_np = []
    for b in range(input_ids.shape[0]):
        # Stack across layers at last_idx[b] -> [num_layers, D]
        acts = torch.stack([hs[l + 1][b, last_idx[b], :] for l in range(num_layers)], dim=0)
        results_np.append(acts.to(torch.float16).cpu().numpy())
    del out, hs
    return results_np


def run_phase4(limit=None, batch_size: int = 8):
    data = load_forced_branches()
    if not data:
        return

    if limit:
        data = data[:limit]

    # Defensive cleanup before model load — Phase 1's HF model may still be
    # holding GPU memory if it wasn't freed cleanly.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            print(f"  [phase4-pre-load] cuda:{i} alloc="
                  f"{torch.cuda.memory_allocated(i)/1e9:.2f}GB  "
                  f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB", flush=True)

    print("\n--- Loading HF model (fp16, sdpa) for Phase 4 ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos_token=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Explicit per-device memory caps. Without this, accelerate's "balanced"
    # heuristic still packs cuda:0 (where lm_head + embeddings live) close to
    # capacity and OOMs on the first forward when activations land. Leaving
    # 4-5 GB free on each GPU for activations + KV cache works on Kaggle T4.
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="balanced",
        max_memory={0: "11GiB", 1: "13GiB", "cpu": "20GiB"},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).eval()
    # Diagnostic: where did the weights actually land?
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  [phase4] cuda:{i} after load: "
                  f"alloc={torch.cuda.memory_allocated(i)/1e9:.2f}GB  "
                  f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB", flush=True)
    num_layers = hf_model.config.num_hidden_layers

    results = []
    checkpoint_path = PHASE4_OUT_DIR / "forced_activations.pkl"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"Resuming with {len(results)} previously processed examples.")
    processed_qids = {r['question_id'] for r in results}

    # Build (q_id, cond_name, trimmed_text) triples — 4 per surviving question.
    # Skip a question entirely if any condition fails the rfind check.
    cond_names = ["cond1", "cond2", "cond3", "cond4"]
    flat_records = []  # list of dicts: {q_id, correct_label, cond_name, text}
    for item in data:
        q_id = item['question_id']
        if q_id in processed_qids:
            continue
        trimmed = {}
        bad = False
        for c in cond_names:
            t = _trim_to_pre_answer(item[c])
            if t is None:
                bad = True
                break
            trimmed[c] = t
        if bad:
            continue
        for c in cond_names:
            flat_records.append({
                "q_id": q_id,
                "correct_label": item['correct_label'],
                "cond_name": c,
                "text": trimmed[c],
            })

    print(f"Extracting activations for {len(flat_records) // 4} questions × 4 conditions "
          f"= {len(flat_records)} forwards (batched).")

    # Group by batches of `batch_size` flat rows; reassemble per-question records as we go.
    pending_per_q = {}  # q_id -> {cond_name: array, "correct_label": int}

    for start in tqdm(range(0, len(flat_records), batch_size), desc="Extracting Forced Branch Activations"):
        batch = flat_records[start:start + batch_size]
        texts = [r['text'] for r in batch]
        acts_list = _extract_last_token_acts_batched(hf_model, tokenizer, texts, num_layers)

        for r, acts in zip(batch, acts_list):
            q_id = r['q_id']
            if q_id not in pending_per_q:
                pending_per_q[q_id] = {"correct_label": r['correct_label']}
            pending_per_q[q_id][f"{r['cond_name']}_act"] = acts

            # If all 4 conditions are filled, flush this question to results.
            if all(f"{c}_act" in pending_per_q[q_id] for c in cond_names):
                pq = pending_per_q.pop(q_id)
                results.append({
                    "question_id": q_id,
                    "correct_label": pq["correct_label"],
                    "cond1_act": pq["cond1_act"],
                    "cond2_act": pq["cond2_act"],
                    "cond3_act": pq["cond3_act"],
                    "cond4_act": pq["cond4_act"],
                })

        if len(results) % 500 == 0 and len(results) > 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Phase 4 complete. Extracted for {len(results)} questions.")
    print("\n--- Phase 4 cleanup: releasing HF model ---")
    _free_model_and_report(hf_model, tag="phase4")


if __name__ == "__main__":
    run_phase4()
