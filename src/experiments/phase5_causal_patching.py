import gc
import os
import pickle
import sys
import time
import traceback

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME, PHASE3_OUT_DIR, PHASE5_OUT_DIR
from src.data.loader import ANSWER_TRIGGER_RE


# ---------------------------------------------------------------------------
# Diagnostics — write to stderr (line-buffered) so the last few lines survive
# a SIGKILL from the kernel OOM killer.
# ---------------------------------------------------------------------------

try:
    import psutil  # type: ignore
    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PROC = None


def _stderr(msg):
    print(msg, file=sys.stderr, flush=True)


def _mem_snapshot(tag):
    parts = [f"[MEM/{tag}]"]
    if psutil is not None:
        vm = psutil.virtual_memory()
        rss = _PROC.memory_info().rss if _PROC else None
        parts.append(
            f"sysRAM used={vm.used/1e9:.2f}GB free={vm.available/1e9:.2f}GB "
            f"total={vm.total/1e9:.2f}GB"
        )
        if rss is not None:
            parts.append(f"procRSS={rss/1e9:.2f}GB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                free, total = -1, -1
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            max_alloc = torch.cuda.max_memory_allocated(i)
            parts.append(
                f"cuda:{i} free={free/1e9:.2f} alloc={alloc/1e9:.2f} "
                f"reserved={reserved/1e9:.2f} maxAlloc={max_alloc/1e9:.2f}"
            )
    _stderr(" ".join(parts))


def _safe_forward(label, fn, *args, **kwargs):
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        _stderr(f"[OOM/{label}] CUDA OOM after {time.time()-t0:.1f}s: {e}")
        _mem_snapshot(f"{label}-OOM")
        if torch.cuda.is_available():
            try:
                _stderr(torch.cuda.memory_summary(abbreviated=True))
            except Exception:
                pass
        raise
    except Exception as e:
        _stderr(f"[ERR/{label}] {type(e).__name__} after {time.time()-t0:.1f}s: {e}")
        _stderr(traceback.format_exc())
        raise
    _stderr(f"[fwd/{label}] elapsed={time.time()-t0:.2f}s")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _resolve_target_token_id(tokenizer, correct_label: int):
    """Return the single-token id of the answer string, trying ' Yes'/' No' first
    (the Qwen tokenizer normally folds the leading space) and falling back to no
    leading space. Returns None if neither tokenizes to a single token."""
    for s in ([" Yes", " No"], ["Yes", "No"]):
        ans = s[0] if correct_label == 1 else s[1]
        ids = tokenizer(ans, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    return None


# ---------------------------------------------------------------------------
# Patch hook on a single transformer block
# ---------------------------------------------------------------------------

def _make_patch_hook(h_C3_b_d: torch.Tensor, last_idx_1):
    """Returns a forward-hook fn that overwrites the block's output residual
    stream at row b, position last_idx_1[b], :, with h_C3_b_d[b, :].

    HF Qwen2 decoder layer's forward returns either a tensor (just the new
    residual) or a tuple whose first element is the residual. We handle both."""

    def hook(module, args, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        # h_C3_b_d may live on a different device than hs (when device_map splits
        # layers across GPUs). Move once.
        h_local = h_C3_b_d.to(hs.device, dtype=hs.dtype, non_blocking=True)
        # In-place patch — output tensors are owned by the layer's forward and
        # safe to mutate before being returned to the next layer.
        for b, idx in enumerate(last_idx_1):
            hs[b, idx, :] = h_local[b]
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    return hook


def run_phase5(limit=None, batch_size: int = 4):
    """Causal patching using HF + forward hooks (TL is no longer used).

    Algorithm per batch of B questions:
      1) Run cond3 with output_hidden_states=True; cache resid_post[l] at
         last_idx_3[b] per layer per batch element  -> h_C3 [L, B, D].
      2) Run cond1 cleanly; record log-prob of the correct-answer token at
         last_idx_1[b] per row.
      3) For each layer l, register a forward hook on hf_model.model.layers[l]
         that overwrites the output residual at last_idx_1[b] with h_C3[l, b].
         Run cond1 forward; record patched log-prob; remove hook.
      4) effect[b, l] = patched_lp - clean_lp.

    HF's `output_hidden_states[l+1]` is the residual-stream output of layer l
    (post-attention + post-MLP + post-residual-add) — exactly what TL calls
    blocks.{l}.hook_resid_post. So patching `model.model.layers[l]`'s output
    matches the original TL semantics."""
    _stderr("=" * 60)
    _stderr(f"Phase 5 starting  batch_size={batch_size}  limit={limit}")
    _mem_snapshot("phase5-entry-raw")
    # Force-release any leftover state from prior phases before measuring
    # what's actually available. Without this, Phase 4's HF model can sit
    # in GPU memory when Phase 5 starts (Python GC + PyTorch caching
    # allocator both hold references past function return).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    _mem_snapshot("phase5-entry-cleaned")

    data = load_forced_branches()
    if not data:
        return

    if limit:
        data = data[:limit]

    _stderr("Phase 5: loading HF model (fp16, sdpa, low_cpu_mem_usage)...")
    _mem_snapshot("pre-HF-load")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos_token=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # See Phase 4 for the device_map rationale and the explicit max_memory
    # caps — without them accelerate packs cuda:0 (where lm_head sits) and
    # OOMs on first forward.
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="balanced",
        max_memory={0: "11GiB", 1: "13GiB", "cpu": "20GiB"},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    _mem_snapshot("post-HF-load")

    num_layers = hf_model.config.num_hidden_layers
    d_model = hf_model.config.hidden_size
    layers = hf_model.model.layers  # list of decoder blocks
    _stderr(f"Phase 5: HF loaded.  n_layers={num_layers}  d_model={d_model}  "
            f"len(layers)={len(layers)}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    results = []
    checkpoint_path = PHASE5_OUT_DIR / "causal_patching.pkl"
    if checkpoint_path.exists():
        _stderr(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        _stderr(f"Resuming with {len(results)} previously processed examples.")
    processed_qids = {r['question_id'] for r in results}

    pending = []
    for item in data:
        q_id = item['question_id']
        if q_id in processed_qids:
            continue
        c1 = _pre_answer_text(item['cond1'])
        c3 = _pre_answer_text(item['cond3'])
        if c1 is None or c3 is None:
            continue
        tgt = _resolve_target_token_id(tokenizer, item['correct_label'])
        if tgt is None:
            continue
        pending.append({
            "q_id": q_id,
            "correct_label": item['correct_label'],
            "cond1_text": c1,
            "cond3_text": c3,
            "target_token_id": tgt,
        })

    _stderr(f"Phase 5: {len(pending)} questions x {num_layers} layers, batches of {batch_size}")
    _mem_snapshot("pre-loop")

    for start in tqdm(range(0, len(pending), batch_size), desc="Causal Patching",
                      file=sys.stderr):
        batch = pending[start:start + batch_size]
        B = len(batch)
        _stderr(f"\n--- batch start={start} B={B} ---")

        # Tokenize cond1 / cond3 separately (different lengths).
        cond1_enc = tokenizer([b['cond1_text'] for b in batch],
                              return_tensors="pt", padding=True,
                              add_special_tokens=False)
        cond3_enc = tokenizer([b['cond3_text'] for b in batch],
                              return_tensors="pt", padding=True,
                              add_special_tokens=False)
        cond1_ids = cond1_enc["input_ids"].to(hf_model.device)
        cond1_attn = cond1_enc["attention_mask"].to(hf_model.device)
        cond3_ids = cond3_enc["input_ids"].to(hf_model.device)
        cond3_attn = cond3_enc["attention_mask"].to(hf_model.device)
        # With left-padding the last non-pad index is always T-1.
        last_idx_1 = [int(cond1_ids.shape[1]) - 1] * B
        last_idx_3 = [int(cond3_ids.shape[1]) - 1] * B
        target_ids = torch.tensor([b['target_token_id'] for b in batch], dtype=torch.long)
        row_idx = torch.arange(B)
        _stderr(f"  cond1_ids.shape={tuple(cond1_ids.shape)}  cond3_ids.shape={tuple(cond3_ids.shape)}")
        _mem_snapshot("post-tokenize")

        # 1) Cache cond3's residual stream at the last-position per layer.
        #    output_hidden_states[l+1] = resid_post of layer l.
        with torch.no_grad():
            cond3_out = _safe_forward(
                "cache_c3", hf_model,
                input_ids=cond3_ids, attention_mask=cond3_attn,
                output_hidden_states=True, use_cache=False, return_dict=True,
            )
        # h_C3: list of [B, D] tensors, one per layer (l = 0..num_layers-1).
        h_C3 = [cond3_out.hidden_states[l + 1][row_idx, last_idx_3, :].detach().clone()
                for l in range(num_layers)]
        del cond3_out, cond3_ids, cond3_attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _mem_snapshot("post-cache")

        # 2) Clean cond1 forward → baseline log-prob of target.
        with torch.no_grad():
            clean_out = _safe_forward(
                "clean", hf_model,
                input_ids=cond1_ids, attention_mask=cond1_attn,
                use_cache=False, return_dict=True,
            )
        clean_lp_full = torch.nn.functional.log_softmax(
            clean_out.logits[row_idx, last_idx_1, :].float().cpu(), dim=-1
        )
        clean_target_lp = clean_lp_full[row_idx, target_ids]  # [B]
        del clean_out, clean_lp_full
        _mem_snapshot("post-clean")

        # 3) Per-layer patched forward.
        per_layer_effect = torch.zeros((B, num_layers), dtype=torch.float32)
        for l in range(num_layers):
            handle = layers[l].register_forward_hook(
                _make_patch_hook(h_C3[l], last_idx_1)
            )
            try:
                with torch.no_grad():
                    patched_out = _safe_forward(
                        f"patched-L{l}", hf_model,
                        input_ids=cond1_ids, attention_mask=cond1_attn,
                        use_cache=False, return_dict=True,
                    )
            finally:
                handle.remove()
            patched_lp_full = torch.nn.functional.log_softmax(
                patched_out.logits[row_idx, last_idx_1, :].float().cpu(), dim=-1
            )
            patched_target_lp = patched_lp_full[row_idx, target_ids]
            per_layer_effect[:, l] = patched_target_lp - clean_target_lp
            del patched_out, patched_lp_full
            if l == 0 or l == num_layers - 1:
                _mem_snapshot(f"post-patch-L{l}")

        del cond1_ids, cond1_attn, h_C3, clean_target_lp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        _mem_snapshot("post-batch-cleanup")

        for b, item in enumerate(batch):
            results.append({
                "question_id": item['q_id'],
                "effects": per_layer_effect[b].tolist(),
            })

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

    _stderr(f"Phase 5 complete. Evaluated {len(results)} questions.")

    # Free the HF model so a chained run_all (or notebook session) gets
    # clean GPUs back. Mirrors the cleanup at the end of Phase 1 and Phase 4.
    _stderr("\n--- Phase 5 cleanup: releasing HF model ---")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            _stderr(f"  [free/phase5] before  cuda:{i} alloc="
                    f"{torch.cuda.memory_allocated(i)/1e9:.2f}GB  "
                    f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB")
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            _stderr(f"  [free/phase5] after   cuda:{i} alloc="
                    f"{torch.cuda.memory_allocated(i)/1e9:.2f}GB  "
                    f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB")


if __name__ == "__main__":
    run_phase5()
