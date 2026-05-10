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

from src.config import PHASE3_OUT_DIR, PHASE5_OUT_DIR
from src.models.loader import load_model
from src.data.loader import ANSWER_TRIGGER_RE


# ---------------------------------------------------------------------------
# Diagnostics — write to stderr (line-buffered) so the last few lines survive
# a SIGKILL from the kernel OOM killer (stdout is fully buffered when piped
# through > log 2>&1).
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
    # System RAM
    if psutil is not None:
        vm = psutil.virtual_memory()
        rss = _PROC.memory_info().rss if _PROC else None
        parts.append(
            f"sysRAM used={vm.used/1e9:.2f}GB free={vm.available/1e9:.2f}GB "
            f"total={vm.total/1e9:.2f}GB"
        )
        if rss is not None:
            parts.append(f"procRSS={rss/1e9:.2f}GB")
    # GPU VRAM per device
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
    """Run a forward pass and print a memory-aware error if it fails. Re-raises
    the exception so callers see it (and the process dies cleanly, with a
    traceback we can read)."""
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
# Helpers (unchanged)
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


def _resolve_target_token_id(model, correct_label: int):
    ans_str = " Yes" if correct_label == 1 else " No"
    tid = model.to_single_token(ans_str)
    if tid is None:
        ans_str = "Yes" if correct_label == 1 else "No"
        tid = model.to_single_token(ans_str)
    return tid


def _left_pad_batch(tokenizer, texts, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    T = input_ids.shape[1]
    last_idx = torch.full((input_ids.shape[0],), T - 1, dtype=torch.long, device=device)
    return input_ids, attn, last_idx


def run_phase5(limit=None, batch_size: int = 4):
    """Causal patching, batched across questions per layer.
    Heavy instrumentation: every forward and every batch boundary logs
    system RAM + per-GPU VRAM to stderr (line-buffered, survives SIGKILL).
    """
    _stderr("=" * 60)
    _stderr(f"Phase 5 starting  batch_size={batch_size}  limit={limit}")
    _mem_snapshot("phase5-entry")

    data = load_forced_branches()
    if not data:
        return

    if limit:
        data = data[:limit]

    _stderr("Phase 5: loading TransformerLens model...")
    _mem_snapshot("pre-TL-load")
    model = load_model()
    num_layers = model.cfg.n_layers
    device = model.cfg.device
    _mem_snapshot("post-TL-load")
    _stderr(f"Phase 5: TL loaded.  n_layers={num_layers}  d_model={model.cfg.d_model}  device={device}")

    # Reset the max_alloc counter so subsequent peaks are visible per-batch.
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    _stderr(f"Phase 5: {len(pending)} questions x {num_layers} layers, batches of {batch_size}")
    _mem_snapshot("pre-loop")

    for start in tqdm(range(0, len(pending), batch_size), desc="Causal Patching",
                      file=sys.stderr):
        batch = pending[start:start + batch_size]
        B = len(batch)
        _stderr(f"\n--- batch start={start} B={B} ---")
        # Show how much padding this batch implies (longer = more wasted compute)
        c1_lens = [len(tokenizer.encode(b['cond1_text'], add_special_tokens=False)) for b in batch]
        c3_lens = [len(tokenizer.encode(b['cond3_text'], add_special_tokens=False)) for b in batch]
        _stderr(f"  cond1 token lens: {c1_lens}  (max={max(c1_lens)})")
        _stderr(f"  cond3 token lens: {c3_lens}  (max={max(c3_lens)})")

        cond1_ids, cond1_attn, last_idx_1 = _left_pad_batch(
            tokenizer, [b['cond1_text'] for b in batch], device
        )
        cond3_ids, cond3_attn, last_idx_3 = _left_pad_batch(
            tokenizer, [b['cond3_text'] for b in batch], device
        )
        target_ids = torch.tensor([b['target_token_id'] for b in batch], device=device, dtype=torch.long)
        row_idx = torch.arange(B, device=device)
        _stderr(f"  cond1_ids.shape={tuple(cond1_ids.shape)}  cond3_ids.shape={tuple(cond3_ids.shape)}")
        _mem_snapshot("post-tokenize")

        # 1) Clean run on cond1
        with torch.no_grad():
            clean_logits = _safe_forward(
                "clean", model, cond1_ids, attention_mask=cond1_attn,
            )
            clean_lp_full = torch.nn.functional.log_softmax(
                clean_logits[row_idx, last_idx_1, :], dim=-1
            )
            clean_target_lp = clean_lp_full[row_idx, target_ids]  # [B]
        del clean_logits, clean_lp_full
        _mem_snapshot("post-clean")

        # 2) Cache cond3 across all resid_post layers
        with torch.no_grad():
            _, cache_c3 = _safe_forward(
                "cache_c3",
                model.run_with_cache,
                cond3_ids,
                attention_mask=cond3_attn,
                names_filter=lambda n: "resid_post" in n,
            )
        _mem_snapshot("post-cache")
        # How big did the cache actually get?
        cache_bytes = sum(t.element_size() * t.nelement() for t in cache_c3.values()
                          if isinstance(t, torch.Tensor))
        _stderr(f"  cache_c3: {len(cache_c3)} tensors, total={cache_bytes/1e9:.3f}GB")

        # 3) Per-layer batched patched forward
        per_layer_effect = torch.zeros((B, num_layers), dtype=torch.float32)
        for l in range(num_layers):
            h_C3_l = cache_c3[f"blocks.{l}.hook_resid_post"][row_idx, last_idx_3, :].detach()  # [B, D]

            def patch_hook(resid_post, hook, _h=h_C3_l, _idx=last_idx_1):
                resid_post[row_idx, _idx, :] = _h
                return resid_post

            with torch.no_grad():
                patched_logits = _safe_forward(
                    f"patched-L{l}",
                    model.run_with_hooks,
                    cond1_ids,
                    attention_mask=cond1_attn,
                    fwd_hooks=[(f"blocks.{l}.hook_resid_post", patch_hook)],
                )
                patched_lp_full = torch.nn.functional.log_softmax(
                    patched_logits[row_idx, last_idx_1, :], dim=-1
                )
                patched_target_lp = patched_lp_full[row_idx, target_ids]
            per_layer_effect[:, l] = (patched_target_lp - clean_target_lp).cpu()
            del patched_logits, patched_lp_full
            # Snapshot only at first & last layer to keep the log readable
            if l == 0 or l == num_layers - 1:
                _mem_snapshot(f"post-patch-L{l}")

        del cache_c3, clean_target_lp, cond1_ids, cond1_attn, cond3_ids, cond3_attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        _mem_snapshot("post-batch-cleanup")

        for b, item in enumerate(batch):
            results.append({
                "question_id": item['q_id'],
                "effects": per_layer_effect[b].tolist(),
            })

        # Save every batch so an OOM kill loses at most one batch.
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


if __name__ == "__main__":
    run_phase5()
