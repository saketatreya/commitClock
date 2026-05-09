#!/usr/bin/env python3
"""End-to-end smoke test for the full CommitClock pipeline.

Run this on the same machine you're benchmarking (e.g. Kaggle T4x2) and paste
the FULL stdout/stderr back. Logs everything needed to diagnose breakage:

    - Environment (Python / torch / transformers / transformer_lens versions,
      CUDA, GPU info, native bf16 support)
    - Per-phase: elapsed time, pre/post VRAM per GPU, exception traceback if any
    - Per-phase output validation (pickle path, record count, schema, shapes,
      dtypes, NaN/Inf check, basic stats, content sample)
    - Numerical sanity check between Phase 1 and Phase 2: loads HF + TL on the
      same example and compares HF hidden_states[l+1] vs TL blocks.{l}.hook_resid_post
    - Final summary table with timings and OK/FAIL per phase

Usage:
    python test_pipeline.py [--limit N] [--fresh] [--no-sanity]

    --limit       StrategyQA examples to process per phase (default: 8)
    --fresh       Delete data/phase*/*.pkl before running (forces full re-run)
    --no-sanity   Skip the HF-vs-TransformerLens numerical comparison
                  (saves ~1 model load worth of time)
"""
import argparse
import gc
import os
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_header(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80, flush=True)


def log_subheader(title):
    print()
    print("-" * 60)
    print(f"  {title}")
    print("-" * 60, flush=True)


def vram_snapshot(tag):
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        print(f"  [VRAM/{tag}] CUDA not available")
        return
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            free, total = -1, -1
        alloc = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        print(
            f"  [VRAM/{tag}] cuda:{i}  "
            f"free={free/1e9:6.2f}GB  total={total/1e9:6.2f}GB  "
            f"alloc={alloc/1e9:6.2f}GB  reserved={reserved/1e9:6.2f}GB",
            flush=True,
        )


def gc_and_empty():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def array_stats(name, arr):
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    orig_dtype = arr.dtype
    n_nan = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    n_inf = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    # Cast to fp32 for stats — fp16 std overflows on (x-mean)^2 for residual-stream
    # magnitudes (~100+) since 100^2 = 10000 sum-squared overflows fast in fp16.
    arr32 = arr.astype(np.float32, copy=False) if np.issubdtype(arr.dtype, np.floating) else arr
    finite = arr32[np.isfinite(arr32)] if np.issubdtype(arr32.dtype, np.floating) else arr32
    if finite.size == 0:
        print(f"  {name}: shape={arr.shape} dtype={orig_dtype} (empty / no finite values)")
        return
    print(
        f"  {name}: shape={arr.shape} dtype={orig_dtype} "
        f"nan={n_nan} inf={n_inf} "
        f"min={float(finite.min()):.4f} max={float(finite.max()):.4f} "
        f"mean={float(finite.mean()):.4f} std={float(finite.std()):.4f}"
    )


# ---------------------------------------------------------------------------
# Environment dump
# ---------------------------------------------------------------------------

def env_dump():
    log_header("ENVIRONMENT")
    print(f"Python:      {sys.version.replace(chr(10), ' ')}")
    print(f"Platform:    {platform.platform()}")
    print(f"Working dir: {REPO_ROOT}")
    print(f"argv:        {sys.argv}")

    try:
        import torch
        print(f"torch:       {torch.__version__}  (CUDA build: {torch.version.cuda})")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(
                    f"  cuda:{i} {p.name}  CC {p.major}.{p.minor}  "
                    f"total mem {p.total_memory/1e9:.2f}GB  multi-proc {p.multi_processor_count}"
                )
            try:
                print(f"  bfloat16 native: {torch.cuda.is_bf16_supported()}")
            except Exception as e:
                print(f"  bfloat16 native: unknown ({e})")
    except Exception as e:
        print(f"torch import failed: {e}")

    for mod in ["transformers", "transformer_lens", "datasets", "sklearn", "matplotlib", "numpy", "tqdm"]:
        try:
            m = __import__(mod)
            print(f"{mod:<18s} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"{mod:<18s} NOT INSTALLED ({e})")


def clear_phase_data():
    log_subheader("Clearing data/phase*/*")
    from src.config import (
        PHASE1_OUT_DIR, PHASE2_OUT_DIR, PHASE3_OUT_DIR,
        PHASE4_OUT_DIR, PHASE5_OUT_DIR, PHASE6_OUT_DIR,
    )
    for d in [PHASE1_OUT_DIR, PHASE2_OUT_DIR, PHASE3_OUT_DIR,
              PHASE4_OUT_DIR, PHASE5_OUT_DIR, PHASE6_OUT_DIR]:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file():
                p.unlink()
                print(f"  removed {p}")


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_step(name, fn, *args, **kwargs):
    log_header(name)
    vram_snapshot("pre")
    t0 = time.time()
    ok = True
    err = None
    out = None
    try:
        out = fn(*args, **kwargs)
    except Exception:
        ok = False
        err = traceback.format_exc()
        print(err, flush=True)
    elapsed = time.time() - t0
    print(f"\n[{name}] elapsed: {elapsed:.1f}s  status: {'OK' if ok else 'FAILED'}", flush=True)
    vram_snapshot("post")
    gc_and_empty()
    vram_snapshot("post-gc")
    return {"name": name, "ok": ok, "elapsed": elapsed, "err": err, "out": out}


# ---------------------------------------------------------------------------
# Per-phase output validation
# ---------------------------------------------------------------------------

def validate_phase1():
    log_subheader("Phase 1 output validation")
    from src.config import PHASE1_OUT_DIR, NUM_FRACTIONAL_POSITIONS
    p_acts = PHASE1_OUT_DIR / "strategyqa_activations.pkl"
    p_gen = PHASE1_OUT_DIR / "generated_texts.pkl"
    print(f"  generated_texts.pkl:        exists={p_gen.exists()}")
    print(f"  strategyqa_activations.pkl: exists={p_acts.exists()}")

    if p_gen.exists():
        with open(p_gen, "rb") as f:
            gen = pickle.load(f)
        print(f"  generated_texts.pkl records: {len(gen)}")
        if gen:
            keys = sorted(gen[0].keys())
            print(f"  gen[0] keys: {keys}")
            parsed = sum(1 for g in gen if g.get('model_answer', -1) != -1)
            print(f"  parsed (model_answer != -1): {parsed}/{len(gen)} "
                  f"({parsed / max(1, len(gen)):.0%})")
            # Show ALL records' continuations (not the few-shot prompt prefix)
            # so failed-parse cases are debuggable.
            print(f"  --- {len(gen)} model continuations (text after prompt) ---")
            for i, g in enumerate(gen):
                cont = g['generated_text']
                if cont.startswith(g['prompt']):
                    cont = cont[len(g['prompt']):]
                tid_len = len(g['token_ids']) if hasattr(g.get('token_ids'), '__len__') else (
                    g['token_ids'].shape[0] if 'token_ids' in g else None
                )
                print(f"  [gen[{i}]] qid={g.get('question_id')} "
                      f"model_answer={g.get('model_answer')} "
                      f"correct_label={g.get('correct_label')} "
                      f"prompt_len={g.get('prompt_len')} "
                      f"token_ids_len={tid_len}")
                # Last ~100 chars of prompt (so we know which question was asked) +
                # first ~800 chars of model continuation.
                print(f"    question : ...{g['prompt'][-100:]!r}")
                print(f"    cont[:800]: {cont[:800]!r}")

    if not p_acts.exists():
        print("  No activations pickle; aborting deeper validation.")
        return None
    with open(p_acts, "rb") as f:
        data = pickle.load(f)
    print(f"  activations records: {len(data)}")
    if not data:
        return data

    rec = data[0]
    expected = {"question_id", "correct_label", "model_answer", "chain_length",
                "position_indices", "activations", "generated_text", "prompt"}
    print(f"  rec[0] keys: {sorted(rec.keys())}")
    print(f"  missing expected keys: {sorted(expected - set(rec.keys()))}")
    print(f"  extra unexpected keys: {sorted(set(rec.keys()) - expected)}")

    acts = rec["activations"]
    array_stats("activations[rec=0]", acts)
    print(f"  acts.shape decomposed: pos={acts.shape[0]}  layers={acts.shape[1]}  d_model={acts.shape[2]}")
    print(f"  expected NUM_FRACTIONAL_POSITIONS={NUM_FRACTIONAL_POSITIONS}  -> match={acts.shape[0] == NUM_FRACTIONAL_POSITIONS}")
    print(f"  position_indices[rec=0]: {rec['position_indices']}")
    print(f"  chain_length[rec=0]: {rec['chain_length']}")

    # Aggregate over all records
    chain_lens = np.array([r['chain_length'] for r in data])
    print(f"  chain_length distribution: min={chain_lens.min()} max={chain_lens.max()} "
          f"mean={chain_lens.mean():.1f}")
    pos = sum(1 for r in data if r['model_answer'] == 1)
    correct = sum(1 for r in data if r['model_answer'] == r['correct_label'])
    print(f"  model_answer balance: {pos} Yes / {len(data)-pos} No")
    print(f"  accuracy on this slice: {correct}/{len(data)} = {correct/max(1,len(data)):.2%}")

    # Sample one full text per answer class
    yes_rec = next((r for r in data if r['model_answer'] == 1), None)
    no_rec = next((r for r in data if r['model_answer'] == 0), None)
    if yes_rec is not None:
        print(f"  sample YES generation (first 300 chars):")
        print(f"    {yes_rec['generated_text'][:300]!r}")
    if no_rec is not None:
        print(f"  sample NO generation (first 300 chars):")
        print(f"    {no_rec['generated_text'][:300]!r}")
    return data


def numerical_sanity_check():
    """Compare HF `hidden_states[l+1]` vs TL `blocks.{l}.hook_resid_post` on a single
    example. This is the critical correctness check for Phase 1's TL drop."""
    log_header("NUMERICAL SANITY: HF hidden_states vs TL resid_post")
    import torch
    from src.config import MODEL_NAME, PHASE1_OUT_DIR

    p = PHASE1_OUT_DIR / "strategyqa_activations.pkl"
    if not p.exists():
        print("  No phase 1 output to compare against; skipping.")
        return
    with open(p, "rb") as f:
        data = pickle.load(f)
    if not data:
        print("  Empty phase 1; skipping.")
        return

    rec = data[0]
    text = rec["generated_text"]
    trigger = "so the answer is "
    idx = text.lower().rfind(trigger)
    if idx == -1:
        print("  No trigger in rec[0]; skipping.")
        return
    text_pref = text[:idx + len(trigger)]
    ans_str = " Yes" if rec["model_answer"] == 1 else " No"
    full_text = text_pref + ans_str
    print(f"  Comparing on text (last 160 chars): {full_text[-160:]!r}")

    # ---- HF forward ----
    print("  Loading HF model (fp16, sdpa)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos_token=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).eval()
    ids = tok(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(hf.device)
    print(f"  input length: {ids.shape[1]} tokens")
    with torch.no_grad():
        out = hf(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
    num_layers = hf.config.num_hidden_layers
    hf_hs = [out.hidden_states[l + 1][0].float().cpu().numpy() for l in range(num_layers)]
    print(f"  HF: num_layers={num_layers}  hidden_states[1].shape={out.hidden_states[1].shape}")
    del hf, out, tok
    gc_and_empty()
    vram_snapshot("after-HF-free")

    # ---- TL forward ----
    print("  Loading TransformerLens model...")
    from src.models.loader import load_model
    tl = load_model()
    tl_tokens = tl.to_tokens(full_text, prepend_bos=False)
    print(f"  TL: tokens.shape={tuple(tl_tokens.shape)}  n_layers={tl.cfg.n_layers}  d_model={tl.cfg.d_model}")
    with torch.no_grad():
        _, cache = tl.run_with_cache(tl_tokens, names_filter=lambda n: "resid_post" in n)
    tl_rs = [cache[f"blocks.{l}.hook_resid_post"][0].float().cpu().numpy() for l in range(num_layers)]
    del tl, cache
    gc_and_empty()
    vram_snapshot("after-TL-free")

    # ---- Compare ----
    print("  Per-layer comparison (sampled layers):")
    print(f"    {'layer':>5} | {'max|d|':>10} | {'mean|d|':>10} | {'mean|a|':>10} | {'rel':>8}")
    layers_to_check = sorted(set([0, 1, num_layers // 4, num_layers // 2,
                                  3 * num_layers // 4, num_layers - 1]))
    worst_max = 0.0
    worst_rel = 0.0
    for l in layers_to_check:
        a = hf_hs[l]
        b = tl_rs[l]
        if a.shape != b.shape:
            print(f"    layer {l}: SHAPE MISMATCH HF={a.shape} TL={b.shape}")
            continue
        d = np.abs(a - b)
        m = float(np.abs(a).mean())
        rel = float(d.mean() / max(m, 1e-9))
        worst_max = max(worst_max, float(d.max()))
        worst_rel = max(worst_rel, rel)
        print(f"    {l:>5d} | {float(d.max()):>10.5f} | {float(d.mean()):>10.5f} | {m:>10.4f} | {rel:>7.3%}")
    print(f"  WORST: max|d|={worst_max:.5f}  rel(mean|d|/mean|a|)={worst_rel:.3%}")
    print("  PASS criterion (fp16 tol): worst rel-diff under ~1% suggests layer-index mapping is correct.")


def validate_phase2():
    log_subheader("Phase 2 output validation")
    from src.config import PHASE2_OUT_DIR
    if not PHASE2_OUT_DIR.exists():
        print("  Phase 2 dir missing.")
        return
    files = sorted(PHASE2_OUT_DIR.iterdir())
    print(f"  files in {PHASE2_OUT_DIR}:")
    for f in files:
        print(f"    {f.name:<40s}  {f.stat().st_size:>10d} bytes")


def validate_phase3():
    log_subheader("Phase 3 output validation")
    from src.config import PHASE3_OUT_DIR
    p = PHASE3_OUT_DIR / "forced_branches.pkl"
    print(f"  pickle path: {p}  exists: {p.exists()}")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    print(f"  records: {len(data)}")
    if not data:
        return data
    rec = data[0]
    print(f"  rec[0] keys: {sorted(rec.keys())}")
    for c in ["cond1", "cond2", "cond3", "cond4"]:
        if c in rec:
            print(f"  rec[0].{c} (last 160 chars): {rec[c][-160:]!r}")
    return data


def validate_phase4():
    log_subheader("Phase 4 output validation")
    from src.config import PHASE4_OUT_DIR
    p = PHASE4_OUT_DIR / "forced_activations.pkl"
    print(f"  pickle path: {p}  exists: {p.exists()}")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    print(f"  records: {len(data)}")
    if not data:
        return data
    rec = data[0]
    print(f"  rec[0] keys: {sorted(rec.keys())}")
    for k in ["cond1_act", "cond2_act", "cond3_act", "cond4_act"]:
        if k in rec:
            array_stats(f"rec[0].{k}", rec[k])
    # Cross-condition diff (sanity: cond1 should differ from cond3)
    if all(k in rec for k in ["cond1_act", "cond3_act"]):
        diff = float(np.abs(rec["cond1_act"] - rec["cond3_act"]).mean())
        print(f"  mean|cond1 - cond3| at rec[0]: {diff:.4f} (should be > 0)")
    return data


def validate_phase5():
    log_subheader("Phase 5 output validation")
    from src.config import PHASE5_OUT_DIR
    p = PHASE5_OUT_DIR / "causal_patching.pkl"
    print(f"  pickle path: {p}  exists: {p.exists()}")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    print(f"  records: {len(data)}")
    if not data:
        return data
    print(f"  rec[0] keys: {sorted(data[0].keys())}")
    effs = np.array([r['effects'] for r in data])
    array_stats("effects (all records)", effs)
    avg = effs.mean(axis=0)
    print(f"  per-layer mean effect (all {len(avg)} layers):")
    for i, v in enumerate(avg):
        print(f"    layer {i:>2d}: {v:>+.4f}")
    plot_path = PHASE5_OUT_DIR / "causal_effect.png"
    print(f"  plot: {plot_path}  exists={plot_path.exists()}")
    return data


def validate_phase6():
    log_subheader("Phase 6 output validation")
    from src.config import PHASE6_OUT_DIR
    p = PHASE6_OUT_DIR / "nonlinearity_results.pkl"
    print(f"  pickle path: {p}  exists: {p.exists()}")
    if not p.exists():
        return
    with open(p, "rb") as f:
        d = pickle.load(f)
    print(f"  keys: {sorted(d.keys())}")
    array_stats("lr_aurocs", d["lr_aurocs"])
    array_stats("mlp_aurocs", d["mlp_aurocs"])
    diff = d["mlp_aurocs"] - d["lr_aurocs"]
    array_stats("mlp - lr (nonlinearity index)", diff)
    plot_path = PHASE6_OUT_DIR / "nonlinearity_index.png"
    print(f"  plot: {plot_path}  exists={plot_path.exists()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8,
                    help="StrategyQA examples per phase (default: 8)")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete all data/phase*/* before running")
    ap.add_argument("--sanity", action="store_true",
                    help="Run HF-vs-TL numerical sanity check inline. "
                         "OFF by default because loading both 7B copies in one "
                         "process can OOM 2x T4. Use test_sanity.py standalone instead.")
    args = ap.parse_args()

    log_header(f"COMMITCLOCK PIPELINE TEST  (limit={args.limit}  fresh={args.fresh}  "
               f"sanity={'yes' if args.sanity else 'no (run test_sanity.py separately)'})")

    env_dump()
    if args.fresh:
        clear_phase_data()

    # Lazy imports so env_dump runs even if a project import fails
    from src.experiments.phase1_free_gen import run_phase1_strategyqa
    from src.experiments.phase2_probe import run_phase2
    from src.experiments.phase3_forced_branch import run_phase3
    from src.experiments.phase4_forced_analysis import run_phase4
    from src.experiments.phase5_causal_patching import run_phase5
    from src.experiments.phase6_nonlinearity import run_phase6

    summary = []

    summary.append(run_step(
        "PHASE 1: free generation + extraction",
        run_phase1_strategyqa, limit=args.limit,
    ))
    try:
        validate_phase1()
    except Exception:
        print(traceback.format_exc())

    if args.sanity:
        try:
            numerical_sanity_check()
        except Exception:
            print(traceback.format_exc())
        gc_and_empty()

    summary.append(run_step("PHASE 2: probe training", run_phase2))
    try:
        validate_phase2()
    except Exception:
        print(traceback.format_exc())

    summary.append(run_step("PHASE 3: forced branch construction", run_phase3))
    try:
        validate_phase3()
    except Exception:
        print(traceback.format_exc())

    summary.append(run_step(
        "PHASE 4: forced branch analysis",
        run_phase4, limit=args.limit,
    ))
    try:
        validate_phase4()
    except Exception:
        print(traceback.format_exc())

    summary.append(run_step(
        "PHASE 5: causal patching",
        run_phase5, limit=args.limit,
    ))
    try:
        validate_phase5()
    except Exception:
        print(traceback.format_exc())

    summary.append(run_step("PHASE 6: nonlinearity", run_phase6))
    try:
        validate_phase6()
    except Exception:
        print(traceback.format_exc())

    # Final summary table
    log_header("FINAL SUMMARY")
    total = 0.0
    n_fail = 0
    for s in summary:
        total += s["elapsed"]
        n_fail += int(not s["ok"])
        print(f"  [{'OK  ' if s['ok'] else 'FAIL'}] {s['name']:<48s} {s['elapsed']:>7.1f}s")
    print(f"  {'-' * 72}")
    print(f"  TOTAL: {total:>7.1f}s   FAILURES: {n_fail}/{len(summary)}")
    print()


if __name__ == "__main__":
    main()
