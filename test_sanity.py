#!/usr/bin/env python3
"""TL-only numerical sanity check for Phase 1's HF activation extraction.

Loads ONLY the TransformerLens model (no second HF copy — keeps us under the
2x T4 OOM limit) and compares its residual-stream activations at the same
positions Phase 1 saved against the values already in
`data/phase1/strategyqa_activations.pkl`.

If the worst per-layer relative diff is under ~1%, the HF
`hidden_states[l + 1]` <-> TL `blocks.{l}.hook_resid_post` mapping in the
new Phase 1 is correct, and downstream probes are using the right activations.

Run AFTER Phase 1 has produced the activations pickle (e.g. after
`python test_pipeline.py --limit 8 --fresh` succeeds).

Usage:
    python test_sanity.py [--n N]

    --n   number of saved Phase 1 records to spot-check (default: 1)
"""
import argparse
import gc
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1,
                    help="Phase 1 records to verify (default: 1)")
    args = ap.parse_args()

    import torch
    from src.config import PHASE1_OUT_DIR

    p = PHASE1_OUT_DIR / "strategyqa_activations.pkl"
    if not p.exists():
        print(f"  Phase 1 activations pickle missing: {p}")
        print(f"  Run `python test_pipeline.py --limit 8 --fresh` first.")
        sys.exit(1)

    with open(p, "rb") as f:
        data = pickle.load(f)
    if not data:
        print("  Phase 1 produced 0 records; nothing to verify.")
        sys.exit(1)

    print(f"  Loaded {len(data)} Phase 1 records; spot-checking first {min(args.n, len(data))}.")

    # Free anything left over before loading the 7B model.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  VRAM before TL load:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"    cuda:{i}  free={free/1e9:.2f}GB  total={total/1e9:.2f}GB")

    print("  Loading TransformerLens model (this is the only 7B copy in memory)...")
    from src.models.loader import load_model
    tl = load_model()
    num_layers = tl.cfg.n_layers
    d_model = tl.cfg.d_model
    print(f"  TL: n_layers={num_layers}  d_model={d_model}")

    print(f"  VRAM after TL load:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"    cuda:{i}  free={free/1e9:.2f}GB  total={total/1e9:.2f}GB")

    n_check = min(args.n, len(data))
    overall_worst_rel = 0.0

    for r_idx in range(n_check):
        rec = data[r_idx]
        # Reconstruct the exact text Phase 1 ran the forward pass on.
        text = rec["generated_text"]
        trigger = "so the answer is "
        idx = text.lower().rfind(trigger)
        if idx == -1:
            print(f"  [rec {r_idx}] no 'so the answer is' trigger; skipping")
            continue
        text_pref = text[:idx + len(trigger)]
        ans_str = " Yes" if rec["model_answer"] == 1 else " No"
        full_text = text_pref + ans_str

        print(f"\n  --- record {r_idx}: qid={rec['question_id']} "
              f"model_answer={rec['model_answer']} chain_length={rec['chain_length']} ---")
        print(f"  full_text last 160 chars: {full_text[-160:]!r}")

        try:
            tl_tokens = tl.to_tokens(full_text, prepend_bos=False)
        except Exception:
            print(traceback.format_exc())
            continue
        print(f"  TL tokenization: tokens.shape={tuple(tl_tokens.shape)}")

        with torch.no_grad():
            _, cache = tl.run_with_cache(tl_tokens, names_filter=lambda n: "resid_post" in n)

        # Phase 1 saved activations at these positions in the HF-padded sequence.
        # In the (single-example, no-padding) TL forward, those same string
        # positions correspond to the same token indices since both tokenizers
        # use the same Qwen vocabulary and we're using add_special_tokens=False.
        positions = rec["position_indices"]
        hf_acts = rec["activations"].astype(np.float32)  # [num_pos, num_layers, D]

        if hf_acts.shape[1] != num_layers:
            print(f"  layer-count mismatch: HF={hf_acts.shape[1]}  TL={num_layers}; skipping")
            continue
        if hf_acts.shape[2] != d_model:
            print(f"  d_model mismatch: HF={hf_acts.shape[2]}  TL={d_model}; skipping")
            continue

        # Sample a few layers across depth for the readout.
        layers_to_check = sorted(set([0, 1, num_layers // 4, num_layers // 2,
                                      3 * num_layers // 4, num_layers - 1]))
        print(f"  per-layer comparison (positions = {positions}):")
        print(f"    {'layer':>5} | {'max|d|':>10} | {'mean|d|':>10} | {'mean|a|':>10} | {'rel':>8}")
        worst_rel = 0.0
        for l in layers_to_check:
            tl_layer = cache[f"blocks.{l}.hook_resid_post"][0].float().cpu().numpy()  # [T, D]
            if tl_tokens.shape[1] <= max(positions):
                print(f"    layer {l}: TL tokenization shorter ({tl_tokens.shape[1]}) "
                      f"than max position ({max(positions)}); skipping")
                continue
            # tl_layer at the same positions. HF acts at this layer:
            hf_layer = hf_acts[:, l, :]                # [num_pos, D]
            tl_at_pos = tl_layer[positions, :]         # [num_pos, D]
            d = np.abs(hf_layer - tl_at_pos)
            mean_a = float(np.abs(hf_layer).mean())
            rel = float(d.mean() / max(mean_a, 1e-9))
            worst_rel = max(worst_rel, rel)
            print(f"    {l:>5d} | {float(d.max()):>10.5f} | {float(d.mean()):>10.5f} | "
                  f"{mean_a:>10.4f} | {rel:>7.3%}")
        print(f"  [rec {r_idx}] WORST rel(mean|d|/mean|a|) across sampled layers: {worst_rel:.3%}")
        overall_worst_rel = max(overall_worst_rel, worst_rel)
        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print(f"  OVERALL WORST rel-diff: {overall_worst_rel:.3%}")
    print("  PASS criterion: under ~1% means HF hidden_states[l+1] == TL")
    print("                  blocks.{l}.hook_resid_post (the layer-index")
    print("                  mapping used in the new Phase 1 is correct).")
    print("  Anything > 5% means downstream probes are reading the wrong layer.")
    print("=" * 60)


if __name__ == "__main__":
    main()
