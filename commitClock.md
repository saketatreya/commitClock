Okay. Here is the full plan. I'll go component by component, being precise about every decision.

---

## The Paper in One Sentence

We introduce the commitment clock — a method for measuring when a behavioral decision commits in LLM activation space — and use it to show that chain-of-thought reasoning is mechanistically post-hoc for a substantial fraction of questions, with the degree of genuine computation varying systematically with reasoning difficulty.

---

## Model

Primary: Qwen2.5-7B-Instruct. Not 1.5B. The claim "CoT is post-hoc at 1.5B" is weak because a 1.5B model arguably never does genuine multi-step reasoning — it's not surprising and it's not interesting. At 7B you have a model that genuinely produces coherent multi-step reasoning and the finding becomes meaningful. 32 layers, 4096 hidden dim, tractable for this work.

Secondary: Qwen2.5-1.5B-Instruct as a scale comparison. If the post-hoc signature strengthens at smaller scale, that's a clean result about the relationship between model capacity and reasoning faithfulness. This is a cheap add once the primary pipeline is running.

Use TransformerLens for activation extraction. It handles residual stream hooking cleanly and the codebase is well-maintained. Do not reimplement hooks from scratch.

---

## Datasets

**StrategyQA** (primary). Binary yes/no questions requiring implicit multi-hop factual reasoning. 2290 training, 490 dev. Ground truth is unambiguous. The binary label makes probe training clean. Download from the original repository.

**BIG-Bench Hard — Logical Deduction** (secondary). 3-object and 5-object variants. Multi-step constraint satisfaction. Ground truth is categorical and unambiguous. Harder than StrategyQA so you should see different commitment profiles.

**Preprocessing.** For each dataset, run the model in free-generation mode on the full set first. Then filter:

Keep questions where the model produces a reasoning chain of at least 40 tokens and at most 250 tokens before the answer. This excludes cases where the model gives a one-sentence answer (no reasoning to analyze) and cases where the chain is so long that fractional position sampling becomes noisy.

Keep questions where model performance on the full filtered set is between 25% and 75% correct. You need genuine difficulty — ceiling and floor questions do not give you interesting activation geometry because there is no variability in commitment.

Target 300–400 questions post-filter for the free-generation set. 150–200 for the forced-branch set (subset of the above).

**Prompt format for StrategyQA:**

```
Q: [question]
A: Let me work through this step by step.
[model generates]
So the answer is Yes. / So the answer is No.
```

The commitment point is the Yes/No token. Be consistent — define it as the first token of the answer word. Pre-commit window is everything before this token in the reasoning chain. Post-commit is after.

**Prompt format for BBH Logical Deduction:**

```
[problem description with constraints]
Let me work through the constraints one by one.
[model generates steps]
Therefore, [X] is in position [Y].
```

Commitment point is the first token of the final answer entity.

---

## Phase 1: Free-Generation Activation Extraction (The Commitment Clock Data)

This is the core data collection phase.

**What you're extracting.** For each question, after the model generates its full CoT response, you re-run the forward pass with the full generated sequence as input and extract hidden states at the residual stream (post-layer, pre-LayerNorm) at specific (position, layer) cells.

**Position sampling.** Do not attempt to store activations at every token — the memory footprint is unmanageable. Instead, divide each reasoning chain into 10 equally-spaced fractional positions: 0%, 11%, 22%, ..., 100% of the chain length (in tokens), where 0% is the first reasoning token and 100% is the last token before the answer token. For each question, identify the 10 token indices corresponding to these positions and extract only at those indices.

This gives you 10 position bins × 32 layers × 4096 hidden dim per question. At 300 questions × 4 bytes × float32, that is around 1.5GB. Manageable.

**Layer selection.** Extract at all 32 layers. You want the full profile, not a sample — the dual-peak pattern you saw in the sycophancy work appeared at specific layers you would not have found by sampling every 4th.

**Storage format.** For each question, store:
```
{
  question_id: str,
  correct_label: int,         # ground truth
  model_answer: int,          # model's answer (correct/incorrect)
  chain_length: int,          # in tokens
  position_indices: list[int], # the 10 token indices sampled
  activations: array[10, 32, 4096],  # float16 to save space
}
```

Store as a flat array of these dicts, one file per dataset.

---

## Phase 2: Probe Training — The Commitment Clock Surface

**What you're training.** For each of the 10 × 32 = 320 (position, layer) cells, train a logistic regression probe. You train two versions:

Probe A: predict model's eventual answer (correct vs. incorrect as a binary label). This measures commitment to what the model will say.

Probe B: predict ground truth label. This measures whether correct information is available in the representation.

The dissociation between A and B is the interesting finding. If Probe B AUROC is high early but Probe A is low, the correct information is present but uncommitted. If both are high early, the model has committed correctly early. If Probe A is high early but to the wrong answer, there is early false commitment.

**Probe specification.** Logistic regression with L2 regularization (C=1.0 default, tune if needed). 5-fold stratified cross-validation. Report mean AUROC across folds with 95% CI (bootstrap the fold AUROCs). BH correction at q=0.05 across all cells to identify cells with significant above-chance decodability.

**The commitment clock figure.** The main figure of the paper is a heatmap: x-axis is fractional position in reasoning chain (0%–100%), y-axis is layer (0–31), color is AUROC for Probe A. The shape of this surface tells you when and where in the network the answer commits. You should see this surface separately for questions the model answers correctly and questions it answers incorrectly.

A secondary figure plots the marginal commitment clock: AUROC averaged across layers at each position bin. This gives the single clearest visualization of early vs. late commitment.

**Stratification.** Split questions into two groups: easy (model gets correct >65% on similar questions, approximated by free-generation accuracy) and hard (correct <45%). Plot separate commitment clock surfaces for each. The hypothesis is that hard questions show later commitment — this is the result that distinguishes genuine computation from retrieval.

---

## Phase 3: Forced Branch Construction

For a subset of 150–200 questions from the free-generation set, construct four forced conditions.

**Condition 1 — Natural correct reasoning.** These are the free-generation outputs where the model got the answer right. You already have these. Use them as-is.

**Condition 4 — No reasoning (direct answer).** Reprompt with a format that elicits direct answers without reasoning:
```
Q: [question]
A: [Yes/No directly]
```
This is your baseline. It tells you what the answer geometry looks like without any chain.

**Condition 3 — Wrong reasoning reaching wrong answer.** Take free-generation outputs from your set where the model got the answer wrong. These are natural, fluent, wrong reasoning chains. Use them as-is as forced chains. The key: for question X, use the wrong reasoning chain from question X itself (from a different seed or sampling run where the model happened to be wrong), not from a different question. This controls for question content.

If you cannot get enough wrong-answer generations from re-sampling (some questions the model almost always gets right), use wrong-answer reasoning chains from thematically similar questions. Document this in the paper.

**Condition 2 — Wrong reasoning reaching correct answer.** This is the hardest to construct. The cleanest approach: take a wrong-reasoning chain from Condition 3 and truncate it before the final conclusion, then append the correct answer. This gives you broken reasoning that nonetheless reaches the right conclusion. Alternatively: take the correct reasoning chain from Condition 1, introduce a factual error in an intermediate step (swap a named entity, negate a fact), and keep the original conclusion. The model, when continuing from this chain, may or may not maintain the correct answer — you want cases where the conclusion says "Yes" or the correct answer despite the broken reasoning.

**How you use these conditions.** For each of the four conditions, prefill the model's context up to and including the reasoning chain (including the trigger phrase like "So the answer is") but stop before the actual answer token. Extract hidden states at the last reasoning token position (the token immediately before the answer token) across all layers. This gives you the answer-decision representation under each reasoning condition.

Storage: same format as Phase 1, but you only need the single pre-answer-token position, not the full chain. 4 conditions × 200 questions × 32 layers × 4096 dim = very manageable.

---

## Phase 4: Forced Branch Analysis

**The core comparison.** For each question, you have four activation vectors at the pre-answer position (one per condition), each of shape [32, 4096]. Use the probe trained in Phase 2 (Probe A, Probe B) to decode the answer from each of these vectors. Plot the distribution of decoded probabilities (correct-answer logit) across conditions.

If reasoning is causally upstream: Condition 1 (correct reasoning) should give higher correct-answer probability than Condition 3 (wrong reasoning). The gap between C1 and C3 is your effect size.

If reasoning is post-hoc: C1, C2, C3 should all give similar correct-answer probabilities, and C4 (no reasoning) should not be dramatically different from C1.

**Layer profile.** Do this comparison at each layer separately. Plot the C1-vs-C3 gap as a function of layer. This gives you the layer profile of reasoning influence — analogous to your pre/post-commit window causal timing result from the sycophancy work, but now as a continuous layer curve rather than three windows.

---

## Phase 5: Causal Patching

This is the formal causal test. For matched pairs (same question, Condition 1 vs. Condition 3):

Run Condition 1 forward pass. Store the residual stream at layer l at the last reasoning token position. Call this h_C1(l).

Run Condition 3 forward pass. Store the same. Call this h_C3(l).

Patched run: run Condition 1 again, but at layer l, replace h_C1(l) with h_C3(l) at the last reasoning token position. Let the model continue forward from there. Measure the change in correct-answer log-probability at the answer token.

Do this for each layer l from 0 to 31. The result is a layer-wise causal effect curve: how much does substituting the Condition 3 representation (wrong reasoning) at layer l shift the answer distribution?

A peak in this curve at early layers indicates that early-layer representations of the reasoning chain causally influence the answer. A flat curve, or a peak only at the final layer, indicates the reasoning chain has little causal influence except at the readout layer.

**Null distribution.** Construct a null by patching from a different question's Condition 1 run (matched for answer label). The content-specific patch should produce larger effects than the question-mismatched null. This is your causal specificity check.

**N required.** You need at least 100 matched pairs for stable causal effect estimates. With 150–200 forced-branch questions, you should have roughly this many where both Condition 1 and Condition 3 are available.

---

## Phase 6: The Nonlinearity Characterization

Carry forward the MLP-linear gap analysis from the sycophancy work. For the same (position, layer) cells where you trained Probe A:

Train a 2-layer ReLU MLP probe (hidden dim 64) under the same cross-validation setup. Compare MLP AUROC against logistic regression AUROC. The gap is your nonlinearity index.

Plot nonlinearity index on the same commitment clock surface. The question: does nonlinearity peak in the same cells as the commitment? If early-layer commitment is accompanied by high nonlinearity and late-layer commitment by low nonlinearity, that is consistent with the Scenario B story from your sycophancy work: nonlinear upstream mechanism, linear downstream readout.

This is a modest add (you already have the probe infrastructure) and it connects the CoT paper to the sycophancy work's findings in a way that gives you a richer mechanistic story.

---

## What the Paper Claims, Precisely

**Claim 1 (Commitment clock).** The answer to a reasoning question commits in residual stream activation space substantially before the end of the expressed reasoning chain, for the majority of questions in the tested difficulty range. This is measured by Probe A AUROC exceeding a significance threshold (BH-corrected) in position bins Q1 or Q2 of the reasoning chain.

**Claim 2 (Difficulty gradient).** The commitment position (defined as the earliest position bin where Probe A AUROC is significantly above chance) is systematically earlier for easier questions than harder questions. This shows that early commitment is not a constant property of the architecture but varies with genuine reasoning demand.

**Claim 3 (Causal test).** Forcing different reasoning chains (Condition 1 vs. Condition 3) produces significantly different answer commitment geometry at the pre-answer position in middle-to-late layers, but smaller effects in early layers. The causal effect curve (patching result) peaks at [layer range TBD]. This shows that reasoning chain content is causally upstream of the answer representation at specific processing stages, even when the final output may not reflect this.

**Claim 4 (Post-hoc characterization).** For questions where commitment is early (pre-Q2), the causal patching effect of wrong-reasoning substitution is significantly smaller than for questions where commitment is late (post-Q3). This is the mechanistic definition of post-hoc rationalization: early commitment with low sensitivity to reasoning content.

**What you are not claiming.** You are not claiming CoT is always post-hoc. You are not claiming this generalizes to all model scales. You are not claiming the commitment clock is a complete theory of reasoning. The paper is a demonstration of the method and a characterization of when and to what degree reasoning is genuine vs. post-hoc for this model and these task types.

---

## Implementation Sequence and Timeline

** 1.** Environment setup, TransformerLens installation and sanity checks, dataset download and preprocessing, prompt format validation. Run the model on full dataset in free-generation mode. Compute filtering statistics and finalize filtered sets. Verify commitment point identification is working correctly on 20 examples by hand.

** 2.** Phase 1 extraction. Write the activation extraction loop with fractional position sampling. Run on full free-generation set for both datasets. Validate storage format and spot-check a sample of activation arrays for sanity (confirm that activations at position 0% vs 100% are actually different, confirm layer dimension is correct).

** 3.** Phase 2 probe training. Write the probe training loop over all (position, layer) cells. Generate the commitment clock heatmap for both Probe A and Probe B. Produce the stratified version (easy vs. hard). This is your first real result — if the commitment clock shows the expected early-commitment pattern, you have the paper's centerpiece.

** 4.** Phase 3 forced branch construction. This is the most labor-intensive phase. Collect Condition 1 and 4 examples (straightforward). Collect Condition 3 examples through re-sampling wrong-answer generations (run model with temperature > 0, collect until you have enough wrong answers). Construct Condition 2 using the truncation+append approach. Validate by hand that the reasoning chains in each condition have the expected properties.

** 5.** Phases 4 and 5. Forced branch activation extraction and comparison. Causal patching implementation and runs. Layer profile plots.

** 6.** Phase 6 nonlinearity characterization. Paper writing begins. Figure finalization.

---

## Figures

Figure 1: The commitment clock surface (AUROC heatmap, position × layer, Probe A). Split into correct-answer and wrong-answer subpanels.

Figure 2: Marginal commitment clock (AUROC averaged over layers, plotted against fractional position). Easy vs. hard questions on the same axes. The single clearest illustration of the main finding.

Figure 3: Forced branch comparison. Boxplots of correct-answer decoded probability at the pre-answer position, for Conditions 1, 2, 3, 4, across layers 8, 16, 24. Shows the effect of reasoning chain content on answer geometry.

Figure 4: Causal patching layer profile. Effect size (change in correct-answer log-prob under C3→C1 patch) as a function of layer, with null distribution shown. Demonstrates which layers are causally upstream.

Figure 5: Nonlinearity profile (MLP-linear gap on the commitment clock surface). Optional depending on space — if the paper is tight, move to appendix.

---

## The Key Risk

The biggest empirical risk: the commitment clock is flat — AUROC is uniformly low across all position bins, meaning the answer is not decodable from intermediate reasoning-token activations at any point. This would mean either the model is doing something genuinely distributed across all layers simultaneously (no localized commitment) or the probe is underpowered.

If this happens: increase N, try a more expressive probe class (MLP), check whether the commitment point definition is correct (maybe the model commits at a different token than assumed). Also check whether the BBH task shows the pattern even if StrategyQA doesn't — the multi-step structure of BBH may produce cleaner commitment signals.

The second risk: the forced branch construction for Condition 3 produces reasoning chains that are so obviously wrong (syntactically broken, incoherent) that the model corrects them at the output level regardless of the activation geometry. Validate this by checking model output token probabilities under forced conditions — if the model almost always outputs the correct answer even under Condition 3, the chain is not penetrating the processing.
