# Running the Commitment Clock Experiments

This document describes how to execute the phases of the Commitment Clock project.

## Prerequisites

1. Set up a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install torch transformers transformer-lens datasets scikit-learn matplotlib pandas tqdm requests
   ```
3. Set your Python path to the root of the project to allow local imports:
   ```bash
   export PYTHONPATH=.
   ```

## Execution Flow

The project is structured sequentially. Phase N generally depends on data generated in Phase N-1.

### Phase 1: Free Generation
Extracts activations during the model's natural chain-of-thought generation.
```bash
python src/experiments/phase1_free_gen.py
```
**Output:** `data/phase1/strategyqa_activations.pkl`

### Phase 2: Probe Training
Trains logistic regression probes and generates the Commitment Clock surface.
```bash
python src/experiments/phase2_probe.py
```
**Output:** `data/phase2/commitment_clock.png`, `data/phase2/marginal_clock.png`

### Phase 3: Forced Branch Construction
Constructs counterfactual reasoning paths (wrong reasoning, right reasoning, no reasoning).
```bash
python src/experiments/phase3_forced_branch.py
```
**Output:** `data/phase3/forced_branches.pkl`

### Phase 4: Forced Branch Analysis
Extracts pre-answer activations for the different forced branches.
```bash
python src/experiments/phase4_forced_analysis.py
```
**Output:** `data/phase4/forced_activations.pkl`

### Phase 5: Causal Patching
Patches wrong-reasoning representations into right-reasoning runs to measure causal effects.
```bash
python src/experiments/phase5_causal_patching.py
```
**Output:** `data/phase5/causal_patching.pkl`, `data/phase5/causal_effect.png`

### Phase 6: Nonlinearity Characterization
Compares linear probes against MLP probes to measure nonlinearity.
```bash
python src/experiments/phase6_nonlinearity.py
```
**Output:** `data/phase6/nonlinearity_results.pkl`, `data/phase6/nonlinearity_index.png`
