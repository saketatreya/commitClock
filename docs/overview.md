# Commitment Clock Project Overview

This project implements the method described in `commitClock.md` for measuring when a behavioral decision commits in LLM activation space.

## Architecture

The project is structured into the following components:

- `src/config.py`: Centralized configuration (model names, hyperparameters, paths).
- `src/data/`: Dataset loading and formatting (StrategyQA, BBH).
- `src/models/`: Loading and interacting with the LLM via TransformerLens.
- `src/experiments/`: The execution scripts for each phase of the project (e.g., free generation, extraction, probing).
- `src/analysis/`: Training probes, running causal patching, and generating figures.

## Pipeline Phases

1.  **Phase 1: Free-Generation Activation Extraction:** Generates CoT reasoning on datasets and extracts hidden states at 10 fractional positions across all layers.
2.  **Phase 2: Probe Training:** Trains logistic regression probes to predict model's answer and ground truth from activations to form the "Commitment Clock Surface".
3.  **Phase 3: Forced Branch Construction:** Prepares alternative reasoning paths (wrong reasoning -> correct answer, etc.).
4.  **Phase 4: Forced Branch Analysis:** Probes activations under different forced reasoning conditions.
5.  **Phase 5: Causal Patching:** Formally tests causal importance of early/late layers by patching wrong reasoning activations into correct runs.
6.  **Phase 6: Nonlinearity Characterization:** Compares MLP probes vs Linear probes.

## Datasets
- StrategyQA (Primary)
- BIG-Bench Hard (BBH) Logical Deduction (Secondary)

## Models
- `Qwen/Qwen2.5-7B-Instruct` (Primary)
- `Qwen/Qwen2.5-1.5B-Instruct` (Secondary, for scale comparison and local testing)
