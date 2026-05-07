import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Model Configuration
# For local testing, smaller models are tractable.
# Changed to Qwen3.5-9B per user request for final run on Kaggle/Cloud
MODEL_NAME = "Qwen/Qwen3.5-9B"

# Extraction configuration
NUM_FRACTIONAL_POSITIONS = 10
# For Qwen2.5-1.5B: 28 layers. Qwen2.5-7B has 32 layers. 
# TransformerLens will auto-detect, but we can set defaults.

# Filtering Configuration
MIN_CHAIN_LENGTH = 40
MAX_CHAIN_LENGTH = 250

# Output Paths
PHASE1_OUT_DIR = DATA_DIR / "phase1"
PHASE2_OUT_DIR = DATA_DIR / "phase2"
PHASE3_OUT_DIR = DATA_DIR / "phase3"
PHASE4_OUT_DIR = DATA_DIR / "phase4"
PHASE5_OUT_DIR = DATA_DIR / "phase5"
PHASE6_OUT_DIR = DATA_DIR / "phase6"

for dir_path in [DATA_DIR, PHASE1_OUT_DIR, PHASE2_OUT_DIR, PHASE3_OUT_DIR, PHASE4_OUT_DIR, PHASE5_OUT_DIR, PHASE6_OUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
