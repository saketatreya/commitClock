import argparse
import time
import sys

from src.experiments.phase1_free_gen import run_phase1_strategyqa
from src.experiments.phase2_probe import run_phase2
from src.experiments.phase3_forced_branch import run_phase3
from src.experiments.phase4_forced_analysis import run_phase4
from src.experiments.phase5_causal_patching import run_phase5
from src.experiments.phase6_nonlinearity import run_phase6

def print_header(phase_name):
    print("\n" + "="*80)
    print(f"🚀 STARTING {phase_name.upper()}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run the full Commitment Clock pipeline.")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of samples processed for a quick dry-run test."
    )
    args = parser.parse_args()

    start_time = time.time()

    if args.limit:
        print(f"⚠️  Running in dry-run mode with a limit of {args.limit} samples per phase.")

    try:
        # Phase 1
        print_header("Phase 1: Free Generation & Extraction")
        run_phase1_strategyqa(limit=args.limit)

        # Phase 2
        print_header("Phase 2: Probe Training")
        run_phase2()  # Uses data generated in Phase 1

        # Phase 3
        print_header("Phase 3: Forced Branch Construction")
        run_phase3()

        # Phase 4
        print_header("Phase 4: Forced Branch Analysis")
        run_phase4(limit=args.limit) # Note: Limit applies if we want to restrict analysis

        # Phase 5
        print_header("Phase 5: Causal Patching")
        run_phase5(limit=args.limit)

        # Phase 6
        print_header("Phase 6: Nonlinearity Characterization")
        run_phase6()

    except Exception as e:
        print("\n❌ PIPELINE FAILED!")
        print(f"Error: {str(e)}")
        sys.exit(1)

    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "="*80)
    print(f"✅ PIPELINE COMPLETE! Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("All outputs and figures have been saved to the 'data/' directory.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
