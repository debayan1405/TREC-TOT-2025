"""
Main execution script for Reciprocal Rank Fusion (RRF) experiments.
This script identifies run files grouped by query re-writer, performs fusion,
and evaluates the results.
"""
from rrf_fusion import RRFusion
import argparse
import sys
from pathlib import Path

# Add project root to the Python path to allow sibling imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sparse_retrieval.config_loader import ConfigLoader

# CLEANED: Add project root to the Python path just once.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def find_env_path():
    """Finds the env.json file by searching in the project structure."""
    env_path = project_root / "env.json"
    if env_path.exists():
        return str(env_path)
    raise FileNotFoundError(
        "env.json not found in the project root directory.")


def main():
    """Main function to run the fusion and evaluation process."""
    parser = argparse.ArgumentParser(
        description="Run RRF Fusion and Evaluation")
    parser.add_argument(
        "--env",
        default=None,
        help="Path to env.json (auto-detected if not provided)"
    )
    parser.add_argument(
        "--dataset",
        default="train",
        choices=["train", "dev-1", "dev-2", "dev-3", "test"],
        help="Dataset version to process (default: train)"
    )
    args = parser.parse_args()

    print("=== RRF Fusion & Evaluation Experiment ===\n")

    try:
        env_path = args.env if args.env else find_env_path()
        print(f"Using env.json at: {env_path}")

        config = ConfigLoader(env_path)
        dataset_version = args.dataset

        print(f"Configuration loaded for dataset: '{dataset_version}'")
        print(f"Sparse Run Directory: {config.get_sparse_run_directory()}")
        print(f"Fusion Run Directory: {config.get_fusion_run_directory()}")
        print("-" * 40)

        # Initialize and run the fusion process
        fusion_process = RRFusion(config, dataset_version)
        fusion_process.run_fusion_and_evaluation()

        print("\n=== Experiment Completed Successfully! ===")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
