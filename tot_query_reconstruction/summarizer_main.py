# summarizer_main.py
import json
from pathlib import Path
import argparse
import sys
from summarizer import summarize_all_datasets
from rewriter import validate_env


def find_env_path():
    """Find env.json file, looking in current directory and parent directory."""
    current_dir = Path(__file__).parent

    # Check current directory first
    env_path = current_dir / "env.json"
    if env_path.exists():
        return str(env_path)

    # Check parent directory
    env_path = current_dir.parent / "env.json"
    if env_path.exists():
        return str(env_path)

    # Check two levels up (for deeply nested structures)
    env_path = current_dir.parent.parent / "env.json"
    if env_path.exists():
        return str(env_path)

    raise FileNotFoundError(
        "env.json not found in current directory, parent directory, or grandparent directory")


def main(env_path: str = None, datasets_to_run: list = None, batch_size: int = 8):
    """
    Main function to run query summarization with optimized hardware utilization.

    Args:
        env_path: Path to env.json file (auto-detected if None)
        datasets_to_run: List of dataset versions to summarize (all if None)
        batch_size: Number of queries to process in parallel
    """
    try:
        # Auto-detect env.json path if not provided
        if env_path is None:
            env_path = find_env_path()
            print(f"Using env.json at: {env_path}")

        # Load and validate environment
        with open(env_path, "r", encoding="utf-8") as f:
            env = json.load(f)
        validate_env(env)

        hf_token = env.get("hf_token", None)
        if hf_token == "":
            hf_token = None

        rewritten_dir = Path(env["paths"]["rewritten_queries_directory"])

        if not rewritten_dir.exists():
            raise FileNotFoundError(
                f"Rewritten queries directory not found: {rewritten_dir}")

        print(f"Looking for rewritten query files in: {rewritten_dir}")
        print(f"Hardware optimization: Using batch size {batch_size} for 2x A6000 GPUs and 700+ GB RAM")

        # List available rewritten query files
        rewritten_files = list(rewritten_dir.glob("*_rewritten_queries.jsonl"))
        if not rewritten_files:
            raise FileNotFoundError(
                f"No rewritten query files found in {rewritten_dir}")

        print(f"Found rewritten query files:")
        for f in sorted(rewritten_files):
            print(f"  - {f.name}")

        # Validate datasets to run
        if datasets_to_run:
            available_datasets = set()
            for f in rewritten_files:
                # Extract dataset version from filename: model_dataset_rewritten_queries.jsonl
                parts = f.stem.split("_")
                if len(parts) >= 3 and parts[-2] == "rewritten":
                    # Handle multi-part dataset names like "dev-1"
                    dataset_version = "_".join(parts[1:-2])
                    available_datasets.add(dataset_version)

            invalid_datasets = [
                d for d in datasets_to_run if d not in available_datasets]
            if invalid_datasets:
                print(
                    f"Warning: No rewritten files found for datasets: {invalid_datasets}")
                datasets_to_run = [
                    d for d in datasets_to_run if d in available_datasets]

            if not datasets_to_run:
                raise ValueError("No valid datasets to process")

        print(f"\n{'='*60}")
        print("Starting query summarization with optimized batch processing")
        print(f"{'='*60}")

        # Run summarization
        results = summarize_all_datasets(
            env=env,
            rewritten_dir=rewritten_dir,
            datasets_to_run=datasets_to_run,
            hf_token=hf_token,
            batch_size=batch_size
        )

        print(f"\n{'='*60}")
        print("Summarization completed!")
        print(f"{'='*60}")

        if results:
            print("Output files created:")
            for dataset, output_path in results.items():
                print(f"  {dataset}: {Path(output_path).name}")
        else:
            print("No datasets were successfully processed.")

    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize rewritten queries from multiple models with optimized batch processing"
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to env.json (auto-detected if not provided)"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=["train", "dev-1", "dev-2", "dev-3", "test"],
        help="Subset of datasets to summarize. All available datasets if not specified."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of queries to process in parallel (default: 8, optimized for high-end GPUs)"
    )

    args = parser.parse_args()
    main(args.env, args.datasets, args.batch_size)
