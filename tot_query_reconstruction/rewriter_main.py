# rewriter_main.py
import json
import os
from pathlib import Path
import argparse
import sys
from rewriter import validate_env, rewrite_topic_set


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


def main(env_path: str = None, models_to_run=None, datasets_to_run=None):
    """
    Main function to run query rewriting.

    Args:
        env_path: Path to env.json file (auto-detected if None)
        models_to_run: List of model names to run (all if None)
        datasets_to_run: List of dataset versions to run (all if None)
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

        paths = env["paths"]
        hf_token = env.get("hf_token", None)
        if hf_token == "":
            hf_token = None

        generation_conf = env.get("generation", {})
        preview_count = env.get("logging", {}).get("preview_count", 5)

        # Dataset version mapping
        version_map = {
            "train": paths["train_topics_path"],
            "dev-1": paths["dev_1_topics_path"],
            "dev-2": paths["dev_2_topics_path"],
            "dev-3": paths["dev_3_topics_path"],
            "test": paths["test_topics_path"]
        }

        # Determine which models to run
        if models_to_run is None:
            models_to_run = list(env["models"].keys())
        else:
            # Validate requested models exist
            invalid_models = [
                m for m in models_to_run if m not in env["models"]]
            if invalid_models:
                raise ValueError(f"Invalid model names: {invalid_models}")

        # Determine which datasets to run
        if datasets_to_run is None:
            datasets_to_run = list(version_map.keys())
        else:
            # Validate requested datasets exist
            invalid_datasets = [
                d for d in datasets_to_run if d not in version_map]
            if invalid_datasets:
                raise ValueError(f"Invalid dataset names: {invalid_datasets}")

        print(f"Running rewriting for:")
        print(f"  Models: {models_to_run}")
        print(f"  Datasets: {datasets_to_run}")

        # Run rewriting for each model and dataset combination
        for model_name in models_to_run:
            print(f"\n{'='*60}")
            print(f"Running rewrites for model: {model_name}")
            print(f"{'='*60}")

            model_cfg = env["models"][model_name]
            hf_id = model_cfg["hf_id"]
            bnb_conf = model_cfg.get("bitsandbytes", {})

            for dataset_version in datasets_to_run:
                print(f"\n{'-'*40}")
                print(f"Dataset version: {dataset_version}")
                print(f"{'-'*40}")

                topic_path = version_map[dataset_version]
                output_dir = paths["rewritten_queries_directory"]

                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)

                try:
                    rewrite_topic_set(
                        model_name=model_name,
                        model_hf_id=hf_id,
                        bnb_conf=bnb_conf,
                        topic_file=topic_path,
                        output_dir=output_dir,
                        generation_conf=generation_conf,
                        dataset_version=dataset_version,
                        preview_count=preview_count,
                        hf_token=hf_token
                    )
                    print(
                        f"✓ Successfully completed {model_name} on {dataset_version}")

                except Exception as e:
                    print(
                        f"✗ Error processing {model_name} on {dataset_version}: {e}")
                    continue

        print(f"\n{'='*60}")
        print("Rewriting process completed!")
        print(f"Results saved in: {paths['rewritten_queries_directory']}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run query rewriting using multiple models and datasets"
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to env.json (auto-detected if not provided)"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Subset of model keys to run (e.g., llama mistral). All models if not specified."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=["train", "dev-1", "dev-2", "dev-3", "test"],
        help="Subset of datasets to run. All datasets if not specified."
    )

    args = parser.parse_args()
    main(args.env, args.models, args.datasets)
