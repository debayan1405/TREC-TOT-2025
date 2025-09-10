"""
Main execution script for PyTerrier sparse retrieval experiments.
Updated to support multiple query sources, dynamic path management, and hardware optimizations.
"""
import os
import sys
from pathlib import Path

# Fix Java environment before importing PyTerrier
def setup_java_environment():
    """Set up Java environment for PyTerrier."""
    
    # Find the correct Java installation
    possible_java_paths = [
        "/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm",
        "/home/ugdf8/anaconda3/pkgs/openjdk-17.0.15-h5ddf6bc_0/lib/jvm",
        "/home/ugdf8/anaconda3/pkgs/openjdk-21.0.6-h38aa4c6_0/lib",
        "/home/ugdf8/.vscode/extensions/redhat.java-1.45.0-linux-x64/jre/21.0.8-linux-x86_64",
    ]
    
    java_home = None
    for path in possible_java_paths:
        libjvm_path = Path(path) / "lib" / "server" / "libjvm.so"
        if libjvm_path.exists():
            java_home = path
            print(f"✓ Found Java at: {java_home}")
            break
    
    if not java_home:
        print("ERROR: Could not find a working Java installation")
        sys.exit(1)
    
    # Set environment variables
    os.environ['JAVA_HOME'] = java_home
    os.environ['LD_LIBRARY_PATH'] = f"{java_home}/lib/server:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ['JVM_PATH'] = f"{java_home}/lib/server/libjvm.so"
    
    print(f"✓ Set JAVA_HOME to: {java_home}")
    return java_home

# Setup Java environment before importing PyTerrier
setup_java_environment()

import pyterrier as pt
import pandas as pd
import argparse
import time
from config_loader import ConfigLoader
from data_loader import DataLoader
from sparse_retrieval import SparseRetrieval


# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE VARIABLES AS NEEDED
# =============================================================================

# Dataset to run experiments on (CHANGE THIS AS NEEDED)
# Options: "train", "dev-1", "dev-2", "dev-3", "test"
DATASET_VERSION = "train"  # <-- CHANGE THIS LINE TO USE DIFFERENT DATASETS

# Query sources to test (None means auto-detect all available sources)
QUERY_SOURCES = ["rewritten_llama"]
# QUERY_SOURCES = ["original", "rewritten_llama", "rewritten_mistral", "rewritten_qwen", "summarized"]

# Models to run (None means all supported models)
# Options: ["BM25", "PL2", "TF_IDF"]
# Set to None for all models, or specify a list like ["BM25", "PL2"]
MODELS_TO_RUN = None  # <-- CHANGE THIS TO SPECIFY SPECIFIC MODELS

# Force rerun even if cached results exist
FORCE_RERUN = False  # <-- CHANGE THIS TO True TO FORCE RERUN

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================


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


def main(env_path: str = None, override_dataset: str = None,
         override_query_sources: list = None, override_models: list = None,
         override_force_rerun: bool = None):
    """
    Main execution function with hard-coded configuration support.

    Args:
        env_path: Path to env.json (auto-detected if None)
        override_dataset: Override hard-coded dataset version
        override_query_sources: Override hard-coded query sources
        override_models: Override hard-coded models
        override_force_rerun: Override hard-coded force rerun setting
    """
    try:
        # Initialize PyTerrier with optimized settings
        if not hasattr(pt, 'java') or not pt.java.started():
            print("🔧 Initializing PyTerrier with optimized settings...")
            pt.init(mem=8192, logging="WARN")  # 8GB memory allocation
            print("✓ PyTerrier initialized successfully")
        else:
            print("✓ PyTerrier already initialized")

        print("=== PyTerrier Sparse Retrieval Experiment ===")
        print("🚀 Optimized for high-end hardware (700+ GB RAM, 2x A6000 GPUs)")

        # Auto-detect env.json path if not provided
        if env_path is None:
            env_path = find_env_path()
            print(f"Using env.json at: {env_path}")

        # Load configuration
        print("Loading configuration...")
        config = ConfigLoader(env_path)
        print(f"Configuration loaded successfully")
        print(f"Index path: {config.get_index_path()}")
        print(f"Sparse run directory: {config.get_sparse_run_directory()}")
        print(f"Evaluation directory: {config.get_evaluation_directory()}")
        print(f"K-sparse: {config.get_k_sparse()}")
        print(f"Evaluation metrics: {config.get_eval_metrics()}\n")

        # Initialize data loader and sparse retrieval
        print("Initializing data loader...")
        data_loader = DataLoader(config)

        print("Loading index...")
        index = data_loader.get_index()
        print(f"Index loaded successfully\n")

        print("Initializing sparse retrieval...")
        sparse_retrieval = SparseRetrieval(config, data_loader)

        # Use hard-coded configuration or overrides
        dataset_version = override_dataset if override_dataset else DATASET_VERSION
        query_sources = override_query_sources if override_query_sources else QUERY_SOURCES
        models_to_run = override_models if override_models else MODELS_TO_RUN
        force_rerun = override_force_rerun if override_force_rerun is not None else FORCE_RERUN

        # Validate dataset
        available_datasets = ["train", "dev-1", "dev-2", "dev-3", "test"]
        if dataset_version not in available_datasets:
            raise ValueError(f"Invalid dataset version: {dataset_version}. "
                             f"Valid options: {available_datasets}")

        # Validate models if specified
        if models_to_run is not None:
            invalid_models = [
                m for m in models_to_run if m not in SparseRetrieval.SUPPORTED_MODELS]
            if invalid_models:
                raise ValueError(f"Invalid models: {invalid_models}. "
                                 f"Valid options: {SparseRetrieval.SUPPORTED_MODELS}")
        else:
            models_to_run = SparseRetrieval.SUPPORTED_MODELS

        # Check if dataset has topics
        try:
            test_topics = data_loader.load_topics(dataset_version, "original")
            print(
                f"Dataset '{dataset_version}' is available with {len(test_topics)} topics")
        except Exception as e:
            raise ValueError(
                f"Dataset '{dataset_version}' is not available: {e}")

        print(f"\n{'='*80}")
        print(f"EXPERIMENT CONFIGURATION")
        print(f"{'='*80}")
        print(f"Dataset version: {dataset_version}")
        print(f"Models to run: {models_to_run}")
        print(
            f"Query sources: {'Auto-detect' if query_sources is None else query_sources}")
        print(f"Force rerun: {force_rerun}")

        # Run experiment for the specified dataset
        print(f"\n{'='*80}")
        print(f"Processing Dataset: {dataset_version}")
        print(f"{'='*80}")

        # Auto-detect available query sources for this dataset if not specified
        if query_sources is None:
            available_sources = data_loader.get_available_rewritten_sources(
                dataset_version)
            print(
                f"Available query sources for {dataset_version}: {available_sources}")
            query_sources = available_sources
        else:
            # Validate specified query sources
            available_sources = data_loader.get_available_rewritten_sources(
                dataset_version)
            invalid_sources = [
                s for s in query_sources if s not in available_sources]
            if invalid_sources:
                print(f"Warning: Invalid query sources: {invalid_sources}")
                query_sources = [
                    s for s in query_sources if s in available_sources]

            if not query_sources:
                raise ValueError("No valid query sources specified")

        print(f"Testing query sources: {query_sources}")

        # Run experiment
        dataset_results = sparse_retrieval.run_single_dataset_experiment(
            dataset_version=dataset_version,
            query_sources=query_sources,
            models=models_to_run,
            force_rerun=force_rerun
        )

        # Display results summary
        print(f"\n{'-'*80}")
        print(f"RESULTS SUMMARY FOR {dataset_version.upper()}")
        print(f"{'-'*80}")

        for source, experiment_result in dataset_results.items():
            rewriter_name = sparse_retrieval._get_rewriter_name(source)
            print(f"\n{rewriter_name} queries:")

            if isinstance(experiment_result, pd.DataFrame) and not experiment_result.empty:
                print(experiment_result.to_string())
            elif isinstance(experiment_result, dict):
                print(
                    f"Retrieval completed for {len(experiment_result)} models")
                for model, results_df in experiment_result.items():
                    if isinstance(results_df, pd.DataFrame):
                        print(f"  {model}: {len(results_df)} results")
                    else:
                        print(f"  {model}: No results")
            else:
                print("No results")

        # Final summary
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")

        successful_sources = len([r for r in dataset_results.values()
                                  if (isinstance(r, pd.DataFrame) and not r.empty) or
                                  (isinstance(r, dict) and r)])

        print(f"Dataset processed: {dataset_version}")
        print(f"Query sources tested: {len(query_sources)}")
        print(f"Successful experiments: {successful_sources}")
        print(f"Models tested: {models_to_run}")
        print(f"Sparse run directory: {config.get_sparse_run_directory()}")
        print(f"Evaluation directory: {config.get_evaluation_directory()}")

        # List output files created
        run_dir = Path(config.get_sparse_run_directory()) / dataset_version
        eval_dir = Path(config.get_evaluation_directory()) / \
            "evals" / dataset_version

        if run_dir.exists():
            run_files = list(run_dir.glob("*.txt"))
            print(f"\nRun files created: {len(run_files)}")
            for run_file in sorted(run_files):
                print(f"  {run_file.name}")

        if eval_dir.exists():
            eval_files = list(eval_dir.glob("*.csv"))
            print(f"\nEvaluation files created: {len(eval_files)}")
            for eval_file in sorted(eval_files):
                print(f"  {eval_file.name}")

        print("\nExperiment completed successfully!")

    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)


def run_command_line_mode():
    """Run with command line arguments (for flexibility)."""
    parser = argparse.ArgumentParser(
        description="Run PyTerrier sparse retrieval experiments"
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to env.json (auto-detected if not provided)"
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "dev-1", "dev-2", "dev-3", "test"],
        help="Dataset version to process (overrides hard-coded value)"
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Query sources to test (e.g., original rewritten_llama summarized)"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=SparseRetrieval.SUPPORTED_MODELS,
        help="Retrieval models to run (overrides hard-coded value)"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-running experiments even if cached results exist"
    )

    args = parser.parse_args()

    main(
        env_path=args.env,
        override_dataset=args.dataset,
        override_query_sources=args.sources,
        override_models=args.models,
        override_force_rerun=args.force_rerun
    )


def run_example_configurations():
    """Examples of different configuration scenarios."""
    print("=== Example Configurations ===\n")

    examples = [
        {
            "name": "Train dataset with all sources",
            "dataset": "train",
            "sources": None,  # Auto-detect
            "models": ["BM25", "PL2"],
        },
        {
            "name": "Dev-1 dataset with original queries only",
            "dataset": "dev-1",
            "sources": ["original"],
            "models": None,  # All models
        },
        {
            "name": "Test dataset with rewritten queries",
            "dataset": "test",
            "sources": ["rewritten_llama", "rewritten_mistral"],
            "models": ["BM25"],
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Dataset: {example['dataset']}")
        print(f"   Sources: {example['sources']}")
        print(f"   Models: {example['models']}")
        print()


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        run_command_line_mode()
    else:
        # Use hard-coded configuration
        main()

    # Uncomment to see example configurations
    # run_example_configurations()
