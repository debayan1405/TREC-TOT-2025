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
import argparse
import time
from config_loader import ConfigLoader
from data_loader import DataLoader
from sparse_retrieval import SparseRetrieval


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


def main(env_path: str = None, datasets_to_run: list = None,
         query_sources: list = None, models_to_run: list = None,
         force_rerun: bool = False, use_parallel: bool = True, n_threads: int = None):
    """
    Main execution function with comprehensive experiment support and hardware optimizations.

    Args:
        env_path: Path to env.json (auto-detected if None)
        datasets_to_run: Dataset versions to process (all if None)
        query_sources: Query sources to test (auto-detected if None)
        models_to_run: Retrieval models to run (all if None)
        force_rerun: Force re-running experiments
        use_parallel: Use parallel processing for better performance
        n_threads: Number of threads for parallel processing
    """
    try:
        # Initialize PyTerrier with better configuration
        if not hasattr(pt, 'java') or not pt.java.started():
            print("🔧 Initializing PyTerrier with optimized settings...")
            pt.init(mem=8192, logging="WARN")  # Use integer for memory in MB
            print("✓ PyTerrier initialized successfully")
        else:
            print("✓ PyTerrier already initialized")

        print("=== PyTerrier Sparse Retrieval Experiment ===")
        print("🚀 Optimized for high-end hardware (700+ GB RAM, 2x A6000 GPUs)")
        print(f"🔧 Parallel processing: {'Enabled' if use_parallel else 'Disabled'}")
        if n_threads:
            print(f"🧵 Using {n_threads} threads")
        print()

        # Auto-detect env.json path if not provided
        if env_path is None:
            env_path = find_env_path()
            print(f"Using env.json at: {env_path}")

        # Load configuration
        print("Loading configuration...")
        config = ConfigLoader(env_path)
        print(f"✓ Configuration loaded successfully")
        print(f"📁 Index path: {config.get_index_path()}")
        print(f"📁 Run directory: {config.get_run_directory()}")
        print(f"📊 K-sparse: {config.get_k_sparse()}")
        print(f"📊 Evaluation metrics: {config.get_eval_metrics()}\n")

        # Initialize data loader and sparse retrieval with optimizations
        print("Initializing components...")
        data_loader = DataLoader(config)

        print("Loading index...")
        index = data_loader.get_index()
        print(f"✓ Index loaded successfully\n")

        print("Initializing optimized sparse retrieval...")
        sparse_retrieval = SparseRetrieval(config, data_loader, n_threads=n_threads)
        print(f"✓ Sparse retrieval initialized with hardware optimizations\n")

        # Determine datasets to process
        available_datasets = ["train", "dev-1", "dev-2", "dev-3", "test"]
        if datasets_to_run is None:
            # Check which datasets have topic files
            datasets_to_run = []
            for dataset in available_datasets:
                try:
                    topics = data_loader.load_topics(dataset, "original")
                    datasets_to_run.append(dataset)
                except Exception:
                    continue
            print(f"Auto-detected datasets: {datasets_to_run}")
        else:
            # Validate requested datasets
            invalid_datasets = [
                d for d in datasets_to_run if d not in available_datasets]
            if invalid_datasets:
                print(f"Warning: Invalid dataset names: {invalid_datasets}")
                datasets_to_run = [
                    d for d in datasets_to_run if d in available_datasets]

        if not datasets_to_run:
            raise ValueError("No valid datasets to process")

        # Validate models
        if models_to_run is None:
            models_to_run = SparseRetrieval.SUPPORTED_MODELS
        else:
            invalid_models = [
                m for m in models_to_run if m not in SparseRetrieval.SUPPORTED_MODELS]
            if invalid_models:
                raise ValueError(f"Invalid models: {invalid_models}")

        print(f"🎯 Processing datasets: {datasets_to_run}")
        print(f"🔍 Using models: {models_to_run}")
        if query_sources:
            print(f"📝 Query sources: {query_sources}")
        print()

        # Check existing files and show status
        print(f"{'='*80}")
        print("CHECKING EXISTING EXPERIMENT RESULTS")
        print(f"{'='*80}")
        
        total_combinations = 0
        existing_combinations = 0
        
        for dataset_version in datasets_to_run:
            available_sources = data_loader.get_available_rewritten_sources(dataset_version)
            sources_to_test = query_sources if query_sources else available_sources
            sources_to_test = [s for s in sources_to_test if s in available_sources]
            
            for source in sources_to_test:
                for model in models_to_run:
                    total_combinations += 1
                    exists = sparse_retrieval._check_existing_output(model, source, dataset_version)
                    if exists:
                        existing_combinations += 1
                    status = "✓ EXISTS" if exists else "✗ MISSING"
                    print(f"  {dataset_version}_{source}_{model}: {status}")
        
        print(f"\nStatus: {existing_combinations}/{total_combinations} combinations already completed")
        if not force_rerun and existing_combinations == total_combinations:
            print("🎉 All experiments already completed! Use --force-rerun to regenerate.")
            return
        print()

        # Main experiment loop with timing
        all_results = {}
        start_time = time.time()

        for i, dataset_version in enumerate(datasets_to_run, 1):
            print(f"\n{'='*80}")
            print(f"Processing Dataset {i}/{len(datasets_to_run)}: {dataset_version}")
            print(f"{'='*80}")

            try:
                # Auto-detect available query sources for this dataset
                available_sources = data_loader.get_available_rewritten_sources(
                    dataset_version)
                print(
                    f"Available query sources for {dataset_version}: {available_sources}")

                # Filter by requested sources if specified
                sources_to_test = query_sources if query_sources else available_sources
                sources_to_test = [
                    s for s in sources_to_test if s in available_sources]

                if not sources_to_test:
                    print(f"No valid query sources for {dataset_version}")
                    continue

                print(f"Testing query sources: {sources_to_test}")

                # Run multi-source experiment for this dataset with optimizations
                dataset_results = sparse_retrieval.run_multi_source_experiment(
                    dataset_version=dataset_version,
                    query_sources=sources_to_test,
                    models=models_to_run,
                    force_rerun=force_rerun,
                    use_parallel=use_parallel
                )

                all_results[dataset_version] = dataset_results

                # Display results summary for this dataset
                print(f"\n{'-'*60}")
                print(f"Results Summary for {dataset_version}:")
                print(f"{'-'*60}")

                for source, experiment_df in dataset_results.items():
                    if not experiment_df.empty:
                        print(f"\n{source} queries:")
                        print(experiment_df.to_string())
                    else:
                        print(f"\n{source} queries: No results")

            except Exception as e:
                print(f"✗ Error processing dataset {dataset_version}: {e}")
                continue

        # Final summary with timing
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("🏁 EXPERIMENT SUMMARY")
        print(f"{'='*80}")

        total_experiments = sum(len(results)
                                for results in all_results.values())
        successful_experiments = sum(
            len([r for r in results.values() if not r.empty])
            for results in all_results.values()
        )

        print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Datasets processed: {len(all_results)}")
        print(f"🧪 Total experiments: {total_experiments}")
        print(f"✅ Successful experiments: {successful_experiments}")
        print(f"🔍 Models tested: {models_to_run}")
        print(f"🚀 Parallel processing: {'Used' if use_parallel else 'Not used'}")
        print(f"💾 Results directory: {config.get_run_directory()}")

        # List all output directories created
        run_dir = Path(config.get_run_directory())
        if run_dir.exists():
            output_dirs = [d for d in run_dir.rglob("*") if d.is_dir()]
            if output_dirs:
                print(f"\n📁 Output directories created:")
                for out_dir in sorted(output_dirs):
                    rel_path = out_dir.relative_to(run_dir)
                    file_count = len([f for f in out_dir.glob("*.txt")])
                    csv_count = len([f for f in out_dir.glob("*.csv")])
                    print(f"   {rel_path}/ ({file_count} result files, {csv_count} experiment files)")

        print(f"\n🎉 Experiment completed successfully in {total_time:.2f}s!")
        
        # Performance summary
        if total_experiments > 0:
            avg_time_per_experiment = total_time / total_experiments
            print(f"⚡ Average time per experiment: {avg_time_per_experiment:.2f}s")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        sys.exit(1)


def run_single_dataset_example(dataset_version: str = "train"):
    """Example of running experiments for a single dataset."""
    try:
        if not hasattr(pt, 'java') or not pt.java.started():
            pt.init(mem=8192, logging="WARN")

        print(f"=== Single Dataset Example: {dataset_version} ===")

        env_path = find_env_path()
        config = ConfigLoader(env_path)
        data_loader = DataLoader(config)
        sparse_retrieval = SparseRetrieval(config, data_loader)

        # Run experiments for all available query sources
        results = sparse_retrieval.run_multi_source_experiment(
            dataset_version=dataset_version,
            force_rerun=False
        )

        print(f"\nResults for {dataset_version}:")
        for source, experiment_df in results.items():
            print(f"\n{source}:")
            if not experiment_df.empty:
                print(experiment_df)
            else:
                print("No results")

    except Exception as e:
        print(f"Error in single dataset example: {e}")


def run_specific_source_example(dataset_version: str = "train", query_source: str = "original"):
    """Example of running with a specific query source."""
    try:
        if not hasattr(pt, 'java') or not pt.java.started():
            pt.init(mem=8192, logging="WARN")

        print(
            f"=== Specific Source Example: {dataset_version} with {query_source} ===")

        env_path = find_env_path()
        config = ConfigLoader(env_path)
        data_loader = DataLoader(config)

        # Load topics and qrels
        topics = data_loader.load_topics(dataset_version, query_source)
        qrels = data_loader.load_qrels(dataset_version)

        sparse_retrieval = SparseRetrieval(config, data_loader)

        # Run experiment
        results = sparse_retrieval.run_experiment(
            topics=topics,
            qrels=qrels,
            query_source=query_source,
            dataset_version=dataset_version
        )

        print(f"\nResults for {dataset_version} with {query_source} queries:")
        print(results)

    except Exception as e:
        print(f"Error in specific source example: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PyTerrier sparse retrieval experiments with multiple query sources and hardware optimizations"
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
        help="Dataset versions to process (all available if not specified)"
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Query sources to test (e.g., original rewritten_llama summarized). Auto-detected if not specified."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=SparseRetrieval.SUPPORTED_MODELS,
        help="Retrieval models to run (all if not specified)"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-running experiments even if cached results exist"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (use sequential processing instead)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for parallel processing (auto-detected if not specified)"
    )
    parser.add_argument(
        "--example",
        choices=["single", "specific"],
        help="Run example instead of main experiment"
    )

    args = parser.parse_args()

    if args.example == "single":
        dataset = args.datasets[0] if args.datasets else "train"
        run_single_dataset_example(dataset)
    elif args.example == "specific":
        dataset = args.datasets[0] if args.datasets else "train"
        source = args.sources[0] if args.sources else "original"
        run_specific_source_example(dataset, source)
    else:
        main(
            env_path=args.env,
            datasets_to_run=args.datasets,
            query_sources=args.sources,
            models_to_run=args.models,
            force_rerun=args.force_rerun,
            use_parallel=not args.no_parallel,
            n_threads=args.threads
        )
