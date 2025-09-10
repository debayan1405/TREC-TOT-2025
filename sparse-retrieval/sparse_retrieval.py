"""
Sparse retrieval module for PyTerrier experiments.
Implements wrapper functions for BM25, PL2, and TF_IDF retrievers with caching.
Updated with dynamic path creation and high-performance optimizations.
"""
import os
import pandas as pd
import pyterrier as pt
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import multiprocessing
from config_loader import ConfigLoader
from data_loader import DataLoader


class SparseRetrieval:
    """Handles sparse retrieval algorithms with dynamic path management and performance optimizations."""

    # Supported sparse retrieval models
    SUPPORTED_MODELS = ["BM25", "PL2", "TF_IDF"]

    def __init__(self, config: ConfigLoader, data_loader: DataLoader, n_threads: int = None):
        """
        Initialize sparse retrieval handler with performance optimizations.

        Args:
            config (ConfigLoader): Configuration loader instance
            data_loader (DataLoader): Data loader instance
            n_threads (int, optional): Number of threads for parallel processing. 
                                     Defaults to CPU count for high-memory systems.
        """
        self.config = config
        self.data_loader = data_loader
        self.index = data_loader.get_index()
        
        # Optimize for high-end hardware with lots of RAM
        if n_threads is None:
            # Use more threads for systems with >64GB RAM
            self.n_threads = min(multiprocessing.cpu_count() * 2, 32)
        else:
            self.n_threads = n_threads
            
        print(f"Initialized SparseRetrieval with {self.n_threads} threads for parallel processing")

        # Set up PyTerrier for high-performance
        self._configure_pyterrier_performance()

    def _configure_pyterrier_performance(self):
        """Configure PyTerrier for optimal performance on high-end hardware."""
        try:
            # Set Java heap size for large memory systems (use up to 64GB)
            import os
            java_opts = [
                "-Xmx64g",  # Maximum heap size
                "-Xms8g",   # Initial heap size  
                "-XX:+UseG1GC",  # Use G1 garbage collector for large heaps
                "-XX:+UseStringDeduplication",  # Reduce memory usage
                "-XX:ParallelGCThreads=16",  # Parallel GC threads
                "-XX:ConcGCThreads=4",  # Concurrent GC threads
                "-Djava.awt.headless=true",  # Headless mode
            ]
            
            # Set environment variable for PyTerrier
            existing_opts = os.environ.get("JAVA_OPTS", "")
            new_opts = " ".join(java_opts)
            if existing_opts:
                os.environ["JAVA_OPTS"] = f"{existing_opts} {new_opts}"
            else:
                os.environ["JAVA_OPTS"] = new_opts
                
            print(f"Configured Java with high-performance settings: {new_opts}")
            
        except Exception as e:
            print(f"Warning: Could not configure Java performance settings: {e}")

    def _check_existing_output(self, algorithm_name: str, query_source: str, dataset_version: str) -> bool:
        """
        Check if output file already exists and is non-empty.
        
        Args:
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version
            
        Returns:
            bool: True if file exists and is non-empty, False otherwise
        """
        output_file = self._get_output_filename(algorithm_name, query_source, dataset_version)
        
        if os.path.exists(output_file):
            try:
                # Check if file is non-empty
                if os.path.getsize(output_file) > 0:
                    # Quick validation by reading first few lines
                    with open(output_file, 'r') as f:
                        lines = [f.readline().strip() for _ in range(3)]
                        if any(lines):  # At least one non-empty line
                            print(f"Found existing output: {output_file}")
                            return True
            except Exception as e:
                print(f"Warning: Error checking existing file {output_file}: {e}")
        
        return False
        
        # Set up PyTerrier for high-performance
        self._configure_pyterrier_performance()

    def _get_output_directory(self, query_source: str, dataset_version: str = "train") -> str:
        """
        Generate dynamic output directory to prevent overwriting.

        Args:
            query_source (str): Source of queries (original, rewritten_<model>, summarized)
            dataset_version (str): Dataset version (train, dev-1, etc.)

        Returns:
            str: Path to output directory
        """
        base_run_dir = Path(self.config.get_run_directory())

        # Create hierarchical structure: run_directory/dataset_version/query_source/
        output_dir = base_run_dir / dataset_version / query_source
        output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir)

    def _get_output_filename(self, algorithm_name: str, query_source: str,
                             dataset_version: str = "train") -> str:
        """
        Generate output filename for algorithm results with dynamic paths.

        Args:
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version

        Returns:
            str: Full path to output file
        """
        output_dir = self._get_output_directory(query_source, dataset_version)
        k_sparse = self.config.get_k_sparse()
        filename = f"{algorithm_name.lower()}_{k_sparse}.txt"
        return os.path.join(output_dir, filename)

    def _save_results(self, results_df: pd.DataFrame, algorithm_name: str,
                      query_source: str, dataset_version: str = "train") -> None:
        """
        Save results dataframe to text file with dynamic naming.

        Args:
            results_df (pd.DataFrame): Results dataframe
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version
        """
        output_file = self._get_output_filename(
            algorithm_name, query_source, dataset_version)
        k_sparse = self.config.get_k_sparse()

        # Get top k_sparse results per query
        if 'qid' in results_df.columns:
            # Ensure we get top k per query, not globally
            top_k_results = (results_df.groupby('qid')
                             .apply(lambda x: x.nsmallest(min(k_sparse, len(x)), 'rank'))
                             .reset_index(drop=True))
        else:
            top_k_results = results_df.head(k_sparse)

        # Save to tab-separated text file in TREC format if possible
        if all(col in top_k_results.columns for col in ['qid', 'docno', 'rank', 'score']):
            # TREC format: qid Q0 docno rank score run_name
            trec_format = top_k_results[[
                'qid', 'docno', 'rank', 'score']].copy()
            trec_format.insert(1, 'Q0', 'Q0')  # Add Q0 column
            trec_format.insert(
                5, 'run_name', f"{algorithm_name}_{query_source}_{dataset_version}")

            trec_format.to_csv(
                output_file,
                sep='\t',
                index=False,
                header=False
            )
        else:
            # Fallback to regular format
            top_k_results.to_csv(
                output_file,
                sep='\t',
                index=False,
                header=True
            )

        print(
            f"Saved {len(top_k_results)} results for {algorithm_name} ({query_source}, {dataset_version}) to {output_file}")

    def _load_cached_results(self, algorithm_name: str, query_source: str,
                             dataset_version: str = "train") -> Optional[pd.DataFrame]:
        """
        Load cached results if they exist.

        Args:
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version

        Returns:
            pd.DataFrame or None: Cached results or None if not found
        """
        output_file = self._get_output_filename(
            algorithm_name, query_source, dataset_version)

        if os.path.exists(output_file):
            try:
                # Try to load as TREC format first (no header)
                cached_df = pd.read_csv(output_file, sep='\t', header=None)

                if len(cached_df.columns) == 6:
                    # TREC format: qid Q0 docno rank score run_name
                    cached_df.columns = ["qid", "Q0",
                                         "docno", "rank", "score", "run_name"]
                    # Keep only needed columns
                    cached_df = cached_df[["qid", "docno", "score", "rank"]]
                else:
                    # Try with header
                    cached_df = pd.read_csv(output_file, sep='\t')

                # Validate expected columns
                expected_cols = ["qid", "docno", "score", "rank"]
                if all(col in cached_df.columns for col in expected_cols):
                    print(
                        f"Loaded cached results for {algorithm_name} ({query_source}, {dataset_version}) from {output_file}")
                    return cached_df
                else:
                    print(
                        f"Warning: Cached file {output_file} has unexpected format. Re-running...")
                    return None
            except Exception as e:
                print(
                    f"Warning: Error loading cached results for {algorithm_name}: {e}. Re-running...")
                return None

        return None

    def _create_retriever(self, wmodel: str) -> Any:
        """
        Create PyTerrier retriever for given weighting model.

        Args:
            wmodel (str): Weighting model name

        Returns:
            PyTerrier retriever object
        """
        if wmodel not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported weighting model: {wmodel}. Supported models: {self.SUPPORTED_MODELS}")

        return pt.terrier.Retriever(self.index, wmodel=wmodel)

    def run_retrieval(self, wmodel: str, topics: pd.DataFrame, query_source: str,
                      dataset_version: str = "train", force_rerun: bool = False) -> pd.DataFrame:
        """
        Run sparse retrieval for given weighting model with dynamic path management and caching.

        Args:
            wmodel (str): Weighting model name (BM25, PL2, TF_IDF)
            topics (pd.DataFrame): Topics dataframe with columns ['qid', 'query']
            query_source (str): Source of queries (original, rewritten_<model>, summarized)
            dataset_version (str): Dataset version (train, dev-1, etc.)
            force_rerun (bool): Force re-running even if cached results exist

        Returns:
            pd.DataFrame: Results dataframe with columns ["qid", "query", "docno", "score", "rank"]
        """
        # Check if output already exists (skip if not forcing rerun)
        if not force_rerun and self._check_existing_output(wmodel, query_source, dataset_version):
            print(f"Output already exists for {wmodel} on {query_source} ({dataset_version}). Skipping...")
            # Load and return existing results
            cached_results = self._load_cached_results(wmodel, query_source, dataset_version)
            if cached_results is not None:
                # Add query text if not present
                if "query" not in cached_results.columns:
                    cached_results = cached_results.merge(
                        topics[['qid', 'query']], on='qid', how='left')
                return cached_results
        
        # Check for cached results first (unless forced to rerun)
        if not force_rerun:
            cached_results = self._load_cached_results(
                wmodel, query_source, dataset_version)
            if cached_results is not None:
                # Add query text if not present
                if "query" not in cached_results.columns:
                    cached_results = cached_results.merge(
                        topics[['qid', 'query']], on='qid', how='left')
                return cached_results

        print(
            f"Running {wmodel} retrieval on {query_source} queries for {dataset_version} ({len(topics)} queries)...")
        
        start_time = time.time()

        # Create retriever
        retriever = self._create_retriever(wmodel)

        # Run retrieval
        try:
            results = retriever.transform(topics)

            # Ensure required columns are present
            required_cols = ["qid", "docno", "score", "rank"]
            missing_cols = [
                col for col in required_cols if col not in results.columns]

            if missing_cols:
                raise ValueError(
                    f"Missing required columns in results: {missing_cols}")

            # Add query text if not present
            if "query" not in results.columns:
                results = results.merge(
                    topics[['qid', 'query']], on='qid', how='left')

            # Ensure proper column order and types
            results = results[["qid", "query",
                               "docno", "score", "rank"]].copy()
            results['qid'] = results['qid'].astype(str)
            results['score'] = pd.to_numeric(results['score'], errors='coerce')
            results['rank'] = pd.to_numeric(results['rank'], errors='coerce')

            # Sort by qid and rank for consistency
            results = results.sort_values(
                ['qid', 'rank']).reset_index(drop=True)

            # Save results with dynamic path
            self._save_results(results, wmodel, query_source, dataset_version)
            
            elapsed_time = time.time() - start_time
            print(f"✓ Completed {wmodel} in {elapsed_time:.2f}s ({len(results)} results)")

            return results

        except Exception as e:
            raise RuntimeError(f"Error running {wmodel} retrieval: {e}")

    def run_all_models_parallel(self, topics: pd.DataFrame, query_source: str,
                               dataset_version: str = "train", force_rerun: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run all supported sparse retrieval models in parallel for better performance.

        Args:
            topics (pd.DataFrame): Topics dataframe
            query_source (str): Source of queries
            dataset_version (str): Dataset version
            force_rerun (bool): Force re-running even if cached results exist

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping model names to result dataframes
        """
        print(f"Running {len(self.SUPPORTED_MODELS)} models in parallel for {query_source} on {dataset_version}")
        
        results = {}
        
        # Use ThreadPoolExecutor for I/O bound PyTerrier operations
        with ThreadPoolExecutor(max_workers=min(len(self.SUPPORTED_MODELS), self.n_threads)) as executor:
            # Submit all model retrieval tasks
            future_to_model = {
                executor.submit(
                    self.run_retrieval, 
                    model, topics, query_source, dataset_version, force_rerun
                ): model for model in self.SUPPORTED_MODELS
            }
            
            # Collect results as they complete
            for future in future_to_model:
                model = future_to_model[future]
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout per model
                    results[model] = result
                    print(f"✓ Successfully completed {model} for {query_source} on {dataset_version}")
                except Exception as e:
                    print(f"✗ Error running {model}: {e}")
                    continue
        
        return results

    def run_all_models(self, topics: pd.DataFrame, query_source: str,
                       dataset_version: str = "train", force_rerun: bool = False, 
                       use_parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run all supported sparse retrieval models with option for parallel execution.

        Args:
            topics (pd.DataFrame): Topics dataframe
            query_source (str): Source of queries
            dataset_version (str): Dataset version
            force_rerun (bool): Force re-running even if cached results exist
            use_parallel (bool): Use parallel execution for better performance

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping model names to result dataframes
        """
        if use_parallel:
            return self.run_all_models_parallel(topics, query_source, dataset_version, force_rerun)
        else:
            # Original sequential implementation
            results = {}
            for model in self.SUPPORTED_MODELS:
                try:
                    results[model] = self.run_retrieval(
                        model, topics, query_source, dataset_version, force_rerun)
                    print(
                        f"✓ Successfully completed {model} for {query_source} on {dataset_version}")
                except Exception as e:
                    print(f"✗ Error running {model}: {e}")
                    continue
            return results

    def run_experiment(self, topics: pd.DataFrame, qrels: pd.DataFrame, query_source: str,
                       dataset_version: str = "train", models: Optional[List[str]] = None,
                       force_rerun: bool = False, use_parallel: bool = True) -> pd.DataFrame:
        """
        Run PyTerrier experiment with multiple retrievers and dynamic result paths.

        Args:
            topics (pd.DataFrame): Topics dataframe
            qrels (pd.DataFrame): Qrels dataframe
            query_source (str): Source of queries
            dataset_version (str): Dataset version
            models (List[str], optional): List of models to run. Defaults to all supported models.
            force_rerun (bool): Force re-running even if cached results exist
            use_parallel (bool): Use parallel execution for better performance

        Returns:
            pd.DataFrame: Experiment evaluation results
        """
        if models is None:
            models = self.SUPPORTED_MODELS

        # Validate models
        invalid_models = [m for m in models if m not in self.SUPPORTED_MODELS]
        if invalid_models:
            raise ValueError(f"Invalid models specified: {invalid_models}")

        print(f"Running experiment for {len(models)} models on {len(topics)} topics with {len(qrels)} qrels")
        experiment_start_time = time.time()

        # Create retrievers
        retrievers = []
        retriever_names = []
        for model in models:
            try:
                retriever = self._create_retriever(model)
                retrievers.append(retriever)
                retriever_names.append(f"{model}_{query_source}")
            except Exception as e:
                print(f"Warning: Could not create {model} retriever: {e}")
                continue

        if not retrievers:
            raise ValueError("No valid retrievers could be created")

        # Run experiment
        eval_metrics = self.config.get_eval_metrics()

        try:
            print(f"Starting PyTerrier experiment with metrics: {eval_metrics}")
            experiment_results = pt.Experiment(
                retrievers,
                topics,
                qrels,
                eval_metrics=eval_metrics,
                names=retriever_names
            )

            # Save experiment results with dynamic naming
            experiment_output_dir = self._get_output_directory(
                query_source, dataset_version)
            experiment_file = Path(
                experiment_output_dir) / f"experiment_results_{dataset_version}_{query_source}.csv"
            experiment_results.to_csv(experiment_file, index=True)
            
            experiment_time = time.time() - experiment_start_time
            print(f"✓ Experiment completed in {experiment_time:.2f}s. Results saved to: {experiment_file}")

            # Cache individual results if not already cached using parallel processing
            if not force_rerun and use_parallel:
                print("Caching individual model results in parallel...")
                self.run_all_models_parallel(topics, query_source, dataset_version, force_rerun=False)

            return experiment_results

        except Exception as e:
            raise RuntimeError(f"Error running experiment: {e}")

    def run_multi_source_experiment(self, dataset_version: str = "train",
                                    query_sources: Optional[List[str]] = None,
                                    models: Optional[List[str]] = None,
                                    force_rerun: bool = False,
                                    use_parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run experiments across multiple query sources for a dataset version with parallel processing.

        Args:
            dataset_version (str): Dataset version to process
            query_sources (List[str], optional): Query sources to test. Auto-detected if None.
            models (List[str], optional): Models to run. All if None.
            force_rerun (bool): Force re-running experiments
            use_parallel (bool): Use parallel execution for better performance

        Returns:
            Dict[str, pd.DataFrame]: Mapping of query source to experiment results
        """
        print(f"Running multi-source experiment for {dataset_version} with hardware optimizations")
        
        # Load qrels once for all experiments
        qrels = self.data_loader.load_qrels(dataset_version)

        # Auto-detect available query sources if not specified
        if query_sources is None:
            query_sources = self.data_loader.get_available_rewritten_sources(
                dataset_version)

        if models is None:
            models = self.SUPPORTED_MODELS

        print(f"Processing {len(query_sources)} query sources: {query_sources}")
        print(f"Using {len(models)} models: {models}")
        print(f"Parallel processing: {'Enabled' if use_parallel else 'Disabled'}")

        results = {}
        total_sources = len(query_sources)

        for i, query_source in enumerate(query_sources, 1):
            print(f"\n{'='*60}")
            print(f"Processing source {i}/{total_sources}: {dataset_version} with {query_source} queries")
            print(f"{'='*60}")

            try:
                # Load topics for this source
                topics = self.data_loader.load_topics(
                    dataset_version, query_source)

                # Validate data consistency
                self.data_loader.validate_data_consistency(
                    topics, qrels, dataset_version)

                # Run experiment with parallel processing
                experiment_result = self.run_experiment(
                    topics=topics,
                    qrels=qrels,
                    query_source=query_source,
                    dataset_version=dataset_version,
                    models=models,
                    force_rerun=force_rerun,
                    use_parallel=use_parallel
                )

                results[query_source] = experiment_result
                print(f"✓ Successfully completed experiment for {query_source} ({i}/{total_sources})")

            except Exception as e:
                print(f"✗ Error in experiment for {query_source}: {e}")
                continue

        return results
