"""
Sparse retrieval module for PyTerrier experiments.
Implements wrapper functions for BM25, PL2, and TF_IDF retrievers with caching.
Updated with dynamic path creation to prevent overwriting results.
"""
import os
import pandas as pd
import pyterrier as pt
from pathlib import Path
from typing import Dict, Any, List, Optional
from config_loader import ConfigLoader
from data_loader import DataLoader


class SparseRetrieval:
    """Handles sparse retrieval algorithms with dynamic path management."""

    # Supported sparse retrieval models
    SUPPORTED_MODELS = ["BM25", "PL2", "TF_IDF"]

    def __init__(self, config: ConfigLoader, data_loader: DataLoader):
        """
        Initialize sparse retrieval handler.

        Args:
            config (ConfigLoader): Configuration loader instance
            data_loader (DataLoader): Data loader instance
        """
        self.config = config
        self.data_loader = data_loader
        self.index = data_loader.get_index()

        # Ensure run directory exists
        self.config.ensure_run_directory_exists()

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
        Run sparse retrieval for given weighting model with dynamic path management.

        Args:
            wmodel (str): Weighting model name (BM25, PL2, TF_IDF)
            topics (pd.DataFrame): Topics dataframe with columns ['qid', 'query']
            query_source (str): Source of queries (original, rewritten_<model>, summarized)
            dataset_version (str): Dataset version (train, dev-1, etc.)
            force_rerun (bool): Force re-running even if cached results exist

        Returns:
            pd.DataFrame: Results dataframe with columns ["qid", "query", "docno", "score", "rank"]
        """
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
            f"Running {wmodel} retrieval on {query_source} queries for {dataset_version}...")

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

            return results

        except Exception as e:
            raise RuntimeError(f"Error running {wmodel} retrieval: {e}")

    def run_all_models(self, topics: pd.DataFrame, query_source: str,
                       dataset_version: str = "train", force_rerun: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run all supported sparse retrieval models.

        Args:
            topics (pd.DataFrame): Topics dataframe
            query_source (str): Source of queries
            dataset_version (str): Dataset version
            force_rerun (bool): Force re-running even if cached results exist

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping model names to result dataframes
        """
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
                       force_rerun: bool = False) -> pd.DataFrame:
        """
        Run PyTerrier experiment with multiple retrievers and dynamic result paths.

        Args:
            topics (pd.DataFrame): Topics dataframe
            qrels (pd.DataFrame): Qrels dataframe
            query_source (str): Source of queries
            dataset_version (str): Dataset version
            models (List[str], optional): List of models to run. Defaults to all supported models.
            force_rerun (bool): Force re-running even if cached results exist

        Returns:
            pd.DataFrame: Experiment evaluation results
        """
        if models is None:
            models = self.SUPPORTED_MODELS

        # Validate models
        invalid_models = [m for m in models if m not in self.SUPPORTED_MODELS]
        if invalid_models:
            raise ValueError(f"Invalid models specified: {invalid_models}")

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
            print(f"Experiment results saved to: {experiment_file}")

            # Cache individual results if not already cached
            if not force_rerun:
                for model in models:
                    if not os.path.exists(self._get_output_filename(model, query_source, dataset_version)):
                        try:
                            self.run_retrieval(
                                model, topics, query_source, dataset_version, force_rerun=False)
                        except Exception as e:
                            print(
                                f"Warning: Could not cache {model} results: {e}")

            return experiment_results

        except Exception as e:
            raise RuntimeError(f"Error running experiment: {e}")

    def run_multi_source_experiment(self, dataset_version: str = "train",
                                    query_sources: Optional[List[str]] = None,
                                    models: Optional[List[str]] = None,
                                    force_rerun: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run experiments across multiple query sources for a dataset version.

        Args:
            dataset_version (str): Dataset version to process
            query_sources (List[str], optional): Query sources to test. Auto-detected if None.
            models (List[str], optional): Models to run. All if None.
            force_rerun (bool): Force re-running experiments

        Returns:
            Dict[str, pd.DataFrame]: Mapping of query source to experiment results
        """
        # Load qrels
        qrels = self.data_loader.load_qrels(dataset_version)

        # Auto-detect available query sources if not specified
        if query_sources is None:
            query_sources = self.data_loader.get_available_rewritten_sources(
                dataset_version)

        if models is None:
            models = self.SUPPORTED_MODELS

        results = {}

        for query_source in query_sources:
            print(f"\n{'='*60}")
            print(
                f"Running experiment: {dataset_version} with {query_source} queries")
            print(f"{'='*60}")

            try:
                # Load topics for this source
                topics = self.data_loader.load_topics(
                    dataset_version, query_source)

                # Validate data consistency
                self.data_loader.validate_data_consistency(
                    topics, qrels, dataset_version)

                # Run experiment
                experiment_result = self.run_experiment(
                    topics=topics,
                    qrels=qrels,
                    query_source=query_source,
                    dataset_version=dataset_version,
                    models=models,
                    force_rerun=force_rerun
                )

                results[query_source] = experiment_result
                print(
                    f"✓ Successfully completed experiment for {query_source}")

            except Exception as e:
                print(f"✗ Error in experiment for {query_source}: {e}")
                continue

        return results
