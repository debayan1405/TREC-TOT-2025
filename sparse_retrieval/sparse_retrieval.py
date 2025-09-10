"""
Sparse retrieval module for PyTerrier experiments.
Implements wrapper functions for BM25, PL2, and TF_IDF retrievers with caching.
Updated with enhanced dynamic path creation and evaluation storage.
"""
import os
import pandas as pd

import pyterrier as pt
from pyterrier.transformer import Transformer

from pathlib import Path
from typing import Dict, Any, List, Optional
from config_loader import ConfigLoader
from data_loader import DataLoader

import re


class QuerySanitiser(Transformer):
    """
    A custom transformer that cleans a query string to make it safe for
    the old Terrier v0.1.5 parser. It removes punctuation and returns a
    single, space-separated string.
    """

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        topics_c = topics.copy()

        # This function cleans the query
        def clean_query(query):
            # Remove apostrophes and other problematic punctuation
            # by keeping only letters, numbers, and spaces
            text = re.sub(r'[^a-zA-Z0-9 ]', ' ', query)
            # Normalize whitespace (e.g., convert multiple spaces to one)
            return " ".join(text.split())

        # Apply the cleaning function to the 'query' column
        topics_c['query'] = topics_c['query'].apply(clean_query)
        return topics_c


class SparseRetrieval:
    """Handles sparse retrieval algorithms with enhanced path management."""

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

        # Ensure directories exist
        self.config.ensure_run_directory_exists()
        self.config.ensure_evaluation_directory_exists()

    def _get_rewriter_name(self, query_source: str) -> str:
        """
        Extract rewriter name from query source.

        Args:
            query_source (str): Source of queries (original, rewritten_<model>, summarized)

        Returns:
            str: Rewriter name for file naming
        """
        if query_source == "original":
            return "original"
        elif query_source == "summarized":
            return "summarized"
        elif query_source.startswith("rewritten_"):
            return query_source.replace("rewritten_", "")
        else:
            return query_source

    def _get_run_output_directory(self, dataset_version: str) -> str:
        """
        Generate run output directory based on dataset version.

        Args:
            dataset_version (str): Dataset version (train, dev-1, etc.)

        Returns:
            str: Path to run output directory
        """
        base_run_dir = Path(self.config.get_sparse_run_directory())
        output_dir = base_run_dir / dataset_version
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def _get_eval_output_directory(self, dataset_version: str) -> str:
        """
        Generate evaluation output directory based on dataset version.

        Args:
            dataset_version (str): Dataset version (train, dev-1, etc.)

        Returns:
            str: Path to evaluation output directory
        """
        base_eval_dir = Path(self.config.get_evaluation_directory())
        eval_dir = base_eval_dir / "evals" / dataset_version
        eval_dir.mkdir(parents=True, exist_ok=True)
        return str(eval_dir)

    def _get_run_filename(self, algorithm_name: str, query_source: str,
                          dataset_version: str) -> str:
        """
        Generate run filename following the naming convention:
        <rewriter_name>_<dataset_version>_<algorithm_name>_<k_sparse>.txt

        Args:
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version

        Returns:
            str: Full path to output file
        """
        output_dir = self._get_run_output_directory(dataset_version)
        rewriter_name = self._get_rewriter_name(query_source)
        k_sparse = self.config.get_k_sparse()

        filename = f"{rewriter_name}_{dataset_version}_{algorithm_name.lower()}_{k_sparse}.txt"
        return os.path.join(output_dir, filename)

    def _get_eval_filename(self, algorithm_name: str, query_source: str,
                           dataset_version: str) -> str:
        """
        Generate evaluation filename following the naming convention:
        <rewriter_name>_<algorithm_name>_<dataset_version>_eval.csv

        Args:
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version

        Returns:
            str: Full path to evaluation file
        """
        eval_dir = self._get_eval_output_directory(dataset_version)
        rewriter_name = self._get_rewriter_name(query_source)

        filename = f"{rewriter_name}_{algorithm_name.lower()}_{dataset_version}_eval.csv"
        return os.path.join(eval_dir, filename)

    def _save_run_results(self, results_df: pd.DataFrame, algorithm_name: str,
                          query_source: str, dataset_version: str) -> None:
        """
        Save run results dataframe to text file with enhanced naming.

        Args:
            results_df (pd.DataFrame): Results dataframe
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version
        """
        output_file = self._get_run_filename(
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

        # Save to tab-separated text file in TREC format
        if all(col in top_k_results.columns for col in ['qid', 'docno', 'rank', 'score']):
            # TREC format: qid Q0 docno rank score run_name
            trec_format = top_k_results[[
                'qid', 'docno', 'rank', 'score']].copy()
            trec_format.insert(1, 'Q0', 'Q0')  # Add Q0 column

            rewriter_name = self._get_rewriter_name(query_source)
            run_name = f"{rewriter_name}_{dataset_version}_{algorithm_name.lower()}"
            trec_format.insert(5, 'run_name', run_name)

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

        print(f"Saved {len(top_k_results)} run results for {algorithm_name} "
              f"({query_source}, {dataset_version}) to {output_file}")

    def _save_eval_results(self, eval_df: pd.DataFrame, algorithm_name: str,
                           query_source: str, dataset_version: str) -> None:
        """
        Save evaluation results to CSV file.

        Args:
            eval_df (pd.DataFrame): Evaluation results dataframe
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version
        """
        eval_file = self._get_eval_filename(
            algorithm_name, query_source, dataset_version)

        eval_df.to_csv(eval_file, index=True)

        print(f"Saved evaluation results for {algorithm_name} "
              f"({query_source}, {dataset_version}) to {eval_file}")

    def _load_cached_run_results(self, algorithm_name: str, query_source: str,
                                 dataset_version: str) -> Optional[pd.DataFrame]:
        """
        Load cached run results if they exist.

        Args:
            algorithm_name (str): Name of the algorithm
            query_source (str): Source of queries
            dataset_version (str): Dataset version

        Returns:
            pd.DataFrame or None: Cached results or None if not found
        """
        output_file = self._get_run_filename(
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
                    print(f"Loaded cached run results for {algorithm_name} "
                          f"({query_source}, {dataset_version}) from {output_file}")
                    return cached_df
                else:
                    print(
                        f"Warning: Cached file {output_file} has unexpected format. Re-running...")
                    return None
            except Exception as e:
                print(
                    f"Warning: Error loading cached run results for {algorithm_name}: {e}. Re-running...")
                return None

        return None

    def _create_retriever(self, wmodel: str) -> Any:
        """
        Create a robust PyTerrier retrieval pipeline that tokenizes
        queries before passing them to the backend.
        """
        if wmodel not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported weighting model: {wmodel}. Supported models: {self.SUPPORTED_MODELS}")

        retriever = pt.BatchRetrieve(self.index, wmodel=wmodel)

        # Use the new QuerySanitiser class
        return QuerySanitiser() >> retriever

    def run_retrieval(self, wmodel: str, topics: pd.DataFrame, query_source: str,
                      dataset_version: str, force_rerun: bool = False) -> pd.DataFrame:
        """
        Run sparse retrieval for given weighting model with enhanced path management.

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
            cached_results = self._load_cached_run_results(
                wmodel, query_source, dataset_version)
            if cached_results is not None:
                # Add query text if not present
                if "query" not in cached_results.columns:
                    cached_results = cached_results.merge(
                        topics[['qid', 'query']], on='qid', how='left')
                return cached_results

        rewriter_name = self._get_rewriter_name(query_source)
        print(
            f"Running {wmodel} retrieval on {rewriter_name} queries for {dataset_version}...")

        # Create retriever
        retriever = self._create_retriever(wmodel)

        # Run retrieval
        try:
            results = retriever.transform(topics, verbose = True)

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

            # Save results with enhanced naming
            self._save_run_results(
                results, wmodel, query_source, dataset_version)

            return results

        except Exception as e:
            raise RuntimeError(f"Error running {wmodel} retrieval: {e}")

    def run_all_models(self, topics: pd.DataFrame, query_source: str,
                       dataset_version: str, force_rerun: bool = False) -> Dict[str, pd.DataFrame]:
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
        rewriter_name = self._get_rewriter_name(query_source)

        for model in self.SUPPORTED_MODELS:
            try:
                results[model] = self.run_retrieval(
                    model, topics, query_source, dataset_version, force_rerun)
                print(
                    f"✓ Successfully completed {model} for {rewriter_name} on {dataset_version}")
            except Exception as e:
                print(f"✗ Error running {model}: {e}")
                continue

        return results

    def run_experiment(self, topics: pd.DataFrame, qrels: pd.DataFrame, query_source: str,
                       dataset_version: str, models: Optional[List[str]] = None,
                       force_rerun: bool = False) -> pd.DataFrame:
        """
        Run PyTerrier experiment with multiple retrievers and enhanced result storage.

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
        rewriter_name = self._get_rewriter_name(query_source)

        for model in models:
            try:
                retriever = self._create_retriever(model)
                retrievers.append(retriever)
                retriever_names.append(f"{rewriter_name}_{model}")
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
                names=retriever_names,
                verbose = True
            )

            # Save individual evaluation results for each model
            for i, model in enumerate(models):
                if i < len(experiment_results):
                    # Extract single model results
                    single_model_eval = experiment_results.iloc[[i]]
                    self._save_eval_results(
                        single_model_eval, model, query_source, dataset_version)

            # Save combined experiment results
            combined_eval_dir = self._get_eval_output_directory(
                dataset_version)
            combined_eval_file = Path(
                combined_eval_dir) / f"combined_{rewriter_name}_{dataset_version}_eval.csv"
            experiment_results.to_csv(combined_eval_file, index=True)
            print(
                f"Combined experiment results saved to: {combined_eval_file}")

            # Cache individual run results if not already cached
            if not force_rerun:
                for model in models:
                    run_file = self._get_run_filename(
                        model, query_source, dataset_version)
                    if not os.path.exists(run_file):
                        try:
                            self.run_retrieval(
                                model, topics, query_source, dataset_version, force_rerun=False)
                        except Exception as e:
                            print(
                                f"Warning: Could not cache {model} results: {e}")

            return experiment_results

        except Exception as e:
            raise RuntimeError(f"Error running experiment: {e}")

    def run_single_dataset_experiment(self, dataset_version: str,
                                      query_sources: Optional[List[str]] = None,
                                      models: Optional[List[str]] = None,
                                      force_rerun: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run experiments across multiple query sources for a single dataset version.

        Args:
            dataset_version (str): Dataset version to process
            query_sources (List[str], optional): Query sources to test. Auto-detected if None.
            models (List[str], optional): Models to run. All if None.
            force_rerun (bool): Force re-running experiments

        Returns:
            Dict[str, pd.DataFrame]: Mapping of query source to experiment results
        """
        # Load qrels (only for datasets that have qrels)
        qrels_versions = ["train", "dev-1", "dev-2", "dev-3"]
        if dataset_version in qrels_versions:
            qrels = self.data_loader.load_qrels(dataset_version)
        else:
            print(
                f"Warning: No qrels available for {dataset_version}. Skipping evaluation.")
            qrels = None

        # Auto-detect available query sources if not specified
        if query_sources is None:
            query_sources = self.data_loader.get_available_rewritten_sources(
                dataset_version)

        if models is None:
            models = self.SUPPORTED_MODELS

        results = {}

        for query_source in query_sources:
            rewriter_name = self._get_rewriter_name(query_source)
            print(f"\n{'='*60}")
            print(
                f"Running experiment: {dataset_version} with {rewriter_name} queries")
            print(f"{'='*60}")

            try:
                # Load topics for this source
                topics = self.data_loader.load_topics(
                    dataset_version, query_source)

                # Validate data consistency only if qrels exist
                if qrels is not None:
                    self.data_loader.validate_data_consistency(
                        topics, qrels, dataset_version)

                    # Run experiment with evaluation
                    experiment_result = self.run_experiment(
                        topics=topics,
                        qrels=qrels,
                        query_source=query_source,
                        dataset_version=dataset_version,
                        models=models,
                        force_rerun=force_rerun
                    )
                    results[query_source] = experiment_result
                else:
                    # Run retrieval only (no evaluation)
                    retrieval_results = self.run_all_models(
                        topics=topics,
                        query_source=query_source,
                        dataset_version=dataset_version,
                        force_rerun=force_rerun
                    )
                    results[query_source] = retrieval_results
                    print(
                        f"✓ Successfully completed retrieval for {rewriter_name} (no evaluation)")

                print(
                    f"✓ Successfully completed experiment for {rewriter_name}")

            except Exception as e:
                print(f"✗ Error in experiment for {query_source}: {e}")
                continue

        return results
