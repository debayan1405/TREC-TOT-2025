"""
Sparse retrieval module for PyTerrier experiments.
Implements wrapper functions for BM25, PL2, and TF_IDF retrievers with caching.
"""
import os
import pandas as pd
import pyterrier as pt
from pathlib import Path
from typing import Dict, Any, List, Optional
from config_loader import ConfigLoader
from data_loader import DataLoader


class SparseRetrieval:
    """Handles sparse retrieval algorithms with caching capabilities."""
    
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
    
    def _get_output_filename(self, algorithm_name: str) -> str:
        """
        Generate output filename for algorithm results.
        
        Args:
            algorithm_name (str): Name of the algorithm
            
        Returns:
            str: Full path to output file
        """
        k_sparse = self.config.get_k_sparse()
        filename = f"{algorithm_name.lower()}_{k_sparse}.txt"
        return os.path.join(self.config.get_run_directory(), filename)
    
    def _save_results(self, results_df: pd.DataFrame, algorithm_name: str) -> None:
        """
        Save results dataframe to text file.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            algorithm_name (str): Name of the algorithm
        """
        output_file = self._get_output_filename(algorithm_name)
        k_sparse = self.config.get_k_sparse()
        
        # Get top k_sparse results
        top_k_results = results_df.head(k_sparse)
        
        # Save to tab-separated text file
        top_k_results.to_csv(
            output_file, 
            sep='\t', 
            index=False, 
            header=True
        )
        
        print(f"Saved {len(top_k_results)} results for {algorithm_name} to {output_file}")
    
    def _load_cached_results(self, algorithm_name: str) -> Optional[pd.DataFrame]:
        """
        Load cached results if they exist.
        
        Args:
            algorithm_name (str): Name of the algorithm
            
        Returns:
            pd.DataFrame or None: Cached results or None if not found
        """
        output_file = self._get_output_filename(algorithm_name)
        
        if os.path.exists(output_file):
            try:
                cached_df = pd.read_csv(output_file, sep='\t')
                
                # Validate expected columns
                expected_cols = ["qid", "query", "docno", "score", "rank"]
                if all(col in cached_df.columns for col in expected_cols):
                    print(f"Loaded cached results for {algorithm_name} from {output_file}")
                    return cached_df
                else:
                    print(f"Warning: Cached file {output_file} has unexpected format. Re-running...")
                    return None
            except Exception as e:
                print(f"Warning: Error loading cached results for {algorithm_name}: {e}. Re-running...")
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
            raise ValueError(f"Unsupported weighting model: {wmodel}. Supported models: {self.SUPPORTED_MODELS}")
        
        return pt.terrier.Retriever(self.index, wmodel=wmodel)
    
    def run_retrieval(self, wmodel: str, topics: pd.DataFrame, force_rerun: bool = False) -> pd.DataFrame:
        """
        Run sparse retrieval for given weighting model.
        
        Args:
            wmodel (str): Weighting model name (BM25, PL2, TF_IDF)
            topics (pd.DataFrame): Topics dataframe with columns ['qid', 'query']
            force_rerun (bool): Force re-running even if cached results exist
            
        Returns:
            pd.DataFrame: Results dataframe with columns ["qid", "query", "docno", "score", "rank"]
        """
        # Check for cached results first (unless forced to rerun)
        if not force_rerun:
            cached_results = self._load_cached_results(wmodel)
            if cached_results is not None:
                return cached_results
        
        print(f"Running {wmodel} retrieval...")
        
        # Create retriever
        retriever = self._create_retriever(wmodel)
        
        # Run retrieval
        try:
            results = retriever.transform(topics)
            
            # Ensure required columns are present
            required_cols = ["qid", "docno", "score", "rank"]
            missing_cols = [col for col in required_cols if col not in results.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns in results: {missing_cols}")
            
            # Add query text if not present
            if "query" not in results.columns:
                results = results.merge(topics[['qid', 'query']], on='qid', how='left')
            
            # Ensure proper column order and types
            results = results[["qid", "query", "docno", "score", "rank"]].copy()
            results['qid'] = results['qid'].astype(str)
            results['score'] = pd.to_numeric(results['score'], errors='coerce')
            results['rank'] = pd.to_numeric(results['rank'], errors='coerce')
            
            # Sort by qid and rank for consistency
            results = results.sort_values(['qid', 'rank']).reset_index(drop=True)
            
            # Save results
            self._save_results(results, wmodel)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Error running {wmodel} retrieval: {e}")
    
    def run_all_models(self, topics: pd.DataFrame, force_rerun: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run all supported sparse retrieval models.
        
        Args:
            topics (pd.DataFrame): Topics dataframe
            force_rerun (bool): Force re-running even if cached results exist
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping model names to result dataframes
        """
        results = {}
        
        for model in self.SUPPORTED_MODELS:
            try:
                results[model] = self.run_retrieval(model, topics, force_rerun)
            except Exception as e:
                print(f"Error running {model}: {e}")
                continue
        
        return results
    
    def run_experiment(self, topics: pd.DataFrame, qrels: pd.DataFrame, 
                      models: Optional[List[str]] = None, 
                      force_rerun: bool = False) -> pd.DataFrame:
        """
        Run PyTerrier experiment with multiple retrievers.
        
        Args:
            topics (pd.DataFrame): Topics dataframe
            qrels (pd.DataFrame): Qrels dataframe
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
        for model in models:
            try:
                retriever = self._create_retriever(model)
                retrievers.append(retriever)
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
                eval_metrics=eval_metrics
            )
            
            # Cache individual results if not already cached
            if not force_rerun:
                for model in models:
                    if not os.path.exists(self._get_output_filename(model)):
                        self.run_retrieval(model, topics, force_rerun=False)
            
            return experiment_results
            
        except Exception as e:
            raise RuntimeError(f"Error running experiment: {e}")