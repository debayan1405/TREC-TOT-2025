"""
Main execution script for PyTerrier sparse retrieval experiments.
Demonstrates usage of the modular retrieval system.
"""
import pyterrier as pt
from pathlib import Path
from config_loader import ConfigLoader
from data_loader import DataLoader
from sparse_retrieval import SparseRetrieval


def main():
    """Main execution function."""
    try:
        # Initialize PyTerrier
        if not pt.started():
            pt.init()
        
        print("=== PyTerrier Sparse Retrieval Experiment ===\n")
        
        # Load configuration
        print("Loading configuration...")
        config = ConfigLoader("env.json")
        print(f"Configuration loaded successfully")
        print(f"Index path: {config.get_index_path()}")
        print(f"Topics path: {config.get_topics_path()}")
        print(f"Qrels path: {config.get_qrels_path()}")
        print(f"Run directory: {config.get_run_directory()}")
        print(f"K-sparse: {config.get_k_sparse()}")
        print(f"Evaluation metrics: {config.get_eval_metrics()}\n")
        
        # Initialize data loader
        print("Initializing data loader...")
        data_loader = DataLoader(config)
        
        # Load data
        print("Loading topics...")
        topics = data_loader.load_topics()
        print(f"Loaded {len(topics)} topics")
        
        print("Loading qrels...")
        qrels = data_loader.load_qrels()
        print(f"Loaded {len(qrels)} relevance judgments")
        
        print("Loading index...")
        index = data_loader.get_index()
        print(f"Index loaded successfully: {index}")
        
        # Validate data consistency
        print("Validating data consistency...")
        data_loader.validate_data_consistency(topics, qrels)
        print("Data validation completed\n")
        
        # Initialize sparse retrieval
        print("Initializing sparse retrieval...")
        sparse_retrieval = SparseRetrieval(config, data_loader)
        
        # Run individual retrievers
        print("=== Running Individual Retrievers ===")
        individual_results = {}
        
        for model in SparseRetrieval.SUPPORTED_MODELS:
            print(f"\nRunning {model}...")
            try:
                results = sparse_retrieval.run_retrieval(model, topics)
                individual_results[model] = results
                print(f"{model} completed: {len(results)} results")
                print(f"Sample results for {model}:")
                print(results[['qid', 'docno', 'score', 'rank']].head(3))
            except Exception as e:
                print(f"Error running {model}: {e}")
        
        # Run full experiment
        print(f"\n=== Running PyTerrier Experiment ===")
        try:
            experiment_results = sparse_retrieval.run_experiment(topics, qrels)
            print("Experiment completed successfully!")
            print("\nExperiment Results:")
            print(experiment_results)
            
            # Save experiment results
            experiment_output = Path(config.get_run_directory()) / "experiment_results.csv"
            experiment_results.to_csv(experiment_output, index=True)
            print(f"\nExperiment results saved to: {experiment_output}")
            
        except Exception as e:
            print(f"Error running experiment: {e}")
        
        # Display summary
        print(f"\n=== Summary ===")
        print(f"Topics processed: {len(topics)}")
        print(f"Qrels loaded: {len(qrels)}")
        print(f"Models run: {list(individual_results.keys())}")
        print(f"Results saved in: {config.get_run_directory()}")
        
        run_dir = Path(config.get_run_directory())
        result_files = list(run_dir.glob("*.txt"))
        if result_files:
            print(f"Output files created:")
            for file in sorted(result_files):
                print(f"  - {file.name}")
        
        print("\nExperiment completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


def run_single_model_example():
    """Example of running a single model."""
    try:
        if not pt.started():
            pt.init()
        
        print("=== Single Model Example ===")
        
        # Load configuration and data
        config = ConfigLoader("env.json")
        data_loader = DataLoader(config)
        topics = data_loader.load_topics()
        
        # Run only BM25
        sparse_retrieval = SparseRetrieval(config, data_loader)
        bm25_results = sparse_retrieval.run_retrieval("BM25", topics)
        
        print(f"BM25 Results: {len(bm25_results)} documents retrieved")
        print(bm25_results.head(10))
        
    except Exception as e:
        print(f"Error in single model example: {e}")


def run_custom_k_example():
    """Example of running with custom k value."""
    try:
        if not pt.started():
            pt.init()
        
        print("=== Custom K Example ===")
        
        # Load and modify configuration
        config = ConfigLoader("env.json")
        
        # Temporarily modify k_sparse for this run
        original_k = config.config["k_sparse"]
        config.config["k_sparse"] = 100  # Use top 100 results
        
        data_loader = DataLoader(config)
        topics = data_loader.load_topics()
        
        sparse_retrieval = SparseRetrieval(config, data_loader)
        
        # Run with custom k
        results = sparse_retrieval.run_retrieval("TF_IDF", topics, force_rerun=True)
        
        print(f"TF_IDF Results with k={config.get_k_sparse()}: {len(results)} documents")
        
        # Restore original k
        config.config["k_sparse"] = original_k
        
    except Exception as e:
        print(f"Error in custom k example: {e}")


if __name__ == "__main__":
    # Run main experiment
    main()
    
    # Uncomment to run examples:
    # run_single_model_example()
    # run_custom_k_example()