# PyTerrier Sparse Retrieval Framework

A modular Python framework for running sparse retrieval experiments using PyTerrier with BM25, PL2, and TF-IDF algorithms.

## Features

- **Modular Architecture**: Low-coupled, independent modules for configuration, data loading, and retrieval
- **Caching Support**: Automatically caches results to avoid re-running expensive operations
- **Flexible Configuration**: JSON-based configuration management
- **Multiple Algorithms**: Support for BM25, PL2, and TF-IDF sparse retrievers
- **Experiment Support**: Built-in PyTerrier experiment functionality with evaluation metrics

## Project Structure

```
├── env.json                 # Configuration file
├── config_loader.py         # Configuration management module
├── data_loader.py          # Data loading and validation module
├── sparse_retrieval.py     # Core sparse retrieval functionality
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
└── run-files/              # Output directory (created automatically)
    ├── bm25_1000.txt      # BM25 results (top k)
    ├── pl2_1000.txt       # PL2 results (top k)
    ├── tf_idf_1000.txt    # TF-IDF results (top k)
    └── experiment_results.csv
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your environment by editing `env.json`:

```json
{
    "index_path": "/path/to/your/data.properties",
    "topics_path": "/path/to/your/topics.jsonl",
    "qrels_path": "/path/to/your/qrels.txt",
    "run_directory": "/path/to/run-files",
    "k_sparse": 1000,
    "eval_metrics": ["map", "ndcg_cut_10", "recip_rank"]
}
```

## Configuration Parameters

- **index_path**: Path to your Terrier index properties file
- **topics_path**: Path to JSONL file containing topics with `q_id` and `query` keys
- **qrels_path**: Path to TREC-format qrels file
- **run_directory**: Directory where result files will be saved
- **k_sparse**: Number of top results to save per algorithm
- **eval_metrics**: List of evaluation metrics for experiments

## Topics File Format

Your topics file should be in JSONL format (one JSON object per line):

```jsonl
{"q_id": "1", "query": "what is information retrieval"}
{"q_id": "2", "query": "machine learning algorithms"}
{"q_id": "3", "query": "natural language processing"}
```

## Usage

### Basic Usage

```python
import pyterrier as pt
from config_loader import ConfigLoader
from data_loader import DataLoader
from sparse_retrieval import SparseRetrieval

# Initialize PyTerrier
if not pt.started():
    pt.init()

# Load configuration
config = ConfigLoader("env.json")

# Initialize modules
data_loader = DataLoader(config)
sparse_retrieval = SparseRetrieval(config, data_loader)

# Load data
topics = data_loader.load_topics()
qrels = data_loader.load_qrels()

# Run single algorithm
bm25_results = sparse_retrieval.run_retrieval("BM25", topics)

# Run all algorithms
all_results = sparse_retrieval.run_all_models(topics)

# Run full experiment with evaluation
experiment_results = sparse_retrieval.run_experiment(topics, qrels)
```

### Running the Complete Pipeline

```bash
python main.py
```

This will:
1. Load configuration from `env.json`
2. Load topics and qrels
3. Run BM25, PL2, and TF-IDF retrievers
4. Save top-k results to individual files
5. Run PyTerrier experiment with evaluation
6. Save experiment results

### Caching Behavior

The framework automatically caches results to improve efficiency:

- **Automatic Caching**: Results are saved as tab-separated files in the run directory
- **Cache Detection**: Existing result files are automatically detected and loaded
- **Force Rerun**: Use `force_rerun=True` to bypass cache and regenerate results
- **File Naming**: Results saved as `{algorithm}_{k}.txt` (e.g., `bm25_1000.txt`)

### Advanced Usage

#### Running Specific Models

```python
# Run only specific models
models_to_run = ["BM25", "TF_IDF"]
experiment_results = sparse_retrieval.run_experiment(
    topics, qrels, models=models_to_run
)
```

#### Custom K Values

```python
# Temporarily change k_sparse
config.config["k_sparse"] = 500
results = sparse_retrieval.run_retrieval("BM25", topics, force_rerun=True)
```

#### Error Handling

```python
try:
    results = sparse_retrieval.run_retrieval("BM25", topics)
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Retrieval error: {e}")
```

## Output Files

### Individual Algorithm Results

Each algorithm generates a tab-separated file with columns:
- `qid`: Query identifier
- `query`: Query text
- `docno`: Document identifier
- `score`: Relevance score
- `rank`: Document rank

Example content:
```
qid	query	docno	score	rank
1	information retrieval	doc123	15.42	1
1	information retrieval	doc456	12.87	2
1	information retrieval	doc789	11.93	3
```

### Experiment Results

The experiment results file contains evaluation metrics for each algorithm:
```
name	map	ndcg_cut_10	recip_rank
BM25	0.2345	0.3456	0.4567
PL2	0.2123	0.3234	0.4345
TF_IDF	0.1987	0.2987	0.3876
```

## Module Documentation

### ConfigLoader

Handles configuration management:
- Loads and validates `env.json`
- Provides typed access to configuration values
- Ensures required parameters are present

### DataLoader

Manages data loading operations:
- Loads topics from JSONL format
- Loads qrels in TREC format
- Manages Terrier index loading
- Validates data consistency

### SparseRetrieval

Core retrieval functionality:
- Supports BM25, PL2, and TF-IDF algorithms
- Implements result caching
- Provides experiment execution
- Handles error recovery

## Troubleshooting

### Common Issues

1. **Index Not Found**
   ```
   FileNotFoundError: Index file not found: /path/to/data.properties
   ```
   - Verify the `index_path` in `env.json`
   - Ensure the index file exists and is readable

2. **Topics Format Error**
   ```
   ValueError: Each topic must have 'q_id' and 'query' keys
   ```
   - Check topics file is valid JSONL format
   - Ensure each line has both `q_id` and `query` fields

3. **PyTerrier Initialization Error**
   ```
   RuntimeError: PyTerrier not started
   ```
   - Call `pt.init()` before using any retrieval functions
   - Check Java installation if PyTerrier fails to start

4. **Memory Issues**
   ```
   OutOfMemoryError: Java heap space
   ```
   - Increase Java heap size: `pt.init(mem=8000)` (8GB)
   - Reduce `k_sparse` value in configuration

### Performance Tips

- Use caching to avoid re-running expensive operations
- Set appropriate `k_sparse` values based on your needs
- Monitor memory usage for large indexes
- Consider running algorithms sequentially for memory-constrained environments

## Dependencies

- **python-terrier**: PyTerrier library for information retrieval
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing support

## License

This framework is designed for research and educational purposes. Please ensure compliance with PyTerrier and Terrier licensing terms.

## Contributing

To extend the framework:

1. **Adding New Algorithms**: Extend `SparseRetrieval.SUPPORTED_MODELS`
2. **Custom Metrics**: Modify `eval_metrics` in configuration
3. **Data Formats**: Extend `DataLoader` for new input formats
4. **Output Formats**: Modify result saving in `SparseRetrieval`

## Example Complete Workflow

```python
#!/usr/bin/env python3

import pyterrier as pt
from config_loader import ConfigLoader
from data_loader import DataLoader
from sparse_retrieval import SparseRetrieval

def complete_workflow():
    """Complete workflow example."""
    
    # Initialize
    if not pt.started():
        pt.init()
    
    # Load configuration
    config = ConfigLoader("env.json")
    data_loader = DataLoader(config)
    sparse_retrieval = SparseRetrieval(config, data_loader)
    
    # Load data
    topics = data_loader.load_topics()
    qrels = data_loader.load_qrels()
    
    # Validate
    data_loader.validate_data_consistency(topics, qrels)
    
    # Run experiments
    results = sparse_retrieval.run_all_models(topics)
    experiment_df = sparse_retrieval.run_experiment(topics, qrels)
    
    # Print summary
    print("Results Summary:")
    for model, df in results.items():
        print(f"{model}: {len(df)} results")
    
    print("\nExperiment Evaluation:")
    print(experiment_df)
    
    return results, experiment_df

if __name__ == "__main__":
    complete_workflow()
```

This framework provides a robust, modular foundation for sparse retrieval experiments with PyTerrier, emphasizing code reusability, maintainability, and extensibility.