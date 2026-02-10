# TREC-TOT 2025: Hierarchical Retrieval Pipeline

This repository contains the implementation for the TREC Tip-of-the-Tongue (ToT) 2025 track. The system implements a sophisticated 4-stage retrieval pipeline designed to resolve complex, descriptive queries using a combination of Sparse Retrieval, Dense Bi-Encoders, Cross-Encoders, and Large Language Models (LLMs).

This documentation outlines the prerequisite hardware/software, the directory structure, and the sequential execution flow for your **Multi-Stage Tip-of-the-Tongue (ToT) Retrieval Pipeline**.

## Directory Structure

The codebase is organized by pipeline stage. Ensure your local environment matches the structure below:

```text
.
├── dense-rerank-stage/         # Stages 2, 3, and 4
│   ├── bi_encoder_stage2.py    # Dense Retrieval & RRF Fusion
│   ├── cross_encoder_stage3.py # MonoT5 Re-ranking
│   └── llm_reranker_stage4.py  # Qwen-72B Listwise Re-ranking
├── evaluations/                # Generated evaluation metrics & charts
├── llm-query-rewriter/         # Pre-processing
│   ├── llm-config.json         # Model paths for rewriting (user supplied)
│   └── rewriter.py             # Query rewriting & summarization script
├── original-queries/           # Input: Original JSONL query files
├── qrel/                       # Input: QREL files for training/eval
├── rewritten-queries/          # Output: Generated rewritten queries
├── runs/                       # Output: TREC run files for every stage
├── sparse-retrieval-stage/     # Stage 1
│   └── sparse_retrieval.py     # BM25/PL2 Parameter Optimization
├── test-environment/           # Final Test Set Inference
│   ├── test_eval_files/
│   ├── test_run_files/
│   └── test_scripts/
│       └── test_pipeline.py    # End-to-end test inference script
└── requirements.txt            # Python Dependencies

```

## Prerequisites & Installation

### Hardware Requirements

* **RAM:** High memory (approx. 200GB+) recommended for PyTerrier `fileinmem` index loading.
* **GPU:** Multi-GPU setup recommended.
* Stage 2 & 3 support DataParallel.
* Stage 4 (Qwen-72B) requires significant VRAM (tested on A6000s or similar) or quantization.



### Software Setup

1. **Java:** A Java Runtime Environment (JRE) 11+ is required for PyTerrier.
2. **Python Environment:**
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```



## Configuration

**Important:** Before running any scripts, you must update the **Absolute Paths** at the top of every `.py` file to match your local machine.

Look for the `PATHS` or `SYSTEM SETUP` sections in the scripts and update:

* `INDEX_PATH`: Path to your PyTerrier index.
* `LOCAL_MODEL_DIR`: Path where HuggingFace models are cached/stored.
* `JAVA_OPTS`: Adjust heap size (`-Xmx`) based on your available RAM.

---

## Running the Pipeline

The pipeline is designed to be run sequentially. Output from one stage becomes the input for the next.

### Phase 0: Query Rewriting (Optional)

Generates query variations (rewrites/summaries) to improve recall.

```bash
# To rewrite original queries using models defined in llm-config.json
python llm-query-rewriter/rewriter.py --task rewrite --datasets train dev-1 --models llama3-8b

# To summarize multiple rewrites into a single query
python llm-query-rewriter/rewriter.py --task summarize --summarizer mistral-7b

```

### Stage 1: Sparse Retrieval Optimization

Optimizes BM25 (`k1`, `b`) and PL2 (`c`) parameters via Grid Search.

```bash
# Run optimization for a specific query variant (e.g., 'original' or 'rewritten-llama')
python sparse-retrieval-stage/sparse_retrieval.py --variant rewritten-llama

```

* **Output:** Saves the best run file to `runs/sparse-retrieval-stage-1/` and optimization logs/charts to `evaluations/`.

### Stage 2: Dense Retrieval & Fusion

Re-ranks Stage 1 candidates using multiple Bi-Encoders (ColBERTv2, Contriever, E5) and performs RRF Fusion.

```bash
# Automatically finds the BEST sparse run from Stage 1 for the given variant
python dense-rerank-stage/bi_encoder_stage2.py --variant rewritten-llama

```

* **Logic:** Retrieves text from the index, encodes queries/docs, calculates similarity, and fuses results.
* **Output:** `runs/be-rerank-stage-2/` (Individual dense runs and the fused run).

### Stage 3: Cross-Encoder Re-ranking

Re-ranks the top results from Stage 2 using a Cross-Encoder (MonoT5-Large).

```bash
# Uses the Fusion run from Stage 2 as input
python dense-rerank-stage/cross_encoder_stage3.py --variant rewritten-llama

```

* **Logic:** Performs a "Virtual Sweep" to determine the optimal depth (`k`) for re-ranking based on validation set performance (Robustness check).
* **Output:** `runs/ce-rerank-stage-3/`.

### Stage 4: LLM Listwise Re-ranking

The final re-ranking stage using a Large Language Model (Qwen-72B-Instruct) via vLLM.

```bash
# Uses the MonoT5 run from Stage 3 as input
python dense-rerank-stage/llm_reranker_stage4.py --variant rewritten-llama

```

* **Logic:** Uses a sliding window approach to re-rank documents. Performs joint hyperparameter tuning to find the optimal cut-offs for Input Depth (`k_input`) and LLM Re-ranking Depth (`k_llm`).
* **Output:** `runs/llm-rerank-stage-4/`.

---

## Evaluation on Test Set

Once models and parameters are finalized, use the **Test Environment** to run the full inference pipeline on the held-out test set without optimization overhead.

```bash
# Run the full end-to-end pipeline (BM25 -> Dense -> CE -> LLM) on the test set
python test-environment/test_scripts/test_pipeline.py --variant rewritten-llama

```

This script:

1. Generates/Loads Stage 1 BM25 results.
2. Generates/Loads Stage 2 Dense Ensemble results.
3. Generates/Loads Stage 3 MonoT5 results.
4. Generates/Loads Stage 4 Qwen results.
5. Calculates final metrics (`nDCG@10`, `Recall@1000`, etc.) and saves them to `test_eval_files/`.

## Metrics

The pipeline utilizes `ir_measures` for evaluation. The primary metrics tracked across all stages are:

* **nDCG@10** (Primary optimization metric for re-rankers)
* **Recall@1000** (Primary optimization metric for retrieval)
* MAP@1000, P@10, Success@10

## Citation & Credits

* **Framework:** [PyTerrier](https://github.com/terrier-org/pyterrier)
* **Models:**
  * Bi-Encoders: [ColBERTv2](https://arxiv.org/abs/2112.01488), [Contriever](https://arxiv.org/pdf/2112.09118), [E5-Large](https://arxiv.org/abs/2212.03533)
  * Cross-Encoder: [MonoT5 (Castorini)](https://arxiv.org/abs/2003.06713)
  * LLMs: [Qwen-2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ), [Mistral-7B-Instruct-v0.3 ](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

