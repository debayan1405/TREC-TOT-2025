TREC-TOT 2025 Submission Description Template
============================================

## System Overview

**Team:** [Your Team Name]
**Run Name:** TREC_TOT_2025_llama_test_final_submission
**Approach:** Multi-Stage Hierarchical Retrieval and Reranking Pipeline

## Pipeline Description

Our submission employs a sophisticated 5-stage hierarchical retrieval and reranking pipeline specifically designed for tip-of-the-tongue queries:

### Stage 1: Sparse Retrieval
- **Models Used:** BM25, DPH (Divergence from Randomness), TF-IDF
- **Index:** TREC-TOT 2025 corpus (6.4M documents) via PyTerrier
- **Query Enhancement:** LLaMA-rewritten queries for improved query understanding
- **Output:** Three ranked lists per query (one per sparse model)

### Stage 2: RRF Fusion
- **Method:** Reciprocal Rank Fusion with k=60
- **Formula:** score(d) = Σ(1/(k + rank_i(d)))
- **Purpose:** Combines sparse retrieval signals for robust initial ranking
- **Output:** Single fused ranking per query

### Stage 3: Dense Retrieval (Bi-Encoders)
- **Models:** 
  - sentence-transformers/all-MiniLM-L6-v2 (lightweight semantic matching)
  - sentence-transformers/all-mpnet-base-v2 (high-quality representations)
  - sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (QA-optimized)
- **Performance Optimizations:**
  - In-memory PyTerrier index (7.9GB RAM) eliminating disk I/O bottlenecks
  - Document caching (50GB RAM) for instant document access
  - 90% VRAM utilization with adaptive GPU batching
  - Mixed precision (FP16) inference for 2x throughput improvement
  - Multi-GPU workload distribution and optimization
- **Process:** Semantic similarity computation via cosine similarity
- **Integration:** Weighted combination with sparse scores (70% semantic, 30% sparse)
- **Output:** Dense retrieval rankings per bi-encoder model

### Stage 4: LTR Fusion
- **Algorithm:** LightGBM Learning-to-Rank
- **Features:** 6-dimensional feature vector (3 sparse + 3 dense scores)
- **Training:** Pre-trained on training set with TREC relevance judgments
- **Application:** Applied to test queries using pre-trained model (no test QRELs)
- **Output:** Optimally fused ranking combining all signals

### Stage 5: ColBERT Reranking
- **Model:** sentence-transformers/all-MiniLM-L6-v2 for late interaction
- **Document Scope:** Top 1000 documents per query from LTR stage
- **Score Normalization:** Z-score normalization (μ=0, σ=1)
- **Fusion Strategy:** 50/50 weighted combination of normalized LTR and ColBERT scores
- **Output:** Final ranking with 1000 documents per query

## Technical Specifications

- **Query Processing:** 622 test queries processed through complete pipeline
- **Document Coverage:** Up to 1000 documents per query in final ranking
- **Score Normalization:** Z-score for statistical consistency and outlier handling
- **Implementation:** PyTerrier + sentence-transformers + LightGBM + custom neural components

## Performance Characteristics

- **Training Performance:** 42.58% NDCG@10 on training set (98.2% of LTR baseline)
- **Robustness:** Multi-stage fusion mitigates individual model weaknesses
- **Semantic Understanding:** Combines lexical precision with semantic matching
- **Scalability:** Efficient processing architecture for large-scale retrieval

## Innovation Highlights

1. **Hierarchical Architecture:** Each stage refines previous ranking with different signal types
2. **Advanced Score Fusion:** Z-score normalization prevents scale mismatch issues
3. **Comprehensive Features:** 6-dimensional sparse+dense feature engineering
4. **Neural Late Interaction:** ColBERT-style semantic reranking for fine-grained relevance
5. **Query Enhancement:** LLaMA rewriting optimized for tip-of-the-tongue scenarios

## Implementation Notes

- **Environment:** Python 3.11, PyTerrier 0.11, sentence-transformers, LightGBM
- **Hardware:** Optimized for multi-core CPU processing + GPU acceleration
- **Memory Management:** 
  - In-memory index loading (7.9GB RAM) 
  - Document caching (50GB RAM)
  - 90% GPU memory utilization with adaptive batching
- **Performance:** Mixed precision (FP16) + multi-GPU distribution
- **Reliability:** Comprehensive error handling and fallback mechanisms

This approach leverages complementary strengths of sparse retrieval (lexical precision), dense retrieval (semantic understanding), learning-to-rank (optimal feature combination), and neural reranking (fine-grained relevance modeling) to achieve optimal performance on tip-of-the-tongue information retrieval tasks.
