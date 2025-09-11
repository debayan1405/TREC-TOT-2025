#!/usr/bin/env python3
"""
Improved ColBERT Reranker for TREC-TOT 2025
- Uses top 1000 documents from LTR fusion
- Removes arbitrary linear combination
- Implements proper score normalization
- Supports ColBERT-only reranking
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Add PyTerrier path
sys.path.append('/home/ugdf8/anaconda3/envs/pyterrier/lib/python3.11/site-packages')
import pyterrier as pt

class ImprovedColBERTReranker:
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2", 
                 use_gpu: bool = True, batch_size: int = 32, progress_logging: bool = False):
        """Initialize the improved ColBERT reranker."""
        if not pt.started():
            pt.init()
        
        self.index = pt.IndexFactory.of(index_path)
        self.metaindex = self.index.getMetaIndex()
        self.batch_size = batch_size
        self.progress_logging = progress_logging
        self.use_gpu = use_gpu
        
        # Set device
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Initialize SentenceTransformer with device
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"✅ Initialized ColBERT reranker with {model_name} on {self.device}")
        
        if use_gpu and torch.cuda.is_available():
            print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            print(f"📦 Batch size: {batch_size}")
    
    def get_document_text(self, docid: str) -> Optional[str]:
        """Retrieve document text by document ID."""
        try:
            doc = self.metaindex.getDocument('docno', docid)
            if doc is None:
                return None
            
            # Extract text content
            text_parts = []
            if hasattr(doc, 'get'):
                # Try common field names
                for field in ['text', 'content', 'body', 'title']:
                    field_content = doc.get(field)
                    if field_content:
                        text_parts.append(field_content)
            
            return ' '.join(text_parts) if text_parts else str(doc)
        except Exception as e:
            print(f"⚠️  Error retrieving document {docid}: {e}")
            return None
    
    def compute_colbert_similarity(self, query: str, doc_text: str) -> float:
        """Compute ColBERT-style similarity between query and document."""
        try:
            # Encode query and document
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            doc_embedding = self.model.encode(doc_text, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), 
                doc_embedding.unsqueeze(0)
            ).item()
            
            return similarity
        except Exception as e:
            print(f"⚠️  Error computing similarity: {e}")
            return 0.0
    
    def normalize_scores(self, scores: List[float], method: str = "minmax") -> List[float]:
        """Normalize scores using specified method."""
        if not scores or len(scores) == 1:
            return scores
        
        scores_array = np.array(scores).reshape(-1, 1)
        
        if method == "minmax":
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(scores_array).flatten()
        elif method == "zscore":
            scaler = StandardScaler()
            normalized = scaler.fit_transform(scores_array).flatten()
        else:
            # No normalization
            normalized = scores_array.flatten()
        
        return normalized.tolist()
    
    def rerank_query(self, query_id: str, query_text: str, ltr_results: List[Tuple[str, float]], 
                    top_k: int = 1000, rerank_method: str = "colbert_only", 
                    normalization: str = "minmax") -> List[Tuple[str, float]]:
        """
        Rerank documents for a single query.
        
        Args:
            query_id: Query identifier
            query_text: Query text
            ltr_results: List of (docid, ltr_score) tuples
            top_k: Number of top documents to rerank
            rerank_method: "colbert_only", "normalized_fusion", or "no_fusion"
            normalization: "minmax", "zscore", or "none"
        """
        # Take top K documents from LTR results
        top_docs = ltr_results[:top_k]
        
        print(f"🔍 Processing query {query_id}: reranking top {len(top_docs)} documents")
        
        # Compute ColBERT scores
        colbert_scores = []
        ltr_scores = []
        valid_docs = []
        
        for docid, ltr_score in top_docs:
            doc_text = self.get_document_text(docid)
            if doc_text:
                colbert_score = self.compute_colbert_similarity(query_text, doc_text)
                colbert_scores.append(colbert_score)
                ltr_scores.append(ltr_score)
                valid_docs.append(docid)
        
        if not valid_docs:
            print(f"⚠️  No valid documents found for query {query_id}")
            return []
        
        # Apply reranking strategy
        if rerank_method == "colbert_only":
            # Use only ColBERT scores
            final_scores = colbert_scores
        elif rerank_method == "normalized_fusion":
            # Normalize both score sets and combine
            norm_colbert = self.normalize_scores(colbert_scores, normalization)
            norm_ltr = self.normalize_scores(ltr_scores, normalization)
            
            # Equal weight combination after normalization
            final_scores = [0.5 * c + 0.5 * l for c, l in zip(norm_colbert, norm_ltr)]
        else:  # no_fusion
            # Use original LTR scores (baseline)
            final_scores = ltr_scores
        
        # Create final ranking
        doc_score_pairs = list(zip(valid_docs, final_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Query {query_id}: reranked {len(doc_score_pairs)} documents")
        return doc_score_pairs
    
    def process_queries(self, queries_file: str, ltr_results_file: str, 
                       output_file: str, top_k: int = 1000, 
                       rerank_method: str = "colbert_only", 
                       normalization: str = "minmax") -> None:
        """Process all queries and save results."""
        
        # Load queries
        queries = {}
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                queries[data['query_id']] = data['query']
        
        # Load LTR fusion results
        ltr_results = defaultdict(list)
        with open(ltr_results_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid = parts[0]
                    docid = parts[2]
                    score = float(parts[4])
                    ltr_results[qid].append((docid, score))
        
        # Sort LTR results by score (descending)
        for qid in ltr_results:
            ltr_results[qid].sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 Loaded {len(queries)} queries and {len(ltr_results)} LTR result sets")
        print(f"🎯 Strategy: {rerank_method} with {normalization} normalization")
        print(f"📈 Reranking top {top_k} documents per query")
        
        # Process each query
        all_results = []
        processed_queries = 0
        
        for query_id in sorted(queries.keys()):
            if query_id in ltr_results:
                query_text = queries[query_id]
                reranked_docs = self.rerank_query(
                    query_id, query_text, ltr_results[query_id],
                    top_k, rerank_method, normalization
                )
                
                # Convert to TREC format
                for rank, (docid, score) in enumerate(reranked_docs, 1):
                    all_results.append(f"{query_id} Q0 {docid} {rank} {score:.6f} improved_colbert\n")
                
                processed_queries += 1
                if processed_queries % 10 == 0:
                    print(f"📈 Processed {processed_queries}/{len(queries)} queries")
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.writelines(all_results)
        
        print(f"✅ Saved {len(all_results)} results to {output_file}")
        print(f"📊 Average docs per query: {len(all_results)/processed_queries:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Improved ColBERT Reranker")
    parser.add_argument("--dataset", choices=["train", "dev1", "dev2", "dev3", "test"], 
                       default="train", help="Dataset to process")
    parser.add_argument("--rewriter", choices=["llama", "chatgpt"], 
                       default="llama", help="Query rewriter used")
    parser.add_argument("--top-k", type=int, default=1000, 
                       help="Number of top documents to rerank")
    parser.add_argument("--method", choices=["colbert_only", "normalized_fusion", "no_fusion"],
                       default="colbert_only", help="Reranking method")
    parser.add_argument("--normalization", choices=["minmax", "zscore", "none"],
                       default="minmax", help="Score normalization method")
    parser.add_argument("--index-path", default="../trec-tot-2025-pyterrier-index",
                       help="Path to PyTerrier index")
    parser.add_argument("--gpu-acceleration", action="store_true",
                       help="Enable GPU acceleration")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for GPU processing")
    parser.add_argument("--progress-logging", action="store_true",
                       help="Enable detailed progress logging")
    
    args = parser.parse_args()
    
    # File paths
    queries_file = f"../rewritten_queries/{args.rewriter}_{args.dataset}_rewritten_queries.jsonl"
    
    # Check for available fusion files (prefer LTR, fallback to RRF)
    ltr_file = f"../dense_run_files/run_files/stage2_fusion/{args.rewriter}_{args.dataset}_ltr_fusion.txt"
    rrf_file = f"../dense_run_files/run_files/stage2_fusion/{args.rewriter}_{args.dataset}_rrf_fusion.txt"
    
    fusion_file = None
    fusion_type = None
    if os.path.exists(ltr_file):
        fusion_file = ltr_file
        fusion_type = "LTR"
    elif os.path.exists(rrf_file):
        fusion_file = rrf_file
        fusion_type = "RRF"
    else:
        raise FileNotFoundError(f"Neither LTR fusion file ({ltr_file}) nor RRF fusion file ({rrf_file}) found")
    
    # Output directory and file
    output_dir = f"improved_colbert_results"
    method_suffix = f"{args.method}_{args.normalization}" if args.method != "colbert_only" else args.method
    output_file = f"{output_dir}/{args.rewriter}_{args.dataset}_{method_suffix}_top{args.top_k}_run.txt"
    
    print(f"🚀 Starting Improved ColBERT Reranking")
    print(f"📂 Queries: {queries_file}")
    print(f"📂 {fusion_type} Results: {fusion_file}")
    print(f"📂 Output: {output_file}")
    
    # Initialize reranker and process
    reranker = ImprovedColBERTReranker(
        args.index_path,
        use_gpu=args.gpu_acceleration,
        batch_size=args.batch_size,
        progress_logging=args.progress_logging
    )
    reranker.process_queries(
        queries_file, fusion_file, output_file,
        args.top_k, args.method, args.normalization
    )
    
    print("🎉 Improved ColBERT reranking completed!")

if __name__ == "__main__":
    main()
