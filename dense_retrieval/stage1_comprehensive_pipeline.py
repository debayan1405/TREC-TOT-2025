#!/usr/bin/env python3
"""
Stage 1 Comprehensive Pipeline: Bi-Encoder Dense Retrieval
=========================================================

This pipeline implements Stage 1 dense retrieval using multiple bi-encoder models:
1. sentence-transformers/all-MiniLM-L6-v2
2. sentence-transformers/all-mpnet-base-v2  
3. sentence-transformers/multi-qa-MiniLM-L6-cos-v1

Input: RRF fused sparse retrieval results
Output: Dense retrieval results for each bi-encoder model
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import torch
import numpy as np

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Sentence transformers for dense retrieval
from sentence_transformers import SentenceTransformer

# PyTerrier for index access
import pyterrier as pt
if not pt.started():
    pt.init()

class Stage1BiEncoderPipeline:
    """Stage 1 Bi-Encoder Dense Retrieval Pipeline"""
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = pt.IndexFactory.of(index_path)
        self.metaindex = self.index.getMetaIndex()
        
        # Bi-encoder models to use
        self.models = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2", 
            "multi-qa-MiniLM-L6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        }
        
        # Load models
        self.encoders = {}
        print("🚀 Loading bi-encoder models...")
        for name, model_path in self.models.items():
            print(f"   Loading {name}...")
            self.encoders[name] = SentenceTransformer(model_path)
        print("✅ All models loaded successfully!")
    
    def get_document_text(self, docid: str) -> str:
        """Retrieve document text by document ID"""
        try:
            doc = self.metaindex.getDocument('docno', docid)
            if doc is None:
                return ""
            
            # Extract text content
            text_parts = []
            if hasattr(doc, 'get'):
                for field in ['text', 'content', 'body', 'title']:
                    field_content = doc.get(field)
                    if field_content:
                        text_parts.append(field_content)
            
            return ' '.join(text_parts) if text_parts else str(doc)
        except Exception as e:
            print(f"⚠️  Error retrieving document {docid}: {e}")
            return ""
    
    def compute_semantic_similarity(self, query: str, doc_text: str, model_name: str) -> float:
        """Compute semantic similarity using specified model"""
        try:
            encoder = self.encoders[model_name]
            
            # Encode query and document
            query_embedding = encoder.encode(query, convert_to_tensor=True)
            doc_embedding = encoder.encode(doc_text, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), 
                doc_embedding.unsqueeze(0)
            ).item()
            
            return similarity
        except Exception as e:
            print(f"⚠️  Error computing similarity: {e}")
            return 0.0
    
    def process_query(self, query_id: str, query_text: str, fused_results: List[Tuple[str, float]], 
                     model_name: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """Process a single query with a specific bi-encoder model"""
        
        print(f"🔍 Processing query {query_id} with {model_name} (top {top_k} docs)")
        
        # Take top-k documents from fused results
        top_docs = fused_results[:top_k]
        
        # Compute semantic similarities
        results = []
        processed = 0
        
        for docid, sparse_score in top_docs:
            doc_text = self.get_document_text(docid)
            if doc_text:
                semantic_score = self.compute_semantic_similarity(query_text, doc_text, model_name)
                
                # Combine scores (weighted combination)
                combined_score = 0.7 * semantic_score + 0.3 * sparse_score
                results.append((docid, combined_score))
            else:
                # Keep original score if no text available
                results.append((docid, sparse_score))
            
            processed += 1
            if processed % 100 == 0:
                print(f"   Processed {processed}/{len(top_docs)} documents")
        
        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Query {query_id}: processed {len(results)} documents with {model_name}")
        return results
    
    def load_queries(self, queries_file: str) -> Dict[str, str]:
        """Load queries from JSONL file"""
        queries = {}
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                queries[data['query_id']] = data['query']
        
        print(f"📖 Loaded {len(queries)} queries from {queries_file}")
        return queries
    
    def load_fused_results(self, fused_file: str) -> Dict[str, List[Tuple[str, float]]]:
        """Load RRF fused results"""
        fused_results = defaultdict(list)
        
        with open(fused_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid = parts[0]
                    docid = parts[2]
                    score = float(parts[4])
                    fused_results[qid].append((docid, score))
        
        # Sort by score (descending)
        for qid in fused_results:
            fused_results[qid].sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 Loaded fused results for {len(fused_results)} queries")
        return fused_results
    
    def save_results(self, results: Dict[str, List[Tuple[str, float]]], 
                    output_file: str, model_name: str) -> None:
        """Save results in TREC format"""
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for query_id in sorted(results.keys()):
                for rank, (docid, score) in enumerate(results[query_id], 1):
                    f.write(f"{query_id} Q0 {docid} {rank} {score:.6f} {model_name}\n")
        
        total_results = sum(len(docs) for docs in results.values())
        print(f"✅ Saved {total_results} results to {output_file}")
    
    def run_pipeline(self, dataset: str, rewriter: str, queries_file: str, 
                    fused_file: str, output_dir: str, top_k: int = 1000) -> None:
        """Run the complete Stage 1 pipeline"""
        
        print(f"🚀 Starting Stage 1 Bi-Encoder Pipeline")
        print(f"📂 Dataset: {dataset}")
        print(f"✍️  Rewriter: {rewriter}")
        print(f"📝 Queries: {queries_file}")
        print(f"🔗 Fused results: {fused_file}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎯 Top-K documents: {top_k}")
        
        # Load data
        queries = self.load_queries(queries_file)
        fused_results = self.load_fused_results(fused_file)
        
        # Process each model
        for model_short_name, model_full_name in self.models.items():
            print(f"\\n{'='*60}")
            print(f"🤖 Processing with {model_short_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            # Process each query
            for query_id in sorted(queries.keys()):
                if query_id in fused_results:
                    query_text = queries[query_id]
                    query_fused_results = fused_results[query_id]
                    
                    # Process with current model
                    model_results[query_id] = self.process_query(
                        query_id, query_text, query_fused_results, 
                        model_short_name, top_k
                    )
                else:
                    print(f"⚠️  No fused results for query {query_id}")
            
            # Save results
            output_file = Path(output_dir) / f"{rewriter}_{dataset}_{model_short_name.replace('/', '_')}_dense.txt"
            self.save_results(model_results, str(output_file), model_short_name)
            
            print(f"✅ {model_short_name} processing completed!")
        
        print(f"\\n🎉 Stage 1 Bi-Encoder Pipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Bi-Encoder Dense Retrieval Pipeline")
    parser.add_argument("--dataset", choices=["train", "dev1", "dev2", "dev3", "test"], 
                       default="test", help="Dataset to process")
    parser.add_argument("--rewriter", choices=["llama", "chatgpt"], 
                       default="llama", help="Query rewriter used")
    parser.add_argument("--queries-file", required=True,
                       help="Path to queries JSONL file")
    parser.add_argument("--fused-file", required=True,
                       help="Path to RRF fused results file")
    parser.add_argument("--index-path", default="../trec-tot-2025-pyterrier-index",
                       help="Path to PyTerrier index")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for dense retrieval results")
    parser.add_argument("--top-k", type=int, default=1000,
                       help="Number of top documents to process per query")
    
    args = parser.parse_args()
    
    print(f"🚀 Stage 1 Bi-Encoder Pipeline Starting")
    print(f"📂 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Rewriter: {args.rewriter}")
    print(f"   Index: {args.index_path}")
    print(f"   Top-K: {args.top_k}")
    
    # Initialize and run pipeline
    pipeline = Stage1BiEncoderPipeline(args.index_path)
    pipeline.run_pipeline(
        args.dataset, args.rewriter, args.queries_file,
        args.fused_file, args.output_dir, args.top_k
    )
    
    print("🎉 Stage 1 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
