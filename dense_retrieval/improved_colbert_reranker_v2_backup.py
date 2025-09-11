#!/usr/bin/env python3
"""
Improved ColBERT Middle Reranking with Proper Document Mapping and Score Handling
Addresses key issues:
1. Uses 1000+ documents from full LTR fusion list
2. Proper document ID mapping using PyTerrier getDocumentId()
3. Direct ColBERT scores without blind mixing
4. Score normalization options
"""

import argparse
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import pyterrier as pt

# ColBERT imports
from colbert import Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedColBERTReranker:
    def __init__(self, index_path: str, model_name: str = "colbert-ir/colbertv2.0"):
        """Initialize improved ColBERT reranker with proper document mapping"""
        self.index_path = index_path
        self.model_name = model_name
        
        # Initialize PyTerrier index for proper document ID mapping
        logger.info("🔧 Initializing PyTerrier index for document mapping...")
        if not pt.started():
            pt.init()
        
        # Load PyTerrier index
        self.pt_index = pt.IndexFactory.of(index_path)
        self.meta_index = self.pt_index.getMetaIndex()
        self.doc_index = self.pt_index.getDocumentIndex()
        self.index_size = self.meta_index.size()
        
        logger.info(f"✅ PyTerrier index loaded: {self.index_size:,} documents")
        
        # Initialize ColBERT model
        logger.info(f"🤖 Loading ColBERT model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Using device: {self.device}")
        
        # Set up ColBERT configuration
        self.config = ColBERTConfig(
            doc_maxlen=512,
            query_maxlen=64,
            dim=128,
            checkpoint=model_name
        )
        
        # Load model for encoding
        try:
            from colbert.modeling.colbert import ColBERT
            self.checkpoint = ColBERT(name=model_name, colbert_config=self.config)
            logger.info("✅ ColBERT model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading ColBERT model: {e}")
            raise
    
    def get_document_text(self, external_docno):
        """
        Retrieve document text using proper PyTerrier mapping
        Uses same logic as LTR phase for consistency
        """
        try:
            # Use metaindex.getDocument for proper mapping (same as LTR phase)
            internal_docid = self.meta_index.getDocument("docno", str(external_docno))
            
            if internal_docid >= 0:
                # Get document text using meta index
                text = self.meta_index.getItem("text", internal_docid)
                if text and len(text.strip()) > 0:
                    return text.strip()
            
            return None
            
        except Exception as e:
            logging.warning(f"Error retrieving document {external_docno}: {e}")
            return None
    
    def load_ltr_fusion_results(self, ltr_file: str, max_docs_per_query: int = 1000) -> pd.DataFrame:
        """Load LTR fusion results in TREC format"""
        logger.info(f"📂 Loading LTR fusion results from: {ltr_file}")
        
        data = []
        with open(ltr_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    data.append({
                        'query_id': parts[0],
                        'doc_id': parts[2],
                        'rank': int(parts[3]),
                        'ltr_score': float(parts[4])
                    })
        
        df = pd.DataFrame(data)
        
        # Limit to max_docs_per_query for each query
        if max_docs_per_query > 0:
            df = df.groupby('query_id').head(max_docs_per_query).reset_index(drop=True)
        
        logger.info(f"✅ Loaded {len(df):,} documents for {df['query_id'].nunique()} queries")
        logger.info(f"📊 Average docs per query: {len(df)/df['query_id'].nunique():.1f}")
        
        return df
    
    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts using ColBERT with batching"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                # Use checkpoint to encode documents
                with torch.no_grad():
                    batch_embeddings = self.checkpoint.docFromText(batch_texts, bsize=len(batch_texts))
                    all_embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding batch {i//batch_size + 1}: {e}")
                # Create dummy embeddings for failed batch
                dummy_emb = torch.zeros((len(batch_texts), 128, self.checkpoint.colbert.dim)).to(self.device)
                all_embeddings.append(dummy_emb)
        
        return torch.cat(all_embeddings, dim=0)
    
    def compute_colbert_scores(self, query_text, doc_texts):
        """Compute ColBERT late interaction scores"""
        try:
            if not doc_texts:
                return []
            
            # Encode query using ColBERT's query method
            query_encodings = self.checkpoint.query(query_text)
            
            # Process documents in batches for memory efficiency
            batch_size = 32
            all_scores = []
            
            for i in range(0, len(doc_texts), batch_size):
                try:
                    batch_texts = doc_texts[i:i + batch_size]
                    
                    # Encode batch of documents using ColBERT's doc method
                    doc_encodings = self.checkpoint.doc(batch_texts, keep_dims='return_mask')
                    
                    # Compute late interaction scores
                    scores = self.checkpoint.score(query_encodings, doc_encodings)
                    
                    # Extract scores and convert to list
                    if hasattr(scores, 'cpu'):
                        batch_scores = scores.cpu().numpy().tolist()
                    elif isinstance(scores, (list, tuple)):
                        batch_scores = list(scores)
                    else:
                        batch_scores = [float(scores)] if isinstance(scores, (int, float)) else []
                    
                    all_scores.extend(batch_scores)
                    
                except Exception as batch_e:
                    logging.error(f"Error encoding batch {i//batch_size}: {batch_e}")
                    # Add default scores for failed batch
                    all_scores.extend([0.0] * len(doc_texts[i:i + batch_size]))
                    
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in ColBERT scoring: {e}")
            return [0.0] * len(doc_texts)
    
    def normalize_scores(self, scores: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize scores using different methods"""
        if method == "minmax":
            scaler = MinMaxScaler()
            return scaler.fit_transform(scores.reshape(-1, 1)).flatten()
        elif method == "zscore":
            return zscore(scores, nan_policy='omit')
        elif method == "none":
            return scores
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def rerank_documents(self, 
                        queries_file: str, 
                        ltr_file: str, 
                        output_file: str,
                        max_docs_per_query: int = 1000,
                        scoring_method: str = "colbert_only",
                        normalization: str = "minmax") -> Dict:
        """
        Rerank documents using improved ColBERT approach
        
        Args:
            queries_file: Path to queries JSONL file
            ltr_file: Path to LTR fusion TREC file
            output_file: Output file prefix
            max_docs_per_query: Maximum documents to rerank per query
            scoring_method: "colbert_only", "ltr_only", or "combined"
            normalization: "minmax", "zscore", or "none"
        """
        logger.info("🚀 Starting improved ColBERT reranking...")
        logger.info(f"   Max docs per query: {max_docs_per_query}")
        logger.info(f"   Scoring method: {scoring_method}")
        logger.info(f"   Score normalization: {normalization}")
        
        # Load queries
        queries = {}
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                queries[str(data['query_id'])] = data['query']
        
        logger.info(f"📝 Loaded {len(queries)} queries")
        
        # Load LTR fusion results
        ltr_df = self.load_ltr_fusion_results(ltr_file, max_docs_per_query)
        
        results = []
        failed_queries = 0
        successful_docs = 0
        failed_docs = 0
        
        for query_id in queries:
            if query_id not in ltr_df['query_id'].values:
                logger.warning(f"Query {query_id} not found in LTR results")
                failed_queries += 1
                continue
            
            query_text = queries[query_id]
            query_docs = ltr_df[ltr_df['query_id'] == query_id].copy()
            
            logger.info(f"🔍 Processing query {query_id}: {len(query_docs)} documents")
            
            # Retrieve document texts using proper mapping
            doc_texts = []
            valid_indices = []
            
            for idx, row in query_docs.iterrows():
                doc_text = self.get_document_text(row['doc_id'])
                if doc_text:
                    doc_texts.append(doc_text)
                    valid_indices.append(idx)
                    successful_docs += 1
                else:
                    failed_docs += 1
            
            if not doc_texts:
                logger.warning(f"No valid documents found for query {query_id}")
                failed_queries += 1
                continue
            
            # Filter to valid documents only
            valid_docs = query_docs.loc[valid_indices].copy()
            
            logger.info(f"📝 Successfully retrieved {len(doc_texts)} documents, computing ColBERT scores...")
            
            # Compute ColBERT scores
            colbert_scores = self.compute_colbert_scores(query_text, doc_texts)
            
            # Prepare scores for combination
            ltr_scores = valid_docs['ltr_score'].values
            colbert_scores = np.array(colbert_scores)
            
            # Apply normalization if combining scores
            if scoring_method == "combined" and normalization != "none":
                ltr_scores_norm = self.normalize_scores(ltr_scores, normalization)
                colbert_scores_norm = self.normalize_scores(colbert_scores, normalization)
            else:
                ltr_scores_norm = ltr_scores
                colbert_scores_norm = colbert_scores
            
            # Determine final scores based on method
            if scoring_method == "colbert_only":
                final_scores = colbert_scores
            elif scoring_method == "ltr_only":
                final_scores = ltr_scores
            elif scoring_method == "combined":
                # 50-50 combination after normalization
                final_scores = 0.5 * ltr_scores_norm + 0.5 * colbert_scores_norm
            else:
                raise ValueError(f"Unknown scoring method: {scoring_method}")
            
            # Create results for this query
            for i, (idx, row) in enumerate(valid_docs.iterrows()):
                results.append({
                    'query_id': query_id,
                    'doc_id': row['doc_id'],
                    'ltr_score': ltr_scores[i],
                    'colbert_score': colbert_scores[i],
                    'final_score': final_scores[i],
                    'original_rank': row['rank']
                })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            logger.error("❌ No results generated!")
            return {}
        
        # Sort by query and final score
        results_df = results_df.sort_values(['query_id', 'final_score'], ascending=[True, False])
        
        # Add new ranks
        results_df['new_rank'] = results_df.groupby('query_id').cumcount() + 1
        
        # Save results
        results_csv = f"{output_file}_results.csv"
        results_df.to_csv(results_csv, index=False)
        logger.info(f"✅ Results saved to: {results_csv}")
        
        # Save TREC format
        trec_file = f"{output_file}_run.txt"
        with open(trec_file, 'w') as f:
            for _, row in results_df.iterrows():
                f.write(f"{row['query_id']}\tQ0\t{row['doc_id']}\t{row['new_rank']}\t{row['final_score']:.6f}\tImprovedColBERT\n")
        
        logger.info(f"✅ TREC format saved to: {trec_file}")
        
        # Generate statistics
        stats = {
            'total_queries': len(queries),
            'processed_queries': results_df['query_id'].nunique(),
            'failed_queries': failed_queries,
            'successful_documents': successful_docs,
            'failed_documents': failed_docs,
            'total_results': len(results_df),
            'avg_docs_per_query': len(results_df) / results_df['query_id'].nunique() if len(results_df) > 0 else 0,
            'scoring_method': scoring_method,
            'normalization': normalization,
            'max_docs_per_query': max_docs_per_query
        }
        
        # Save statistics
        stats_file = f"{output_file}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("📊 Processing Statistics:")
        logger.info(f"   Processed queries: {stats['processed_queries']}/{stats['total_queries']}")
        logger.info(f"   Document retrieval success rate: {successful_docs/(successful_docs+failed_docs)*100:.1f}%")
        logger.info(f"   Average docs per query: {stats['avg_docs_per_query']:.1f}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Improved ColBERT Middle Reranking")
    parser.add_argument("--dataset", choices=["train", "dev1", "dev2", "dev3", "test"], 
                       default="train", help="Dataset to process")
    parser.add_argument("--rewriter", choices=["llama", "openai"], 
                       default="llama", help="Query rewriter used")
    parser.add_argument("--index-path", 
                       default="/home/ugdf8/IRIS/TREC-TOT-2025/trec-tot-2025-pyterrier-index",
                       help="Path to PyTerrier index")
    parser.add_argument("--max-docs", type=int, default=1000,
                       help="Maximum documents to rerank per query")
    parser.add_argument("--scoring", choices=["colbert_only", "ltr_only", "combined"],
                       default="colbert_only", help="Scoring method")
    parser.add_argument("--normalization", choices=["minmax", "zscore", "none"],
                       default="minmax", help="Score normalization method")
    parser.add_argument("--model", default="colbert-ir/colbertv2.0",
                       help="ColBERT model to use")
    
    args = parser.parse_args()
    
    # Setup file paths
    queries_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/rewritten_queries/{args.rewriter}_{args.dataset}_rewritten_queries.jsonl"
    ltr_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/dense_run_files/run_files/stage2_fusion/{args.rewriter}_{args.dataset}_ltr_fusion.txt"
    output_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/dense_retrieval/improved_colbert_results/improved_{args.rewriter}_{args.dataset}_{args.scoring}_{args.max_docs}"
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize reranker
    reranker = ImprovedColBERTReranker(args.index_path, args.model)
    
    # Run reranking
    start_time = time.time()
    stats = reranker.rerank_documents(
        queries_file=queries_file,
        ltr_file=ltr_file,
        output_file=output_file,
        max_docs_per_query=args.max_docs,
        scoring_method=args.scoring,
        normalization=args.normalization
    )
    
    duration = time.time() - start_time
    
    logger.info(f"✅ Improved ColBERT reranking completed in {duration:.1f} seconds!")
    
    # Evaluate results if QREL file exists
    qrel_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/qrel/{args.dataset}-2025-qrel.txt"
    if Path(qrel_file).exists():
        logger.info("📊 Evaluating results...")
        from evaluation_utils import evaluate_trec_run
        
        trec_file = f"{output_file}_run.txt"
        metrics = evaluate_trec_run(trec_file, qrel_file)
        
        logger.info("📈 Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        # Save metrics
        metrics_file = f"{output_file}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"💾 Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
