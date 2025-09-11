#!/usr/bin/env python3
"""
ColBERT Middle Reranking - With Real Document Retrieval
Reranks LTR fusion results using ColBERT with actual document content from PyTerrier index
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from document_retriever import DocumentRetriever

try:
    import torch
except ImportError:
    torch = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ltr_results(ltr_file: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load LTR fusion results from TREC format file"""
    query_docs = {}
    
    try:
        with open(ltr_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    query_id = parts[0]
                    doc_id = parts[2] 
                    score = float(parts[4])
                    
                    if query_id not in query_docs:
                        query_docs[query_id] = []
                    
                    query_docs[query_id].append((doc_id, score))
        
        # Sort by score for each query
        for query_id in query_docs:
            query_docs[query_id].sort(key=lambda x: x[1], reverse=True)
            
        logger.info(f"✅ Loaded LTR results: {len(query_docs)} queries")
        return query_docs
        
    except Exception as e:
        logger.error(f"❌ Failed to load LTR results: {e}")
        return {}

def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from JSONL file"""
    queries = {}
    
    try:
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = str(data['query_id'])
                query_text = data['query']
                queries[query_id] = query_text
                
        logger.info(f"✅ Loaded queries: {len(queries)} queries")
        return queries
        
    except Exception as e:
        logger.error(f"❌ Failed to load queries: {e}")
        return {}

class SimpleColBERTReranker:
    """Simplified ColBERT reranker that handles API correctly"""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        try:
            self._init_colbert()
        except Exception as e:
            logger.error(f"ColBERT initialization failed: {e}")
            self.model = None
    
    def _init_colbert(self):
        """Initialize ColBERT with proper API"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            logger.info(f"Loading ColBERT model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("✅ Model loaded on GPU")
            else:
                logger.info("✅ Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"ColBERT initialization failed: {e}")
            raise
    
    def encode_texts(self, texts: List[str], max_length: int = 512):
        """Encode texts using ColBERT-style encoding"""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        if torch.cuda.is_available() and self.model.device.type == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state
            embeddings = outputs.last_hidden_state
            
        return embeddings
    
    def compute_late_interaction_score(self, query_embedding, doc_embedding) -> float:
        """Compute late interaction score between query and document"""
        import torch
        
        # Query embedding: [1, query_len, hidden_size]
        # Doc embedding: [1, doc_len, hidden_size]
        
        # Compute similarity matrix
        # Shape: [query_len, doc_len]
        similarity_matrix = torch.matmul(
            query_embedding[0],  # [query_len, hidden_size]
            doc_embedding[0].transpose(0, 1)  # [hidden_size, doc_len]
        )
        
        # Max pooling over document tokens for each query token
        max_scores = torch.max(similarity_matrix, dim=1)[0]  # [query_len]
        
        # Sum over query tokens
        score = torch.sum(max_scores).item()
        
        return score
    
    def rerank_documents(self, query_text: str, doc_ids: List[str], 
                        initial_scores: List[float], retriever: DocumentRetriever, 
                        top_k: int = 100) -> List[Dict]:
        """Rerank documents using ColBERT with real document content"""
        
        if self.model is None:
            logger.warning("ColBERT model not available, using initial scores")
            return self._fallback_rerank(doc_ids, initial_scores, top_k)
        
        try:
            # Retrieve actual document texts
            logger.info(f"🔍 Retrieving {len(doc_ids)} documents from index...")
            doc_texts = []
            for doc_id in doc_ids:
                doc_text = retriever.get_document_text(doc_id)
                doc_texts.append(doc_text)
            
            # Filter out documents that couldn't be retrieved properly
            valid_docs = []
            valid_scores = []
            for i, (doc_id, doc_text, score) in enumerate(zip(doc_ids, doc_texts, initial_scores)):
                if doc_text and not doc_text.startswith("Document") or "error" not in doc_text:
                    valid_docs.append((doc_id, doc_text))
                    valid_scores.append(score)
            
            if not valid_docs:
                logger.warning("No valid documents retrieved, falling back to initial scores")
                return self._fallback_rerank(doc_ids, initial_scores, top_k)
            
            # Separate IDs and texts
            valid_doc_ids = [item[0] for item in valid_docs]
            valid_doc_texts = [item[1] for item in valid_docs]
            
            logger.info(f"📝 Successfully retrieved {len(valid_docs)} documents, encoding with ColBERT...")
            
            # Encode query
            query_embedding = self.encode_texts([query_text], max_length=64)
            
            # Encode documents in batches
            batch_size = 16
            colbert_scores = []
            
            for i in range(0, len(valid_doc_texts), batch_size):
                batch_texts = valid_doc_texts[i:i + batch_size]
                doc_embeddings = self.encode_texts(batch_texts, max_length=512)
                
                # Compute scores for batch
                for j in range(len(batch_texts)):
                    doc_emb = doc_embeddings[j:j+1]  # Keep batch dimension
                    score = self.compute_late_interaction_score(query_embedding, doc_emb)
                    colbert_scores.append(score)
            
            # Combine with initial scores
            results = []
            for i, doc_id in enumerate(valid_doc_ids):
                colbert_score = colbert_scores[i]
                initial_score = valid_scores[i]
                
                # Weighted combination: 70% ColBERT, 30% initial
                combined_score = 0.7 * colbert_score + 0.3 * initial_score
                
                results.append({
                    'docid': doc_id,
                    'colbert_score': colbert_score,
                    'initial_score': initial_score,
                    'combined_score': combined_score
                })
            
            # Sort and rank
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            for i, result in enumerate(results[:top_k]):
                result['rank'] = i + 1
                result['method'] = 'ColBERT'
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"ColBERT reranking failed: {e}")
            return self._fallback_rerank(doc_ids, initial_scores, top_k)
    
    def _fallback_rerank(self, doc_ids: List[str], initial_scores: List[float], 
                        top_k: int) -> List[Dict]:
        """Fallback reranking using initial scores only"""
        
        results = []
        for i, doc_id in enumerate(doc_ids):
            results.append({
                'docid': doc_id,
                'colbert_score': initial_scores[i],  # Use initial as fallback
                'initial_score': initial_scores[i],
                'combined_score': initial_scores[i]
            })
        
        # Sort and rank
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        for i, result in enumerate(results[:top_k]):
            result['rank'] = i + 1
            result['method'] = 'Fallback'
        
        return results[:top_k]

def evaluate_results(results_file: str, qrel_file: str) -> Dict[str, float]:
    """Evaluate results against QREL using pytrec_eval"""
    try:
        import pytrec_eval
        
        # Load qrels
        qrels = {}
        with open(qrel_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id = parts[0]
                    doc_id = parts[2]
                    relevance = int(parts[3])
                    
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = relevance
        
        # Load results
        run = {}
        with open(results_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    query_id = parts[0]
                    doc_id = parts[2]
                    score = float(parts[4])
                    
                    if query_id not in run:
                        run[query_id] = {}
                    run[query_id][doc_id] = score
        
        # Evaluate
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recip_rank', 'map'})
        evaluation = evaluator.evaluate(run)
        
        # Aggregate results
        metrics = {}
        for measure in ['ndcg_cut_10', 'recip_rank', 'map']:
            metrics[measure] = np.mean([evaluation[qid][measure] for qid in evaluation])
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="ColBERT Middle Reranking")
    parser.add_argument("--dataset", required=True, choices=["train", "dev1", "dev2", "dev3"], 
                       help="Dataset to process")
    parser.add_argument("--rewriter", required=True, choices=["llama"], 
                       help="Query rewriter used")
    parser.add_argument("--ltr-file", help="LTR fusion results file")
    parser.add_argument("--queries-file", help="Rewritten queries file")
    parser.add_argument("--qrel-file", help="QREL file for evaluation")
    parser.add_argument("--output-dir", default="colbert_results", help="Output directory")
    parser.add_argument("--top-k", type=int, default=100, help="Number of documents to rerank")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if not args.ltr_file:
        args.ltr_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/dense_run_files/run_files/stage2_fusion/{args.rewriter}_{args.dataset}_ltr_fusion.txt"
    
    if not args.queries_file:
        args.queries_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/rewritten_queries/{args.rewriter}_{args.dataset}_rewritten_queries.jsonl"
    
    if not args.qrel_file:
        args.qrel_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/qrel/{args.dataset}-2025-qrel.txt"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=== ColBERT Middle Reranking ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Rewriter: {args.rewriter}")
    logger.info(f"LTR file: {args.ltr_file}")
    logger.info(f"Queries file: {args.queries_file}")
    logger.info(f"QREL file: {args.qrel_file}")
    
    # Load data
    logger.info("📂 Loading data...")
    ltr_results = load_ltr_results(args.ltr_file)
    queries = load_queries(args.queries_file)
    
    if not ltr_results or not queries:
        logger.error("❌ Failed to load required data")
        return 1
    
    # Initialize ColBERT reranker
    logger.info("🤖 Initializing ColBERT reranker...")
    reranker = SimpleColBERTReranker()
    
    # Initialize document retriever
    logger.info("📚 Initializing document retriever...")
    index_path = "/home/ugdf8/IRIS/TREC-TOT-2025/trec-tot-2025-pyterrier-index"
    retriever = DocumentRetriever(index_path)
    
    # Process queries
    logger.info("🔄 Processing queries...")
    all_results = []
    
    for query_id, query_text in queries.items():
        if query_id not in ltr_results:
            logger.warning(f"⚠️  Query {query_id} not found in LTR results")
            continue
        
        # Get top documents from LTR
        ltr_docs = ltr_results[query_id][:args.top_k]
        doc_ids = [doc[0] for doc in ltr_docs]
        initial_scores = [doc[1] for doc in ltr_docs]
        
        logger.info(f"Processing query {query_id}: {len(doc_ids)} documents")
        
        # Rerank with ColBERT using real document content
        reranked = reranker.rerank_documents(query_text, doc_ids, initial_scores, retriever, args.top_k)
        
        # Add to results
        for result in reranked:
            all_results.append({
                'query_id': query_id,
                'query_text': query_text,
                **result
            })
    
    # Save results
    logger.info("💾 Saving results...")
    
    # Save as CSV
    csv_file = output_dir / f"colbert_{args.rewriter}_{args.dataset}_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)
    logger.info(f"✅ Results saved to {csv_file}")
    
    # Save as TREC format
    trec_file = output_dir / f"colbert_{args.rewriter}_{args.dataset}_run.txt"
    with open(trec_file, 'w') as f:
        for result in all_results:
            f.write(f"{result['query_id']} Q0 {result['docid']} {result['rank']} {result['combined_score']:.6f} ColBERT\n")
    logger.info(f"✅ TREC format saved to {trec_file}")
    
    # Evaluate if QREL file exists
    if os.path.exists(args.qrel_file):
        logger.info("📊 Evaluating results...")
        metrics = evaluate_results(str(trec_file), args.qrel_file)
        
        if metrics:
            logger.info("=== Evaluation Results ===")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Save metrics
            metrics_file = output_dir / f"colbert_{args.rewriter}_{args.dataset}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"✅ Metrics saved to {metrics_file}")
    else:
        logger.warning(f"⚠️  QREL file not found: {args.qrel_file}")
    
    logger.info("✅ ColBERT middle reranking completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
