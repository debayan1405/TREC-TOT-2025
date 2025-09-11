#!/usr/bin/env python3
"""
Fixed ColBERT Reranker for TREC TOT 2025 - Stage 3 Dense Retrieval
This script implements middle reranking using ColBERT-style semantic similarity
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pyterrier as pt
from sentence_transformers import SentenceTransformer, util
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FunctionalColBERTReranker:
    """Functional ColBERT reranker using sentence transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = None):
        """Initialize the reranker"""
        self.model_name = model_name
        self.model = None
        self.index = None
        
        # Load sentence transformer model
        logging.info(f"🔄 Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logging.info(f"✅ Model loaded successfully")
        
        # Load PyTerrier index
        if index_path:
            self.load_index(index_path)
    
    def load_index(self, index_path: str):
        """Load PyTerrier index"""
        logging.info(f"📂 Loading PyTerrier index from: {index_path}")
        
        if not pt.started():
            pt.init()
        
        self.index = pt.IndexFactory.of(index_path)
        doc_count = self.index.getCollectionStatistics().getNumberOfDocuments()
        logging.info(f"✅ Index loaded with {doc_count:,} documents")
    
    def compute_similarity_scores(self, query_text: str, doc_texts: list) -> list:
        """Compute semantic similarity scores between query and documents"""
        
        try:
            # Encode query and documents
            query_embedding = self.model.encode([query_text], convert_to_tensor=True)
            doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
            
            # Compute cosine similarities
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            logging.error(f"Error computing similarities: {e}")
            return [0.0] * len(doc_texts)
    
    def rerank_documents(self, queries_file: str, ltr_fusion_file: str, max_docs: int = 1000) -> tuple:
        """Rerank documents using semantic similarity"""
        
        # Load queries
        logging.info("📝 Loading queries...")
        queries = {}
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                queries[str(data['query_id'])] = data['query']
        
        logging.info(f"✅ Loaded {len(queries)} queries")
        
        # Load LTR fusion results (TREC format)
        logging.info(f"📂 Loading LTR fusion results from: {ltr_fusion_file}")
        ltr_data = []
        with open(ltr_fusion_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, docno, rank, score, run_name = parts[:6]
                    ltr_data.append({
                        'qid': int(qid),
                        'docno': docno,
                        'rank': int(rank),
                        'score': float(score)
                    })
        
        ltr_results = pd.DataFrame(ltr_data)
        logging.info(f"✅ Loaded {len(ltr_results)} documents for {len(ltr_results['qid'].unique())} queries")
        
        # Limit documents per query
        limited_results = []
        for qid in ltr_results['qid'].unique():
            query_results = ltr_results[ltr_results['qid'] == qid].head(max_docs)
            limited_results.append(query_results)
        
        ltr_results = pd.concat(limited_results, ignore_index=True)
        logging.info(f"📊 Limited to {len(ltr_results)} documents ({max_docs} per query)")
        
        # Process each query
        all_results = []
        processed_queries = 0
        
        for qid, query_results in ltr_results.groupby('qid'):
            if str(qid) not in queries:
                continue
            
            query_text = queries[str(qid)]
            doc_count = len(query_results)
            
            logging.info(f"🔍 Processing query {qid}: {doc_count} documents")
            
            # Get document texts
            doc_texts = []
            valid_docs = []
            success_count = 0
            failed_count = 0
            
            for _, row in query_results.iterrows():
                docid = str(row['docno'])
                try:
                    # Get document using correct API
                    metaindex = self.index.getMetaIndex()
                    docid_int = metaindex.getDocument('docno', docid)
                    if docid_int >= 0:
                        text = metaindex.getItem('text', docid_int)
                        if text:
                            doc_texts.append(text)
                            valid_docs.append(row)
                            success_count += 1
                        else:
                            logging.warning(f"Document {docid} has no text")
                            failed_count += 1
                    else:
                        logging.warning(f"Document {docid} not found in index")
                        failed_count += 1
                except Exception as e:
                    logging.error(f"Error retrieving document {docid}: {e}")
                    failed_count += 1
            
            if not doc_texts:
                logging.warning(f"No valid documents for query {qid}")
                continue
            
            logging.info(f"📝 Retrieved {len(doc_texts)} valid documents ({success_count} success, {failed_count} failed), computing similarity scores...")
            
            # Compute similarity scores
            similarity_scores = self.compute_similarity_scores(query_text, doc_texts)
            
            # Create results with new scores
            for i, doc_row in enumerate(valid_docs):
                similarity_score = similarity_scores[i] if i < len(similarity_scores) else 0.0
                
                all_results.append({
                    'qid': str(qid),
                    'docno': str(doc_row['docno']),
                    'original_score': float(doc_row['score']),
                    'similarity_score': float(similarity_score),
                    'final_score': float(similarity_score)  # Use similarity score only
                })
            
            processed_queries += 1
            if processed_queries % 10 == 0:
                logging.info(f"📊 Processed {processed_queries}/{len(queries)} queries")
        
        # Sort results by query and score
        results_df = pd.DataFrame(all_results)
        
        # Sort within each query by final score
        sorted_results = []
        for qid in results_df['qid'].unique():
            query_results = results_df[results_df['qid'] == qid].copy()
            query_results = query_results.sort_values('final_score', ascending=False)
            query_results['rank'] = range(1, len(query_results) + 1)
            sorted_results.append(query_results)
        
        final_results = pd.concat(sorted_results, ignore_index=True)
        
        stats = {
            'processed_queries': processed_queries,
            'total_documents': len(final_results),
            'avg_docs_per_query': len(final_results) / processed_queries if processed_queries > 0 else 0,
            'model_name': self.model_name
        }
        
        return final_results, stats

def save_results(results_df: pd.DataFrame, base_filename: str):
    """Save results in multiple formats"""
    
    # Create results directory
    results_dir = Path("fixed_colbert_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save CSV
    csv_path = results_dir / f"{base_filename}.csv"
    results_df.to_csv(csv_path, index=False)
    logging.info(f"💾 Saved CSV results: {csv_path}")
    
    # Save TREC format
    trec_path = results_dir / f"{base_filename}_run.txt"
    with open(trec_path, 'w') as f:
        for _, row in results_df.iterrows():
            f.write(f"{row['qid']} Q0 {row['docno']} {row['rank']} {row['final_score']:.6f} fixed_colbert\n")
    
    logging.info(f"💾 Saved TREC results: {trec_path}")
    
    return csv_path, trec_path

def evaluate_results(trec_file: str, qrel_file: str, output_file: str):
    """Evaluate results using pytrec_eval"""
    
    try:
        import pytrec_eval
        
        logging.info("📊 Starting evaluation...")
        
        # Load qrels
        qrels = {}
        with open(qrel_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docno, relevance = parts[:4]
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][docno] = int(relevance)
        
        # Load run
        run = {}
        with open(trec_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, docno, rank, score, _ = parts[:6]
                    if qid not in run:
                        run[qid] = {}
                    run[qid][docno] = float(score)
        
        # Evaluate
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'recip_rank', 'ndcg_cut.10'})
        results = evaluator.evaluate(run)
        
        # Aggregate results
        metrics = {}
        for measure in ['map', 'recip_rank', 'ndcg_cut_10']:
            scores = [query_measures[measure] for query_measures in results.values()]
            metrics[measure] = sum(scores) / len(scores) if scores else 0.0
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"✅ Evaluation completed:")
        logging.info(f"   MAP: {metrics['map']:.4f}")
        logging.info(f"   MRR: {metrics['recip_rank']:.4f}")
        logging.info(f"   NDCG@10: {metrics['ndcg_cut_10']:.4f}")
        
        return metrics
        
    except ImportError:
        logging.error("pytrec_eval not available, using trec_eval")
        return None

def main():
    """Main execution function"""
    
    logging.info("🚀 Starting Fixed ColBERT Reranker...")
    
    # Configuration
    queries_file = "../rewritten_queries/llama_train_rewritten_queries.jsonl"
    ltr_fusion_file = "../dense_run_files/run_files/stage2_fusion/llama_train_ltr_fusion.txt"
    index_path = "../trec-tot-2025-pyterrier-index"
    qrel_file = "../qrel/train-2025-qrel.txt"
    
    # Initialize reranker
    reranker = FunctionalColBERTReranker(
        model_name='all-MiniLM-L6-v2',
        index_path=index_path
    )
    
    # Rerank documents
    logging.info("🔄 Starting reranking process...")
    start_time = time.time()
    
    results_df, stats = reranker.rerank_documents(
        queries_file=queries_file,
        ltr_fusion_file=ltr_fusion_file,
        max_docs=100  # Limit to 100 docs per query for faster processing
    )
    
    rerank_time = time.time() - start_time
    
    # Save results
    base_filename = "fixed_llama_train_colbert_100"
    csv_path, trec_path = save_results(results_df, base_filename)
    
    # Evaluate results
    metrics_file = f"fixed_colbert_results/{base_filename}_metrics.json"
    metrics = evaluate_results(str(trec_path), qrel_file, metrics_file)
    
    # Print final summary
    logging.info("🎉 ColBERT Reranking Complete!")
    logging.info(f"⏱️  Processing time: {rerank_time:.2f} seconds")
    logging.info(f"📊 Processed {stats['processed_queries']} queries")
    logging.info(f"📄 Generated {stats['total_documents']} ranked documents")
    logging.info(f"📈 Average docs per query: {stats['avg_docs_per_query']:.1f}")
    
    if metrics:
        logging.info(f"🎯 Performance Metrics:")
        logging.info(f"   MAP: {metrics['map']:.4f}")
        logging.info(f"   MRR: {metrics['recip_rank']:.4f}")
        logging.info(f"   NDCG@10: {metrics['ndcg_cut_10']:.4f}")

if __name__ == "__main__":
    main()
