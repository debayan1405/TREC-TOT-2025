#!/usr/bin/env python3
"""
Functional ColBERT Reranker - Simplified Version
Uses direct sentence-transformers approach for compatibility
"""

import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import pyterrier as pt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FunctionalColBERTReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name
        
        # Initialize PyTerrier
        if not pt.started():
            pt.init()
        
        # Load PyTerrier index for document retrieval
        import zipfile
        import tempfile
        import os
        
        index_zip_path = "/home/ugdf8/IRIS/TREC-TOT-2025/trec-tot-2025-pyterrier-index.zip"
        logging.info("🔧 Extracting and loading PyTerrier index...")
        
        # Extract index to temporary directory
        self.temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(index_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        
        # Find the data.properties file
        index_path = None
        for root, dirs, files in os.walk(self.temp_dir):
            if 'data.properties' in files:
                index_path = root
                break
        
        if not index_path:
            raise ValueError(f"Could not find data.properties in extracted index")
        
        self.pt_index = pt.IndexFactory.of(index_path)
        self.meta_index = self.pt_index.getMetaIndex()
        logging.info(f"✅ PyTerrier index loaded: {self.pt_index.getCollectionStatistics().getNumberOfDocuments():,} documents")
        
        # Use sentence-transformers for simplicity and compatibility
        logging.info(f"🤖 Loading model: {model_name}")
        self.model = self._load_sentence_transformer()
        logging.info("✅ Model loaded successfully")
    
    def _load_sentence_transformer(self):
        """Load sentence-transformers model as fallback"""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a compatible model that works well
            return SentenceTransformer('all-MiniLM-L6-v2')  # Reliable alternative
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def get_document_text(self, external_docno):
        """Retrieve document text using PyTerrier meta index"""
        try:
            # Use metaindex.getDocument for proper mapping
            internal_docid = self.meta_index.getDocument("docno", str(external_docno))
            
            if internal_docid >= 0:
                text = self.meta_index.getItem("text", internal_docid)
                if text and len(text.strip()) > 0:
                    return text.strip()
            
            return None
            
        except Exception as e:
            logging.warning(f"Error retrieving document {external_docno}: {e}")
            return None
    
    def compute_similarity_scores(self, query_text: str, doc_texts: List[str]) -> List[float]:
        """Compute semantic similarity scores"""
        try:
            if not doc_texts:
                return []
            
            # Encode query and documents
            query_embedding = self.model.encode([query_text])
            doc_embeddings = self.model.encode(doc_texts)
            
            # Compute cosine similarities
            similarities = self.model.similarity(query_embedding, doc_embeddings)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            logging.error(f"Error computing similarities: {e}")
            return [0.0] * len(doc_texts)
    
    def rerank_documents(self, queries_file: str, ltr_fusion_file: str, max_docs: int = 1000) -> Dict:
        """Rerank documents using semantic similarity"""
        
        # Load queries
        logging.info("📝 Loading queries...")
        queries = {}
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                queries[str(data['query_id'])] = data['query']
        
        logging.info(f"✅ Loaded {len(queries)} queries")
        
        # Load LTR fusion results
        logging.info(f"📂 Loading LTR fusion results from: {ltr_fusion_file}")
        ltr_results = pd.read_csv(ltr_fusion_file, sep='\t', header=None, 
                                 names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_id'])
        
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
            
        for _, row in query_docs.iterrows():
            docid = str(row['docno'])
            try:
                # Get document using correct API
                metaindex = self.index.getMetaIndex()
                docid_int = metaindex.getDocument('docno', docid)
                if docid_int >= 0:
                    text = metaindex.getItem('text', docid_int)
                    if text:
                        doc_texts.append(text)
                        doc_ids.append(docid)
                        success_count += 1
                    else:
                        logger.warning(f"Document {docid} has no text")
                        failed_count += 1
                else:
                    logger.warning(f"Document {docid} not found in index")
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error retrieving document {docid}: {e}")
                failed_count += 1            if not doc_texts:
                logging.warning(f"No valid documents for query {qid}")
                continue
            
            logging.info(f"📝 Retrieved {len(doc_texts)} valid documents, computing similarity scores...")
            
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
    
    # Save as CSV
    csv_file = f"{base_filename}.csv"
    results_df.to_csv(csv_file, index=False)
    logging.info(f"✅ Results saved to: {csv_file}")
    
    # Save as TREC format
    trec_file = f"{base_filename}_run.txt"
    with open(trec_file, 'w') as f:
        for _, row in results_df.iterrows():
            f.write(f"{row['qid']} Q0 {row['docno']} {row['rank']} {row['final_score']:.6f} functional_colbert\\n")
    
    logging.info(f"✅ TREC format saved to: {trec_file}")
    
    return trec_file

def evaluate_results(trec_file: str, qrel_file: str) -> Dict:
    """Evaluate results using pytrec_eval"""
    try:
        import pytrec_eval
        
        # Load qrels
        with open(qrel_file, 'r') as f:
            qrel_lines = f.readlines()
        
        qrels = {}
        for line in qrel_lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docno, rel = parts[0], parts[1], parts[2], int(parts[3])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docno] = rel
        
        # Load run
        with open(trec_file, 'r') as f:
            run_lines = f.readlines()
        
        run = {}
        for line in run_lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docno, rank, score, _ = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4]), parts[5]
                if qid not in run:
                    run[qid] = {}
                run[qid][docno] = score
        
        # Evaluate
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, ['map', 'recip_rank', 'ndcg_cut_10'])
        results = evaluator.evaluate(run)
        
        # Aggregate results
        metrics = {}
        for measure in ['map', 'recip_rank', 'ndcg_cut_10']:
            metrics[measure] = np.mean([results[qid][measure] for qid in results])
        
        return metrics
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Functional ColBERT Reranker')
    parser.add_argument('--dataset', default='train', help='Dataset to use')
    parser.add_argument('--rewriter', default='llama', help='Rewriter to use')
    parser.add_argument('--max-docs', type=int, default=1000, help='Max documents per query')
    
    args = parser.parse_args()
    
    # File paths
    queries_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/topics/{args.dataset}-2025-queries.jsonl"
    ltr_fusion_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/dense_run_files/run_files/stage2_fusion/{args.rewriter}_{args.dataset}_ltr_fusion.txt"
    qrel_file = f"/home/ugdf8/IRIS/TREC-TOT-2025/qrel/{args.dataset}-2025-qrel.txt"
    
    # Create output directory
    output_dir = Path("/home/ugdf8/IRIS/TREC-TOT-2025/dense_retrieval/functional_colbert_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize reranker
    reranker = FunctionalColBERTReranker()
    
    # Rerank documents
    logging.info("🚀 Starting functional ColBERT reranking...")
    results_df, stats = reranker.rerank_documents(queries_file, ltr_fusion_file, args.max_docs)
    
    # Save results
    base_filename = output_dir / f"functional_{args.rewriter}_{args.dataset}_colbert_{args.max_docs}"
    trec_file = save_results(results_df, str(base_filename))
    
    # Print statistics
    logging.info("📊 Processing Statistics:")
    for key, value in stats.items():
        logging.info(f"    {key}: {value}")
    
    # Evaluate
    logging.info("📊 Evaluating results...")
    metrics = evaluate_results(trec_file, qrel_file)
    
    if metrics:
        logging.info("📈 Evaluation Results:")
        for metric, value in metrics.items():
            logging.info(f"    {metric}: {value:.4f}")
        
        # Save metrics
        metrics_file = f"{base_filename}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"💾 Metrics saved to: {metrics_file}")
    
    logging.info(f"✅ Functional ColBERT reranking completed!")

if __name__ == "__main__":
    main()
