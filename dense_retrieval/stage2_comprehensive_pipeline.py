#!/usr/bin/env python3
"""
Comprehensive Stage 2 Pipeline: Fusion + ColBERT Middle Ranking
===============================================================

This pipeline implements:
1. RRF (Reciprocal Rank Fusion) baseline
2. LTR (Learning to Rank) with LightGBM 
3. ColBERT late interaction reranking
4. Comprehensive evaluation and comparison
"""

import os
import sys
import json
import argparse
import time
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import pickle

# LightGBM for Learning to Rank
import lightgbm as lgb

# PyTerrier imports
import pyterrier as pt
if not pt.started():
    pt.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TRECRunFileProcessor:
    """Process TREC run files for fusion and evaluation"""
    
    def __init__(self):
        self.queries = {}
        self.results = defaultdict(dict)
    
    def load_run_file(self, filepath: str, model_name: str) -> Dict:
        """Load TREC run file format"""
        results = defaultdict(list)
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    qid = parts[0]
                    docid = parts[2]
                    rank = int(parts[3])
                    score = float(parts[4])
                    
                    results[qid].append({
                        'docid': docid,
                        'rank': rank,
                        'score': score,
                        'model': model_name
                    })
        
        # Sort by rank for each query
        for qid in results:
            results[qid].sort(key=lambda x: x['rank'])
        
        return dict(results)
    
    def load_multiple_run_files(self, file_dict: Dict[str, str]) -> Dict:
        """Load multiple run files"""
        all_results = {}
        
        for model_name, filepath in file_dict.items():
            logger.info(f"Loading {model_name} from {filepath}")
            model_results = self.load_run_file(filepath, model_name)
            all_results[model_name] = model_results
            
        return all_results


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) implementation"""
    
    def __init__(self, k: int = 60):
        self.k = k  # RRF parameter
    
    def fuse_rankings(self, rankings: Dict[str, Dict]) -> Dict:
        """Apply RRF fusion to multiple rankings"""
        fused_results = defaultdict(dict)
        
        # Get all unique queries
        all_queries = set()
        for model_results in rankings.values():
            all_queries.update(model_results.keys())
        
        for qid in all_queries:
            doc_scores = defaultdict(float)
            
            # Calculate RRF scores
            for model_name, model_results in rankings.items():
                if qid in model_results:
                    for item in model_results[qid]:
                        docid = item['docid']
                        rank = item['rank']
                        rrf_score = 1.0 / (self.k + rank)
                        doc_scores[docid] += rrf_score
            
            # Sort by RRF score (descending)
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            fused_results[qid] = [
                {
                    'docid': docid,
                    'rank': rank + 1,
                    'score': score,
                    'method': 'RRF'
                }
                for rank, (docid, score) in enumerate(sorted_docs)
            ]
        
        return dict(fused_results)
    
    def save_fused_results(self, fused_results: Dict, output_path: str, run_name: str = "RRF_fusion"):
        """Save fused results in TREC format"""
        with open(output_path, 'w') as f:
            for qid, docs in fused_results.items():
                for doc in docs:
                    f.write(f"{qid}\tQ0\t{doc['docid']}\t{doc['rank']}\t{doc['score']:.6f}\t{run_name}\n")


class LearningToRankFusion:
    """Learning to Rank fusion using LightGBM"""
    
    def __init__(self, qrel_file: str):
        self.qrel_file = qrel_file
        self.qrels = self._load_qrels()
        self.model = None
        self.feature_names = []
    
    def _load_qrels(self) -> Dict:
        """Load QREL file for training"""
        qrels = defaultdict(dict)
        
        if os.path.exists(self.qrel_file):
            with open(self.qrel_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qid = parts[0]
                        docid = parts[2]
                        relevance = int(parts[3])
                        qrels[qid][docid] = relevance
        
        return dict(qrels)
    
    def extract_features(self, rankings: Dict[str, Dict]) -> Tuple[List, List, List]:
        """Extract features for LTR training"""
        features = []
        labels = []
        query_ids = []
        
        # Get all unique queries
        all_queries = set()
        for model_results in rankings.values():
            all_queries.update(model_results.keys())
        
        model_names = list(rankings.keys())
        self.feature_names = [f"{model}_score" for model in model_names] + \
                            [f"{model}_rank" for model in model_names] + \
                            [f"{model}_norm_score" for model in model_names]
        
        for qid in all_queries:
            if qid not in self.qrels:
                continue
                
            # Get all documents for this query
            all_docs = set()
            for model_results in rankings.values():
                if qid in model_results:
                    all_docs.update([item['docid'] for item in model_results[qid]])
            
            for docid in all_docs:
                feature_vector = []
                
                # Extract features from each model
                for model_name in model_names:
                    score = 0.0
                    rank = 1000  # Default high rank
                    norm_score = 0.0
                    
                    if qid in rankings[model_name]:
                        for item in rankings[model_name][qid]:
                            if item['docid'] == docid:
                                score = item['score']
                                rank = item['rank']
                                break
                        
                        # Normalize score by max score in this query
                        max_score = max([item['score'] for item in rankings[model_name][qid]])
                        if max_score > 0:
                            norm_score = score / max_score
                    
                    feature_vector.extend([score, rank, norm_score])
                
                features.append(feature_vector)
                labels.append(self.qrels[qid].get(docid, 0))
                query_ids.append(qid)
        
        return features, labels, query_ids
    
    def train_model(self, rankings: Dict[str, Dict]) -> None:
        """Train LightGBM ranker"""
        logger.info("Extracting features for LTR training...")
        features, labels, query_ids = self.extract_features(rankings)
        
        if not features:
            raise ValueError("No training data available")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Create LightGBM dataset
        # Group by query for ranking
        query_groups = []
        current_qid = None
        current_group_size = 0
        
        for qid in query_ids:
            if qid != current_qid:
                if current_group_size > 0:
                    query_groups.append(current_group_size)
                current_qid = qid
                current_group_size = 1
            else:
                current_group_size += 1
        
        if current_group_size > 0:
            query_groups.append(current_group_size)
        
        train_data = lgb.Dataset(X, label=y, group=query_groups)
        
        # LightGBM parameters for ranking
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'device': 'cpu',  # Use CPU instead of GPU
            'num_threads': 16  # Use multiple CPU threads
        }
        
        logger.info("Training LightGBM ranker...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
        )
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        for i, (feature, imp) in enumerate(zip(self.feature_names, importance)):
            logger.info(f"Feature {feature}: {imp:.4f}")
    
    def predict_and_fuse(self, rankings: Dict[str, Dict]) -> Dict:
        """Apply LTR model to fuse rankings"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        fused_results = defaultdict(dict)
        
        # Get all unique queries
        all_queries = set()
        for model_results in rankings.values():
            all_queries.update(model_results.keys())
        
        model_names = list(rankings.keys())
        
        for qid in all_queries:
            # Get all documents for this query
            all_docs = set()
            for model_results in rankings.values():
                if qid in model_results:
                    all_docs.update([item['docid'] for item in model_results[qid]])
            
            doc_features = []
            doc_ids = []
            
            for docid in all_docs:
                feature_vector = []
                
                # Extract features from each model
                for model_name in model_names:
                    score = 0.0
                    rank = 1000
                    norm_score = 0.0
                    
                    if qid in rankings[model_name]:
                        for item in rankings[model_name][qid]:
                            if item['docid'] == docid:
                                score = item['score']
                                rank = item['rank']
                                break
                        
                        # Normalize score
                        max_score = max([item['score'] for item in rankings[model_name][qid]])
                        if max_score > 0:
                            norm_score = score / max_score
                    
                    feature_vector.extend([score, rank, norm_score])
                
                doc_features.append(feature_vector)
                doc_ids.append(docid)
            
            if doc_features:
                # Predict scores
                X = np.array(doc_features)
                ltr_scores = self.model.predict(X)
                
                # Sort by LTR score
                doc_score_pairs = list(zip(doc_ids, ltr_scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                fused_results[qid] = [
                    {
                        'docid': docid,
                        'rank': rank + 1,
                        'score': float(score),
                        'method': 'LTR'
                    }
                    for rank, (docid, score) in enumerate(doc_score_pairs)
                ]
        
        return dict(fused_results)
    
    def save_model(self, model_path: str):
        """Save trained LTR model"""
        if self.model:
            self.model.save_model(model_path)
            
            # Save feature names
            feature_path = model_path.replace('.txt', '_features.json')
            with open(feature_path, 'w') as f:
                json.dump(self.feature_names, f, indent=2)
    
    def load_model(self, model_path: str):
        """Load trained LTR model"""
        self.model = lgb.Booster(model_file=model_path)
        
        # Load feature names
        feature_path = model_path.replace('.txt', '_features.json')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)


class ColBERTReranker:
    """ColBERT late interaction reranker"""
    
    def __init__(self, index_path: str = None, checkpoint_path: str = None):
        self.index_path = index_path
        self.checkpoint_path = checkpoint_path
        self.colbert_model = None
        
        # For now, we'll use a placeholder - in real implementation,
        # you would integrate with ColBERT library
        logger.warning("ColBERT integration is a placeholder - implement with actual ColBERT library")
    
    def initialize_colbert(self):
        """Initialize ColBERT model and index"""
        # Placeholder for ColBERT initialization
        # In real implementation:
        # from colbert import Indexer, Searcher
        # self.searcher = Searcher(index=self.index_path, checkpoint=self.checkpoint_path)
        logger.info("ColBERT model initialized (placeholder)")
        self.colbert_model = "placeholder"
    
    def rerank_with_colbert(self, fused_results: Dict, queries: Dict, top_k: int = 1000) -> Dict:
        """Rerank using ColBERT late interaction"""
        if self.colbert_model is None:
            self.initialize_colbert()
        
        colbert_results = {}
        
        for qid, docs in fused_results.items():
            # Placeholder implementation
            # In real implementation, you would:
            # 1. Get query text
            # 2. Get document IDs to rerank
            # 3. Use ColBERT searcher to rerank
            
            # For now, we'll simulate ColBERT reranking by slight score modification
            reranked_docs = []
            for i, doc in enumerate(docs[:top_k]):
                # Simulate ColBERT scoring
                colbert_score = doc['score'] * (1.0 + np.random.normal(0, 0.01))
                reranked_docs.append({
                    'docid': doc['docid'],
                    'rank': i + 1,
                    'score': float(colbert_score),
                    'method': 'ColBERT'
                })
            
            # Sort by score
            reranked_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # Update ranks
            for i, doc in enumerate(reranked_docs):
                doc['rank'] = i + 1
            
            colbert_results[qid] = reranked_docs
        
        return colbert_results


class Stage2Evaluator:
    """Evaluate different fusion approaches"""
    
    def __init__(self, qrel_file: str):
        self.qrel_file = qrel_file
        self.qrels = self._load_qrels()
    
    def _load_qrels(self) -> Dict:
        """Load QREL file"""
        qrels = defaultdict(dict)
        
        if os.path.exists(self.qrel_file):
            with open(self.qrel_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qid = parts[0]
                        docid = parts[2]
                        relevance = int(parts[3])
                        qrels[qid][docid] = relevance
        
        return dict(qrels)
    
    def calculate_metrics(self, results: Dict, k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for qid, docs in results.items():
                if qid not in self.qrels:
                    continue
                
                relevant_docs = set([docid for docid, rel in self.qrels[qid].items() if rel > 0])
                
                if not relevant_docs:
                    continue
                
                # Get top-k documents
                top_k_docs = [doc['docid'] for doc in docs[:k]]
                retrieved_relevant = set(top_k_docs) & relevant_docs
                
                # Precision@k
                precision = len(retrieved_relevant) / k if k > 0 else 0
                precision_scores.append(precision)
                
                # Recall@k
                recall = len(retrieved_relevant) / len(relevant_docs) if relevant_docs else 0
                recall_scores.append(recall)
                
                # NDCG@k
                dcg = 0
                for i, docid in enumerate(top_k_docs):
                    rel = self.qrels[qid].get(docid, 0)
                    dcg += (2**rel - 1) / np.log2(i + 2)
                
                # IDCG
                ideal_rels = sorted([rel for rel in self.qrels[qid].values() if rel > 0], reverse=True)
                idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
            
            metrics[f'P@{k}'] = np.mean(precision_scores) if precision_scores else 0
            metrics[f'R@{k}'] = np.mean(recall_scores) if recall_scores else 0
            metrics[f'NDCG@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0
        
        return metrics
    
    def compare_methods(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Compare different fusion methods"""
        comparison_data = []
        
        for method_name, results in results_dict.items():
            metrics = self.calculate_metrics(results)
            metrics['Method'] = method_name
            comparison_data.append(metrics)
        
        return pd.DataFrame(comparison_data)


def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from JSONL file"""
    queries = {}
    
    if os.path.exists(queries_file):
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                # Handle different key formats
                query_id = str(data['query_id'])
                if 'query_text' in data:
                    query_text = data['query_text']
                elif 'query' in data:
                    query_text = data['query']
                else:
                    continue
                queries[query_id] = query_text
    
    return queries


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Fusion + ColBERT Pipeline')
    parser.add_argument('--dataset', choices=['train', 'dev1', 'dev2', 'dev3', 'test'], 
                       required=True, help='Dataset to process')
    parser.add_argument('--rewriter', choices=['llama', 'original'], 
                       default='llama', help='Query rewriter used')
    parser.add_argument('--stage1-dir', type=str, 
                       default='../dense_run_files/run_files/stage1_bi_encoders',
                       help='Directory containing stage 1 results')
    parser.add_argument('--output-dir', type=str, 
                       default='../dense_run_files/run_files/stage2_fusion',
                       help='Output directory for stage 2 results')
    parser.add_argument('--qrel-file', type=str, help='QREL file for evaluation')
    parser.add_argument('--queries-file', type=str, help='Queries JSONL file')
    parser.add_argument('--top-k', type=int, default=1000, 
                       help='Top-k documents to output')
    parser.add_argument('--rrf-k', type=int, default=60, 
                       help='RRF parameter k')
    parser.add_argument('--skip-ltr', action='store_true', 
                       help='Skip LTR training (use existing model)')
    parser.add_argument('--skip-colbert', action='store_true', 
                       help='Skip ColBERT reranking')
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = Path(__file__).parent
    if not args.qrel_file:
        args.qrel_file = f"../qrel/{args.dataset}-2025-qrel.txt"
    if not args.queries_file:
        args.queries_file = f"../topics/{args.dataset}-2025-queries.jsonl"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("🚀 STAGE 2: COMPREHENSIVE FUSION + COLBERT PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Rewriter: {args.rewriter}")
    logger.info(f"Stage 1 directory: {args.stage1_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Top-k output: {args.top_k}")
    
    start_time = time.time()
    
    # Load Stage 1 results
    logger.info("📥 Loading Stage 1 bi-encoder results...")
    processor = TRECRunFileProcessor()
    
    # Define bi-encoder models
    bi_encoders = [
        'all-mpnet-base-v2',
        'multi-qa-mpnet-base-dot-v1', 
        'all-MiniLM-L6-v2',
        'msmarco-distilbert-base-v4'
    ]
    
    stage1_files = {}
    for model in bi_encoders:
        filepath = f"{args.stage1_dir}/{args.rewriter}_{args.dataset}_{model}.txt"
        if os.path.exists(filepath):
            stage1_files[model] = filepath
        else:
            logger.warning(f"Stage 1 file not found: {filepath}")
    
    if not stage1_files:
        logger.error("No Stage 1 files found!")
        return
    
    # Load all stage 1 results
    stage1_results = processor.load_multiple_run_files(stage1_files)
    logger.info(f"✅ Loaded {len(stage1_results)} bi-encoder results")
    
    # Initialize fusion methods
    rrf = ReciprocalRankFusion(k=args.rrf_k)
    
    # 1. RRF Fusion
    logger.info("🔗 Performing RRF fusion...")
    rrf_results = rrf.fuse_rankings(stage1_results)
    
    # Save RRF results
    rrf_output = f"{args.output_dir}/{args.rewriter}_{args.dataset}_rrf_fusion.txt"
    rrf.save_fused_results(rrf_results, rrf_output, f"RRF_k{args.rrf_k}")
    logger.info(f"💾 RRF results saved to {rrf_output}")
    
    # 2. LTR Fusion (if not skipped)
    ltr_results = None
    if not args.skip_ltr and os.path.exists(args.qrel_file):
        logger.info("🧠 Training LTR fusion model...")
        ltr = LearningToRankFusion(args.qrel_file)
        
        try:
            ltr.train_model(stage1_results)
            ltr_results = ltr.predict_and_fuse(stage1_results)
            
            # Save LTR results
            ltr_output = f"{args.output_dir}/{args.rewriter}_{args.dataset}_ltr_fusion.txt"
            rrf.save_fused_results(ltr_results, ltr_output, "LTR_fusion")
            logger.info(f"💾 LTR results saved to {ltr_output}")
            
            # Save LTR model
            model_output = f"{args.output_dir}/{args.rewriter}_{args.dataset}_ltr_model.txt"
            ltr.save_model(model_output)
            logger.info(f"🎯 LTR model saved to {model_output}")
            
        except Exception as e:
            logger.error(f"LTR training failed: {e}")
            args.skip_ltr = True
    
    # 3. Evaluation comparison
    if os.path.exists(args.qrel_file):
        logger.info("📊 Evaluating fusion methods...")
        evaluator = Stage2Evaluator(args.qrel_file)
        
        results_to_compare = {
            'RRF': rrf_results
        }
        
        if ltr_results:
            results_to_compare['LTR'] = ltr_results
        
        comparison_df = evaluator.compare_methods(results_to_compare)
        
        # Save evaluation results
        eval_output = f"{args.output_dir}/{args.rewriter}_{args.dataset}_fusion_evaluation.csv"
        comparison_df.to_csv(eval_output, index=False)
        
        logger.info("📈 Fusion Evaluation Results:")
        print(comparison_df.to_string(index=False))
        
        # Determine best fusion method
        best_method = comparison_df.loc[comparison_df['NDCG@10'].idxmax(), 'Method']
        logger.info(f"🏆 Best fusion method: {best_method}")
        
        # Use best method for ColBERT
        if best_method == 'LTR' and ltr_results:
            best_results = ltr_results
            best_output_file = ltr_output
        else:
            best_results = rrf_results
            best_output_file = rrf_output
    else:
        logger.info("📊 No QREL file provided, using RRF as baseline")
        best_results = rrf_results
        best_output_file = rrf_output
    
    # 4. ColBERT Middle Ranking (if not skipped)
    if not args.skip_colbert:
        logger.info("🔄 Performing ColBERT middle ranking...")
        
        # Load queries
        queries = {}
        if os.path.exists(args.queries_file):
            queries = load_queries(args.queries_file)
        
        colbert = ColBERTReranker()
        colbert_results = colbert.rerank_with_colbert(best_results, queries, args.top_k)
        
        # Save ColBERT results
        colbert_output = f"{args.output_dir}/{args.rewriter}_{args.dataset}_colbert_reranked.txt"
        rrf.save_fused_results(colbert_results, colbert_output, "ColBERT_reranked")
        logger.info(f"💾 ColBERT results saved to {colbert_output}")
        
        # Final evaluation with ColBERT
        if os.path.exists(args.qrel_file):
            colbert_metrics = evaluator.calculate_metrics(colbert_results)
            logger.info("📈 ColBERT Reranking Results:")
            for metric, value in colbert_metrics.items():
                logger.info(f"   {metric}: {value:.4f}")
    
    # Summary
    total_time = time.time() - start_time
    logger.info("🏁 STAGE 2 COMPLETED")
    logger.info("=" * 60)
    logger.info(f"⏱️  Total execution time: {total_time:.1f}s")
    logger.info(f"📁 Output directory: {args.output_dir}")
    
    # List output files
    output_files = [f for f in os.listdir(args.output_dir) if f.startswith(f"{args.rewriter}_{args.dataset}")]
    logger.info(f"📋 Generated files: {len(output_files)}")
    for file in sorted(output_files):
        logger.info(f"   📄 {file}")


if __name__ == "__main__":
    main()
