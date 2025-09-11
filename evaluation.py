"""
Simple evaluation script that works without PyTerrier
Manually implements basic evaluation metrics for the TREC-ToT pipeline
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import time
import math
from collections import defaultdict

def load_qrels(qrels_file):
    """Load QRELs file manually"""
    qrels = {}
    
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                docno = parts[2]  
                rel = int(parts[3])
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docno] = rel
    
    return qrels

def load_run_results(run_file):
    """Load run results from TREC format file"""
    try:
        # Read as string first to handle potential issues
        df = pd.read_csv(
            run_file,
            sep='\\s+',
            header=None,
            names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name'],
            dtype=str
        )
        
        # Convert types carefully
        df['qid'] = df['qid'].astype(str)
        df['docno'] = df['docno'].astype(str)
        
        # Handle rank column - replace any non-numeric values
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        df = df.dropna(subset=['rank'])  # Drop rows with invalid ranks
        df['rank'] = df['rank'].astype(int)
        
        # Handle score column
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df = df.dropna(subset=['score'])  # Drop rows with invalid scores
        df['score'] = df['score'].astype(float)
        
        return df
    except Exception as e:
        print(f"Error loading {run_file}: {e}")
        return None

def calculate_metrics(run_df, qrels, metrics=['ndcg_cut_10', 'recall_1000', 'P_10']):
    """Calculate evaluation metrics manually"""
    
    results = {}
    all_qids = set(run_df['qid'].unique()) & set(qrels.keys())
    
    if not all_qids:
        return {'error': 'No matching QIDs between run and qrels'}
    
    # Calculate per-query metrics
    query_metrics = defaultdict(list)
    
    for qid in all_qids:
        query_run = run_df[run_df['qid'] == qid].sort_values('rank')
        query_qrels = qrels.get(qid, {})
        
        if not query_qrels:
            continue
        
        # Get retrieved documents and their relevance
        retrieved_docs = query_run['docno'].tolist()
        relevance_scores = [query_qrels.get(doc, 0) for doc in retrieved_docs]
        
        # Calculate NDCG@10
        if 'ndcg_cut_10' in metrics:
            ndcg_10 = calculate_ndcg(relevance_scores[:10])
            query_metrics['ndcg_cut_10'].append(ndcg_10)
        
        # Calculate Recall@1000
        if 'recall_1000' in metrics:
            relevant_docs = [doc for doc, rel in query_qrels.items() if rel > 0]
            retrieved_relevant = [doc for doc in retrieved_docs[:1000] if doc in relevant_docs]
            recall_1000 = len(retrieved_relevant) / len(relevant_docs) if relevant_docs else 0.0
            query_metrics['recall_1000'].append(recall_1000)
        
        # Calculate P@10
        if 'P_10' in metrics:
            relevant_in_top10 = sum(1 for doc in retrieved_docs[:10] if query_qrels.get(doc, 0) > 0)
            p_10 = relevant_in_top10 / min(10, len(retrieved_docs))
            query_metrics['P_10'].append(p_10)
    
    # Calculate mean metrics
    for metric in metrics:
        if metric in query_metrics and query_metrics[metric]:
            results[metric] = sum(query_metrics[metric]) / len(query_metrics[metric])
        else:
            results[metric] = 0.0
    
    return results

def calculate_ndcg(relevance_scores):
    """Calculate NDCG for a list of relevance scores"""
    
    def dcg(scores):
        return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(scores))
    
    if not relevance_scores:
        return 0.0
    
    # Calculate DCG
    dcg_value = dcg(relevance_scores)
    
    # Calculate IDCG (ideal DCG)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg_value = dcg(ideal_scores)
    
    # Calculate NDCG
    if idcg_value == 0:
        return 0.0
    else:
        return dcg_value / idcg_value

def collect_all_results(dataset_version):
    """Collect all available results for a dataset"""
    
    print(f"\\n📊 Collecting results for {dataset_version} dataset:")
    
    results = {
        'sparse': {},
        'fusion': {},
        'dense_cross': {}
    }
    
    # Sparse retrieval results
    sparse_dir = Path(f"sparse_run_files/{dataset_version}")
    if sparse_dir.exists():
        sparse_files = list(sparse_dir.glob("*.txt"))
        print(f"   Found {len(sparse_files)} sparse retrieval files")
        
        for file in sparse_files:
            # Extract method and rewriter from filename
            parts = file.stem.split('_')
            if len(parts) >= 3:
                rewriter = parts[0]
                method = parts[-1]
                key = f"{rewriter}_{method}"
                results['sparse'][key] = str(file)
    
    # Fusion results
    fusion_dir = Path(f"fused_run_files/{dataset_version}")
    if fusion_dir.exists():
        fusion_files = list(fusion_dir.glob("*_fused.txt"))
        print(f"   Found {len(fusion_files)} fusion files")
        
        for file in fusion_files:
            rewriter = file.stem.replace(f"_{dataset_version}_fused", "")
            results['fusion'][rewriter] = str(file)
    
    # Dense reranking results
    dense_dir = Path(f"dense_run_files/{dataset_version}")
    if dense_dir.exists():
        cross_files = list(dense_dir.glob("*_dense_cross-encoder.txt"))
        
        print(f"   Found {len(cross_files)} cross-encoder files")
        
        for file in cross_files:
            rewriter = file.stem.replace(f"_{dataset_version}_dense_cross-encoder", "")
            results['dense_cross'][rewriter] = str(file)
    
    return results

def evaluate_all_results(dataset_version, results):
    """Evaluate all collected results"""
    
    print(f"\\n🎯 Evaluating all results for {dataset_version}:")
    
    # Load QRELs
    qrels_file = f"qrel/{dataset_version}-2025-qrel.txt"
    
    if not Path(qrels_file).exists():
        print(f"   ⚠️  No QRELs available for {dataset_version}")
        return {}
    
    qrels = load_qrels(qrels_file)
    print(f"   Loaded {sum(len(q) for q in qrels.values())} relevance judgments")
    
    evaluations = {}
    
    for approach, files in results.items():
        print(f"\\n   {approach.upper()} Results:")
        evaluations[approach] = {}
        
        for name, file_path in files.items():
            print(f"      Evaluating {name}...")
            
            run_df = load_run_results(file_path)
            if run_df is not None:
                eval_result = calculate_metrics(run_df, qrels)
                evaluations[approach][name] = eval_result
                
                if 'error' not in eval_result:
                    # Print key metrics
                    ndcg_10 = eval_result.get('ndcg_cut_10', 'N/A')
                    recall_1000 = eval_result.get('recall_1000', 'N/A')
                    p_10 = eval_result.get('P_10', 'N/A')
                    print(f"         NDCG@10: {ndcg_10:.4f}" if isinstance(ndcg_10, float) else f"         NDCG@10: {ndcg_10}")
                    print(f"         Recall@1000: {recall_1000:.4f}" if isinstance(recall_1000, float) else f"         Recall@1000: {recall_1000}")
                    print(f"         P@10: {p_10:.4f}" if isinstance(p_10, float) else f"         P@10: {p_10}")
                else:
                    print(f"         Error: {eval_result['error']}")
            else:
                evaluations[approach][name] = {'error': 'Failed to load run file'}
    
    return evaluations

def create_comparison_table(evaluations):
    """Create a comparison table of all results"""
    
    print(f"\\n📋 COMPREHENSIVE RESULTS COMPARISON:")
    print("=" * 80)
    
    # Collect all metrics
    all_metrics = set()
    for approach_results in evaluations.values():
        for result in approach_results.values():
            if isinstance(result, dict) and 'error' not in result:
                all_metrics.update(result.keys())
    
    if not all_metrics:
        print("No valid evaluation results found")
        return None
    
    # Focus on key metrics
    key_metrics = ['ndcg_cut_10', 'recall_1000', 'P_10']
    available_metrics = [m for m in key_metrics if m in all_metrics]
    
    if not available_metrics:
        print("No key metrics available")
        return None
    
    # Create comparison data
    comparison_data = []
    
    for approach, approach_results in evaluations.items():
        for name, result in approach_results.items():
            if isinstance(result, dict) and 'error' not in result:
                row = {
                    'Approach': approach,
                    'Configuration': name,
                }
                
                for metric in available_metrics:
                    value = result.get(metric, 0.0)
                    row[metric] = f"{value:.4f}" if isinstance(value, float) else str(value)
                
                comparison_data.append(row)
    
    # Create DataFrame and display
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Sort by approach and configuration
        approach_order = ['sparse', 'fusion', 'dense_cross']
        df['approach_order'] = df['Approach'].map({a: i for i, a in enumerate(approach_order)})
        df = df.sort_values(['approach_order', 'Configuration']).drop('approach_order', axis=1)
        
        print(df.to_string(index=False))
        
        return df
    else:
        print("No valid results to compare")
        return None

def generate_summary_analysis(evaluations, dataset_version):
    """Generate summary analysis and insights"""
    
    print(f"\\n🔍 ANALYSIS SUMMARY for {dataset_version}:")
    print("=" * 50)
    
    # Find best performing configurations
    best_results = {}
    
    for approach, approach_results in evaluations.items():
        approach_best = {}
        
        for name, result in approach_results.items():
            if isinstance(result, dict) and 'error' not in result:
                for metric in ['ndcg_cut_10', 'recall_1000', 'P_10']:
                    if metric in result:
                        if metric not in approach_best or result[metric] > approach_best[metric]['score']:
                            approach_best[metric] = {
                                'config': name,
                                'score': result[metric]
                            }
        
        if approach_best:
            best_results[approach] = approach_best
    
    # Print best results per approach
    for approach, metrics in best_results.items():
        print(f"\\n   {approach.upper()} - Best Results:")
        for metric, info in metrics.items():
            print(f"      {metric}: {info['score']:.4f} ({info['config']})")
    
    # Overall best
    print(f"\\n   OVERALL BEST RESULTS:")
    for metric in ['ndcg_cut_10', 'recall_1000', 'P_10']:
        overall_best = None
        
        for approach, metrics in best_results.items():
            if metric in metrics:
                score = metrics[metric]['score']
                if overall_best is None or score > overall_best['score']:
                    overall_best = {
                        'approach': approach,
                        'config': metrics[metric]['config'],
                        'score': score
                    }
        
        if overall_best:
            print(f"      {metric}: {overall_best['score']:.4f} ({overall_best['approach']} - {overall_best['config']})")

def save_comprehensive_evaluation(dataset_version, results, evaluations, comparison_df):
    """Save comprehensive evaluation results"""
    
    output_data = {
        'dataset': dataset_version,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'file_inventory': results,
        'evaluation_results': evaluations,
        'comparison_table': comparison_df.to_dict() if comparison_df is not None else None
    }
    
    output_file = f"simple_comprehensive_evaluation_{dataset_version}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\\n💾 Comprehensive evaluation saved to: {output_file}")

def main():
    """Main evaluation function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Comprehensive Evaluation Pipeline")
    parser.add_argument("--dataset", default="train", 
                       choices=["train", "dev-1", "dev-2", "dev-3", "test"],
                       help="Dataset to evaluate")
    
    args = parser.parse_args()
    
    print("🎯 SIMPLE COMPREHENSIVE EVALUATION PIPELINE")
    print("=" * 60)
    
    # Collect all results
    results = collect_all_results(args.dataset)
    
    # Evaluate all results
    evaluations = evaluate_all_results(args.dataset, results)
    
    # Create comparison table
    comparison_df = create_comparison_table(evaluations)
    
    # Generate analysis
    generate_summary_analysis(evaluations, args.dataset)
    
    # Save comprehensive results
    save_comprehensive_evaluation(args.dataset, results, evaluations, comparison_df)
    
    print(f"\\n✅ SIMPLE COMPREHENSIVE EVALUATION COMPLETED!")
    print(f"   Dataset: {args.dataset}")

if __name__ == "__main__":
    main()
