#!/usr/bin/env python3
"""
Evaluation script to compare Fixed ColBERT performance with LTR baseline
"""

import os
import json
import pandas as pd
import pytrec_eval

# Configuration
BASE_DIR = "/home/ugdf8/IRIS/TREC-TOT-2025"
QREL_FILE = f"{BASE_DIR}/qrel/train-2025-qrel.txt"
LTR_RESULTS_FILE = f"{BASE_DIR}/dense_run_files/run_files/stage2_fusion/llama_train_ltr_fusion.txt"
COLBERT_RESULTS_FILE = f"{BASE_DIR}/dense_retrieval/fixed_colbert_results/fixed_llama_train_colbert_100_run.txt"

def load_qrel(file_path):
    """Load QREL file into pytrec_eval format"""
    qrel = {}
    with open(file_path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][docid] = int(rel)
    return qrel

def load_run_file(file_path):
    """Load TREC run file into pytrec_eval format"""
    run = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts[:6]
                if qid not in run:
                    run[qid] = {}
                run[qid][docid] = float(score)
    return run

def evaluate_run(qrel, run, run_name):
    """Evaluate a run file against qrel"""
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'recip_rank', 'ndcg_cut_10'})
    metrics = evaluator.evaluate(run)
    
    # Aggregate metrics
    map_score = sum([query_metrics['map'] for query_metrics in metrics.values()]) / len(metrics)
    mrr_score = sum([query_metrics['recip_rank'] for query_metrics in metrics.values()]) / len(metrics)
    ndcg10_score = sum([query_metrics['ndcg_cut_10'] for query_metrics in metrics.values()]) / len(metrics)
    
    results = {
        'run_name': run_name,
        'MAP': map_score,
        'MRR': mrr_score,
        'NDCG@10': ndcg10_score,
        'num_queries': len(metrics)
    }
    
    return results, metrics

def main():
    print("🚀 Starting Fixed ColBERT vs LTR Baseline Evaluation")
    print("=" * 60)
    
    # Load QREL
    print(f"📂 Loading QREL from: {QREL_FILE}")
    qrel = load_qrel(QREL_FILE)
    print(f"✅ Loaded QREL with {len(qrel)} queries")
    
    # Evaluate LTR Fusion
    print(f"\n📂 Loading LTR results from: {LTR_RESULTS_FILE}")
    ltr_run = load_run_file(LTR_RESULTS_FILE)
    print(f"✅ Loaded LTR results with {len(ltr_run)} queries")
    
    ltr_results, ltr_metrics = evaluate_run(qrel, ltr_run, "LTR Fusion")
    
    # Evaluate Fixed ColBERT
    print(f"\n📂 Loading Fixed ColBERT results from: {COLBERT_RESULTS_FILE}")
    colbert_run = load_run_file(COLBERT_RESULTS_FILE)
    print(f"✅ Loaded Fixed ColBERT results with {len(colbert_run)} queries")
    
    colbert_results, colbert_metrics = evaluate_run(qrel, colbert_run, "Fixed ColBERT")
    
    # Results comparison
    print("\n📊 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"🔍 LTR Fusion (Baseline):")
    print(f"   MAP: {ltr_results['MAP']:.4f}")
    print(f"   MRR: {ltr_results['MRR']:.4f}")
    print(f"   NDCG@10: {ltr_results['NDCG@10']:.4f}")
    print(f"   Queries: {ltr_results['num_queries']}")
    
    print(f"\n🤖 Fixed ColBERT Reranking:")
    print(f"   MAP: {colbert_results['MAP']:.4f}")
    print(f"   MRR: {colbert_results['MRR']:.4f}")
    print(f"   NDCG@10: {colbert_results['NDCG@10']:.4f}")
    print(f"   Queries: {colbert_results['num_queries']}")
    
    # Performance comparison
    print(f"\n📈 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    map_improvement = ((colbert_results['MAP'] - ltr_results['MAP']) / ltr_results['MAP']) * 100
    mrr_improvement = ((colbert_results['MRR'] - ltr_results['MRR']) / ltr_results['MRR']) * 100
    ndcg10_improvement = ((colbert_results['NDCG@10'] - ltr_results['NDCG@10']) / ltr_results['NDCG@10']) * 100
    
    print(f"MAP: {colbert_results['MAP']:.4f} vs {ltr_results['MAP']:.4f} ({map_improvement:+.2f}%)")
    print(f"MRR: {colbert_results['MRR']:.4f} vs {ltr_results['MRR']:.4f} ({mrr_improvement:+.2f}%)")
    print(f"NDCG@10: {colbert_results['NDCG@10']:.4f} vs {ltr_results['NDCG@10']:.4f} ({ndcg10_improvement:+.2f}%)")
    
    # Create comparison summary
    comparison_data = {
        'System': ['LTR Fusion', 'Fixed ColBERT', 'Improvement'],
        'MAP': [f"{ltr_results['MAP']:.4f}", f"{colbert_results['MAP']:.4f}", f"{map_improvement:+.2f}%"],
        'MRR': [f"{ltr_results['MRR']:.4f}", f"{colbert_results['MRR']:.4f}", f"{mrr_improvement:+.2f}%"],
        'NDCG@10': [f"{ltr_results['NDCG@10']:.4f}", f"{colbert_results['NDCG@10']:.4f}", f"{ndcg10_improvement:+.2f}%"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save results
    os.makedirs("fixed_colbert_results", exist_ok=True)
    
    # Save detailed metrics
    with open("fixed_colbert_results/fixed_colbert_evaluation_metrics.json", "w") as f:
        json.dump({
            "ltr_results": ltr_results,
            "colbert_results": colbert_results,
            "improvements": {
                "MAP": map_improvement,
                "MRR": mrr_improvement,
                "NDCG@10": ndcg10_improvement
            }
        }, f, indent=2)
    
    # Save comparison table
    comparison_df.to_csv("fixed_colbert_results/fixed_colbert_comparison.csv", index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   📄 Detailed metrics: fixed_colbert_results/fixed_colbert_evaluation_metrics.json")
    print(f"   📊 Comparison table: fixed_colbert_results/fixed_colbert_comparison.csv")
    
    print("\n📋 COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    print(f"\n🎯 CONCLUSION")
    print("=" * 60)
    if ndcg10_improvement > 0:
        print(f"✅ Fixed ColBERT improves NDCG@10 by {ndcg10_improvement:.2f}%")
    else:
        print(f"❌ Fixed ColBERT decreases NDCG@10 by {abs(ndcg10_improvement):.2f}%")
    
    if map_improvement > 0:
        print(f"✅ Fixed ColBERT improves MAP by {map_improvement:.2f}%")
    else:
        print(f"❌ Fixed ColBERT decreases MAP by {abs(map_improvement):.2f}%")

if __name__ == "__main__":
    main()
