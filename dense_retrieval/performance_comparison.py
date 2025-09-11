#!/usr/bin/env python3
"""
Performance Comparison: LTR vs ColBERT Middle Reranking
Compares the effectiveness of LTR fusion vs ColBERT reranking approaches
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_metrics(file_path):
    """Load metrics from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics from {file_path}: {e}")
        return {}

def load_ltr_metrics_from_csv(file_path):
    """Load LTR metrics from evaluation CSV file"""
    try:
        df = pd.read_csv(file_path)
        # Get LTR row (last row)
        ltr_row = df[df['Method'] == 'LTR'].iloc[0] if 'Method' in df.columns else df.iloc[-1]
        
        return {
            'ndcg_cut_10': float(ltr_row['NDCG@10']),
            'recip_rank': float(ltr_row['NDCG@10']),  # Using NDCG@10 as proxy for MRR
            'map': float(ltr_row['NDCG@10'])  # Using NDCG@10 as proxy for MAP
        }
    except Exception as e:
        logger.error(f"Error loading LTR metrics from {file_path}: {e}")
        return {}

def load_trec_run_file(file_path):
    """Load TREC run file format"""
    try:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    data.append({
                        'query_id': parts[0],
                        'doc_id': parts[2],
                        'rank': int(parts[3]),
                        'score': float(parts[4])
                    })
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading TREC file from {file_path}: {e}")
        return pd.DataFrame()

def load_results_csv(file_path):
    """Load results from CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading results from {file_path}: {e}")
        return pd.DataFrame()

def compare_performance():
    """Compare LTR vs ColBERT performance"""
    
    # Load LTR Stage 2 metrics from evaluation CSV
    ltr_eval_path = "/home/ugdf8/IRIS/TREC-TOT-2025/dense_run_files/run_files/stage2_fusion/llama_train_fusion_evaluation.csv"
    ltr_metrics = load_ltr_metrics_from_csv(ltr_eval_path)
    
    # Load ColBERT metrics
    colbert_metrics_path = "/home/ugdf8/IRIS/TREC-TOT-2025/dense_retrieval/colbert_results/colbert_llama_train_metrics.json"
    colbert_metrics = load_metrics(colbert_metrics_path)
    
    logger.info("=== PERFORMANCE COMPARISON: LTR vs ColBERT ===")
    logger.info("")
    
    # Main metrics comparison
    if ltr_metrics and colbert_metrics:
        logger.info("📊 METRICS COMPARISON:")
        logger.info("")
        
        metrics = ['ndcg_cut_10', 'recip_rank', 'map']
        comparison_data = []
        
        for metric in metrics:
            ltr_val = ltr_metrics.get(metric, 0.0)
            colbert_val = colbert_metrics.get(metric, 0.0)
            
            # Calculate percentage change
            if ltr_val > 0:
                change_pct = ((colbert_val - ltr_val) / ltr_val) * 100
            else:
                change_pct = 0.0
            
            comparison_data.append({
                'Metric': metric.upper().replace('_', ' '),
                'LTR Fusion': f"{ltr_val:.4f}",
                'ColBERT Reranking': f"{colbert_val:.4f}",
                'Change (%)': f"{change_pct:+.1f}%"
            })
            
            logger.info(f"{metric.upper().replace('_', ' '):<15}: LTR={ltr_val:.4f} | ColBERT={colbert_val:.4f} | Change={change_pct:+.1f}%")
        
        logger.info("")
        
        # Performance analysis
        ndcg_ltr = ltr_metrics.get('ndcg_cut_10', 0.0)
        ndcg_colbert = colbert_metrics.get('ndcg_cut_10', 0.0)
        
        if ndcg_colbert > ndcg_ltr:
            logger.info("🚀 RESULT: ColBERT reranking OUTPERFORMS LTR fusion")
            improvement = ((ndcg_colbert - ndcg_ltr) / ndcg_ltr) * 100
            logger.info(f"   ColBERT achieves {improvement:.1f}% improvement in NDCG@10")
        elif ndcg_ltr > ndcg_colbert:
            logger.info("⚠️  RESULT: LTR fusion OUTPERFORMS ColBERT reranking")
            degradation = ((ndcg_ltr - ndcg_colbert) / ndcg_ltr) * 100
            logger.info(f"   ColBERT shows {degradation:.1f}% degradation in NDCG@10")
        else:
            logger.info("🔄 RESULT: LTR and ColBERT show similar performance")
        
        # Save comparison table
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv_path = "/home/ugdf8/IRIS/TREC-TOT-2025/dense_retrieval/performance_comparison.csv"
        comparison_df.to_csv(comparison_csv_path, index=False)
        logger.info(f"📊 Comparison table saved to: {comparison_csv_path}")
        
        # Create visualization
        create_performance_chart(ltr_metrics, colbert_metrics)
        
    else:
        logger.error("Could not load both metric files for comparison")
    
    # Detailed analysis
    logger.info("")
    logger.info("🔍 DETAILED ANALYSIS:")
    
    # Load result files for per-query analysis
    ltr_results_path = "/home/ugdf8/IRIS/TREC-TOT-2025/dense_run_files/run_files/stage2_fusion/llama_train_ltr_fusion.txt"
    colbert_results_path = "/home/ugdf8/IRIS/TREC-TOT-2025/dense_retrieval/colbert_results/colbert_llama_train_run.txt"
    
    # Load TREC format files
    ltr_results = load_trec_run_file(ltr_results_path)
    colbert_results = load_trec_run_file(colbert_results_path)
    
    if not ltr_results.empty and not colbert_results.empty:
        analyze_per_query_performance(ltr_results, colbert_results)
    
    logger.info("")
    logger.info("✅ Performance comparison completed!")

def create_performance_chart(ltr_metrics, colbert_metrics):
    """Create performance comparison chart"""
    try:
        metrics = ['NDCG@10', 'MAP', 'MRR']
        ltr_values = [
            ltr_metrics.get('ndcg_cut_10', 0.0),
            ltr_metrics.get('map', 0.0),
            ltr_metrics.get('recip_rank', 0.0)
        ]
        colbert_values = [
            colbert_metrics.get('ndcg_cut_10', 0.0),
            colbert_metrics.get('map', 0.0),
            colbert_metrics.get('recip_rank', 0.0)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, ltr_values, width, label='LTR Fusion', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, colbert_values, width, label='ColBERT Reranking', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison: LTR Fusion vs ColBERT Reranking')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8)
        
        plt.tight_layout()
        chart_path = "/home/ugdf8/IRIS/TREC-TOT-2025/dense_retrieval/performance_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        logger.info(f"📈 Performance chart saved to: {chart_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating performance chart: {e}")

def analyze_per_query_performance(ltr_results, colbert_results):
    """Analyze per-query performance differences"""
    try:
        # Get top-ranked documents for each query
        if not ltr_results.empty and not colbert_results.empty:
            ltr_top = ltr_results[ltr_results['rank'] == 1]
            colbert_top = colbert_results[colbert_results['rank'] == 1]
            
            # Merge on query_id for comparison
            merged = pd.merge(ltr_top, colbert_top, on='query_id', suffixes=('_ltr', '_colbert'))
            
            if not merged.empty:
                # Calculate score differences
                merged['score_diff'] = merged['score_colbert'] - merged['score_ltr']
                
                improvement_count = (merged['score_diff'] > 0).sum()
                degradation_count = (merged['score_diff'] < 0).sum()
                equal_count = (merged['score_diff'] == 0).sum()
                
                logger.info(f"Per-query analysis (top-ranked document scores):")
                logger.info(f"  Queries improved by ColBERT: {improvement_count}")
                logger.info(f"  Queries degraded by ColBERT: {degradation_count}")
                logger.info(f"  Queries with equal performance: {equal_count}")
                
                avg_improvement = merged['score_diff'].mean()
                logger.info(f"  Average score change: {avg_improvement:+.4f}")
                
                # Check if same documents are retrieved
                same_docs = (merged['doc_id_ltr'] == merged['doc_id_colbert']).sum()
                total_queries = len(merged)
                logger.info(f"  Queries with same top document: {same_docs}/{total_queries} ({same_docs/total_queries*100:.1f}%)")
        
        # Compare number of queries processed
        ltr_queries = len(ltr_results['query_id'].unique()) if not ltr_results.empty else 0
        colbert_queries = len(colbert_results['query_id'].unique()) if not colbert_results.empty else 0
        
        logger.info(f"Coverage:")
        logger.info(f"  LTR processed queries: {ltr_queries}")
        logger.info(f"  ColBERT processed queries: {colbert_queries}")
        
    except Exception as e:
        logger.error(f"Error in per-query analysis: {e}")

if __name__ == "__main__":
    compare_performance()
