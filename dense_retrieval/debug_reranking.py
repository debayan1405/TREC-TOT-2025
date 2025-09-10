#!/usr/bin/env python3
"""
Comprehensive debugging script for dense re-ranking pipeline.
This script helps diagnose why dense re-ranking is failing by checking:
1. Document ID alignment between reranker output and QRELs
2. Candidate recall before/after reranking
3. Query text validation
4. Score distributions
5. Output format validation
"""

import pandas as pd
import pyterrier as pt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse
import sys

def load_qrels(qrels_path):
    """Load QREL file into a set of (qid, docno) pairs for quick lookup."""
    qrels_df = pd.read_csv(qrels_path, sep=r'\s+', header=None, 
                          names=['qid', 'iteration', 'docno', 'relevance'],
                          dtype={'qid': str, 'docno': str})
    # Filter for relevant documents (relevance > 0)
    relevant_docs = set((str(qid), str(docno)) 
                       for qid, docno in zip(qrels_df['qid'], qrels_df['docno']) 
                       if qrels_df[(qrels_df['qid'] == qid) & (qrels_df['docno'] == docno)]['relevance'].iloc[0] > 0)
    return relevant_docs, qrels_df

def load_run_file(run_path):
    """Load run file with proper data types."""
    df = pd.read_csv(run_path, sep=r'\s+', header=None,
                    names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name'],
                    dtype={'qid': str, 'docno': str, 'rank': int})
    return df

def check_docid_alignment(run_df, qrels_set, top_k=10):
    """
    Check if document IDs in run file match those in QRELs.
    Returns alignment stats and examples of mismatches.
    """
    print(f"\n=== DOCID ALIGNMENT CHECK (Top-{top_k}) ===")
    
    # Get top-k docs per query from run
    top_docs = run_df.groupby('qid').head(top_k)
    
    # Check alignment
    run_pairs = set((str(row['qid']), str(row['docno'])) for _, row in top_docs.iterrows())
    aligned_pairs = run_pairs & qrels_set
    
    print(f"Total top-{top_k} (qid, docno) pairs in run: {len(run_pairs)}")
    print(f"Pairs that match QREL document IDs: {len(aligned_pairs)}")
    print(f"Alignment rate: {len(aligned_pairs) / len(run_pairs) * 100:.2f}%")
    
    # Show examples of run docnos vs qrel docnos
    sample_qid = top_docs['qid'].iloc[0]
    run_docs_sample = set(top_docs[top_docs['qid'] == sample_qid]['docno'].head(5))
    qrel_docs_sample = set(docno for qid, docno in qrels_set if qid == sample_qid)
    
    print(f"\nExample for query {sample_qid}:")
    print(f"Run top-5 docnos: {list(run_docs_sample)}")
    print(f"QREL docnos for this query: {list(qrel_docs_sample)[:5] if qrel_docs_sample else 'None'}")
    
    return len(aligned_pairs) / len(run_pairs) if run_pairs else 0

def compute_recall_at_k(run_df, qrels_set, k_values=[10, 100, 1000, 2000]):
    """Compute Recall@K for different K values."""
    print(f"\n=== RECALL@K ANALYSIS ===")
    
    results = {}
    for k in k_values:
        top_k_docs = run_df.groupby('qid').head(k)
        
        recall_per_query = []
        for qid in run_df['qid'].unique():
            # Get relevant docs for this query from QRELs
            relevant_for_qid = set(docno for q, docno in qrels_set if q == str(qid))
            
            if not relevant_for_qid:  # Skip queries with no relevant docs
                continue
                
            # Get retrieved docs for this query
            retrieved_for_qid = set(top_k_docs[top_k_docs['qid'] == qid]['docno'].astype(str))
            
            # Calculate recall
            retrieved_relevant = len(relevant_for_qid & retrieved_for_qid)
            recall = retrieved_relevant / len(relevant_for_qid) if relevant_for_qid else 0
            recall_per_query.append(recall)
        
        avg_recall = np.mean(recall_per_query) if recall_per_query else 0
        results[f'Recall@{k}'] = avg_recall
        print(f"Recall@{k}: {avg_recall:.4f}")
    
    return results

def analyze_score_distribution(run_df, output_dir):
    """Analyze the distribution of reranking scores."""
    print(f"\n=== SCORE DISTRIBUTION ANALYSIS ===")
    
    scores = run_df['score'].values
    print(f"Score statistics:")
    print(f"  Min: {scores.min():.6f}")
    print(f"  Max: {scores.max():.6f}")
    print(f"  Mean: {scores.mean():.6f}")
    print(f"  Std: {scores.std():.6f}")
    print(f"  Unique values: {len(np.unique(scores))}")
    
    # Check for problematic patterns
    if scores.std() < 1e-6:
        print("⚠️  WARNING: All scores are nearly identical! Reranker may not be working.")
    
    if np.all(scores <= 0):
        print("⚠️  WARNING: All scores are <= 0. Check reranker output.")
    
    if len(np.unique(scores)) < len(scores) * 0.1:
        print("⚠️  WARNING: Very few unique scores. Reranker may be producing constant output.")
    
    # Plot score distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=50, alpha=0.7)
    plt.title('Score Distribution')
    plt.xlabel('Reranking Score')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    # Score distribution per query (sample 5 queries)
    sample_qids = run_df['qid'].unique()[:5]
    for qid in sample_qids:
        query_scores = run_df[run_df['qid'] == qid]['score']
        plt.plot(query_scores.values, alpha=0.7, label=f'Query {qid}')
    plt.title('Score Distribution by Query (Sample)')
    plt.xlabel('Document Rank')
    plt.ylabel('Reranking Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Score distribution plot saved to: {output_dir / 'score_distribution.png'}")

def validate_output_format(run_df):
    """Validate the format of the reranker output."""
    print(f"\n=== OUTPUT FORMAT VALIDATION ===")
    
    # Check basic structure
    expected_cols = ['qid', 'Q0', 'docno', 'rank', 'score', 'run_name']
    missing_cols = set(expected_cols) - set(run_df.columns)
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    
    # Check data types
    if run_df['qid'].dtype != 'object':
        print("⚠️  WARNING: qid should be string/object type")
    if run_df['docno'].dtype != 'object':
        print("⚠️  WARNING: docno should be string/object type")
    
    # Check for missing values
    null_counts = run_df.isnull().sum()
    if null_counts.any():
        print(f"❌ Null values found: {null_counts[null_counts > 0].to_dict()}")
        return False
    
    # Check rank ordering
    rank_issues = 0
    for qid in run_df['qid'].unique():
        query_data = run_df[run_df['qid'] == qid].sort_values('rank')
        expected_ranks = list(range(1, len(query_data) + 1))
        actual_ranks = query_data['rank'].tolist()
        if actual_ranks != expected_ranks:
            rank_issues += 1
            if rank_issues <= 3:  # Show first 3 examples
                print(f"❌ Rank ordering issue for query {qid}: expected {expected_ranks[:5]}, got {actual_ranks[:5]}")
    
    if rank_issues == 0:
        print("✅ Output format validation passed")
        return True
    else:
        print(f"❌ Rank ordering issues found in {rank_issues} queries")
        return False

def compare_before_after_reranking(fused_path, reranked_path, qrels_set):
    """Compare retrieval effectiveness before and after reranking."""
    print(f"\n=== BEFORE vs AFTER RERANKING COMPARISON ===")
    
    try:
        fused_df = load_run_file(fused_path)
        reranked_df = load_run_file(reranked_path)
        
        print("BEFORE RERANKING (Fused):")
        fused_recall = compute_recall_at_k(fused_df, qrels_set)
        
        print("\nAFTER RERANKING:")
        reranked_recall = compute_recall_at_k(reranked_df, qrels_set)
        
        # Show the difference
        print("\nCHANGE (After - Before):")
        for metric in fused_recall:
            change = reranked_recall[metric] - fused_recall[metric]
            print(f"{metric}: {change:+.4f}")
            
    except Exception as e:
        print(f"Error comparing before/after: {e}")

def check_candidate_set_quality(fused_path, qrels_set):
    """Check if the candidate set from fusion contains relevant documents."""
    print(f"\n=== CANDIDATE SET QUALITY CHECK ===")
    
    try:
        fused_df = load_run_file(fused_path)
        
        # Check how many queries have at least one relevant doc in top-K
        queries_with_relevant = defaultdict(int)
        total_queries = len(fused_df['qid'].unique())
        
        for k in [10, 100, 1000, 2000]:
            top_k = fused_df.groupby('qid').head(k)
            
            for qid in fused_df['qid'].unique():
                query_docs = set(top_k[top_k['qid'] == qid]['docno'].astype(str))
                relevant_docs = set(docno for q, docno in qrels_set if q == str(qid))
                
                if query_docs & relevant_docs:  # Has at least one relevant doc
                    queries_with_relevant[k] += 1
            
            coverage = queries_with_relevant[k] / total_queries * 100
            print(f"Queries with ≥1 relevant doc in top-{k}: {queries_with_relevant[k]}/{total_queries} ({coverage:.1f}%)")
            
    except Exception as e:
        print(f"Error checking candidate set: {e}")

def main():
    parser = argparse.ArgumentParser(description="Debug dense reranking pipeline")
    parser.add_argument("--reranked_run", required=True, help="Path to reranked run file")
    parser.add_argument("--fused_run", required=True, help="Path to fused run file (before reranking)")
    parser.add_argument("--qrels", required=True, help="Path to QREL file")
    parser.add_argument("--output_dir", default="./debug_output", help="Output directory for plots")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 DENSE RERANKING DEBUG ANALYSIS")
    print("=" * 50)
    
    # Load data
    try:
        print(f"Loading QRELs from: {args.qrels}")
        qrels_set, qrels_df = load_qrels(args.qrels)
        print(f"✅ Loaded {len(qrels_set)} relevant (qid, docno) pairs")
        
        print(f"Loading reranked run from: {args.reranked_run}")
        reranked_df = load_run_file(args.reranked_run)
        print(f"✅ Loaded reranked run with {len(reranked_df)} entries")
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # Run all diagnostic checks
    validate_output_format(reranked_df)
    alignment_rate = check_docid_alignment(reranked_df, qrels_set)
    compute_recall_at_k(reranked_df, qrels_set)
    analyze_score_distribution(reranked_df, output_dir)
    
    # Compare before/after if fused file provided
    if Path(args.fused_run).exists():
        check_candidate_set_quality(args.fused_run, qrels_set)
        compare_before_after_reranking(args.fused_run, args.reranked_run, qrels_set)
    
    print(f"\n🏁 DIAGNOSIS SUMMARY:")
    print(f"=" * 30)
    
    if alignment_rate < 0.1:
        print("🚨 CRITICAL: Document ID alignment is very poor (<10%)")
        print("   → Check if reranker uses different document ID format")
        print("   → Verify document metadata extraction in dense_reranker.py")
    elif alignment_rate < 0.5:
        print("⚠️  WARNING: Document ID alignment is low (<50%)")
    else:
        print("✅ Document ID alignment looks reasonable")
    
    print(f"\n💡 RECOMMENDED NEXT STEPS:")
    print("1. Fix document ID alignment if critical")
    print("2. Consider using cross-encoder instead of bi-encoder for reranking")
    print("3. Increase candidate set size if recall@2000 is low")
    print("4. Fine-tune reranking model on your domain/QRELs")

if __name__ == "__main__":
    main()