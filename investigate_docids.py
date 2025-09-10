#!/usr/bin/env python3
"""
Script to investigate the document ID mismatch between fused runs, 
reranked runs, and QRELs.
"""
import pandas as pd
from collections import Counter, defaultdict

def load_run_file(file_path):
    """Load run file."""
    return pd.read_csv(file_path, sep=r'\s+', header=None,
                      names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name'],
                      dtype={'qid': str, 'docno': str})

def load_qrels(file_path):
    """Load QREL file."""
    return pd.read_csv(file_path, sep=r'\s+', header=None,
                      names=['qid', 'iteration', 'docno', 'relevance'],
                      dtype={'qid': str, 'docno': str})

def analyze_docid_patterns():
    print("🔍 DOCUMENT ID MISMATCH INVESTIGATION")
    print("=" * 60)
    
    # Load files
    print("Loading files...")
    fused_run = load_run_file("fused_run_files/train/mistral_train_fused.txt")
    reranked_run = load_run_file("dense_run_files/train/mistral_train_all-mpnet-base-v2_reranked.txt")
    qrels = load_qrels("qrel/train-2025-qrel.txt")
    
    print(f"✅ Fused run: {len(fused_run)} entries, {fused_run['qid'].nunique()} queries")
    print(f"✅ Reranked run: {len(reranked_run)} entries, {reranked_run['qid'].nunique()} queries")
    print(f"✅ QRELs: {len(qrels)} entries, {qrels['qid'].nunique()} queries")
    
    # Analyze document ID patterns
    print(f"\n📊 DOCUMENT ID PATTERNS:")
    
    fused_docnos = set(fused_run['docno'].unique())
    reranked_docnos = set(reranked_run['docno'].unique())
    qrel_docnos = set(qrels['docno'].unique())
    
    print(f"Unique docnos in fused run: {len(fused_docnos)}")
    print(f"Unique docnos in reranked run: {len(reranked_docnos)}")
    print(f"Unique docnos in QRELs: {len(qrel_docnos)}")
    
    # Check overlap
    fused_vs_reranked = len(fused_docnos & reranked_docnos)
    fused_vs_qrels = len(fused_docnos & qrel_docnos)
    reranked_vs_qrels = len(reranked_docnos & qrel_docnos)
    
    print(f"\n🔗 DOCUMENT ID OVERLAPS:")
    print(f"Fused ∩ Reranked: {fused_vs_reranked} ({fused_vs_reranked/len(fused_docnos)*100:.1f}%)")
    print(f"Fused ∩ QRELs: {fused_vs_qrels} ({fused_vs_qrels/len(fused_docnos)*100:.1f}%)")
    print(f"Reranked ∩ QRELs: {reranked_vs_qrels} ({reranked_vs_qrels/len(reranked_docnos)*100:.1f}%)")
    
    # Show examples
    print(f"\n📋 SAMPLE DOCUMENT IDs:")
    print("Fused run samples:", list(fused_docnos)[:10])
    print("Reranked run samples:", list(reranked_docnos)[:10])
    print("QREL samples:", list(qrel_docnos)[:10])
    
    # Check if document IDs are being transformed
    print(f"\n🔄 DOCUMENT ID TRANSFORMATION CHECK:")
    sample_query = fused_run['qid'].iloc[0]
    
    fused_docs_q = set(fused_run[fused_run['qid'] == sample_query]['docno'].head(10))
    reranked_docs_q = set(reranked_run[reranked_run['qid'] == sample_query]['docno'].head(10))
    
    print(f"Query {sample_query}:")
    print(f"  Fused top-10 docnos: {sorted(list(fused_docs_q))}")
    print(f"  Reranked top-10 docnos: {sorted(list(reranked_docs_q))}")
    
    if fused_docs_q == reranked_docs_q:
        print("  ✅ Document IDs are preserved during reranking")
    else:
        print("  ❌ Document IDs are being changed during reranking!")
        print(f"  Missing from reranked: {fused_docs_q - reranked_docs_q}")
        print(f"  Added in reranked: {reranked_docs_q - fused_docs_q}")
    
    # Check query coverage
    print(f"\n📈 QUERY COVERAGE:")
    
    for qid in list(qrels['qid'].unique())[:5]:  # Check first 5 queries
        qrel_docs = set(qrels[qrels['qid'] == qid]['docno'])
        
        fused_docs = set(fused_run[fused_run['qid'] == qid]['docno'])
        reranked_docs = set(reranked_run[reranked_run['qid'] == qid]['docno'])
        
        fused_matches = len(qrel_docs & fused_docs)
        reranked_matches = len(qrel_docs & reranked_docs)
        
        print(f"Query {qid}:")
        print(f"  Relevant docs in QRELs: {qrel_docs}")
        print(f"  Fused run matches: {fused_matches}/{len(qrel_docs)}")
        print(f"  Reranked run matches: {reranked_matches}/{len(qrel_docs)}")
        
        if fused_matches > 0 and reranked_matches == 0:
            print(f"  🚨 PROBLEM: Relevant docs found in fused but lost in reranked!")
    
    # Length analysis
    print(f"\n📏 DOCUMENT ID LENGTH ANALYSIS:")
    
    fused_lengths = [len(docno) for docno in list(fused_docnos)[:100]]
    reranked_lengths = [len(docno) for docno in list(reranked_docnos)[:100]]
    qrel_lengths = [len(docno) for docno in list(qrel_docnos)]
    
    print(f"Fused docno lengths: min={min(fused_lengths)}, max={max(fused_lengths)}, avg={sum(fused_lengths)/len(fused_lengths):.1f}")
    print(f"Reranked docno lengths: min={min(reranked_lengths)}, max={max(reranked_lengths)}, avg={sum(reranked_lengths)/len(reranked_lengths):.1f}")
    print(f"QREL docno lengths: min={min(qrel_lengths)}, max={max(qrel_lengths)}, avg={sum(qrel_lengths)/len(qrel_lengths):.1f}")
    
    # Character pattern analysis
    print(f"\n🔤 CHARACTER PATTERN ANALYSIS:")
    
    def analyze_patterns(docno_set, name):
        patterns = {
            'all_numeric': sum(1 for d in list(docno_set)[:100] if d.isdigit()),
            'contains_letters': sum(1 for d in list(docno_set)[:100] if any(c.isalpha() for c in d)),
            'contains_dash': sum(1 for d in list(docno_set)[:100] if '-' in d),
            'contains_underscore': sum(1 for d in list(docno_set)[:100] if '_' in d),
            'starts_with_digit': sum(1 for d in list(docno_set)[:100] if d[0].isdigit()),
        }
        print(f"{name}: {patterns}")
        return patterns
    
    fused_patterns = analyze_patterns(fused_docnos, "Fused")
    reranked_patterns = analyze_patterns(reranked_docnos, "Reranked") 
    qrel_patterns = analyze_patterns(qrel_docnos, "QRELs")

if __name__ == "__main__":
    analyze_docid_patterns()