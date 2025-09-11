#!/usr/bin/env python3
"""
Fix rankings in TREC submission after deduplication
"""

from collections import defaultdict
from tqdm import tqdm

def main():
    print("🔧 FIXING RANKINGS AFTER DEDUPLICATION")
    print("=" * 50)
    
    input_file = "/home/ugdf8/IRIS/TREC-TOT-2025/final_run_deduplicated.txt"
    output_file = "/home/ugdf8/IRIS/TREC-TOT-2025/final_run.txt"
    
    print(f"📋 Input file: {input_file}")
    print(f"📋 Output file: {output_file}")
    
    print("\n🔍 Step 1: Loading and grouping by query...")
    
    # Load all lines and group by query
    query_results = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc="Loading lines"):
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id = parts[0]
                q0 = parts[1]
                doc_id = parts[2]
                rank = int(parts[3])
                score = float(parts[4])
                run_tag = parts[5]
                
                query_results[query_id].append({
                    'q0': q0,
                    'doc_id': doc_id,
                    'original_rank': rank,
                    'score': score,
                    'run_tag': run_tag,
                    'original_line': line.strip()
                })
    
    print(f"   📊 Loaded results for {len(query_results)} queries")
    
    print("\n🔍 Step 2: Sorting by score and fixing rankings...")
    
    total_results = 0
    
    with open(output_file, 'w') as outfile:
        for query_id in tqdm(sorted(query_results.keys(), key=int), desc="Processing queries"):
            # Sort results by score (descending) to ensure proper ranking
            results = sorted(query_results[query_id], key=lambda x: x['score'], reverse=True)
            
            # Rewrite with correct rankings
            for new_rank, result in enumerate(results, 1):
                outfile.write(f"{query_id} {result['q0']} {result['doc_id']} {new_rank} {result['score']} {result['run_tag']}\n")
                total_results += 1
    
    print(f"   ✅ Fixed rankings for {total_results:,} results")
    
    print("\n🔍 Step 3: Final verification...")
    
    # Verify no duplicates and proper rankings
    query_docs = defaultdict(set)
    query_counts = defaultdict(int)
    duplicates_found = 0
    
    with open(output_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                
                # Check for duplicates
                if doc_id in query_docs[query_id]:
                    duplicates_found += 1
                    print(f"   ⚠️  Duplicate found at line {line_num}: doc {doc_id} in query {query_id}")
                else:
                    query_docs[query_id].add(doc_id)
                
                query_counts[query_id] += 1
    
    print(f"   📊 Total queries: {len(query_counts)}")
    print(f"   📊 Total results: {sum(query_counts.values())}")
    print(f"   📊 Results per query: {min(query_counts.values())} - {max(query_counts.values())}")
    print(f"   ❌ Duplicates found: {duplicates_found}")
    
    # Check file size
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"   📁 File size: {file_size_mb:.1f} MB")
    
    # Show sample
    print(f"\n📋 Sample output:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 3 or i >= sum(query_counts.values()) - 3:
                print(f"   Line {i+1}: {line.strip()}")
            elif i == 3:
                print("   ...")
    
    print(f"\n🎉 FINAL SUBMISSION READY: {output_file}")
    if duplicates_found == 0:
        print("   ✅ No duplicate documents per query")
        print("   ✅ Proper ranking order maintained")
        print("   ✅ Ready for TREC-TOT 2025 submission!")
    else:
        print(f"   ⚠️  {duplicates_found} duplicates still found - needs further investigation")

if __name__ == "__main__":
    main()
