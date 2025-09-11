#!/usr/bin/env python3
"""
Fix TREC submission by removing duplicate document IDs per query
"""

from collections import defaultdict
from tqdm import tqdm

def main():
    print("🔧 FIXING TREC SUBMISSION - REMOVING DUPLICATES")
    print("=" * 60)
    
    input_file = "/home/ugdf8/IRIS/TREC-TOT-2025/final_run.txt"
    output_file = "/home/ugdf8/IRIS/TREC-TOT-2025/final_run_deduplicated.txt"
    
    print(f"📋 Input file: {input_file}")
    print(f"📋 Output file: {output_file}")
    
    # Track seen documents per query
    query_docs = defaultdict(set)
    total_lines = 0
    duplicate_count = 0
    kept_lines = 0
    
    print("\n🔍 Step 1: Processing and deduplicating...")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, desc="Processing lines"):
            total_lines += 1
            parts = line.strip().split()
            
            if len(parts) >= 6:
                query_id = parts[0]
                doc_id = parts[2]
                
                # Check if this document has already been seen for this query
                if doc_id not in query_docs[query_id]:
                    # First time seeing this doc for this query - keep it
                    query_docs[query_id].add(doc_id)
                    outfile.write(line)
                    kept_lines += 1
                else:
                    # Duplicate document for this query - skip it
                    duplicate_count += 1
            else:
                # Write malformed lines as-is (shouldn't happen)
                outfile.write(line)
                kept_lines += 1
    
    print(f"\n✅ DEDUPLICATION COMPLETED!")
    print(f"   📊 Total input lines: {total_lines:,}")
    print(f"   ❌ Duplicate lines removed: {duplicate_count:,}")
    print(f"   ✅ Lines kept: {kept_lines:,}")
    
    print("\n🔍 Step 2: Verification...")
    
    # Verify the output
    query_counts = defaultdict(int)
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id = parts[0]
                query_counts[query_id] += 1
    
    print(f"   📊 Unique queries in output: {len(query_counts)}")
    print(f"   📊 Results per query range: {min(query_counts.values())} - {max(query_counts.values())}")
    print(f"   📊 Average results per query: {sum(query_counts.values()) / len(query_counts):.1f}")
    
    # Show some specific query counts
    print(f"   📋 Sample query result counts:")
    for i, (query_id, count) in enumerate(list(query_counts.items())[:5]):
        print(f"      Query {query_id}: {count} results")
    
    # Check file size
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"   📁 Output file size: {file_size_mb:.1f} MB")
    
    # Show first and last lines
    print(f"\n📋 Sample output:")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print(f"   First line: {lines[0].strip()}")
        print(f"   Last line: {lines[-1].strip()}")
    
    print(f"\n🎉 DEDUPLICATED SUBMISSION READY: {output_file}")
    print("   ✅ No duplicate documents per query")
    print("   ✅ Ready for TREC-TOT 2025 submission!")

if __name__ == "__main__":
    main()
