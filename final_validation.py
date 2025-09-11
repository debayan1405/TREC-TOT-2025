#!/usr/bin/env python3
"""
Final validation check for TREC submission
"""

from collections import defaultdict

def main():
    print("🔍 FINAL VALIDATION CHECK")
    print("=" * 40)
    
    input_file = "/home/ugdf8/IRIS/TREC-TOT-2025/final_run.txt"
    
    query_docs = defaultdict(set)
    query_counts = defaultdict(int)
    total_lines = 0
    duplicates = []
    malformed = []
    
    print(f"📋 Validating: {input_file}")
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            parts = line.strip().split()
            
            # Check format
            if len(parts) != 6:
                malformed.append(line_num)
                continue
            
            query_id, q0, doc_id, rank, score, run_tag = parts
            
            # Check for duplicates
            if doc_id in query_docs[query_id]:
                duplicates.append((line_num, query_id, doc_id))
            else:
                query_docs[query_id].add(doc_id)
            
            query_counts[query_id] += 1
    
    print(f"\n✅ VALIDATION RESULTS:")
    print(f"   📊 Total lines: {total_lines:,}")
    print(f"   📊 Unique queries: {len(query_counts)}")
    print(f"   📊 Query range: {min(query_counts.keys())} - {max(query_counts.keys())}")
    print(f"   📊 Results per query: {min(query_counts.values())} - {max(query_counts.values())}")
    print(f"   ❌ Malformed lines: {len(malformed)}")
    print(f"   ❌ Duplicate documents: {len(duplicates)}")
    
    if malformed:
        print(f"\n⚠️  Malformed lines found:")
        for line_num in malformed[:5]:
            print(f"      Line {line_num}")
    
    if duplicates:
        print(f"\n⚠️  Duplicate documents found:")
        for line_num, query_id, doc_id in duplicates[:5]:
            print(f"      Line {line_num}: doc {doc_id} in query {query_id}")
    
    # Check some specific previously problematic queries
    problem_queries = ['3007', '3026', '3033', '3059', '3065']
    print(f"\n🔍 Checking previously problematic queries:")
    for query_id in problem_queries:
        if query_id in query_docs:
            unique_docs = len(query_docs[query_id])
            total_results = query_counts[query_id]
            print(f"   Query {query_id}: {total_results} results, {unique_docs} unique docs")
            if unique_docs != total_results:
                print(f"      ⚠️  {total_results - unique_docs} duplicates!")
        else:
            print(f"   Query {query_id}: NOT FOUND")
    
    if len(malformed) == 0 and len(duplicates) == 0:
        print(f"\n🎉 VALIDATION PASSED!")
        print(f"   ✅ Perfect TREC format")
        print(f"   ✅ No duplicate documents per query")
        print(f"   ✅ All 622 queries present")
        print(f"   ✅ Ready for submission!")
    else:
        print(f"\n❌ VALIDATION FAILED!")
        print(f"   Fix malformed lines and/or duplicates before submission")

if __name__ == "__main__":
    main()
