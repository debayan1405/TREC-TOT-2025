#!/usr/bin/env python3
"""
Debug document ID mapping in PyTerrier index
"""
import os
import sys
from pathlib import Path

# Set Java environment before any PyTerrier imports
os.environ['JAVA_HOME'] = '/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm'
os.environ['JVM_PATH'] = '/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm/lib/server/libjvm.so'

import pyterrier as pt

# Initialize PyTerrier
if not pt.java.started():
    pt.java.init()

def debug_docno_mapping():
    print("🔍 DEBUGGING DOCNO MAPPING IN PYTERRIER INDEX")
    print("=" * 60)
    
    # Load index
    index_path = "/home/ugdf8/IRIS/TREC-TOT-2025/trec-tot-2025-pyterrier-index"
    index = pt.IndexFactory.of(index_path)
    
    print(f"Total documents: {index.getCollectionStatistics().getNumberOfDocuments()}")
    
    # Get different index structures
    doc_index = index.getDocumentIndex()
    meta_index = index.getMetaIndex()
    
    print("\n📋 Available metadata keys:")
    print(f"Keys: {meta_index.getKeys()}")
    
    print("\n🧪 Testing different docno access methods:")
    
    # Test first 10 documents
    for docid in range(10):
        print(f"\n--- Document ID {docid} ---")
        
        # Method 1: Direct meta access
        try:
            docno_meta = meta_index.getItem("docno", docid)
            print(f"  meta_index.getItem('docno', {docid}) = '{docno_meta}'")
        except Exception as e:
            print(f"  meta_index.getItem ERROR: {e}")
        
        # Method 2: Get all metadata
        try:
            all_meta = meta_index.getAllItems(docid)
            print(f"  meta_index.getAllItems({docid}) = {all_meta}")
        except Exception as e:
            print(f"  meta_index.getAllItems ERROR: {e}")
    
    print("\n🔍 Searching for sample QRELs docnos in index:")
    sample_qrels_docnos = ['16742289', '30523669', '1705452', '4891218', '4815950']
    
    for target_docno in sample_qrels_docnos:
        print(f"\nSearching for docno '{target_docno}':")
        
        # Linear search through index (expensive but thorough)
        found = False
        for docid in range(min(1000, index.getCollectionStatistics().getNumberOfDocuments())):
            try:
                docno = meta_index.getItem("docno", docid)
                if str(docno) == str(target_docno):
                    print(f"  ✅ FOUND: docno '{target_docno}' at docid {docid}")
                    found = True
                    break
            except:
                continue
        
        if not found:
            print(f"  ❌ NOT FOUND in first 1000 documents")

if __name__ == "__main__":
    debug_docno_mapping()
