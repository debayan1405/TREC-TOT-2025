#!/usr/bin/env python3
"""
Investigate what's actually in the PyTerrier index vs what's expected.
"""
import sys
import os
from pathlib import Path
import pandas as pd

# Set Java environment before any PyTerrier imports
os.environ['JAVA_HOME'] = '/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm'
os.environ['JVM_PATH'] = '/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm/lib/server/libjvm.so'

# Add project paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from sparse_retrieval.config_loader import ConfigLoader
from sparse_retrieval.data_loader import DataLoader

def investigate_index():
    print("🔍 INVESTIGATING PYTERRIER INDEX vs RUNS vs QRELS")
    print("=" * 60)
    
    try:
        # Use absolute index path directly
        absolute_index_path = str(project_root / "trec-tot-2025-pyterrier-index")
        print(f"Using index path: {absolute_index_path}")
        
        # Initialize PyTerrier
        import pyterrier as pt
        if not pt.started():
            pt.init()
        
        print("Loading PyTerrier index...")
        index = pt.IndexFactory.of(absolute_index_path)
        
        print("Loading fused run...")
        fused_df = pd.read_csv("fused_run_files/train/mistral_train_fused.txt", 
                              sep=r'\s+', header=None,
                              names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name'],
                              dtype={'qid': str, 'docno': str})
        
        print("Loading QRELs...")
        qrels_df = pd.read_csv("qrel/train-2025-qrel.txt", 
                              sep=r'\s+', header=None,
                              names=['qid', 'iteration', 'docno', 'relevance'],
                              dtype={'qid': str, 'docno': str})
        
        print("\n📊 BASIC STATISTICS:")
        print(f"PyTerrier index documents: {index.getCollectionStatistics().getNumberOfDocuments()}")
        print(f"Fused run unique docnos: {fused_df['docno'].nunique()}")
        print(f"QRELs unique docnos: {qrels_df['docno'].nunique()}")
        
        # Test document retrieval
        print("\n🔍 TESTING DOCUMENT RETRIEVAL:")
        
        meta_index = index.getMetaIndex()
        doc_index = index.getDocumentIndex()
        
        # Test some docnos from fused run
        test_docnos = fused_df['docno'].head(10).tolist()
        print(f"\nTesting docnos from fused run: {test_docnos}")
        
        found_in_index = 0
        for docno in test_docnos:
            try:
                docid = doc_index.getDocumentId(str(docno))
                if docid >= 0:
                    found_in_index += 1
                    # Try to get metadata
                    try:
                        text = meta_index.getItem("text", docid)
                        text_preview = str(text)[:100] + "..." if text else "No text"
                        print(f"  ✅ {docno} -> docid={docid}, text='{text_preview}'")
                    except:
                        print(f"  ✅ {docno} -> docid={docid}, text=ERROR")
                else:
                    print(f"  ❌ {docno} -> NOT FOUND")
            except Exception as e:
                print(f"  ❌ {docno} -> ERROR: {e}")
        
        print(f"\nFound {found_in_index}/{len(test_docnos)} docnos from fused run in index")
        
        # Test some docnos from QRELs
        test_qrel_docnos = qrels_df['docno'].head(10).tolist()
        print(f"\nTesting docnos from QRELs: {test_qrel_docnos}")
        
        found_qrel_in_index = 0
        for docno in test_qrel_docnos:
            try:
                docid = doc_index.getDocumentId(str(docno))
                if docid >= 0:
                    found_qrel_in_index += 1
                    print(f"  ✅ {docno} -> docid={docid}")
                else:
                    print(f"  ❌ {docno} -> NOT FOUND")
            except Exception as e:
                print(f"  ❌ {docno} -> ERROR: {e}")
        
        print(f"\nFound {found_qrel_in_index}/{len(test_qrel_docnos)} docnos from QRELs in index")
        
        # Sample actual docnos from index
        print(f"\n📋 SAMPLE DOCNOS AND DOC_IDS FROM INDEX:")
        try:
            # Get some actual docnos and doc_ids from the index
            sample_docids = list(range(min(10, index.getCollectionStatistics().getNumberOfDocuments())))
            for docid in sample_docids:
                try:
                    # Get both docno and doc_id fields
                    docno_field = meta_index.getItem("docno", docid)
                    doc_id_field = meta_index.getItem("doc_id", docid)
                    
                    print(f"  docid={docid} -> docno='{docno_field}' | doc_id='{doc_id_field}'")
                    
                except Exception as e:
                    print(f"  docid={docid} -> ERROR: {e}")
        except Exception as e:
            print(f"Error getting index docnos: {e}")
        
        # CRITICAL TEST: Check if doc_id matches QRELs/fused run docnos
        print(f"\n🎯 TESTING DOC_ID FIELD AGAINST QRELS/FUSED RUN:")
        
        # Test some docnos from fused run against doc_id field
        test_docnos = fused_df['docno'].head(10).tolist()
        print(f"\nSearching fused run docnos in doc_id field: {test_docnos}")
        
        found_in_doc_id = 0
        for target_docno in test_docnos:
            found = False
            # Search through index for matching doc_id
            for docid in range(min(1000, index.getCollectionStatistics().getNumberOfDocuments())):
                try:
                    doc_id_value = meta_index.getItem("doc_id", docid)
                    if str(doc_id_value) == str(target_docno):
                        found_in_doc_id += 1
                        print(f"  ✅ FOUND: fused_docno '{target_docno}' matches doc_id at docid={docid}")
                        found = True
                        break
                except:
                    continue
            
            if not found:
                print(f"  ❌ NOT FOUND: fused_docno '{target_docno}' in doc_id field (searched first 1000)")
        
        print(f"\nFound {found_in_doc_id}/{len(test_docnos)} fused run docnos in doc_id field")
        
        # Test some docnos from QRELs against doc_id field  
        test_qrel_docnos = qrels_df['docno'].head(5).tolist()
        print(f"\nSearching QRELs docnos in doc_id field: {test_qrel_docnos}")
        
        found_qrel_in_doc_id = 0
        for target_docno in test_qrel_docnos:
            found = False
            # Search through index for matching doc_id
            for docid in range(min(1000, index.getCollectionStatistics().getNumberOfDocuments())):
                try:
                    doc_id_value = meta_index.getItem("doc_id", docid)
                    if str(doc_id_value) == str(target_docno):
                        found_qrel_in_doc_id += 1
                        print(f"  ✅ FOUND: qrel_docno '{target_docno}' matches doc_id at docid={docid}")
                        found = True
                        break
                except:
                    continue
            
            if not found:
                print(f"  ❌ NOT FOUND: qrel_docno '{target_docno}' in doc_id field (searched first 1000)")
        
        print(f"\nFound {found_qrel_in_doc_id}/{len(test_qrel_docnos)} QRELs docnos in doc_id field")
        
        # Check metadata fields available
        print(f"\n🏷️  INDEX METADATA FIELDS:")
        try:
            # This is a bit hacky but should work
            for field in ["docno", "text", "title", "body", "content", "id", "doc_id"]:
                try:
                    test_val = meta_index.getItem(field, 0)
                    print(f"  ✅ {field}: available (sample: '{str(test_val)[:50]}...')")
                except:
                    print(f"  ❌ {field}: not available")
        except Exception as e:
            print(f"Error checking metadata fields: {e}")
            
        # Check overlap between different sources
        print(f"\n🔗 DOCNO OVERLAPS:")
        fused_docnos = set(fused_df['docno'])
        qrel_docnos = set(qrels_df['docno'])
        
        overlap = fused_docnos & qrel_docnos
        print(f"Fused ∩ QRELs: {len(overlap)}/{len(fused_docnos)} ({len(overlap)/len(fused_docnos)*100:.2f}%)")
        
        if overlap:
            print(f"Example overlapping docnos: {list(overlap)[:5]}")
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_index()