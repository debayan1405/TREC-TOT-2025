#!/usr/bin/env python3
"""
Rebuild PyTerrier index with correct document ID mapping for TREC-ToT 2025
"""
import os
import sys
from pathlib import Path

# Set Java environment
os.environ['JAVA_HOME'] = '/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm'
os.environ['JVM_PATH'] = '/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm/lib/server/libjvm.so'

import pyterrier as pt
import ir_datasets

def rebuild_index_with_correct_docnos():
    """Rebuild index using ir_datasets with correct document IDs."""
    
    print("🔧 REBUILDING PYTERRIER INDEX WITH CORRECT DOCNOS")
    print("=" * 60)
    
    # Initialize PyTerrier
    if not pt.java.started():
        pt.java.init()
    
    # Load the TREC-ToT dataset
    print("📥 Loading TREC-ToT dataset...")
    try:
        # Try to load the dataset (may need ir_datasets extension)
        ir_dataset = ir_datasets.load("trec-tot/2025/train")
        print(f"✅ Dataset loaded: {ir_dataset}")
    except Exception as e:
        print(f"❌ Could not load trec-tot/2025/train dataset: {e}")
        print("💡 You may need to install the TREC-ToT ir_datasets extension")
        return False
    
    # Set up index directory
    index_directory = Path("trec-tot-2025-pyterrier-index-corrected")
    print(f"📁 Index directory: {index_directory}")
    
    # Create indexer with metadata
    print("🚀 Creating PyTerrier indexer...")
    indexer = pt.IterDictIndexer(
        str(index_directory), 
        overwrite=True, 
        meta={"docno": 100, "text": 20480}
    )
    
    # Prepare documents with correct docno mapping
    print("📝 Preparing documents with correct docno mapping...")
    def doc_generator():
        count = 0
        for doc in ir_dataset.docs_iter():
            if count % 100000 == 0:
                print(f"   Processed {count} documents...")
            
            yield {
                "docno": doc.doc_id,  # Use original doc_id as docno
                "text": doc.default_text()
            }
            count += 1
        print(f"✅ Total documents processed: {count}")
    
    # Index the documents
    print("⚙️ Indexing documents...")
    indexer.index(doc_generator())
    
    print(f"✅ Index built successfully in: {index_directory}")
    
    # Verify the index
    print("\n🔍 Verifying index...")
    index = pt.IndexFactory.of(str(index_directory))
    print(f"📊 Total documents: {index.getCollectionStatistics().getNumberOfDocuments()}")
    
    # Check sample docnos
    meta_index = index.getMetaIndex()
    print("\n📋 Sample docnos from corrected index:")
    for docid in range(min(10, index.getCollectionStatistics().getNumberOfDocuments())):
        docno = meta_index.getItem("docno", docid)
        print(f"  docid={docid} -> docno='{docno}'")
    
    return True

if __name__ == "__main__":
    rebuild_index_with_correct_docnos()
