#!/usr/bin/env python3
"""
Test ir_datasets to understand the document ID structure for TREC-ToT
"""
import ir_datasets

def investigate_ir_datasets():
    print("🔍 INVESTIGATING IR_DATASETS FOR TREC-TOT")
    print("=" * 60)
    
    try:
        # Check available datasets
        print("📋 Available TREC-ToT datasets:")
        available_datasets = []
        for dataset_id in ir_datasets.registry._registered:
            if 'trec-tot' in dataset_id or 'trec25' in dataset_id or 'tot' in dataset_id:
                available_datasets.append(dataset_id)
        
        if available_datasets:
            for ds in available_datasets:
                print(f"  - {ds}")
        else:
            print("  ❌ No TREC-ToT datasets found in ir_datasets registry")
            print("  💡 May need to install TREC-ToT extension")
        
        # Try to load the dataset
        print("\n🔄 Attempting to load trec-tot/2025/train...")
        try:
            dataset = ir_datasets.load("trec-tot/2025/train")
            print(f"✅ Dataset loaded: {dataset}")
            
            # Sample documents
            print("\n📄 Sample documents:")
            doc_count = 0
            for doc in dataset.docs_iter():
                if doc_count >= 5:
                    break
                print(f"  doc_id='{doc.doc_id}', text='{doc.default_text()[:100]}...'")
                doc_count += 1
            
            # Sample queries
            print("\n🔍 Sample queries:")
            query_count = 0
            for query in dataset.queries_iter():
                if query_count >= 3:
                    break
                print(f"  query_id='{query.query_id}', text='{query.default_text()}'")
                query_count += 1
                
        except Exception as e:
            print(f"❌ Could not load dataset: {e}")
            print("💡 This might explain the document ID issue!")
            
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_ir_datasets()
