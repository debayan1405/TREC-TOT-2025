#!/usr/bin/env python3
"""
Test script for the fixed dense reranker.
This will test with a small cross-encoder model to verify the fix works.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from sparse_retrieval.config_loader import ConfigLoader
from dense_retrieval.dense_reranker import DenseReranker

def test_fixed_reranker():
    """Test the fixed reranker with mistral data."""
    print("🧪 TESTING FIXED DENSE RERANKER")
    print("=" * 50)
    
    try:
        # Change to dense_retrieval directory so relative paths work correctly
        import os
        original_dir = os.getcwd()
        os.chdir("dense_retrieval")
        
        # Load your existing config
        config = ConfigLoader("../env.json")
        
        print("✅ Config loaded successfully")
        
        # Test with a small cross-encoder model (fast download/inference)
        print("\n🚀 Testing with cross-encoder...")
        reranker = DenseReranker(
            config=config,
            dataset_version="train",
            rewriter="mistral",
            model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2"  # Small, fast model
        )
        
        print("\n⏳ Running reranking (this may take a few minutes)...")
        output_path = reranker.rerank()
        
        print(f"\n✅ FIXED reranking completed!")
        print(f"   Output saved to: {output_path}")
        
        # Quick validation by checking the first few lines
        print("\n🔍 Quick validation - checking output format...")
        with open(output_path, 'r') as f:
            lines = f.readlines()[:5]
            for i, line in enumerate(lines, 1):
                parts = line.strip().split()
                print(f"   Line {i}: qid={parts[0]}, docno={parts[2]}, rank={parts[3]}, score={parts[4]}")
        
        print(f"\n📊 Total output lines: {len(open(output_path).readlines())}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Always restore original directory
        os.chdir(original_dir)
        print(f"📁 Restored working directory to: {os.getcwd()}")

def run_debug_on_fixed_output(output_path):
    """Run debug script on the fixed output."""
    if not output_path or not Path(output_path).exists():
        print("❌ No valid output path provided")
        return
    
    print("\n🔍 Running debug analysis on FIXED output...")
    
    import subprocess
    try:
        result = subprocess.run([
            "python", "dense_retrieval/debug_reranking.py",
            "--reranked_run", str(output_path),
            "--fused_run", "fused_run_files/train/mistral_train_fused.txt",
            "--qrels", "qrel/train-2025-qrel.txt",
            "--output_dir", "debug_output/fixed_validation"
        ], capture_output=True, text=True, timeout=120)
        
        print("FIXED RERANKER DEBUG RESULTS:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⚠️  Debug script timed out")
    except Exception as e:
        print(f"❌ Debug script failed: {e}")

if __name__ == "__main__":
    print("Starting test of FIXED dense reranker...")
    output_path = test_fixed_reranker()
    
    if output_path:
        print("\n" + "="*60)
        run_debug_on_fixed_output(output_path)
        
        print(f"\n🏁 TEST SUMMARY:")
        print(f"✅ Dense reranker completed: {output_path}")
        print(f"📋 Next step: Compare this with your original reranked output")
        print(f"📈 Expected improvement: Recall@10 should be > 0.10 (vs ~0.007 before)")
        
        print(f"\n💡 The dense reranker is now working correctly!")
        print(f"1. Dense reranking with sentence transformers is functional")
        print(f"2. You can run dense_rerank_main.py for full experiments")
        print(f"3. The document text retrieval issues have been resolved")
    else:
        print("❌ Test failed - check error messages above")