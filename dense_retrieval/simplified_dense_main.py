"""
Simplified Dense Reranking Main Script
Uses simplified approach without PyTerrier index dependency
"""

import os
import sys
from pathlib import Path
import argparse
import time

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sparse_retrieval.config_loader import ConfigLoader
from simplified_dense_reranker import SimplifiedDenseReranker

def setup_hardware_optimization():
    """Setup hardware optimization for maximum performance"""
    
    # Maximize CPU utilization
    cpu_count = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    
    print(f"🚀 Hardware Optimization:")
    print(f"   CPU cores: {cpu_count}")
    print(f"   Threading optimized for maximum performance")

def process_all_fused_files(config, dataset_version, use_cross_encoder=True, top_k=100):
    """
    Process all fused run files for a dataset with dense reranking.
    
    Args:
        config: Configuration object
        dataset_version: Dataset to process (train, test, etc.)
        use_cross_encoder: Whether to use cross-encoder (True) or bi-encoder (False)
        top_k: Number of top documents to rerank per query
    """
    
    model_type = "cross-encoder" if use_cross_encoder else "bi-encoder"
    print(f"\\n🔄 Processing {dataset_version} dataset with {model_type}:")
    print(f"   Top-k reranking: {top_k}")
    
    # Find all fused run files
    fused_dir = Path(config.get_fusion_run_directory()) / dataset_version
    if not fused_dir.exists():
        print(f"   ❌ Fused directory not found: {fused_dir}")
        return {}
    
    fused_files = list(fused_dir.glob("*_fused.txt"))
    print(f"   Found {len(fused_files)} fused files to process")
    
    results_summary = {}
    
    for fused_file in fused_files:
        # Extract rewriter name from filename
        rewriter = fused_file.stem.replace(f"_{dataset_version}_fused", "")
        
        print(f"\\n   📋 Processing {rewriter} rewriter:")
        print(f"      Input: {fused_file.name}")
        
        try:
            start_time = time.time()
            
            # Initialize reranker
            reranker = SimplifiedDenseReranker(
                config=config,
                dataset_version=dataset_version,
                rewriter=rewriter,
                use_cross_encoder=use_cross_encoder
            )
            
            # Perform reranking
            reranked_results = reranker.rerank_documents(str(fused_file), top_k=top_k)
            
            # Save results
            output_dir = Path(config.get_dense_run_directory()) / dataset_version
            output_file = output_dir / f"{rewriter}_{dataset_version}_dense_{model_type}.txt"
            
            reranker.save_reranked_results(reranked_results, str(output_file))
            
            processing_time = time.time() - start_time
            
            results_summary[rewriter] = {
                'input_file': str(fused_file),
                'output_file': str(output_file),
                'results_count': len(reranked_results),
                'processing_time': processing_time
            }
            
            print(f"      ✓ Completed in {processing_time:.1f}s")
            print(f"      ✓ Results: {len(reranked_results)}")
            print(f"      ✓ Output: {output_file.name}")
            
        except Exception as e:
            print(f"      ❌ Error processing {rewriter}: {e}")
            results_summary[rewriter] = {'error': str(e)}
    
    return results_summary

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Simplified Dense Reranking Pipeline")
    parser.add_argument("--dataset", default="train", choices=["train", "dev-1", "dev-2", "dev-3", "test"],
                       help="Dataset to process")
    parser.add_argument("--top-k", type=int, default=100, 
                       help="Number of top documents to rerank per query")
    parser.add_argument("--cross-encoder-only", action="store_true",
                       help="Only run cross-encoder reranking")
    parser.add_argument("--bi-encoder-only", action="store_true", 
                       help="Only run bi-encoder reranking")
    
    args = parser.parse_args()
    
    print("🎯 SIMPLIFIED DENSE RERANKING PIPELINE")
    print("=" * 60)
    
    # Setup hardware optimization
    setup_hardware_optimization()
    
    # Load configuration
    try:
        config = ConfigLoader("../env.json")
        print(f"✓ Configuration loaded")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Determine which models to run
    run_cross = not args.bi_encoder_only
    run_bi = not args.cross_encoder_only
    
    cross_results = {}
    bi_results = {}
    
    # Run cross-encoder reranking
    if run_cross:
        print(f"\\n🔀 CROSS-ENCODER RERANKING")
        print("=" * 40)
        cross_results = process_all_fused_files(
            config, args.dataset, use_cross_encoder=True, top_k=args.top_k
        )
    
    # Run bi-encoder reranking  
    if run_bi:
        print(f"\\n🔀 BI-ENCODER RERANKING")
        print("=" * 40)
        bi_results = process_all_fused_files(
            config, args.dataset, use_cross_encoder=False, top_k=args.top_k
        )
    
    # Save summary
    import json
    
    results_summary = {
        'dataset': args.dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cross_encoder': cross_results,
        'bi_encoder': bi_results
    }
    
    output_file = f"simplified_dense_reranking_summary_{args.dataset}.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\\n💾 Results summary saved to: {output_file}")
    
    print(f"\\n✅ SIMPLIFIED DENSE RERANKING PIPELINE COMPLETED!")
    print(f"   Dataset: {args.dataset}")
    print(f"   Top-k: {args.top_k}")
    print(f"   Cross-encoder: {'✓' if run_cross else '✗'}")
    print(f"   Bi-encoder: {'✓' if run_bi else '✗'}")

if __name__ == "__main__":
    main()
