"""
Enhanced Dense Reranking Main Script
Processes all fused run files with both cross-encoder and bi-encoder reranking.
Optimized for high-end hardware (700+ GB RAM, 2x A6000 GPUs).
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
from enhanced_dense_reranker import EnhancedDenseReranker

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
            reranker = EnhancedDenseReranker(
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

def run_evaluation(config, dataset_version, model_type):
    """
    Run evaluation on dense reranked results.
    
    Args:
        config: Configuration object
        dataset_version: Dataset version
        model_type: "cross-encoder" or "bi-encoder"
    """
    
    print(f"\\n📊 Evaluating {model_type} results for {dataset_version}:")
    
    try:
        # Import evaluation utilities
        from sparse_retrieval.data_loader import DataLoader
        import pandas as pd
        import pyterrier as pt
        
        data_loader = DataLoader(config)
        
        # Load qrels if available
        qrels_df = data_loader.load_qrels(dataset_version)
        if qrels_df is None:
            print(f"   ⚠️  No QRELs available for {dataset_version}, skipping evaluation")
            return {}
        
        print(f"   Loaded {len(qrels_df)} relevance judgments")
        
        # Find dense run files
        dense_dir = Path(config.get_dense_run_directory()) / dataset_version
        dense_files = list(dense_dir.glob(f"*_dense_{model_type}.txt"))
        
        if not dense_files:
            print(f"   ❌ No dense run files found for {model_type}")
            return {}
        
        print(f"   Found {len(dense_files)} dense run files")
        
        evaluation_results = {}
        
        for dense_file in dense_files:
            rewriter = dense_file.stem.replace(f"_{dataset_version}_dense_{model_type}", "")
            
            try:
                # Load run results
                run_df = pd.read_csv(
                    dense_file,
                    sep='\\t',
                    header=None,
                    names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name']
                )
                
                # Convert to PyTerrier format
                run_df['qid'] = run_df['qid'].astype(str)
                run_df['docno'] = run_df['docno'].astype(str)
                
                # Evaluate
                eval_metrics = config.get_evaluation_metrics()
                evaluation = pt.Evaluate(
                    run_df, 
                    qrels_df, 
                    metrics=eval_metrics,
                    perquery=False
                )
                
                evaluation_results[rewriter] = evaluation
                
                print(f"   ✓ {rewriter}: {evaluation}")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {rewriter}: {e}")
                evaluation_results[rewriter] = {'error': str(e)}
        
        return evaluation_results
        
    except Exception as e:
        print(f"   ❌ Evaluation error: {e}")
        return {}

def save_comprehensive_results(dataset_version, cross_results, bi_results, cross_eval, bi_eval):
    """Save comprehensive results and analysis"""
    
    import json
    
    results_summary = {
        'dataset': dataset_version,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cross_encoder': {
            'processing_results': cross_results,
            'evaluation_results': cross_eval
        },
        'bi_encoder': {
            'processing_results': bi_results,
            'evaluation_results': bi_eval
        }
    }
    
    # Save summary
    output_file = f"dense_reranking_summary_{dataset_version}.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\\n💾 Comprehensive results saved to: {output_file}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Enhanced Dense Reranking Pipeline")
    parser.add_argument("--dataset", default="train", choices=["train", "dev-1", "dev-2", "dev-3", "test"],
                       help="Dataset to process")
    parser.add_argument("--top-k", type=int, default=100, 
                       help="Number of top documents to rerank per query")
    parser.add_argument("--cross-encoder-only", action="store_true",
                       help="Only run cross-encoder reranking")
    parser.add_argument("--bi-encoder-only", action="store_true", 
                       help="Only run bi-encoder reranking")
    
    args = parser.parse_args()
    
    print("🎯 ENHANCED DENSE RERANKING PIPELINE")
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
    cross_eval = {}
    bi_eval = {}
    
    # Run cross-encoder reranking
    if run_cross:
        print(f"\\n🔀 CROSS-ENCODER RERANKING")
        print("=" * 40)
        cross_results = process_all_fused_files(
            config, args.dataset, use_cross_encoder=True, top_k=args.top_k
        )
        cross_eval = run_evaluation(config, args.dataset, "cross-encoder")
    
    # Run bi-encoder reranking  
    if run_bi:
        print(f"\\n🔀 BI-ENCODER RERANKING")
        print("=" * 40)
        bi_results = process_all_fused_files(
            config, args.dataset, use_cross_encoder=False, top_k=args.top_k
        )
        bi_eval = run_evaluation(config, args.dataset, "bi-encoder")
    
    # Save comprehensive results
    save_comprehensive_results(args.dataset, cross_results, bi_results, cross_eval, bi_eval)
    
    print(f"\\n✅ DENSE RERANKING PIPELINE COMPLETED!")
    print(f"   Dataset: {args.dataset}")
    print(f"   Top-k: {args.top_k}")
    print(f"   Cross-encoder: {'✓' if run_cross else '✗'}")
    print(f"   Bi-encoder: {'✓' if run_bi else '✗'}")

if __name__ == "__main__":
    main()
