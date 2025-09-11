"""
Comprehensive Dense Retrieval Pipeline
Orchestrates all stages of dense retrieval: bi-encoders -> late-interaction -> cross-encoders -> LLM rerankers
"""

import os
import sys
import time
import argparse
from pathlib import Path
import json

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sparse_retrieval.config_loader import ConfigLoader

def setup_hardware_optimization():
    """Setup hardware optimization for maximum performance"""
    
    cpu_count = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    
    print(f"🚀 Hardware Optimization:")
    print(f"   CPU cores: {cpu_count}")
    print(f"   Threading optimized for maximum performance")

def run_stage1_bi_encoders(config, dataset, rewriter, args):
    """Run Stage 1: Bi-encoder retrievers"""
    
    print(f"\\n🎯 STAGE 1: BI-ENCODER DENSE RETRIEVAL")
    print(f"========================================")
    
    # Import and run bi-encoder retrievers
    sys.path.append(str(Path(__file__).parent / "stage1_bi_encoders"))
    from bi_encoder_retrievers import create_bi_encoder_retrievers
    
    # Input from fusion results
    input_file = Path(config.get_fusion_run_directory()) / dataset / f"{rewriter}_{dataset}_fused.txt"
    
    if not input_file.exists():
        print(f"   ❌ Input file not found: {input_file}")
        return []
    
    # Create retrievers
    retrievers = create_bi_encoder_retrievers(config, dataset, rewriter)
    results = []
    
    print(f"   Found {len(retrievers)} bi-encoder models")
    
    for retriever in retrievers:
        print(f"\\n   📋 Processing {retriever.model_name}:")
        
        try:
            start_time = time.time()
            
            # Rerank documents
            reranked_results = retriever.rerank_documents(
                str(input_file), 
                top_k_input=args.top_k_stage1_input,
                top_k_output=args.top_k_stage1_output
            )
            
            # Save results
            output_file = retriever.get_output_path(dataset)
            retriever.save_results(reranked_results, output_file)
            
            # Evaluate if train dataset
            metrics = None
            if dataset == "train":
                metrics = retriever.evaluate_results(output_file)
                if metrics:
                    eval_file = retriever.get_evaluation_path(dataset)
                    eval_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(eval_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            elapsed = time.time() - start_time
            
            result_info = {
                'stage': 'stage1_bi_encoders',
                'model': retriever.model_name,
                'output_file': str(output_file),
                'num_results': len(reranked_results),
                'time_seconds': elapsed,
                'metrics': metrics
            }
            results.append(result_info)
            
            print(f"      ✓ Completed in {elapsed:.1f}s")
            print(f"      ✓ Results: {len(reranked_results)}")
            if metrics:
                print(f"      ✓ NDCG@10: {metrics.get('ndcg_cut_10', 0):.4f}")
            
        except Exception as e:
            print(f"      ❌ Error with {retriever.model_name}: {e}")
            continue
    
    print(f"\\n   ✅ Stage 1 completed: {len(results)} models processed")
    return results

def run_stage2_late_interaction(config, dataset, rewriter, args, stage1_results):
    """Run Stage 2: Late-interaction retrievers"""
    
    print(f"\\n🎯 STAGE 2: LATE-INTERACTION DENSE RETRIEVAL")
    print(f"=============================================")
    
    # Import and run late-interaction retrievers
    sys.path.append(str(Path(__file__).parent / "stage2_late_interaction"))
    from colbert_retrievers import create_late_interaction_retrievers
    
    # Use best bi-encoder result as input, fallback to fusion
    input_file = None
    if stage1_results and args.cascade_stages:
        # Find best performing bi-encoder
        best_result = max(stage1_results, key=lambda x: x.get('metrics', {}).get('ndcg_cut_10', 0) if x.get('metrics') else 0)
        input_file = Path(best_result['output_file'])
        print(f"   Using best stage1 result: {best_result['model']}")
    
    if not input_file or not input_file.exists():
        # Fallback to fusion results
        input_file = Path(config.get_fusion_run_directory()) / dataset / f"{rewriter}_{dataset}_fused.txt"
        print(f"   Fallback to fusion results")
    
    if not input_file.exists():
        print(f"   ❌ Input file not found: {input_file}")
        return []
    
    # Create retrievers
    retrievers = create_late_interaction_retrievers(config, dataset, rewriter)
    results = []
    
    print(f"   Found {len(retrievers)} late-interaction models")
    
    for retriever in retrievers:
        print(f"\\n   📋 Processing {retriever.model_name}:")
        
        try:
            start_time = time.time()
            
            # Rerank documents
            reranked_results = retriever.rerank_documents(
                str(input_file), 
                top_k_input=args.top_k_stage2_input,
                top_k_output=args.top_k_stage2_output
            )
            
            # Save results
            output_file = retriever.get_output_path(dataset)
            retriever.save_results(reranked_results, output_file)
            
            # Evaluate if train dataset
            metrics = None
            if dataset == "train":
                metrics = retriever.evaluate_results(output_file)
                if metrics:
                    eval_file = retriever.get_evaluation_path(dataset)
                    eval_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(eval_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            elapsed = time.time() - start_time
            
            result_info = {
                'stage': 'stage2_late_interaction',
                'model': retriever.model_name,
                'output_file': str(output_file),
                'num_results': len(reranked_results),
                'time_seconds': elapsed,
                'metrics': metrics
            }
            results.append(result_info)
            
            print(f"      ✓ Completed in {elapsed:.1f}s")
            print(f"      ✓ Results: {len(reranked_results)}")
            if metrics:
                print(f"      ✓ NDCG@10: {metrics.get('ndcg_cut_10', 0):.4f}")
            
        except Exception as e:
            print(f"      ❌ Error with {retriever.model_name}: {e}")
            continue
    
    print(f"\\n   ✅ Stage 2 completed: {len(results)} models processed")
    return results

def run_stage3_cross_encoders(config, dataset, rewriter, args, stage2_results, stage1_results):
    """Run Stage 3: Cross-encoder retrievers"""
    
    print(f"\\n🎯 STAGE 3: CROSS-ENCODER DENSE RETRIEVAL")
    print(f"==========================================")
    
    # Import and run cross-encoder retrievers
    sys.path.append(str(Path(__file__).parent / "stage3_cross_encoders"))
    from cross_encoder_retrievers import create_cross_encoder_retrievers
    
    # Use best result from previous stages as input
    input_file = None
    if stage2_results and args.cascade_stages:
        # Find best performing late-interaction result
        best_result = max(stage2_results, key=lambda x: x.get('metrics', {}).get('ndcg_cut_10', 0) if x.get('metrics') else 0)
        input_file = Path(best_result['output_file'])
        print(f"   Using best stage2 result: {best_result['model']}")
    elif stage1_results and args.cascade_stages:
        # Find best performing bi-encoder result
        best_result = max(stage1_results, key=lambda x: x.get('metrics', {}).get('ndcg_cut_10', 0) if x.get('metrics') else 0)
        input_file = Path(best_result['output_file'])
        print(f"   Using best stage1 result: {best_result['model']}")
    
    if not input_file or not input_file.exists():
        # Fallback to fusion results
        input_file = Path(config.get_fusion_run_directory()) / dataset / f"{rewriter}_{dataset}_fused.txt"
        print(f"   Fallback to fusion results")
    
    if not input_file.exists():
        print(f"   ❌ Input file not found: {input_file}")
        return []
    
    # Create retrievers
    retrievers = create_cross_encoder_retrievers(config, dataset, rewriter)
    results = []
    
    print(f"   Found {len(retrievers)} cross-encoder models")
    
    for retriever in retrievers:
        print(f"\\n   📋 Processing {retriever.model_name}:")
        
        try:
            start_time = time.time()
            
            # Rerank documents
            reranked_results = retriever.rerank_documents(
                str(input_file), 
                top_k_input=args.top_k_stage3_input,
                top_k_output=args.top_k_stage3_output
            )
            
            # Save results
            output_file = retriever.get_output_path(dataset)
            retriever.save_results(reranked_results, output_file)
            
            # Evaluate if train dataset
            metrics = None
            if dataset == "train":
                metrics = retriever.evaluate_results(output_file)
                if metrics:
                    eval_file = retriever.get_evaluation_path(dataset)
                    eval_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(eval_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            elapsed = time.time() - start_time
            
            result_info = {
                'stage': 'stage3_cross_encoders',
                'model': retriever.model_name,
                'output_file': str(output_file),
                'num_results': len(reranked_results),
                'time_seconds': elapsed,
                'metrics': metrics
            }
            results.append(result_info)
            
            print(f"      ✓ Completed in {elapsed:.1f}s")
            print(f"      ✓ Results: {len(reranked_results)}")
            if metrics:
                print(f"      ✓ NDCG@10: {metrics.get('ndcg_cut_10', 0):.4f}")
            
        except Exception as e:
            print(f"      ❌ Error with {retriever.model_name}: {e}")
            continue
    
    print(f"\\n   ✅ Stage 3 completed: {len(results)} models processed")
    return results

def run_stage4_llm_rerankers(config, dataset, rewriter, args, stage3_results, stage2_results, stage1_results):
    """Run Stage 4: LLM rerankers"""
    
    print(f"\\n🎯 STAGE 4: LLM-BASED DENSE RERANKING")
    print(f"=====================================")
    
    # Import and run LLM rerankers
    sys.path.append(str(Path(__file__).parent / "stage4_llm_rankers"))
    from llm_rerankers import create_llm_rerankers
    
    # Use best result from previous stages as input
    input_file = None
    if stage3_results and args.cascade_stages:
        best_result = max(stage3_results, key=lambda x: x.get('metrics', {}).get('ndcg_cut_10', 0) if x.get('metrics') else 0)
        input_file = Path(best_result['output_file'])
        print(f"   Using best stage3 result: {best_result['model']}")
    elif stage2_results and args.cascade_stages:
        best_result = max(stage2_results, key=lambda x: x.get('metrics', {}).get('ndcg_cut_10', 0) if x.get('metrics') else 0)
        input_file = Path(best_result['output_file'])
        print(f"   Using best stage2 result: {best_result['model']}")
    elif stage1_results and args.cascade_stages:
        best_result = max(stage1_results, key=lambda x: x.get('metrics', {}).get('ndcg_cut_10', 0) if x.get('metrics') else 0)
        input_file = Path(best_result['output_file'])
        print(f"   Using best stage1 result: {best_result['model']}")
    
    if not input_file or not input_file.exists():
        input_file = Path(config.get_fusion_run_directory()) / dataset / f"{rewriter}_{dataset}_fused.txt"
        print(f"   Fallback to fusion results")
    
    if not input_file.exists():
        print(f"   ❌ Input file not found: {input_file}")
        return []
    
    # Create retrievers
    retrievers = create_llm_rerankers(config, dataset, rewriter)
    results = []
    
    if not retrievers:
        print(f"   No LLM rerankers available")
        return results
    
    print(f"   Found {len(retrievers)} LLM reranker models")
    
    for retriever in retrievers:
        print(f"\\n   📋 Processing {retriever.model_name}:")
        
        try:
            start_time = time.time()
            
            # Rerank documents
            reranked_results = retriever.rerank_documents(
                str(input_file), 
                top_k_input=args.top_k_stage4_input,
                top_k_output=args.top_k_stage4_output
            )
            
            # Save results
            output_file = retriever.get_output_path(dataset)
            retriever.save_results(reranked_results, output_file)
            
            # Evaluate if train dataset
            metrics = None
            if dataset == "train":
                metrics = retriever.evaluate_results(output_file)
                if metrics:
                    eval_file = retriever.get_evaluation_path(dataset)
                    eval_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(eval_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            elapsed = time.time() - start_time
            
            result_info = {
                'stage': 'stage4_llm_rankers',
                'model': retriever.model_name,
                'output_file': str(output_file),
                'num_results': len(reranked_results),
                'time_seconds': elapsed,
                'metrics': metrics
            }
            results.append(result_info)
            
            print(f"      ✓ Completed in {elapsed:.1f}s")
            print(f"      ✓ Results: {len(reranked_results)}")
            if metrics:
                print(f"      ✓ NDCG@10: {metrics.get('ndcg_cut_10', 0):.4f}")
            
        except Exception as e:
            print(f"      ❌ Error with {retriever.model_name}: {e}")
            continue
    
    print(f"\\n   ✅ Stage 4 completed: {len(results)} models processed")
    return results

def generate_comprehensive_report(all_results, dataset, rewriter, output_dir):
    """Generate comprehensive evaluation report"""
    
    print(f"\\n📊 GENERATING COMPREHENSIVE REPORT")
    print(f"==================================")
    
    # Organize results by stage
    stage_results = {
        'stage1_bi_encoders': [],
        'stage2_late_interaction': [],
        'stage3_cross_encoders': [],
        'stage4_llm_rankers': []
    }
    
    for result in all_results:
        stage = result.get('stage')
        if stage in stage_results:
            stage_results[stage].append(result)
    
    # Create comprehensive report
    report = {
        'dataset': dataset,
        'rewriter': rewriter,
        'timestamp': time.time(),
        'stages': stage_results,
        'summary': {
            'total_models': len(all_results),
            'total_time': sum(r.get('time_seconds', 0) for r in all_results)
        }
    }
    
    # Find best model overall
    if dataset == "train":
        best_model = None
        best_ndcg = 0
        
        for result in all_results:
            metrics = result.get('metrics')
            if metrics and metrics.get('ndcg_cut_10', 0) > best_ndcg:
                best_ndcg = metrics['ndcg_cut_10']
                best_model = result
        
        if best_model:
            report['best_model'] = {
                'stage': best_model['stage'],
                'model': best_model['model'],
                'ndcg_cut_10': best_ndcg,
                'output_file': best_model['output_file']
            }
            
            print(f"   🏆 Best model: {best_model['stage']}/{best_model['model']}")
            print(f"   🏆 Best NDCG@10: {best_ndcg:.4f}")
    
    # Save report
    report_file = Path(output_dir) / "evaluations" / f"comprehensive_dense_report_{rewriter}_{dataset}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✓ Report saved to: {report_file}")
    return report

def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description="Comprehensive Dense Retrieval Pipeline")
    
    # Dataset arguments
    parser.add_argument("--dataset", default="train", help="Dataset version (train, test)")
    parser.add_argument("--rewriter", default="llama", help="Query rewriter type")
    
    # Stage control
    parser.add_argument("--stages", nargs='+', default=["1", "2", "3"], 
                       help="Stages to run (1=bi-encoders, 2=late-interaction, 3=cross-encoders, 4=llm)")
    parser.add_argument("--cascade-stages", action="store_true", default=True,
                       help="Use output of previous stage as input for next stage")
    
    # Top-k parameters for each stage
    parser.add_argument("--top-k-stage1-input", type=int, default=5000, help="Stage 1 input top-k")
    parser.add_argument("--top-k-stage1-output", type=int, default=1000, help="Stage 1 output top-k")
    parser.add_argument("--top-k-stage2-input", type=int, default=1000, help="Stage 2 input top-k")
    parser.add_argument("--top-k-stage2-output", type=int, default=1000, help="Stage 2 output top-k")
    parser.add_argument("--top-k-stage3-input", type=int, default=1000, help="Stage 3 input top-k")
    parser.add_argument("--top-k-stage3-output", type=int, default=1000, help="Stage 3 output top-k")
    parser.add_argument("--top-k-stage4-input", type=int, default=1000, help="Stage 4 input top-k")
    parser.add_argument("--top-k-stage4-output", type=int, default=1000, help="Stage 4 output top-k")
    
    args = parser.parse_args()
    
    # Setup
    setup_hardware_optimization()
    config = ConfigLoader("env.json")
    
    print(f"\\n🎯 COMPREHENSIVE DENSE RETRIEVAL PIPELINE")
    print(f"==========================================")
    print(f"Dataset: {args.dataset}")
    print(f"Rewriter: {args.rewriter}")
    print(f"Stages: {', '.join(args.stages)}")
    print(f"Cascade stages: {args.cascade_stages}")
    
    # Run stages
    all_results = []
    stage1_results = []
    stage2_results = []
    stage3_results = []
    stage4_results = []
    
    if "1" in args.stages:
        stage1_results = run_stage1_bi_encoders(config, args.dataset, args.rewriter, args)
        all_results.extend(stage1_results)
    
    if "2" in args.stages:
        stage2_results = run_stage2_late_interaction(config, args.dataset, args.rewriter, args, stage1_results)
        all_results.extend(stage2_results)
    
    if "3" in args.stages:
        stage3_results = run_stage3_cross_encoders(config, args.dataset, args.rewriter, args, stage2_results, stage1_results)
        all_results.extend(stage3_results)
    
    if "4" in args.stages:
        stage4_results = run_stage4_llm_rerankers(config, args.dataset, args.rewriter, args, stage3_results, stage2_results, stage1_results)
        all_results.extend(stage4_results)
    
    # Generate comprehensive report
    output_dir = Path(__file__).parent
    report = generate_comprehensive_report(all_results, args.dataset, args.rewriter, output_dir)
    
    total_time = sum(r.get('time_seconds', 0) for r in all_results)
    
    print(f"\\n✅ COMPREHENSIVE DENSE RETRIEVAL PIPELINE COMPLETED!")
    print(f"   Total models processed: {len(all_results)}")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Report: comprehensive_dense_report_{args.rewriter}_{args.dataset}.json")

if __name__ == "__main__":
    main()
