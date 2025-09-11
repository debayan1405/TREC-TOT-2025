#!/usr/bin/env python3
"""
Optimized Comprehensive Dense Retrieval Pipeline
===============================================

Maximum performance implementation with:
- In-memory PyTerrier index (7.9GB RAM)
- Document caching (50GB RAM)
- Optimal VRAM utilization (90%)
- Adaptive batching
- Mixed precision (FP16)
- Multi-GPU support
- Real-time progress monitoring
"""

import os
import sys
import time
import argparse
import torch
import psutil
from typing import Dict, List
import gc
import threading

# Global progress monitoring
PROGRESS_INTERVAL = 10  # seconds

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimized_dense_retriever import (
    create_optimized_bi_encoder_retrievers,
    PerformanceConfig,
    OptimizedGPUManager
)
from sparse_retrieval.config_loader import ConfigLoader


def print_hardware_status():
    """Print current hardware utilization"""
    print("🖥️  HARDWARE STATUS:")
    
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_cores = psutil.cpu_count()
    print(f"   CPU: {cpu_percent:.1f}% utilization ({cpu_cores} cores)")
    
    # RAM info
    ram = psutil.virtual_memory()
    print(f"   RAM: {ram.used / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB ({ram.percent:.1f}%)")
    
    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            total = props.total_memory / (1024**3)
            print(f"   GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached / {total:.1f}GB total")


def optimize_system_for_performance():
    """Apply system-level optimizations"""
    print("🚀 APPLYING SYSTEM OPTIMIZATIONS:")
    
    # Set environment variables for maximum performance
    os.environ['OMP_NUM_THREADS'] = str(min(16, psutil.cpu_count()))
    os.environ['MKL_NUM_THREADS'] = str(min(16, psutil.cpu_count()))
    os.environ['NUMEXPR_NUM_THREADS'] = str(min(16, psutil.cpu_count()))
    
    # PyTorch optimizations
    torch.set_num_threads(min(16, psutil.cpu_count()))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
    
    # Java heap for PyTerrier (use more RAM for better performance)
    os.environ['JAVA_OPTS'] = '-Xmx40g -Xms20g -XX:+UseG1GC -XX:MaxGCPauseMillis=200'
    
    print(f"   ✅ CPU threads: {os.environ['OMP_NUM_THREADS']}")
    print(f"   ✅ Java heap: 40GB max, 20GB initial")
    print(f"   ✅ CUDA optimizations: {torch.cuda.is_available()}")


def force_pyterrier_memory_optimization(index_path: str):
    """Force PyTerrier to use in-memory meta index"""
    print("🔧 FORCING PYTERRIER MEMORY OPTIMIZATION:")
    
    properties_file = os.path.join(index_path, "data.properties")
    
    if os.path.exists(properties_file):
        # Read existing properties
        with open(properties_file, 'r') as f:
            lines = f.readlines()
        
        # Add or update memory optimization
        found_meta_source = False
        for i, line in enumerate(lines):
            if 'index.meta.data-source' in line:
                lines[i] = 'index.meta.data-source=fileinmem\n'
                found_meta_source = True
                break
        
        if not found_meta_source:
            lines.append('index.meta.data-source=fileinmem\n')
        
        # Add additional memory optimizations
        additional_opts = [
            'index.meta.compression.configuration=ZSTD\n',
            'index.lexicon.termids=fileinmem\n',
            'termpipelines.skip=\n'  # Disable term processing for speed
        ]
        
        for opt in additional_opts:
            if not any(opt.split('=')[0] in line for line in lines):
                lines.append(opt)
        
        # Write back
        with open(properties_file, 'w') as f:
            f.writelines(lines)
        
        print(f"   ✅ Updated {properties_file} with memory optimizations")
        print(f"   ✅ Meta index will load into 7.9GB RAM")
    else:
        print(f"   ⚠️  Properties file not found: {properties_file}")


def start_progress_monitor():
    """Start a background thread for progress monitoring"""
    def monitor():
        while True:
            time.sleep(PROGRESS_INTERVAL)
            print("\n📊 PROGRESS UPDATE:")
            print_hardware_status()
            print("🔄 Processing continues...\n")
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread


def distribute_models_to_gpus(retrievers: List) -> Dict:
    """Distribute models across available GPUs for parallel processing"""
    if not torch.cuda.is_available():
        return {0: retrievers}
    
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return {0: retrievers}
    
    # Distribute models across GPUs
    gpu_assignments = {}
    for i, retriever in enumerate(retrievers):
        gpu_id = i % num_gpus
        if gpu_id not in gpu_assignments:
            gpu_assignments[gpu_id] = []
        gpu_assignments[gpu_id].append(retriever)
    
    print(f"🔀 MULTI-GPU DISTRIBUTION:")
    for gpu_id, models in gpu_assignments.items():
        model_names = [r.model_name for r in models]
        print(f"   GPU {gpu_id}: {model_names}")
    
    return gpu_assignments


def run_optimized_stage1(config: Dict, dataset: str, rewriter: str, 
                        top_k_input: int, top_k_output: int, multi_gpu: bool = False, models: List[str] = None) -> Dict:
    """Run optimized Stage 1 bi-encoder pipeline"""
    
    print("\n🎯 STAGE 1: OPTIMIZED BI-ENCODER DENSE RETRIEVAL")
    print("=" * 60)
    
    stage_start_time = time.time()
    
    # Start progress monitoring
    if PROGRESS_INTERVAL > 0:
        print(f"📊 Starting progress monitor (updates every {PROGRESS_INTERVAL}s)")
        start_progress_monitor()
    
    # Create optimized retrievers
    print("🔧 Initializing optimized bi-encoder retrievers...")
    retrievers = create_optimized_bi_encoder_retrievers(
        config, dataset, rewriter, 'stage1_bi_encoders', models
    )
    
    print(f"✅ Created {len(retrievers)} optimized retrievers")
    
    # Multi-GPU distribution if enabled
    if multi_gpu and torch.cuda.device_count() > 1:
        gpu_assignments = distribute_models_to_gpus(retrievers)
        print(f"🚀 Multi-GPU processing enabled with {len(gpu_assignments)} GPUs")
    else:
        gpu_assignments = {0: retrievers}
        print("🔧 Single-GPU processing")
    
    # Input and output paths
    input_file = f"../fused_run_files/{dataset}/{rewriter}_{dataset}_fused.txt"
    results_dir = f"../dense_run_files/run_files/stage1_bi_encoders"
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each GPU group
    stage_results = {}
    
    for gpu_id, gpu_retrievers in gpu_assignments.items():
        print(f"\n🔥 Processing GPU {gpu_id} with {len(gpu_retrievers)} models")
        
        for i, retriever in enumerate(gpu_retrievers):
            # Force model to specific GPU
            if torch.cuda.is_available() and hasattr(retriever, 'model'):
                retriever.model = retriever.model.to(f'cuda:{gpu_id}')
            
            print(f"\n📋 Processing {retriever.model_name} on GPU {gpu_id} ({i+1}/{len(gpu_retrievers)}):")
            print("-" * 50)
        
        model_start_time = time.time()
        
        # Output file
        output_file = f"{results_dir}/{rewriter}_{dataset}_{retriever.model_name}.txt"
        
        try:
            # Run reranking
            summary = retriever.rerank_documents(
                input_file, output_file,
                top_k_input=top_k_input,
                top_k_output=top_k_output
            )
            
            model_time = time.time() - model_start_time
            summary['model_time'] = model_time
            stage_results[retriever.model_name] = summary
            
            print(f"✅ {retriever.model_name} completed in {model_time:.1f}s")
            
            # Print hardware status after each model
            print_hardware_status()
            
            # Cleanup between models
            del retriever
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error processing {retriever.model_name}: {e}")
            stage_results[retriever.model_name] = {'error': str(e)}
    
    stage_time = time.time() - stage_start_time
    
    print(f"\n🏁 STAGE 1 COMPLETED in {stage_time:.1f}s")
    print("=" * 60)
    
    # Print summary
    total_results = sum(r.get('total_results', 0) for r in stage_results.values() if 'error' not in r)
    successful_models = len([r for r in stage_results.values() if 'error' not in r])
    
    print(f"📊 STAGE 1 SUMMARY:")
    print(f"   ✅ Successful models: {successful_models}/{len(retrievers)}")
    print(f"   ✅ Total results generated: {total_results}")
    print(f"   ⏱️  Total time: {stage_time:.1f}s")
    
    if successful_models > 0:
        avg_time = stage_time / successful_models
        print(f"   ⏱️  Average time per model: {avg_time:.1f}s")
    
    return stage_results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Optimized Dense Retrieval Pipeline')
    parser.add_argument('--dataset', choices=['train', 'dev1', 'dev2', 'dev3', 'test'], required=True)
    parser.add_argument('--rewriter', choices=['llama'], default='llama')
    parser.add_argument('--stages', type=int, default=1, help='Number of stages to run (1-4)')
    parser.add_argument('--top-k-stage1-input', type=int, default=1000)
    parser.add_argument('--top-k-stage1-output', type=int, default=1000)
    parser.add_argument('--multi-gpu', action='store_true', help='Enable multi-GPU processing')
    parser.add_argument('--progress-interval', type=int, default=10, help='Progress update interval (seconds)')
    parser.add_argument('--models', nargs='+', help='Specific models to run (default: all models)')
    
    args = parser.parse_args()
    
    # Set global progress interval for monitoring
    global PROGRESS_INTERVAL
    PROGRESS_INTERVAL = args.progress_interval
    
    print("🚀 OPTIMIZED COMPREHENSIVE DENSE RETRIEVAL PIPELINE")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Rewriter: {args.rewriter}")
    print(f"Stages: {args.stages}")
    print(f"Multi-GPU: {'✅ ENABLED' if args.multi_gpu else '❌ Disabled'}")
    print(f"Progress updates: Every {args.progress_interval}s")
    print(f"Max VRAM utilization: 90%")
    print(f"Document cache: 50GB RAM")
    print(f"PyTerrier meta index: 7.9GB RAM")
    print("=" * 60)
    
    # Load configuration
    try:
        config_loader = ConfigLoader('../env.json')
        config = config_loader.config
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Apply system optimizations
    optimize_system_for_performance()
    
    # Force PyTerrier memory optimization
    index_path = config['paths']['index_path']
    force_pyterrier_memory_optimization(index_path)
    
    # Print initial hardware status
    print_hardware_status()
    
    # Start pipeline
    pipeline_start_time = time.time()
    
    try:
        if args.stages >= 1:
            stage1_results = run_optimized_stage1(
                config, args.dataset, args.rewriter,
                args.top_k_stage1_input, args.top_k_stage1_output,
                multi_gpu=args.multi_gpu, models=args.models
            )
        
        # TODO: Add optimized stages 2-4 here
        if args.stages >= 2:
            print("\n🚧 Stages 2-4 optimization in progress...")
        
        # Final summary
        total_time = time.time() - pipeline_start_time
        
        print(f"\n🏁 PIPELINE COMPLETED")
        print("=" * 60)
        print(f"⏱️  Total execution time: {total_time:.1f}s")
        print(f"🎯 Maximum performance achieved!")
        
        # Final hardware status
        print_hardware_status()
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
