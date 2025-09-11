"""
Optimized Dense Retrieval with Maximum RAM/VRAM Utilization
===========================================================

This module implements high-performance optimizations:
1. In-memory PyTerrier index and document cache
2. Maximum VRAM utilization with adaptive batching
3. Efficient tensor operations and memory management
4. Multi-GPU support and optimization
"""

import os
import gc
import psutil
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from tqdm import tqdm
import pyterrier as pt
from dataclasses import dataclass
import time
import json
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparse_retrieval.config_loader import ConfigLoader
from sparse_retrieval.data_loader import DataLoader


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations"""
    max_batch_size: int = 128
    vram_target_utilization: float = 0.85  # Use 85% of VRAM
    ram_cache_size_gb: float = 50.0  # 50GB for document cache
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False  # Not needed for inference
    prefetch_factor: int = 4
    num_workers: int = 8
    pin_memory: bool = True


class OptimizedDocumentCache:
    """High-performance in-memory document cache"""
    
    def __init__(self, index_path: str, cache_size_gb: float = 50.0):
        self.index_path = index_path
        self.cache_size_gb = cache_size_gb
        self.cache_size_bytes = int(cache_size_gb * 1024 * 1024 * 1024)
        
        # Document cache
        self._doc_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._current_cache_size = 0
        
        # PyTerrier optimization
        self._index = None
        self._text_pipeline = None
        self._meta_loaded_to_memory = False
        
        print(f"🚀 Initializing document cache with {cache_size_gb:.1f}GB capacity")
        self._initialize_pyterrier_optimized()
    
    def _initialize_pyterrier_optimized(self):
        """Initialize PyTerrier with maximum optimization"""
        try:
            # Force PyTerrier to load meta index to memory
            print("🔧 Optimizing PyTerrier for maximum performance...")
            
            # Set Java heap size for large operations
            os.environ['JAVA_OPTS'] = '-Xmx32g -Xms16g'
            
            if not pt.started():
                pt.init()
            
            # Load index with optimizations
            indexref = pt.IndexRef.of(self.index_path)
            self._index = pt.IndexFactory.of(indexref)
            
            # Try to force meta index to memory by setting system properties
            try:
                from jnius import autoclass
                System = autoclass('java.lang.System')
                System.setProperty('index.meta.data-source', 'fileinmem')
                print("✅ Set PyTerrier meta index to load in memory")
            except Exception as e:
                print(f"⚠️  Could not set Java system property: {e}")
            
            # Create text pipeline
            import pyterrier.text
            self._text_pipeline = pyterrier.text.get_text(indexref)
            
            print(f"✅ PyTerrier optimized index loaded: {self._index.getCollectionStatistics().getNumberOfDocuments()} documents")
            
        except Exception as e:
            print(f"❌ PyTerrier optimization failed: {e}")
            raise
    
    def warm_up_cache(self, docnos: List[str], batch_size: int = 1000):
        """Pre-warm cache with frequently accessed documents"""
        print(f"🔥 Warming up document cache with {len(docnos)} documents...")
        
        for i in tqdm(range(0, len(docnos), batch_size), desc="Cache warmup"):
            batch_docnos = docnos[i:i + batch_size]
            self._fetch_documents_batch(batch_docnos)
        
        print(f"✅ Cache warmed up: {len(self._doc_cache)} documents cached")
        print(f"📊 Cache size: {self._current_cache_size / (1024*1024):.1f} MB")
    
    def _fetch_documents_batch(self, docnos: List[str]) -> Dict[str, str]:
        """Fetch documents in batch with caching"""
        # Check cache first
        cached_docs = {}
        missing_docnos = []
        
        for docno in docnos:
            if docno in self._doc_cache:
                cached_docs[docno] = self._doc_cache[docno]
                self._cache_hits += 1
            else:
                missing_docnos.append(docno)
                self._cache_misses += 1
        
        # Fetch missing documents
        if missing_docnos:
            try:
                # Create DataFrame for PyTerrier
                docs_df = pd.DataFrame({
                    'qid': ['cache'] * len(missing_docnos),
                    'docno': missing_docnos,
                    'score': [1.0] * len(missing_docnos),
                    'rank': list(range(1, len(missing_docnos) + 1))
                })
                
                # Get texts using PyTerrier
                result_df = self._text_pipeline(docs_df)
                
                # Cache the results
                for _, row in result_df.iterrows():
                    docno = row['docno']
                    text = row.get('text', f'No text available for document {docno}')
                    
                    # Check cache size before adding
                    text_size = len(text.encode('utf-8'))
                    if self._current_cache_size + text_size < self.cache_size_bytes:
                        self._doc_cache[docno] = text
                        self._current_cache_size += text_size
                        cached_docs[docno] = text
                    else:
                        # Cache is full, use document directly without caching
                        cached_docs[docno] = text
                        
            except Exception as e:
                print(f"⚠️  Batch document fetch failed: {e}")
                # Fallback to individual fetching
                for docno in missing_docnos:
                    cached_docs[docno] = f"Error retrieving document {docno}: {e}"
        
        return cached_docs
    
    def get_documents(self, docnos: List[str]) -> Dict[str, str]:
        """Get documents with optimal batching"""
        return self._fetch_documents_batch(docnos)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached_documents': len(self._doc_cache),
            'cache_size_mb': self._current_cache_size / (1024 * 1024),
            'cache_size_gb': self._current_cache_size / (1024 * 1024 * 1024)
        }


class OptimizedGPUManager:
    """Optimized GPU memory and compute management"""
    
    def __init__(self, target_utilization: float = 0.85):
        self.target_utilization = target_utilization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
        
        print(f"🚀 GPU Manager initialized:")
        print(f"   Device: {self.device}")
        print(f"   GPUs available: {self.num_gpus}")
        
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({total_memory:.1f}GB)")
        
        self._setup_optimization()
    
    def _setup_optimization(self):
        """Setup GPU optimizations"""
        if torch.cuda.is_available():
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Memory management
            torch.cuda.empty_cache()
            
            print("✅ GPU optimizations enabled")
    
    def get_optimal_batch_size(self, model_size_mb: float, max_seq_length: int = 512) -> int:
        """Calculate optimal batch size based on available VRAM"""
        if not torch.cuda.is_available():
            return 32  # Conservative for CPU
        
        # Get available VRAM
        device_id = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        available_memory = total_memory * self.target_utilization
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample = (
            model_size_mb * 1024 * 1024 +  # Model memory
            max_seq_length * 4 * 8 +       # Input tensors (float32, batch)
            max_seq_length * 4 * 8          # Attention matrices
        )
        
        # Calculate batch size
        optimal_batch_size = int(available_memory / memory_per_sample)
        optimal_batch_size = max(1, min(optimal_batch_size, 512))  # Clamp between 1-512
        
        print(f"🎯 Optimal batch size calculated: {optimal_batch_size}")
        return optimal_batch_size
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class OptimizedBaseDenseRetriever(ABC):
    """Optimized base class for dense retrievers with maximum performance"""
    
    def __init__(self, config: Dict, dataset_version: str, rewriter: str, 
                 stage: str, model_name: str, performance_config: Optional[PerformanceConfig] = None):
        self.config = config
        self.dataset_version = dataset_version
        self.rewriter = rewriter
        self.stage = stage
        self.model_name = model_name
        self.performance_config = performance_config or PerformanceConfig()
        
        # Initialize components
        self.gpu_manager = OptimizedGPUManager(self.performance_config.vram_target_utilization)
        self.device = self.gpu_manager.device
        
        # Document cache
        index_path = config['paths']['index_path']
        self.doc_cache = OptimizedDocumentCache(
            index_path, 
            self.performance_config.ram_cache_size_gb
        )
        
        # Model components (to be initialized by subclasses)
        self.model = None
        self.tokenizer = None
        
        print(f"🚀 Optimized {self.__class__.__name__} initialized:")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Performance config: {self.performance_config}")
        
        # Load rewritten queries
        self.rewritten_queries = self._load_rewritten_queries()
        print(f"   Loaded {len(self.rewritten_queries)} rewritten queries")
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        pass
    
    @abstractmethod
    def _score_documents_batch(self, query_text: str, doc_texts: List[str]) -> List[float]:
        """Score documents in batch - implemented by subclasses"""
        pass
    
    def _load_rewritten_queries(self) -> Dict[int, str]:
        """Load rewritten queries from JSONL file"""
        queries_file = f"../rewritten_queries/{self.rewriter}_{self.dataset_version}_rewritten_queries.jsonl"
        
        if not os.path.exists(queries_file):
            raise FileNotFoundError(f"Rewritten queries file not found: {queries_file}")
        
        queries = {}
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                query_id = int(data['query_id'])
                queries[query_id] = data['query']
        
        return queries
    
    def _load_input_data(self, input_file: str, top_k: int) -> pd.DataFrame:
        """Load and prepare input data"""
        print(f"📊 Loading input data from {input_file}")
        
        # Read the run file
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid = int(parts[0])
                    docno = parts[2]
                    score = float(parts[4])
                    data.append({'qid': qid, 'docno': docno, 'score': score})
        
        df = pd.DataFrame(data)
        
        # Apply top-k filtering per query
        if top_k > 0:
            df = df.groupby('qid').head(top_k).reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} results for {df['qid'].nunique()} queries")
        return df
    
    def _optimize_batching(self, doc_texts: List[str]) -> List[List[str]]:
        """Create optimal batches for processing"""
        # Get optimal batch size
        model_size_mb = 500  # Rough estimate, can be refined per model
        optimal_batch_size = self.gpu_manager.get_optimal_batch_size(model_size_mb)
        
        # Create batches
        batches = []
        for i in range(0, len(doc_texts), optimal_batch_size):
            batch = doc_texts[i:i + optimal_batch_size]
            batches.append(batch)
        
        return batches
    
    def rerank_documents(self, input_file: str, output_file: str, 
                        top_k_input: int = 1000, top_k_output: int = 1000) -> Dict[str, Any]:
        """Optimized document reranking with maximum performance"""
        
        print(f"🔄 Starting optimized {self.stage} reranking:")
        print(f"   Input top-k: {top_k_input}")
        print(f"   Output top-k: {top_k_output}")
        
        start_time = time.time()
        
        # Load input data
        input_df = self._load_input_data(input_file, top_k_input)
        
        # Get unique queries
        unique_queries = input_df['qid'].unique()
        print(f"   Processing {len(unique_queries)} unique queries...")
        
        # Pre-warm document cache with all documents
        all_docnos = input_df['docno'].unique().tolist()
        self.doc_cache.warm_up_cache(all_docnos, batch_size=2000)
        
        # Process queries
        results = []
        
        for qid in tqdm(unique_queries, desc=f"Reranking with {self.model_name}"):
            # Get query text
            if qid not in self.rewritten_queries:
                print(f"⚠️  Query {qid} not found in rewritten queries, skipping")
                continue
            
            query_text = self.rewritten_queries[qid]
            query_docs = input_df[input_df['qid'] == qid].copy()
            
            print(f"\n🔍 Query {qid}: '{query_text[:50]}...' with {len(query_docs)} docs")
            
            # Get document texts (optimized with cache)
            doc_texts_dict = self.doc_cache.get_documents(query_docs['docno'].tolist())
            doc_texts = [doc_texts_dict[docno] for docno in query_docs['docno']]
            
            print(f"📄 Retrieved {len(doc_texts)} document texts from cache")
            
            # Score documents in optimized batches
            try:
                all_scores = []
                doc_batches = self._optimize_batching(doc_texts)
                
                for batch_idx, doc_batch in enumerate(doc_batches):
                    batch_scores = self._score_documents_batch(query_text, doc_batch)
                    all_scores.extend(batch_scores)
                    
                    # Cleanup after each batch
                    if batch_idx % 5 == 0:  # Every 5 batches
                        self.gpu_manager.cleanup_memory()
                
                # Ensure all scores are Python floats to avoid array comparison issues
                all_scores = [float(score) for score in all_scores]
                
                # Update scores
                query_docs['dense_score'] = all_scores
                
                # Sort by score and take top-k
                query_docs = query_docs.nlargest(top_k_output, 'dense_score')
                
                score_range = f"{min(all_scores):.4f} to {max(all_scores):.4f}"
                print(f"   ✅ Processed {len(query_docs)} docs, score range: {score_range}")
                
                # Add to results
                for rank, (_, row) in enumerate(query_docs.iterrows(), 1):
                    results.append({
                        'qid': qid,
                        'docno': row['docno'],
                        'rank': rank,
                        'score': float(row['dense_score'])  # Ensure Python float
                    })
                
            except Exception as e:
                print(f"      ❌ Error in scoring: {e}")
                # Continue with next query
                continue
        
        # Save results
        print(f"\n💾 Saving {len(results)} results to {output_file}")
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if there's a directory part
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(f"{result['qid']}\tQ0\t{result['docno']}\t{result['rank']}\t{result['score']:.6f}\t{self.model_name}_{self.stage}\n")
        
        # Performance summary
        total_time = time.time() - start_time
        cache_stats = self.doc_cache.get_cache_stats()
        
        summary = {
            'total_results': len(results),
            'total_time': total_time,
            'queries_processed': len(unique_queries),
            'time_per_query': total_time / len(unique_queries) if len(unique_queries) > 0 else 0,
            'cache_stats': cache_stats
        }
        
        print(f"\n📊 Performance Summary:")
        print(f"   ✓ Completed in {total_time:.1f}s")
        print(f"   ✓ Results: {len(results)}")
        print(f"   ✓ Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"   ✓ Cache size: {cache_stats['cache_size_gb']:.2f}GB")
        
        return summary


class OptimizedBiEncoderRetriever(OptimizedBaseDenseRetriever):
    """Optimized bi-encoder with maximum VRAM utilization"""
    
    def __init__(self, config: Dict, dataset_version: str, rewriter: str, 
                 stage: str, model_name: str, performance_config: Optional[PerformanceConfig] = None):
        super().__init__(config, dataset_version, rewriter, stage, model_name, performance_config)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize bi-encoder model with optimizations"""
        from sentence_transformers import SentenceTransformer
        
        print(f"🔧 Loading bi-encoder: sentence-transformers/{self.model_name}")
        
        # Load model with optimizations
        self.model = SentenceTransformer(f'sentence-transformers/{self.model_name}')
        self.model = self.model.to(self.device)
        
        # Enable optimizations
        if self.performance_config.enable_mixed_precision:
            self.model.half()  # Use FP16
            print("✅ Mixed precision (FP16) enabled")
        
        self.model.eval()
        
        # Get model info
        max_seq_length = getattr(self.model, 'max_seq_length', 512)
        model_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Bi-encoder model ready")
        print(f"   Model dimension: {model_dim}")
        print(f"   Max sequence length: {max_seq_length}")
        
        # Calculate optimal batch size
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        self.optimal_batch_size = self.gpu_manager.get_optimal_batch_size(model_size_mb, max_seq_length)
    
    def _score_documents_batch(self, query_text: str, doc_texts: List[str]) -> List[float]:
        """Score documents using optimized bi-encoder"""
        try:
            with torch.no_grad():
                # Encode query
                query_embedding = self.model.encode(
                    [query_text], 
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=1
                )
                
                # Encode documents in smaller batches if needed
                doc_embeddings_list = []
                batch_size = min(self.optimal_batch_size, len(doc_texts))
                
                for i in range(0, len(doc_texts), batch_size):
                    batch = doc_texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        batch_size=len(batch)
                    )
                    doc_embeddings_list.append(batch_embeddings)
                
                # Concatenate all document embeddings
                doc_embeddings = torch.cat(doc_embeddings_list, dim=0)
                
                # Compute cosine similarity
                similarities = torch.cosine_similarity(
                    query_embedding.expand_as(doc_embeddings), 
                    doc_embeddings
                )
                
                # Convert to list
                scores = similarities.cpu().tolist()
                
                return scores
                
        except Exception as e:
            print(f"❌ Error in bi-encoder scoring: {e}")
            # Return zeros as fallback
            return [0.0] * len(doc_texts)


def create_optimized_bi_encoder_retrievers(config: Dict, dataset_version: str, 
                                         rewriter: str, stage: str, models_filter: List[str] = None) -> List[OptimizedBiEncoderRetriever]:
    """Create optimized bi-encoder retrievers"""
    
    all_models = [
        'all-mpnet-base-v2',
        'multi-qa-mpnet-base-dot-v1', 
        'all-MiniLM-L6-v2',
        'msmarco-distilbert-base-v4'
    ]
    
    # Filter models if specified
    if models_filter:
        models = [m for m in all_models if m in models_filter]
        print(f"🎯 Running specific models: {models}")
    else:
        models = all_models
        print(f"🎯 Running all models: {models}")
    
    # Create high-performance config
    perf_config = PerformanceConfig(
        max_batch_size=128,
        vram_target_utilization=0.90,  # Use 90% of VRAM
        ram_cache_size_gb=50.0,        # 50GB document cache
        enable_mixed_precision=True,
        prefetch_factor=4,
        num_workers=8,
        pin_memory=True
    )
    
    retrievers = []
    for model_name in models:
        retriever = OptimizedBiEncoderRetriever(
            config, dataset_version, rewriter, stage, model_name, perf_config
        )
        retrievers.append(retriever)
    
    return retrievers


if __name__ == "__main__":
    """Test the optimized dense retriever"""
    
    # Load configuration
    config_loader = ConfigLoader('../env.json')
    config = config_loader.config
    
    # Create test retriever
    retrievers = create_optimized_bi_encoder_retrievers(
        config, 'train', 'llama', 'stage1_bi_encoders'
    )
    
    print(f"Created {len(retrievers)} optimized retrievers")
    
    # Test with first retriever
    test_retriever = retrievers[0]
    
    # Run test reranking
    input_file = "../fused_run_files/train/llama_train_fused.txt"
    output_file = f"../dense_run_files/run_files/stage1_bi_encoders/optimized_test_{test_retriever.model_name}.txt"
    
    summary = test_retriever.rerank_documents(
        input_file, output_file, 
        top_k_input=100,  # Test with smaller set first
        top_k_output=50
    )
    
    print(f"Test completed: {summary}")
