"""
Enhanced Dense Reranker with Cross-Encoder Support
This module implements both bi-encoder and cross-encoder reranking for optimal performance.
"""
import pandas as pd
import pyterrier as pt
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from pathlib import Path
from tqdm import tqdm
import os
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sparse_retrieval.config_loader import ConfigLoader
from sparse_retrieval.data_loader import DataLoader

class EnhancedDenseReranker:
    """
    Enhanced dense re-ranker supporting both bi-encoders and cross-encoders.
    Optimized for high-end hardware (700+ GB RAM, 2x A6000 GPUs).
    """

    def __init__(self, config: ConfigLoader, dataset_version: str, rewriter: str, use_cross_encoder: bool = True):
        self.config = config
        self.data_loader = DataLoader(config)
        self.dataset_version = dataset_version
        self.rewriter = rewriter
        self.use_cross_encoder = use_cross_encoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🚀 Initializing Enhanced Dense Reranker:")
        print(f"   Dataset: {dataset_version}")
        print(f"   Rewriter: {rewriter}")
        print(f"   Device: {self.device}")
        print(f"   Cross-encoder mode: {use_cross_encoder}")
        
        # Hardware optimizations
        if self.device == 'cuda':
            # Optimize for A6000 GPUs
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.batch_size = 32 if use_cross_encoder else 512  # Cross-encoders need smaller batches
            print(f"   Batch size: {self.batch_size}")
        else:
            self.batch_size = 8 if use_cross_encoder else 64
        
        # Initialize models based on mode
        if use_cross_encoder:
            self._init_cross_encoder()
        else:
            self._init_bi_encoder()
    
    def _init_cross_encoder(self):
        """Initialize cross-encoder models for more accurate reranking"""
        
        # High-performance cross-encoders for different use cases
        self.cross_encoder_models = {
            'ms-marco': 'cross-encoder/ms-marco-MiniLM-L-12-v2',  # Fast and accurate
            'msmarco-distilbert': 'cross-encoder/msmarco-distilbert-base-v4',  # Balanced
            'stsb': 'cross-encoder/stsb-distilroberta-base'  # General purpose
        }
        
        # Use the best performing cross-encoder
        self.model_name = self.cross_encoder_models['ms-marco']
        print(f"   Loading cross-encoder: {self.model_name}")
        
        self.model = CrossEncoder(self.model_name, device=self.device)
        
        # Enable model compilation for faster inference
        if hasattr(torch, 'compile'):
            try:
                self.model.model = torch.compile(self.model.model, mode='max-autotune')
                print("   ✓ Model compiled with max-autotune")
            except Exception as e:
                print(f"   Note: Compilation failed: {e}")
    
    def _init_bi_encoder(self):
        """Initialize bi-encoder models for embedding-based reranking"""
        
        # High-performance bi-encoders
        self.bi_encoder_models = [
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'sentence-transformers/all-distilroberta-v1'
        ]
        
        self.model_name = self.bi_encoder_models[1]  # Best for Q&A
        print(f"   Loading bi-encoder: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Enable model compilation
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                print("   ✓ Model compiled with max-autotune")
            except Exception as e:
                print(f"   Note: Compilation failed: {e}")

    def rerank_documents(self, fused_run_file: str, top_k: int = 100) -> pd.DataFrame:
        """
        Re-rank documents from fused run file using dense models.
        
        Args:
            fused_run_file (str): Path to the fused run file
            top_k (int): Number of top documents to rerank per query
            
        Returns:
            pd.DataFrame: Reranked results
        """
        print(f"🔄 Starting dense reranking:")
        print(f"   Input file: {fused_run_file}")
        print(f"   Top-k reranking: {top_k}")
        
        # Load fused results
        fused_results = pd.read_csv(
            fused_run_file, 
            sep='\\s+', 
            header=None,
            names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name']
        )
        
        print(f"   Loaded {len(fused_results)} fused results")
        
        # Load index for document text retrieval
        print("   Loading PyTerrier index...")
        index_path = self.config.get_index_path()
        index = pt.IndexFactory.of(f"{index_path}/data.properties")
        metaindex = index.getMetaIndex()
        
        # Load topics for query text
        topics_df = self.data_loader.load_topics(self.dataset_version, "original")
        if topics_df is None or len(topics_df) == 0:
            # Try loading from fused results qids
            unique_qids = fused_results['qid'].unique()
            print(f"   Warning: No topics loaded, using {len(unique_qids)} QIDs from results")
            topics_dict = {qid: f"Query {qid}" for qid in unique_qids}
        else:
            topics_dict = dict(zip(topics_df['qid'], topics_df['query']))
            print(f"   Loaded {len(topics_dict)} topics")
        
        # Process queries in batches
        reranked_results = []
        unique_qids = fused_results['qid'].unique()
        
        print(f"   Processing {len(unique_qids)} queries...")
        
        for qid in tqdm(unique_qids, desc="Reranking queries"):
            # Get top-k results for this query
            query_results = fused_results[fused_results['qid'] == qid].head(top_k)
            
            if len(query_results) == 0:
                continue
            
            # Get query text
            query_text = topics_dict.get(str(qid), f"Query {qid}")
            
            # Get document texts
            doc_texts = []
            valid_docs = []
            
            for _, row in query_results.iterrows():
                try:
                    doc_idx = metaindex.getDocument("docno", str(row['docno']))
                    if doc_idx >= 0:
                        text = metaindex.getItem("text", doc_idx)
                        if text and len(text) > 0:
                            # Truncate long documents for efficiency
                            doc_text = text[:2000] if len(text) > 2000 else text
                            doc_texts.append(doc_text)
                            valid_docs.append(row)
                except Exception:
                    continue
            
            if len(doc_texts) == 0:
                continue
            
            # Rerank using selected method
            if self.use_cross_encoder:
                scores = self._rerank_cross_encoder(query_text, doc_texts)
            else:
                scores = self._rerank_bi_encoder(query_text, doc_texts)
            
            # Create reranked results
            for i, (score, doc_row) in enumerate(zip(scores, valid_docs)):
                reranked_results.append({
                    'qid': qid,
                    'Q0': 'Q0',
                    'docno': doc_row['docno'],
                    'rank': i,
                    'score': float(score),
                    'run_name': f"{self.rewriter}_dense_{'cross' if self.use_cross_encoder else 'bi'}"
                })
        
        reranked_df = pd.DataFrame(reranked_results)
        print(f"   ✓ Reranked {len(reranked_df)} results")
        
        return reranked_df
    
    def _rerank_cross_encoder(self, query: str, doc_texts: list) -> list:
        """Rerank using cross-encoder"""
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in doc_texts]
        
        # Get similarity scores in batches
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch_pairs)
            scores.extend(batch_scores)
        
        return scores
    
    def _rerank_bi_encoder(self, query: str, doc_texts: list) -> list:
        """Rerank using bi-encoder"""
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        
        # Encode documents in batches
        doc_embeddings = []
        for i in range(0, len(doc_texts), self.batch_size):
            batch_docs = doc_texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch_docs, convert_to_tensor=True)
            doc_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        if doc_embeddings:
            all_doc_embeddings = torch.cat(doc_embeddings, dim=0)
            
            # Compute similarities
            similarities = cos_sim(query_embedding, all_doc_embeddings)[0]
            return similarities.cpu().numpy().tolist()
        
        return []

    def save_reranked_results(self, reranked_df: pd.DataFrame, output_path: str):
        """Save reranked results in TREC format"""
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Sort by qid and score (descending)
        reranked_df = reranked_df.sort_values(['qid', 'score'], ascending=[True, False])
        
        # Update ranks within each query
        reranked_df['rank'] = reranked_df.groupby('qid').cumcount()
        
        # Save in TREC format
        reranked_df.to_csv(
            output_path, 
            sep='\\t', 
            header=False, 
            index=False,
            columns=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name']
        )
        
        print(f"   ✓ Saved reranked results to: {output_path}")

def main():
    """Main function for testing the enhanced dense reranker"""
    
    # Load configuration
    config = ConfigLoader("../env.json")
    
    # Test both cross-encoder and bi-encoder modes
    for use_cross_encoder in [True, False]:
        model_type = "cross-encoder" if use_cross_encoder else "bi-encoder"
        print(f"\\n🧪 Testing {model_type} reranking:")
        
        # Initialize reranker
        reranker = EnhancedDenseReranker(
            config=config,
            dataset_version="train",
            rewriter="mistral",  # Use best performing rewriter
            use_cross_encoder=use_cross_encoder
        )
        
        # Test reranking
        fused_file = "../fused_run_files/train/mistral_train_fused.txt"
        if Path(fused_file).exists():
            reranked_results = reranker.rerank_documents(fused_file, top_k=50)
            
            # Save results
            output_path = f"../dense_run_files/train/mistral_train_dense_{model_type}.txt"
            reranker.save_reranked_results(reranked_results, output_path)
        else:
            print(f"   ❌ Fused file not found: {fused_file}")

if __name__ == "__main__":
    main()
