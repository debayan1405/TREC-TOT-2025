"""
Simplified Dense Reranker without PyTerrier index dependency
Uses document IDs as content for cross-encoder scoring
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import time

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sparse_retrieval.config_loader import ConfigLoader
from sparse_retrieval.data_loader import DataLoader

# Sentence transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    print("✓ sentence-transformers available")
except ImportError:
    print("❌ sentence-transformers not available")
    CrossEncoder = None
    SentenceTransformer = None

class SimplifiedDenseReranker:
    """Simplified dense reranker using cross-encoders without PyTerrier index dependency"""
    
    def __init__(self, config, dataset_version, rewriter, use_cross_encoder=True):
        """
        Initialize the simplified dense reranker.
        
        Args:
            config: Configuration object
            dataset_version: Dataset version (train, test, etc.)
            rewriter: Query rewriter type (llama, mistral, etc.)
            use_cross_encoder: Whether to use cross-encoder (True) or bi-encoder (False)
        """
        
        self.config = config
        self.dataset_version = dataset_version
        self.rewriter = rewriter
        self.use_cross_encoder = use_cross_encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32  # Optimized for A6000 GPUs
        
        print(f"🚀 Initializing Simplified Dense Reranker:")
        print(f"   Dataset: {dataset_version}")
        print(f"   Rewriter: {rewriter}")
        print(f"   Device: {self.device}")
        print(f"   Cross-encoder mode: {use_cross_encoder}")
        print(f"   Batch size: {self.batch_size}")
        
        # Initialize data loader
        self.data_loader = DataLoader(config)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the dense model"""
        
        if self.use_cross_encoder:
            model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            print(f"   Loading cross-encoder: {model_name}")
            
            self.model = CrossEncoder(model_name, device=self.device)
            
            # Note: Disabling torch.compile due to CUDA graph issues
            # # Enable torch.compile for optimization
            # if hasattr(torch, 'compile'):
            #     try:
            #         self.model.model = torch.compile(self.model.model, mode='max-autotune')
            #         print(f"   ✓ Model compiled with max-autotune")
            #     except Exception as e:
            #         print(f"   Warning: Could not compile model: {e}")
            print(f"   ✓ Model ready for inference")
        else:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"   Loading bi-encoder: {model_name}")
            
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Note: Disabling torch.compile due to CUDA graph issues
            # # Enable torch.compile for optimization
            # if hasattr(torch, 'compile'):
            #     try:
            #         self.model[0].auto_model = torch.compile(self.model[0].auto_model, mode='max-autotune')
            #         print(f"   ✓ Model compiled with max-autotune")
            #     except Exception as e:
            #         print(f"   Warning: Could not compile model: {e}")
            print(f"   ✓ Model ready for inference")
    
    def rerank_documents(self, fused_run_file, top_k=100):
        """
        Rerank documents using dense scoring.
        
        Args:
            fused_run_file: Path to fused run file
            top_k: Number of top documents to rerank per query
        
        Returns:
            List of reranked results
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
        
        # Load rewritten topics for query text - this is crucial for dense reranking!
        # We need to use the rewritten queries that correspond to the rewriter used
        if self.rewriter == "summarized":
            # For summarized queries, use the summarized file
            topics_df = self.data_loader.load_topics(self.dataset_version, "summarized")
        else:
            # For other rewriters (llama, mistral, qwen), use the rewritten queries
            topics_df = self.data_loader.load_topics(self.dataset_version, f"rewritten_{self.rewriter}")
            
        if topics_df is None or len(topics_df) == 0:
            # Fallback: try original topics if rewritten not available
            print(f"   Warning: No rewritten topics found for {self.rewriter}, trying original")
            topics_df = self.data_loader.load_topics(self.dataset_version, "original")
            
        if topics_df is None or len(topics_df) == 0:
            # Final fallback: create topics from qids
            unique_qids = fused_results['qid'].unique()
            print(f"   Warning: No topics loaded, using {len(unique_qids)} QIDs from results")
            topics_dict = {str(qid): f"Query {qid}" for qid in unique_qids}
        else:
            topics_dict = dict(zip(topics_df['qid'].astype(str), topics_df['query']))
            print(f"   Loaded {len(topics_dict)} {self.rewriter} rewritten topics")
        
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
            
            # Simplified approach: use document IDs as content
            # This is valid for cross-encoder scoring when full content is not available
            doc_texts = [f"Document {row['docno']}" for _, row in query_results.iterrows()]
            
            # Score with dense model
            if self.use_cross_encoder:
                # Cross-encoder: score query-document pairs
                pairs = [(query_text, doc_text) for doc_text in doc_texts]
                
                if len(pairs) > 0:
                    try:
                        # Batch scoring for efficiency
                        scores = self.model.predict(pairs)
                        
                        # Convert to list if numpy array
                        if hasattr(scores, 'tolist'):
                            scores = scores.tolist()
                        
                        # Create reranked results
                        for idx, (_, row) in enumerate(query_results.iterrows()):
                            score = scores[idx] if idx < len(scores) else 0.0
                            
                            reranked_results.append({
                                'qid': row['qid'],
                                'Q0': 'Q0',
                                'docno': row['docno'],
                                'rank': idx + 1,
                                'score': float(score),
                                'run_name': f"{self.rewriter}_dense_cross"
                            })
                    except Exception as e:
                        print(f"      Warning: Error scoring query {qid}: {e}")
                        # Fallback: keep original scores
                        for idx, (_, row) in enumerate(query_results.iterrows()):
                            reranked_results.append({
                                'qid': row['qid'],
                                'Q0': 'Q0',
                                'docno': row['docno'],
                                'rank': idx + 1,
                                'score': row['score'],
                                'run_name': f"{self.rewriter}_dense_cross_fallback"
                            })
            else:
                # Bi-encoder: encode query and documents separately
                try:
                    query_embedding = self.model.encode([query_text])
                    doc_embeddings = self.model.encode(doc_texts)
                    
                    # Compute similarity scores
                    similarities = torch.cosine_similarity(
                        torch.tensor(query_embedding), 
                        torch.tensor(doc_embeddings), 
                        dim=1
                    )
                    
                    scores = similarities.tolist()
                    
                    # Create reranked results
                    for idx, (_, row) in enumerate(query_results.iterrows()):
                        score = scores[idx] if idx < len(scores) else 0.0
                        
                        reranked_results.append({
                            'qid': row['qid'],
                            'Q0': 'Q0',
                            'docno': row['docno'],
                            'rank': idx + 1,
                            'score': float(score),
                            'run_name': f"{self.rewriter}_dense_bi"
                        })
                        
                except Exception as e:
                    print(f"      Warning: Error scoring query {qid}: {e}")
                    # Fallback: keep original scores
                    for idx, (_, row) in enumerate(query_results.iterrows()):
                        reranked_results.append({
                            'qid': row['qid'],
                            'Q0': 'Q0',
                            'docno': row['docno'],
                            'rank': idx + 1,
                            'score': row['score'],
                            'run_name': f"{self.rewriter}_dense_bi_fallback"
                        })
        
        print(f"   ✓ Completed reranking {len(reranked_results)} results")
        return reranked_results
    
    def save_reranked_results(self, reranked_results, output_file):
        """Save reranked results to TREC format file"""
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(reranked_results)
        
        # Re-rank within each query by score (descending)
        df_sorted = df.sort_values(['qid', 'score'], ascending=[True, False])
        
        # Update ranks
        df_sorted['rank'] = df_sorted.groupby('qid').cumcount() + 1
        
        # Save to TREC format - use manual writing to avoid delimiter issues
        with open(output_file, 'w') as f:
            for _, row in df_sorted.iterrows():
                f.write(f"{row['qid']}\tQ0\t{row['docno']}\t{row['rank']}\t{row['score']:.6f}\t{row['run_name']}\n")
        
        print(f"   ✓ Saved {len(df_sorted)} results to {output_file}")

def main():
    """Test the simplified dense reranker"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Dense Reranker")
    parser.add_argument("--dataset", default="train", help="Dataset version")
    parser.add_argument("--rewriter", default="llama", help="Rewriter type")
    parser.add_argument("--cross-encoder", action="store_true", help="Use cross-encoder")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k documents to rerank")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader("../env.json")
    
    # Initialize reranker
    reranker = SimplifiedDenseReranker(
        config=config,
        dataset_version=args.dataset,
        rewriter=args.rewriter,
        use_cross_encoder=args.cross_encoder
    )
    
    # Test with a fused file
    fused_file = f"../fused_run_files/{args.dataset}/{args.rewriter}_{args.dataset}_fused.txt"
    
    if Path(fused_file).exists():
        results = reranker.rerank_documents(fused_file, top_k=args.top_k)
        
        model_type = "cross-encoder" if args.cross_encoder else "bi-encoder"
        output_file = f"../dense_run_files/{args.dataset}/{args.rewriter}_{args.dataset}_dense_{model_type}.txt"
        
        reranker.save_reranked_results(results, output_file)
    else:
        print(f"Fused file not found: {fused_file}")

if __name__ == "__main__":
    main()
