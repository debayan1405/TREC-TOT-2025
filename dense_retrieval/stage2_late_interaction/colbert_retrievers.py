"""
Stage 2: Late-interaction Dense Retrievers (ColBERT)
Implements ColBERT v2 for late-interaction dense retrieval
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from base_dense_retriever import BaseDenseRetriever

# Try to import ColBERT
try:
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    print("✓ ColBERT available for late-interaction")
    COLBERT_AVAILABLE = True
except ImportError:
    print("❌ ColBERT not available - install with: pip install colbert-ai")
    COLBERT_AVAILABLE = False

class ColBERTRetriever(BaseDenseRetriever):
    """ColBERT late-interaction dense retriever"""
    
    def __init__(self, config, dataset_version, rewriter, checkpoint_name="colbert-ir/colbertv2.0"):
        """
        Initialize ColBERT retriever.
        
        Args:
            config: Configuration object
            dataset_version: Dataset version
            rewriter: Query rewriter type
            checkpoint_name: ColBERT checkpoint name
        """
        
        if not COLBERT_AVAILABLE:
            raise ImportError("ColBERT is not available. Install with: pip install colbert-ai")
        
        self.checkpoint_name = checkpoint_name
        
        super().__init__(config, dataset_version, rewriter, "stage2_late_interaction", "colbertv2")
    
    def _initialize_model(self):
        """Initialize the ColBERT model"""
        
        print(f"   Loading ColBERT checkpoint: {self.checkpoint_name}")
        
        try:
            # Configure ColBERT
            self.colbert_config = ColBERTConfig(
                doc_maxlen=180,  # Maximum document length
                query_maxlen=32,  # Maximum query length
                amp=True,        # Automatic mixed precision
                similarity='cosine'
            )
            
            # Load checkpoint
            self.checkpoint = Checkpoint(
                self.checkpoint_name, 
                colbert_config=self.colbert_config
            )
            
            print(f"   ✓ ColBERT model ready")
            print(f"   ✓ Query maxlen: {self.colbert_config.query_maxlen}")
            print(f"   ✓ Doc maxlen: {self.colbert_config.doc_maxlen}")
            
        except Exception as e:
            print(f"   ❌ Error loading ColBERT: {e}")
            raise
    
    def _score_documents(self, query_text, doc_texts):
        """
        Score documents using ColBERT late-interaction.
        
        Args:
            query_text: Query text
            doc_texts: List of document texts
        
        Returns:
            List of similarity scores
        """
        
        try:
            # Encode query
            query_ids, query_mask = self.checkpoint.query_tokenizer.tensorize([query_text])
            query_embeddings = self.checkpoint.query(query_ids)
            
            # Encode documents
            doc_ids, doc_mask = self.checkpoint.doc_tokenizer.tensorize(doc_texts)
            doc_embeddings = self.checkpoint.doc(doc_ids, keep_dims='return_mask')
            
            # Compute late-interaction scores
            scores = []
            for i, doc_emb in enumerate(doc_embeddings):
                # ColBERT scoring: max-sim between query and document tokens
                score = self.checkpoint.score(query_embeddings, doc_emb.unsqueeze(0))
                scores.append(float(score))
            
            return scores
            
        except Exception as e:
            print(f"      Error in ColBERT scoring: {e}")
            # Return zeros as fallback
            return [0.0] * len(doc_texts)

class ColBERTv2Retriever(ColBERTRetriever):
    """ColBERT v2.0 retriever"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, "colbert-ir/colbertv2.0")

def create_late_interaction_retrievers(config, dataset_version, rewriter):
    """
    Factory function to create late-interaction retrievers.
    
    Returns:
        List of late-interaction retriever instances
    """
    
    retrievers = []
    
    if COLBERT_AVAILABLE:
        try:
            retriever = ColBERTv2Retriever(config, dataset_version, rewriter)
            retrievers.append(retriever)
        except Exception as e:
            print(f"Warning: Could not initialize ColBERTv2Retriever: {e}")
    else:
        print("Warning: ColBERT not available, skipping late-interaction stage")
    
    return retrievers

def main():
    """Test late-interaction retrievers"""
    
    import argparse
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    
    from sparse_retrieval.config_loader import ConfigLoader
    
    parser = argparse.ArgumentParser(description="Stage 2: Late-interaction Dense Retrievers")
    parser.add_argument("--dataset", default="train", help="Dataset version")
    parser.add_argument("--rewriter", default="llama", help="Rewriter type")
    parser.add_argument("--input-stage", default="stage1", help="Input stage (stage1 or fusion)")
    parser.add_argument("--input-model", default="all-mpnet-base-v2", help="Input model name")
    parser.add_argument("--top-k-input", type=int, default=1000, help="Top-k documents to load from input")
    parser.add_argument("--top-k-output", type=int, default=1000, help="Top-k documents to output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader("../../env.json")
    
    # Determine input file
    if args.input_stage == "fusion":
        input_file = f"../../fused_run_files/{args.dataset}/{args.rewriter}_{args.dataset}_fused.txt"
    else:
        input_file = f"../run_files/stage1_bi_encoders/{args.rewriter}_{args.dataset}_{args.input_model}.txt"
    
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        return
    
    # Create late-interaction retrievers
    retrievers = create_late_interaction_retrievers(config, args.dataset, args.rewriter)
    
    if not retrievers:
        print("No late-interaction retrievers available")
        return
    
    print(f"\\n🎯 STAGE 2: LATE-INTERACTION DENSE RETRIEVAL")
    print(f"=============================================")
    print(f"Dataset: {args.dataset}")
    print(f"Rewriter: {args.rewriter}")
    print(f"Input: {input_file}")
    print(f"Models: {len(retrievers)}")
    
    for retriever in retrievers:
        print(f"\\n📋 Processing {retriever.model_name}:")
        
        try:
            # Rerank documents
            results = retriever.rerank_documents(
                input_file, 
                top_k_input=args.top_k_input,
                top_k_output=args.top_k_output
            )
            
            # Save results
            output_file = retriever.get_output_path(args.dataset)
            retriever.save_results(results, output_file)
            
            # Evaluate if train dataset
            if args.dataset == "train":
                metrics = retriever.evaluate_results(output_file)
                if metrics:
                    # Save evaluation
                    eval_file = retriever.get_evaluation_path(args.dataset)
                    eval_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    import json
                    with open(eval_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            print(f"   ✓ Completed {retriever.model_name}")
            
        except Exception as e:
            print(f"   ❌ Error with {retriever.model_name}: {e}")
            continue
    
    print(f"\\n✅ STAGE 2 COMPLETED: Late-interaction dense retrieval")

if __name__ == "__main__":
    main()
