"""
Stage 1: Bi-encoder Dense Retrievers
Implements various bi-encoder models for dense retrieval
"""

import torch
import numpy as np
from pathlib import Path
from base_dense_retriever import BaseDenseRetriever

# Import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers available for bi-encoders")
except ImportError:
    print("❌ sentence-transformers not available")
    SentenceTransformer = None

class BiEncoderRetriever(BaseDenseRetriever):
    """Bi-encoder dense retriever using sentence transformers"""
    
    def __init__(self, config, dataset_version, rewriter, model_name):
        """
        Initialize bi-encoder retriever.
        
        Args:
            config: Configuration object
            dataset_version: Dataset version
            rewriter: Query rewriter type
            model_name: Specific bi-encoder model name
        """
        
        # Model mapping for different bi-encoders
        self.model_mapping = {
            'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
            'multi-qa-mpnet-base-dot-v1': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
            'all-MiniLM-L12-v2': 'sentence-transformers/all-MiniLM-L12-v2',
            'msmarco-distilbert-base-v4': 'sentence-transformers/msmarco-distilbert-base-v4',
            'msmarco-MiniLM-L6-cos-v5': 'sentence-transformers/msmarco-MiniLM-L6-cos-v5'
        }
        
        self.hf_model_name = self.model_mapping.get(model_name, model_name)
        
        super().__init__(config, dataset_version, rewriter, "stage1_bi_encoders", model_name)
        
        # Initialize the model after base class setup
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the bi-encoder model"""
        
        print(f"   Loading bi-encoder: {self.hf_model_name}")
        
        try:
            self.model = SentenceTransformer(self.hf_model_name, device=self.device)
            
            # Optimize for inference
            self.model.eval()
            
            print(f"   ✓ Bi-encoder model ready")
            print(f"   ✓ Model dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise
    
    def _score_documents(self, query_text, doc_texts):
        """
        Score documents using bi-encoder similarity.
        
        Args:
            query_text: Query text
            doc_texts: List of document texts
        
        Returns:
            List of similarity scores
        """
        
        try:
            # Encode query and documents
            query_embedding = self.model.encode([query_text], convert_to_tensor=True)
            doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarities = torch.cosine_similarity(
                query_embedding.unsqueeze(0), 
                doc_embeddings.unsqueeze(1), 
                dim=2
            ).squeeze()
            
            # Handle single document case
            if similarities.dim() == 0:
                similarities = similarities.unsqueeze(0)
            
            return similarities.cpu().tolist()
            
        except Exception as e:
            print(f"      Error in bi-encoder scoring: {e}")
            # Return zeros as fallback
            return [0.0] * len(doc_texts)

class AllMPNetRetriever(BiEncoderRetriever):
    """All-MPNet-base-v2 retriever"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 'all-mpnet-base-v2')

class MultiQAMPNetRetriever(BiEncoderRetriever):
    """Multi-QA-MPNet-base-dot-v1 retriever"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 'multi-qa-mpnet-base-dot-v1')

class MiniLML6Retriever(BiEncoderRetriever):
    """All-MiniLM-L6-v2 retriever"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 'all-MiniLM-L6-v2')

class MSMarcoDistilBERTRetriever(BiEncoderRetriever):
    """MS MARCO DistilBERT retriever"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 'msmarco-distilbert-base-v4')

def create_bi_encoder_retrievers(config, dataset_version, rewriter):
    """
    Factory function to create all bi-encoder retrievers.
    
    Returns:
        List of bi-encoder retriever instances
    """
    
    retrievers = []
    
    # Available bi-encoder models
    retriever_classes = [
        AllMPNetRetriever,
        MultiQAMPNetRetriever,
        MiniLML6Retriever,
        MSMarcoDistilBERTRetriever
    ]
    
    for retriever_class in retriever_classes:
        try:
            retriever = retriever_class(config, dataset_version, rewriter)
            retrievers.append(retriever)
        except Exception as e:
            print(f"Warning: Could not initialize {retriever_class.__name__}: {e}")
    
    return retrievers

def main():
    """Test bi-encoder retrievers"""
    
    import argparse
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    
    from sparse_retrieval.config_loader import ConfigLoader
    
    parser = argparse.ArgumentParser(description="Stage 1: Bi-encoder Dense Retrievers")
    parser.add_argument("--dataset", default="train", help="Dataset version")
    parser.add_argument("--rewriter", default="llama", help="Rewriter type")
    parser.add_argument("--top-k-input", type=int, default=5000, help="Top-k documents to load from input")
    parser.add_argument("--top-k-output", type=int, default=1000, help="Top-k documents to output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader("../env.json")
    
    # Input file (from fusion results)
    input_file = f"../fused_run_files/{args.dataset}/{args.rewriter}_{args.dataset}_fused.txt"
    
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        return
    
    # Create and run all bi-encoder retrievers
    retrievers = create_bi_encoder_retrievers(config, args.dataset, args.rewriter)
    
    print(f"\\n🎯 STAGE 1: BI-ENCODER DENSE RETRIEVAL")
    print(f"========================================")
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
    
    print(f"\\n✅ STAGE 1 COMPLETED: Bi-encoder dense retrieval")

if __name__ == "__main__":
    main()
