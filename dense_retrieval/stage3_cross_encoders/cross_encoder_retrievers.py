"""
Stage 3: Cross-encoder Dense Retrievers
Implements various cross-encoder models including MonoBERT, MonoT5, etc.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from base_dense_retriever import BaseDenseRetriever

# Import transformers and sentence transformers
try:
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
    import torch.nn.functional as F
    print("✓ transformers and sentence-transformers available for cross-encoders")
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("❌ transformers or sentence-transformers not available")
    TRANSFORMERS_AVAILABLE = False

class CrossEncoderRetriever(BaseDenseRetriever):
    """Base cross-encoder retriever using sentence transformers"""
    
    def __init__(self, config, dataset_version, rewriter, model_name, hf_model_name):
        """
        Initialize cross-encoder retriever.
        
        Args:
            config: Configuration object
            dataset_version: Dataset version
            rewriter: Query rewriter type
            model_name: Short model name
            hf_model_name: HuggingFace model name
        """
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Required libraries not available")
        
        self.hf_model_name = hf_model_name
        
        super().__init__(config, dataset_version, rewriter, "stage3_cross_encoders", model_name)
    
    def _initialize_model(self):
        """Initialize the cross-encoder model"""
        
        print(f"   Loading cross-encoder: {self.hf_model_name}")
        
        try:
            self.model = CrossEncoder(self.hf_model_name, device=self.device)
            
            print(f"   ✓ Cross-encoder model ready")
            
        except Exception as e:
            print(f"   ❌ Error loading cross-encoder: {e}")
            raise
    
    def _score_documents(self, query_text, doc_texts):
        """
        Score documents using cross-encoder.
        
        Args:
            query_text: Query text
            doc_texts: List of document texts
        
        Returns:
            List of relevance scores
        """
        
        try:
            # Create query-document pairs
            pairs = [(query_text, doc_text) for doc_text in doc_texts]
            
            # Score all pairs
            scores = self.model.predict(pairs)
            
            # Convert to list if numpy array
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            return scores
            
        except Exception as e:
            print(f"      Error in cross-encoder scoring: {e}")
            # Return zeros as fallback
            return [0.0] * len(doc_texts)

class MonoT5Retriever(BaseDenseRetriever):
    """MonoT5 cross-encoder retriever with T5 generation approach"""
    
    def __init__(self, config, dataset_version, rewriter, model_name="castorini/monot5-base-msmarco"):
        """Initialize MonoT5 retriever"""
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        self.hf_model_name = model_name
        
        super().__init__(config, dataset_version, rewriter, "stage3_cross_encoders", "monot5-base")
    
    def _initialize_model(self):
        """Initialize the MonoT5 model"""
        
        print(f"   Loading MonoT5: {self.hf_model_name}")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.hf_model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.hf_model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # MonoT5 tokens
            self.token_false = self.tokenizer.encode("false")[0]
            self.token_true = self.tokenizer.encode("true")[0]
            
            print(f"   ✓ MonoT5 model ready")
            
        except Exception as e:
            print(f"   ❌ Error loading MonoT5: {e}")
            raise
    
    def _score_documents(self, query_text, doc_texts):
        """
        Score documents using MonoT5.
        
        Args:
            query_text: Query text
            doc_texts: List of document texts
        
        Returns:
            List of relevance scores
        """
        
        try:
            scores = []
            
            for doc_text in doc_texts:
                # Format input for MonoT5
                input_text = f"Query: {query_text} Document: {doc_text} Relevant:"
                
                # Tokenize
                inputs = self.tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=2,
                        num_return_sequences=1,
                        do_sample=False
                    )
                
                # Get logits for true/false tokens
                logits = self.model(input_ids=inputs['input_ids'], decoder_input_ids=outputs[:, :1]).logits[0, -1, :]
                
                # Calculate probability of "true"
                prob_true = F.softmax(logits[[self.token_false, self.token_true]], dim=0)[1].item()
                scores.append(float(prob_true))
            
            return scores
            
        except Exception as e:
            print(f"      Error in MonoT5 scoring: {e}")
            # Return zeros as fallback
            return [0.0] * len(doc_texts)

class MSMarcoMiniLMRetriever(CrossEncoderRetriever):
    """MS MARCO MiniLM cross-encoder"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 
                        "msmarco-minilm-l6", "cross-encoder/ms-marco-MiniLM-L-6-v2")

class MSMarcoMiniLML12Retriever(CrossEncoderRetriever):
    """MS MARCO MiniLM L12 cross-encoder"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 
                        "msmarco-minilm-l12", "cross-encoder/ms-marco-MiniLM-L-12-v2")

class MSMarcoTinyBERTRetriever(CrossEncoderRetriever):
    """MS MARCO TinyBERT cross-encoder"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 
                        "msmarco-tinybert", "cross-encoder/ms-marco-TinyBERT-L-2-v2")

class RankLlamaRetriever(CrossEncoderRetriever):
    """RankLlama cross-encoder"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 
                        "rankllama", "castorini/rankllama-v1-7b-lora-passage")

def create_cross_encoder_retrievers(config, dataset_version, rewriter):
    """
    Factory function to create cross-encoder retrievers.
    
    Returns:
        List of cross-encoder retriever instances
    """
    
    retrievers = []
    
    if not TRANSFORMERS_AVAILABLE:
        print("Warning: Transformers not available, skipping cross-encoder stage")
        return retrievers
    
    # Available cross-encoder models
    retriever_configs = [
        (MSMarcoMiniLML12Retriever, "MS MARCO MiniLM L12"),
        (MSMarcoMiniLMRetriever, "MS MARCO MiniLM L6"),
        (MSMarcoTinyBERTRetriever, "MS MARCO TinyBERT"),
        (MonoT5Retriever, "MonoT5 Base"),
    ]
    
    for retriever_class, description in retriever_configs:
        try:
            retriever = retriever_class(config, dataset_version, rewriter)
            retrievers.append(retriever)
            print(f"✓ Initialized {description}")
        except Exception as e:
            print(f"Warning: Could not initialize {description}: {e}")
    
    return retrievers

def main():
    """Test cross-encoder retrievers"""
    
    import argparse
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    
    from sparse_retrieval.config_loader import ConfigLoader
    
    parser = argparse.ArgumentParser(description="Stage 3: Cross-encoder Dense Retrievers")
    parser.add_argument("--dataset", default="train", help="Dataset version")
    parser.add_argument("--rewriter", default="llama", help="Rewriter type")
    parser.add_argument("--input-stage", default="stage2", help="Input stage (stage1, stage2, or fusion)")
    parser.add_argument("--input-model", default="colbertv2", help="Input model name")
    parser.add_argument("--top-k-input", type=int, default=1000, help="Top-k documents to load from input")
    parser.add_argument("--top-k-output", type=int, default=1000, help="Top-k documents to output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader("../../env.json")
    
    # Determine input file
    if args.input_stage == "fusion":
        input_file = f"../../fused_run_files/{args.dataset}/{args.rewriter}_{args.dataset}_fused.txt"
    elif args.input_stage == "stage1":
        input_file = f"../run_files/stage1_bi_encoders/{args.rewriter}_{args.dataset}_{args.input_model}.txt"
    elif args.input_stage == "stage2":
        input_file = f"../run_files/stage2_late_interaction/{args.rewriter}_{args.dataset}_{args.input_model}.txt"
    else:
        print(f"Unknown input stage: {args.input_stage}")
        return
    
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        return
    
    # Create cross-encoder retrievers
    retrievers = create_cross_encoder_retrievers(config, args.dataset, args.rewriter)
    
    if not retrievers:
        print("No cross-encoder retrievers available")
        return
    
    print(f"\\n🎯 STAGE 3: CROSS-ENCODER DENSE RETRIEVAL")
    print(f"==========================================")
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
    
    print(f"\\n✅ STAGE 3 COMPLETED: Cross-encoder dense retrieval")

if __name__ == "__main__":
    main()
