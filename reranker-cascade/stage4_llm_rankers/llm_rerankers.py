"""
Stage 4: LLM-based Dense Rerankers
Implements LLM-based rerankers like RankZephyr, RankGPT, etc.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from base_dense_retriever import BaseDenseRetriever

# Import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch.nn.functional as F
    print("✓ transformers available for LLM rerankers")
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("❌ transformers not available")
    TRANSFORMERS_AVAILABLE = False

class LLMReranker(BaseDenseRetriever):
    """Base LLM reranker"""
    
    def __init__(self, config, dataset_version, rewriter, model_name, hf_model_name):
        """
        Initialize LLM reranker.
        
        Args:
            config: Configuration object
            dataset_version: Dataset version
            rewriter: Query rewriter type
            model_name: Short model name
            hf_model_name: HuggingFace model name
        """
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        self.hf_model_name = hf_model_name
        
        super().__init__(config, dataset_version, rewriter, "stage4_llm_rankers", model_name)
    
    def _initialize_model(self):
        """Initialize the LLM model"""
        
        print(f"   Loading LLM reranker: {self.hf_model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"   ✓ LLM reranker model ready")
            
        except Exception as e:
            print(f"   ❌ Error loading LLM reranker: {e}")
            raise
    
    def _score_documents(self, query_text, doc_texts):
        """
        Score documents using LLM reranker.
        
        Args:
            query_text: Query text
            doc_texts: List of document texts
        
        Returns:
            List of relevance scores
        """
        
        try:
            scores = []
            
            for doc_text in doc_texts:
                # Create ranking prompt
                prompt = self._create_ranking_prompt(query_text, doc_text)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=True
                ).to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Extract and parse response
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                score = self._parse_response(response)
                scores.append(score)
            
            return scores
            
        except Exception as e:
            print(f"      Error in LLM reranker scoring: {e}")
            # Return zeros as fallback
            return [0.0] * len(doc_texts)
    
    def _create_ranking_prompt(self, query_text, doc_text):
        """Create ranking prompt for LLM"""
        return f"Query: {query_text}\\nDocument: {doc_text}\\nIs this document relevant to the query? Answer with a score from 0-10:"
    
    def _parse_response(self, response):
        """Parse LLM response to extract relevance score"""
        try:
            # Extract number from response
            import re
            numbers = re.findall(r'\\d+', response)
            if numbers:
                score = float(numbers[0]) / 10.0  # Normalize to 0-1
                return min(max(score, 0.0), 1.0)
        except:
            pass
        return 0.5  # Default score

class RankZephyrRetriever(LLMReranker):
    """RankZephyr LLM reranker"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 
                        "rankzephyr", "castorini/rank_zephyr_7b_v1_full")
    
    def _create_ranking_prompt(self, query_text, doc_text):
        """Create RankZephyr-specific prompt"""
        prompt = f"""<|user|>
I will provide you with a query and a document. Your task is to determine if the document is relevant to the query. Please respond with 'Yes' if relevant or 'No' if not relevant.

Query: {query_text}

Document: {doc_text}

<|assistant|>
"""
        return prompt
    
    def _parse_response(self, response):
        """Parse RankZephyr response"""
        response_lower = response.lower().strip()
        if 'yes' in response_lower:
            return 1.0
        elif 'no' in response_lower:
            return 0.0
        else:
            return 0.5

class RankVicunaRetriever(LLMReranker):
    """RankVicuna LLM reranker"""
    
    def __init__(self, config, dataset_version, rewriter):
        super().__init__(config, dataset_version, rewriter, 
                        "rankvicuna", "castorini/rankvicuna-7b-v1")
    
    def _create_ranking_prompt(self, query_text, doc_text):
        """Create RankVicuna-specific prompt"""
        prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: I will provide you with a query and a document. Please determine if the document is relevant to the query and provide a relevance score from 0 to 10, where 0 means not relevant at all and 10 means highly relevant.

Query: {query_text}

Document: {doc_text}

ASSISTANT: """
        return prompt

def create_llm_rerankers(config, dataset_version, rewriter):
    """
    Factory function to create LLM rerankers.
    
    Returns:
        List of LLM reranker instances
    """
    
    retrievers = []
    
    if not TRANSFORMERS_AVAILABLE:
        print("Warning: Transformers not available, skipping LLM reranker stage")
        return retrievers
    
    # Available LLM reranker models (commented out heavy models for now)
    retriever_configs = [
        # (RankZephyrRetriever, "RankZephyr 7B"),  # Requires significant GPU memory
        # (RankVicunaRetriever, "RankVicuna 7B"),  # Requires significant GPU memory
    ]
    
    for retriever_class, description in retriever_configs:
        try:
            retriever = retriever_class(config, dataset_version, rewriter)
            retrievers.append(retriever)
            print(f"✓ Initialized {description}")
        except Exception as e:
            print(f"Warning: Could not initialize {description}: {e}")
    
    # For now, return empty list to avoid memory issues
    print("Note: LLM rerankers disabled to avoid GPU memory issues. Enable by uncommenting in create_llm_rerankers()")
    return []

def main():
    """Test LLM rerankers"""
    
    import argparse
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    
    from sparse_retrieval.config_loader import ConfigLoader
    
    parser = argparse.ArgumentParser(description="Stage 4: LLM-based Dense Rerankers")
    parser.add_argument("--dataset", default="train", help="Dataset version")
    parser.add_argument("--rewriter", default="llama", help="Rewriter type")
    parser.add_argument("--input-stage", default="stage3", help="Input stage")
    parser.add_argument("--input-model", default="msmarco-minilm-l12", help="Input model name")
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
    elif args.input_stage == "stage3":
        input_file = f"../run_files/stage3_cross_encoders/{args.rewriter}_{args.dataset}_{args.input_model}.txt"
    else:
        print(f"Unknown input stage: {args.input_stage}")
        return
    
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        return
    
    # Create LLM rerankers
    retrievers = create_llm_rerankers(config, args.dataset, args.rewriter)
    
    if not retrievers:
        print("No LLM rerankers available or enabled")
        return
    
    print(f"\\n🎯 STAGE 4: LLM-BASED DENSE RERANKING")
    print(f"=====================================")
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
    
    print(f"\\n✅ STAGE 4 COMPLETED: LLM-based dense reranking")

if __name__ == "__main__":
    main()
