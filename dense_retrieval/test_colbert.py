#!/usr/bin/env python3
"""
Simple ColBERT test script to verify the installation and basic functionality
"""

import sys
import logging
import json
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_colbert_imports():
    """Test if ColBERT can be imported successfully"""
    try:
        logger.info("Testing ColBERT imports...")
        
        # Test basic imports
        from colbert import Indexer, Searcher
        from colbert.infra import Run, RunConfig, ColBERTConfig
        from colbert.data import Queries, Collection
        
        logger.info("✓ ColBERT basic imports successful")
        
        # Test model loading
        from transformers import AutoTokenizer, AutoModel
        model_name = "colbert-ir/colbertv2.0"
        
        logger.info(f"Testing model loading: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        logger.info("✓ ColBERT model loading successful")
        logger.info(f"Model config: {model.config}")
        
        return True
        
    except Exception as e:
        logger.error(f"ColBERT import/loading failed: {e}")
        return False

def test_simple_scoring():
    """Test ColBERT scoring with simple text examples"""
    try:
        logger.info("Testing simple ColBERT scoring...")
        
        from colbert.modeling.colbert import ColBERT
        from colbert.infra.config import ColBERTConfig
        import torch
        
        # Simple configuration
        config = ColBERTConfig(
            doc_maxlen=300,
            query_maxlen=64,
            dim=128,
            checkpoint="colbert-ir/colbertv2.0"
        )
        
        # Initialize model
        colbert = ColBERT(name="colbert-ir/colbertv2.0", colbert_config=config)
        
        # Test simple query and document
        queries = ["What is artificial intelligence?"]
        documents = [
            "Artificial intelligence is the simulation of human intelligence in machines.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers."
        ]
        
        logger.info(f"Testing with {len(queries)} queries and {len(documents)} documents")
        
        # Encode query
        query_encodings = colbert.query(*queries)
        logger.info(f"Query encoding shape: {query_encodings.shape}")
        
        # Encode documents  
        doc_encodings = colbert.doc(documents, keep_dims='return_mask')
        logger.info(f"Document encoding shape: {doc_encodings[0].shape}")
        
        # Compute scores
        scores = colbert.score(query_encodings, doc_encodings)
        logger.info(f"Scores: {scores}")
        
        logger.info("✓ Simple ColBERT scoring successful")
        return True
        
    except Exception as e:
        logger.error(f"Simple ColBERT scoring failed: {e}")
        return False

def test_environment_info():
    """Display environment information"""
    try:
        import torch
        import transformers
        import colbert
        
        logger.info("=== Environment Information ===")
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"Transformers: {transformers.__version__}")
        logger.info(f"ColBERT: {colbert.__version__ if hasattr(colbert, '__version__') else 'version unknown'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment info failed: {e}")
        return False

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test ColBERT installation and functionality")
    parser.add_argument("--simple", action="store_true", help="Run simple scoring test")
    parser.add_argument("--env", action="store_true", help="Show environment info")
    args = parser.parse_args()
    
    logger.info("Starting ColBERT tests...")
    
    results = {}
    
    # Test environment info
    if args.env or not any([args.simple]):
        results['environment'] = test_environment_info()
    
    # Test imports
    results['imports'] = test_colbert_imports()
    
    # Test simple scoring if requested
    if args.simple:
        results['simple_scoring'] = test_simple_scoring()
    
    # Summary
    logger.info("=== Test Results ===")
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
