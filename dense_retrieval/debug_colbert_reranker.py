#!/usr/bin/env python3
"""
Debug ColBERT Reranker for TREC TOT 2025 - Stage 3 Dense Retrieval
This script debugs why ColBERT results are incomplete
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pyterrier as pt
from sentence_transformers import SentenceTransformer, util
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_setup():
    """Debug the setup and data loading"""
    
    # PyTerrier setup
    if not pt.started():
        logger.info("Starting PyTerrier...")
        pt.init()
    
    # Load LTR results
    ltr_results_path = "../dense_run_files/run_files/stage2_fusion/llama_train_ltr_fusion.txt"
    if not os.path.exists(ltr_results_path):
        logger.error(f"LTR results file not found: {ltr_results_path}")
        return None
    
    logger.info(f"Loading LTR results from: {ltr_results_path}")
    # Load TREC format file
    ltr_data = []
    with open(ltr_results_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docno, rank, score, run_name = parts[:6]
                ltr_data.append({
                    'qid': int(qid),
                    'docno': docno,
                    'rank': int(rank),
                    'score': float(score)
                })
    ltr_df = pd.DataFrame(ltr_data)
    logger.info(f"LTR results shape: {ltr_df.shape}")
    logger.info(f"LTR columns: {ltr_df.columns.tolist()}")
    logger.info(f"Unique queries in LTR: {ltr_df['qid'].nunique()}")
    
    # Check PyTerrier index
    index_path = "../trec-tot-2025-pyterrier-index"
    if not os.path.exists(index_path):
        logger.error(f"PyTerrier index not found: {index_path}")
        return None
    
    logger.info(f"Loading PyTerrier index from: {index_path}")
    index = pt.IndexFactory.of(index_path)
    logger.info(f"Index loaded with {index.getCollectionStatistics().getNumberOfDocuments()} documents")
    
    return ltr_df, index

def debug_first_few_queries(ltr_df, index, max_queries=5):
    """Debug the first few queries to see what's happening"""
    
    logger.info("Testing document retrieval for first few queries...")
    
    # Get unique queries
    unique_queries = sorted(ltr_df['qid'].unique())[:max_queries]
    logger.info(f"Testing queries: {unique_queries}")
    
    for qid in unique_queries:
        logger.info(f"\n--- Processing Query {qid} ---")
        
        # Get documents for this query
        query_docs = ltr_df[ltr_df['qid'] == qid].head(10)  # Just first 10 docs
        logger.info(f"Number of docs for query {qid}: {len(query_docs)}")
        
        # Test document retrieval
        doc_texts = []
        doc_ids = []
        
        for _, row in query_docs.iterrows():
            docid = str(row['docno'])
            try:
                # Try to get document using correct API
                metaindex = index.getMetaIndex()
                docid_int = metaindex.getDocument('docno', docid)
                if docid_int >= 0:
                    text = metaindex.getItem('text', docid_int)
                    if text:
                        doc_texts.append(text)
                        doc_ids.append(docid)
                        logger.info(f"Doc {docid}: {len(text)} chars - '{text[:50]}...'")
                    else:
                        logger.warning(f"Document {docid} has no text")
                else:
                    logger.warning(f"Document {docid} not found in index")
            except Exception as e:
                logger.error(f"Error retrieving document {docid}: {e}")
        
        logger.info(f"Successfully retrieved {len(doc_texts)} documents for query {qid}")
        
        if len(doc_texts) == 0:
            logger.error(f"No documents retrieved for query {qid} - this will cause ColBERT to fail!")
            return False
    
    return True

def main():
    """Main debug function"""
    logger.info("Starting ColBERT Debug...")
    
    # Debug setup
    result = debug_setup()
    if result is None:
        logger.error("Setup failed!")
        return
    
    ltr_df, index = result
    
    # Debug first few queries
    success = debug_first_few_queries(ltr_df, index)
    if not success:
        logger.error("Document retrieval test failed!")
        return
    
    logger.info("Debug completed successfully!")

if __name__ == "__main__":
    main()
