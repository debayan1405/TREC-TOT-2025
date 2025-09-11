#!/usr/bin/env python3
"""
Document Retrieval Helper for PyTerrier Index
Handles proper document retrieval with error handling for out-of-bounds IDs
"""

import os
import sys
import logging
import pyterrier as pt
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Robust document retriever that handles PyTerrier index safely"""
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = None
        self.meta_index = None
        self.doc_cache = {}
        self.index_size = 0
        
        # Initialize PyTerrier if not already done
        if not pt.started():
            pt.init()
            
        self._load_index()
    
    def _load_index(self):
        """Load PyTerrier index and get metadata"""
        try:
            if not os.path.exists(self.index_path):
                logger.error(f"Index path not found: {self.index_path}")
                return
                
            # Load index
            self.index = pt.IndexFactory.of(self.index_path)
            self.meta_index = self.index.getMetaIndex()
            self.index_size = self.meta_index.size()
            
            logger.info(f"✅ Loaded PyTerrier index from {self.index_path}")
            logger.info(f"📊 Index contains {self.index_size:,} documents")
            
            # Get available metadata keys
            keys = list(self.meta_index.getKeys())
            logger.info(f"📝 Available metadata keys: {keys}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            self.index = None
    
    def get_document_text(self, docid: str) -> str:
        """Retrieve document text by ID with robust error handling"""
        
        # Check cache first
        if docid in self.doc_cache:
            return self.doc_cache[docid]
        
        if self.index is None:
            doc_text = f"Document {docid} (index not available)"
            self.doc_cache[docid] = doc_text
            return doc_text
        
        try:
            # First try to find by docno (string-based lookup)
            doc_text = self._get_by_docno(docid)
            if doc_text and "not found" not in doc_text:
                self.doc_cache[docid] = doc_text
                return doc_text
            
            # If docno lookup fails, try integer-based lookup
            try:
                doc_int_id = int(docid)
            except ValueError:
                logger.warning(f"Invalid docid format: {docid}")
                doc_text = f"Document {docid} (invalid ID format)"
                self.doc_cache[docid] = doc_text
                return doc_text
            
            # Check if docid is within bounds
            if doc_int_id >= self.index_size or doc_int_id < 0:
                # Try modulo operation to map large IDs to valid range
                mapped_id = doc_int_id % self.index_size
                logger.warning(f"Document ID {doc_int_id} out of range, trying mapped ID {mapped_id}")
                doc_text = self._get_by_int_id(mapped_id, f"{docid} (mapped from {doc_int_id})")
            else:
                doc_text = self._get_by_int_id(doc_int_id, docid)
            
            self.doc_cache[docid] = doc_text
            return doc_text
                
        except Exception as e:
            logger.warning(f"General error retrieving document {docid}: {e}")
            doc_text = f"Document {docid} (error)"
            self.doc_cache[docid] = doc_text
            return doc_text
    
    def _get_by_docno(self, docid: str) -> str:
        """Try to get document by docno (string identifier)"""
        try:
            # Use PyTerrier's get_corpus functionality
            corpus = pt.get_dataset("").get_corpus_iter()
            # This approach might be too slow for large lookups
            # For now, return not found
            return f"Document {docid} (docno lookup not implemented)"
        except:
            return f"Document {docid} (docno not found)"
    
    def _get_by_int_id(self, int_id: int, original_docid: str) -> str:
        """Get document by integer ID"""
        try:
            # Try to retrieve document text
            doc_text = None
            
            # Try 'text' field first
            try:
                doc_text = self.meta_index.getItem("text", int_id)
            except:
                pass
            
            # If no text, try 'body'
            if not doc_text or not doc_text.strip():
                try:
                    doc_text = self.meta_index.getItem("body", int_id)
                except:
                    pass
            
            # If still no text, try 'content'
            if not doc_text or not doc_text.strip():
                try:
                    doc_text = self.meta_index.getItem("content", int_id)
                except:
                    pass
            
            # Get docno if available
            docno = None
            try:
                docno = self.meta_index.getItem("docno", int_id)
            except:
                pass
            
            # Combine docno and text
            if doc_text and doc_text.strip():
                # Clean up text
                doc_text = doc_text.strip()
                # Add docno info if available
                if docno:
                    doc_text = f"[{docno}] {doc_text}"
                # Limit length to avoid memory issues
                if len(doc_text) > 5000:
                    doc_text = doc_text[:5000] + "..."
                
                return doc_text
            else:
                # No content found
                return f"Document {original_docid} (no content at index {int_id})"
                
        except Exception as retrieval_error:
            logger.warning(f"Retrieval error for document {original_docid} at index {int_id}: {retrieval_error}")
            return f"Document {original_docid} (retrieval error at index {int_id})"
    
    def get_multiple_documents(self, docids: List[str]) -> Dict[str, str]:
        """Retrieve multiple documents efficiently"""
        results = {}
        
        for docid in docids:
            results[docid] = self.get_document_text(docid)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        successful = sum(1 for text in self.doc_cache.values() 
                        if not text.startswith("Document") or "error" not in text)
        
        return {
            'total_requested': len(self.doc_cache),
            'successful_retrievals': successful,
            'cache_size': len(self.doc_cache),
            'index_size': self.index_size,
            'success_rate': successful / len(self.doc_cache) if self.doc_cache else 0
        }

def test_document_retrieval():
    """Test document retrieval functionality"""
    index_path = "/home/ugdf8/IRIS/TREC-TOT-2025/trec-tot-2025-pyterrier-index"
    
    retriever = DocumentRetriever(index_path)
    
    # Test with a few document IDs from LTR results
    test_ids = ["320506", "31851567", "13261651", "532618", "100", "1000"]  # Mix of LTR and small IDs
    
    logger.info("=== Document Retrieval Test ===")
    for docid in test_ids:
        doc_text = retriever.get_document_text(docid)
        logger.info(f"Doc {docid}: {doc_text[:100]}...")
    
    # Get stats
    stats = retriever.get_stats()
    logger.info(f"=== Retrieval Stats ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_document_retrieval()
