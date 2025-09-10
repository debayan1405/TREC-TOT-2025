"""
Core logic for dense re-ranking of documents using sentence-transformer models.
"""
import pandas as pd
import pyterrier as pt
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
from pathlib import Path
from tqdm import tqdm

from sparse_retrieval.config_loader import ConfigLoader
from sparse_retrieval.data_loader import DataLoader

class DenseReranker:
    """
    Handles the dense re-ranking process for a given fused run file.
    """

    def __init__(self, config: ConfigLoader, dataset_version: str, rewriter: str, model_name: str):
        self.config = config
        self.data_loader = DataLoader(config)
        self.dataset_version = dataset_version
        self.rewriter = rewriter
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Initializing dense re-ranker with model: {self.model_name} on device: {self.device}")
        
        # Optimize for high-end hardware (700+ GB RAM, 2x A6000 GPUs)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Enable optimizations for maximum performance
        if self.device == 'cuda':
            # Use compilation for faster inference
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                print("✓ Model compiled with max-autotune for optimal performance")
            except:
                print("Note: torch.compile not available, using standard model")
            
            # Set optimal batch size for A6000 GPUs (48GB VRAM each)
            self.batch_size = 512  # Large batch for maximum throughput
            
            # Enable tensor cores and mixed precision
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.batch_size = 64
        
        # Load PyTerrier index for fetching document text
        self.index = self.data_loader.get_index()

        # Define and create output directories
        self.run_dir = Path(self.config.get_dense_run_directory()) / self.dataset_version
        self.eval_dir = Path(self.config.get_dense_eval_directory()) / self.dataset_version
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def _get_document_texts(self, docnos: list) -> dict:
        """Fetch the 'text' metadata for a list of docnos from the index."""
        texts = {}
        # Try to get document metadata using PyTerrier's index
        try:
            meta_index = self.index.getMetaIndex()
            doc_index = self.index.getDocumentIndex()
            
            for docno in docnos:
                try:
                    # Get document ID from docno
                    docid = doc_index.getDocumentId(docno)
                    if docid >= 0:
                        # Try to get the text metadata
                        meta_data = meta_index.getItem("text", docid)
                        if meta_data:
                            texts[docno] = str(meta_data).strip()
                        else:
                            # Try alternative metadata fields
                            title = meta_index.getItem("title", docid)
                            texts[docno] = str(title).strip() if title else f"Document {docno}"
                    else:
                        texts[docno] = f"Document {docno}"
                except Exception:
                    texts[docno] = f"Document {docno}"
        except Exception as e:
            print(f"Warning: Could not fetch document texts: {e}")
            # Fallback: use docno as text for reranking
            texts = {docno: f"Document {docno}" for docno in docnos}
        return texts

    def rerank(self) -> Path:
        """
        Performs the entire re-ranking process for the specified configuration.
        """
        # Convert rewriter to proper source type
        if self.rewriter == "original":
            source_type = "original"
        elif self.rewriter == "summarized":
            source_type = "summarized"  
        else:
            source_type = f"rewritten_{self.rewriter}"
            
        # Load the corresponding query topics
        topics_df = self.data_loader.load_topics(self.dataset_version, source_type)
        
        # Load the fused run file to be re-ranked
        fused_run_path = Path(self.config.get_fusion_run_directory()) / self.dataset_version / f"{self.rewriter}_{self.dataset_version}_fused.txt"
        if not fused_run_path.exists():
            raise FileNotFoundError(f"Fused run file not found: {fused_run_path}")
        
        run_df = pt.io.read_results(str(fused_run_path))

        reranked_results = []

        # Group by query and re-rank documents for each
        for qid, group in tqdm(run_df.groupby('qid'), desc=f"Re-ranking for {self.rewriter} with {self.model_name}"):
            query_text = topics_df[topics_df['qid'] == qid]['query'].iloc[0]
            
            docnos = group['docno'].tolist()
            doc_texts = self._get_document_texts(docnos)
            
            # Clean and prepare texts for encoding
            query_text_clean = str(query_text).strip()
            if not query_text_clean:
                query_text_clean = "empty query"
                
            doc_texts_clean = []
            for docno in docnos:
                doc_text = doc_texts.get(docno, f"Document {docno}")
                doc_text_clean = str(doc_text).strip()
                if not doc_text_clean:
                    doc_text_clean = f"Document {docno}"
                doc_texts_clean.append(doc_text_clean)
            
            try:
                # Encode query and document texts with batch processing
                query_embedding = self.model.encode(
                    query_text_clean, 
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=1
                )
                
                # Batch encode documents for maximum throughput
                doc_embeddings = self.model.encode(
                    doc_texts_clean, 
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=min(self.batch_size, len(doc_texts_clean))
                )
                
                # Compute cosine similarity
                scores = cos_sim(query_embedding, doc_embeddings)[0].cpu().tolist()
                
                # Create a new ranked list
                for i, docno in enumerate(docnos):
                    reranked_results.append({'qid': qid, 'docno': docno, 'score': scores[i]})
                    
            except Exception as e:
                print(f"Warning: Failed to encode texts for query {qid}: {e}")
                # Fallback: use original scores from fusion
                for i, docno in enumerate(docnos):
                    original_score = group[group['docno'] == docno]['score'].iloc[0]
                    reranked_results.append({'qid': qid, 'docno': docno, 'score': original_score})

        # Create DataFrame and calculate new ranks
        reranked_df = pd.DataFrame(reranked_results)
        reranked_df['rank'] = reranked_df.groupby('qid')['score'].rank(method='first', ascending=False).astype(int)
        reranked_df = reranked_df.sort_values(['qid', 'rank'])

        # Save the re-ranked run file
        model_tag = self.model_name.split('/')[-1] # Use model name for file tagging
        output_filename = f"{self.rewriter}_{self.dataset_version}_{model_tag}_reranked.txt"
        output_path = self.run_dir / output_filename
        
        with open(output_path, 'w') as f_out:
            for _, row in reranked_df.iterrows():
                f_out.write(f"{row['qid']} Q0 {row['docno']} {row['rank']} {row['score']:.6f} {model_tag}\n")
        
        print(f"Saved dense re-ranked run file to: {output_path}")
        return output_path