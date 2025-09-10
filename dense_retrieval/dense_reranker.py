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
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Load PyTerrier index for fetching document text
        self.index = self.data_loader.get_index()

        # Define and create output directories
        self.run_dir = Path(self.config.get_path("dense_run_directory")) / self.dataset_version
        self.eval_dir = Path(self.config.get_path("dense_eval_directory")) / self.dataset_version
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def _get_document_texts(self, docnos: list) -> dict:
        """Fetch the 'text' metadata for a list of docnos from the index."""
        texts = {}
        # PyTerrier's get_meta is efficient for batch lookups
        meta_df = self.index.getMetaIndex().getDocuments("docno", docnos)
        for _, row in meta_df.iterrows():
            texts[row['docno']] = row['text']
        return texts

    def rerank(self) -> Path:
        """
        Performs the entire re-ranking process for the specified configuration.
        """
        # Load the corresponding query topics
        topics_df = self.data_loader.load_topics(self.dataset_version, self.rewriter)
        
        # Load the fused run file to be re-ranked
        fused_run_path = Path(self.config.get_path("fusion_run_directory")) / self.dataset_version / f"{self.rewriter}_{self.dataset_version}_fused.txt"
        if not fused_run_path.exists():
            raise FileNotFoundError(f"Fused run file not found: {fused_run_path}")
        
        run_df = pt.io.read_results(str(fused_run_path))

        reranked_results = []

        # Group by query and re-rank documents for each
        for qid, group in tqdm(run_df.groupby('qid'), desc=f"Re-ranking for {self.rewriter} with {self.model_name}"):
            query_text = topics_df[topics_df['qid'] == qid]['query'].iloc[0]
            
            docnos = group['docno'].tolist()
            doc_texts = self._get_document_texts(docnos)
            
            # Encode query and document texts
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)
            doc_embeddings = self.model.encode([doc_texts.get(d, "") for d in docnos], convert_to_tensor=True)
            
            # Compute cosine similarity
            scores = cos_sim(query_embedding, doc_embeddings)[0].cpu().tolist()
            
            # Create a new ranked list
            for i, docno in enumerate(docnos):
                reranked_results.append({'qid': qid, 'docno': docno, 'score': scores[i]})

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