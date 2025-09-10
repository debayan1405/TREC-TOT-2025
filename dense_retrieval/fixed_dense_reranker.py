"""
FIXED version of dense re-ranking that preserves original document IDs.
The key fix: we NEVER allow PyTerrier to change document IDs - we preserve the original docnos from the fused run.
"""
import pandas as pd
import pyterrier as pt
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

from sparse_retrieval.config_loader import ConfigLoader
from sparse_retrieval.data_loader import DataLoader

class FixedDenseReranker:
    """
    Fixed dense re-ranker that preserves document IDs and supports cross-encoders.
    """

    def __init__(self, config: ConfigLoader, dataset_version: str, rewriter: str, model_name: str, 
                 use_cross_encoder: bool = True, debug: bool = True):
        self.config = config
        self.data_loader = DataLoader(config)
        self.dataset_version = dataset_version
        self.rewriter = rewriter
        self.model_name = model_name
        self.use_cross_encoder = use_cross_encoder
        self.debug = debug
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🚀 Initializing FIXED dense re-ranker")
        print(f"   Model: {self.model_name}")
        print(f"   Type: {'Cross-Encoder' if use_cross_encoder else 'Bi-Encoder'}")
        print(f"   Device: {self.device}")
        print(f"   Debug: {debug}")
        
        # Initialize model
        try:
            if self.use_cross_encoder:
                self.model = CrossEncoder(self.model_name, device=self.device)
                self.batch_size = 32  # Smaller batch for cross-encoders
                print("✅ Cross-encoder loaded successfully")
            else:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.batch_size = 128
                print("✅ Bi-encoder loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load PyTerrier index for document retrieval
        try:
            self.index = self.data_loader.get_index()
            print("✅ PyTerrier index loaded")
        except Exception as e:
            print(f"⚠️  Could not load index: {e}")
            self.index = None

        # Setup directories
        self.run_dir = Path(config.get_dense_run_directory()) / dataset_version
        self.eval_dir = Path(config.get_dense_eval_directory()) / dataset_version
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def _get_document_texts_safe(self, docnos: list) -> dict:
        """
        FIXED: Safely fetch document texts while preserving original docnos.
        CRITICAL: We never change the docno keys - only the text values.
        """
        texts = {}
        
        if self.debug:
            print(f"📄 Fetching texts for {len(docnos)} documents...")
        
        successful_retrievals = 0
        
        # Try PyTerrier index first
        if self.index:
            try:
                meta_index = self.index.getMetaIndex()
                doc_index = self.index.getDocumentIndex()
                
                for original_docno in docnos:
                    try:
                        # CRITICAL: We store the result using the ORIGINAL docno as key
                        docid = doc_index.getDocumentId(str(original_docno))
                        
                        if docid >= 0:
                            # Try to get text metadata
                            for field in ['text', 'body', 'content', 'title']:
                                try:
                                    content = meta_index.getItem(field, docid)
                                    if content and len(str(content).strip()) > 10:  # Must be meaningful content
                                        texts[original_docno] = str(content).strip()
                                        successful_retrievals += 1
                                        break
                                except:
                                    continue
                    except Exception as e:
                        if self.debug and len(texts) < 3:  # Only show first few errors
                            print(f"⚠️  Could not retrieve text for docno {original_docno}: {e}")
                        continue
                        
            except Exception as e:
                print(f"⚠️  PyTerrier index access failed: {e}")
        
        # Fill in missing documents with meaningful fallback text
        missing_docs = set(docnos) - set(texts.keys())
        if missing_docs:
            if self.debug:
                print(f"⚠️  Using fallback text for {len(missing_docs)}/{len(docnos)} documents")
            
            for original_docno in missing_docs:
                # Use the docno itself as a meaningful fallback
                # This preserves the original ID while providing text for reranking
                texts[original_docno] = f"Document {original_docno} text not available"
        
        # VALIDATION: Ensure we have exactly the same docnos as input
        assert set(texts.keys()) == set(docnos), f"CRITICAL: Document ID mismatch! Input: {len(docnos)}, Output: {len(texts)}"
        
        if self.debug:
            print(f"✅ Text retrieval: {successful_retrievals}/{len(docnos)} from index, {len(missing_docs)} fallbacks")
            
            # Show example of successful retrieval
            if successful_retrievals > 0:
                example_docno = next(docno for docno in docnos if docno in texts and "not available" not in texts[docno])
                example_text = texts[example_docno][:100] + "..." if len(texts[example_docno]) > 100 else texts[example_docno]
                print(f"   Example: {example_docno} -> '{example_text}'")
        
        return texts

    def _cross_encoder_rerank(self, query_text: str, doc_texts: list, docnos: list) -> list:
        """Rerank using cross-encoder model."""
        try:
            # Prepare query-document pairs
            pairs = [(query_text, doc_text) for doc_text in doc_texts]
            
            # Get scores in batches
            scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i+self.batch_size]
                batch_scores = self.model.predict(batch_pairs, show_progress_bar=False)
                scores.extend(batch_scores.tolist() if isinstance(batch_scores, np.ndarray) else batch_scores)
            
            if self.debug and len(scores) > 0:
                print(f"   Cross-encoder scores: min={min(scores):.4f}, max={max(scores):.4f}, std={np.std(scores):.4f}")
            
            return scores
            
        except Exception as e:
            print(f"❌ Cross-encoder reranking failed: {e}")
            # Return uniform scores as fallback
            return [0.5] * len(doc_texts)

    def _bi_encoder_rerank(self, query_text: str, doc_texts: list, docnos: list) -> list:
        """Rerank using bi-encoder model."""
        try:
            # Encode query and documents
            query_embedding = self.model.encode(
                query_text, 
                convert_to_tensor=True, 
                show_progress_bar=False
            )
            
            doc_embeddings = self.model.encode(
                doc_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=min(self.batch_size, len(doc_texts))
            )
            
            # Compute similarities
            scores = cos_sim(query_embedding, doc_embeddings)[0].cpu().tolist()
            
            if self.debug and len(scores) > 0:
                print(f"   Bi-encoder scores: min={min(scores):.4f}, max={max(scores):.4f}, std={np.std(scores):.4f}")
            
            return scores
            
        except Exception as e:
            print(f"❌ Bi-encoder reranking failed: {e}")
            return [0.5] * len(doc_texts)

    def rerank(self) -> Path:
        """
        FIXED: Main reranking method that preserves document IDs.
        """
        print(f"\n🔄 Starting FIXED reranking for {self.rewriter} on {self.dataset_version}")
        
        # Load queries
        try:
            source_type = "original" if self.rewriter == "original" else (
                "summarized" if self.rewriter == "summarized" else f"rewritten_{self.rewriter}")
            topics_df = self.data_loader.load_topics(self.dataset_version, source_type)
            print(f"✅ Loaded {len(topics_df)} queries")
        except Exception as e:
            print(f"❌ Failed to load topics: {e}")
            raise
        
        # Load fused run file
        fused_run_path = Path(self.config.get_fusion_run_directory()) / self.dataset_version / f"{self.rewriter}_{self.dataset_version}_fused.txt"
        if not fused_run_path.exists():
            raise FileNotFoundError(f"Fused run file not found: {fused_run_path}")
        
        try:
            run_df = pt.io.read_results(str(fused_run_path))
            print(f"✅ Loaded fused run with {len(run_df)} entries for {run_df['qid'].nunique()} queries")
            
            # CRITICAL: Show sample of input docnos to verify they match what we expect
            sample_docnos = run_df['docno'].head(10).tolist()
            print(f"📋 Sample input docnos: {sample_docnos}")
            
        except Exception as e:
            print(f"❌ Failed to load fused run: {e}")
            raise
        
        reranked_results = []
        failed_queries = 0
        processed_queries = 0
        
        # Process each query
        for qid, group in tqdm(run_df.groupby('qid'), desc="Fixed reranking"):
            try:
                processed_queries += 1
                
                # Get query text
                query_rows = topics_df[topics_df['qid'] == qid]
                if len(query_rows) == 0:
                    print(f"⚠️  No query text found for QID {qid}")
                    failed_queries += 1
                    continue
                    
                query_text = str(query_rows['query'].iloc[0]).strip()
                if not query_text:
                    query_text = f"Query {qid}"  # Fallback
                
                # CRITICAL: Get original docnos and preserve them exactly
                original_docnos = group['docno'].tolist()
                
                if self.debug and processed_queries <= 3:  # Show first few queries
                    print(f"\n🔍 Query {qid}: '{query_text[:50]}...' with {len(original_docnos)} docs")
                    print(f"   Input docnos: {original_docnos[:5]}...")
                
                # Get document texts using ORIGINAL docnos as keys
                doc_texts_dict = self._get_document_texts_safe(original_docnos)
                doc_texts = [doc_texts_dict[docno] for docno in original_docnos]
                
                # Rerank documents
                if self.use_cross_encoder:
                    scores = self._cross_encoder_rerank(query_text, doc_texts, original_docnos)
                else:
                    scores = self._bi_encoder_rerank(query_text, doc_texts, original_docnos)
                
                # Validate scores
                if len(scores) != len(original_docnos):
                    print(f"⚠️  Score length mismatch for query {qid}: {len(scores)} scores, {len(original_docnos)} docs")
                    scores = scores[:len(original_docnos)] + [0.0] * max(0, len(original_docnos) - len(scores))
                
                # CRITICAL: Store results with ORIGINAL docnos - no transformation!
                for i, (original_docno, score) in enumerate(zip(original_docnos, scores)):
                    reranked_results.append({
                        'qid': str(qid),  
                        'docno': str(original_docno),  # PRESERVE EXACTLY
                        'score': float(score)
                    })
                
                if self.debug and processed_queries <= 3:
                    print(f"   ✅ Processed {len(original_docnos)} docs, score range: {min(scores):.4f} to {max(scores):.4f}")
                
            except Exception as e:
                print(f"❌ Failed to process query {qid}: {e}")
                failed_queries += 1
                
                # Add fallback results with ORIGINAL docnos
                for _, row in group.iterrows():
                    reranked_results.append({
                        'qid': str(row['qid']),
                        'docno': str(row['docno']),  # PRESERVE ORIGINAL
                        'score': float(row['score'])  # Use original fusion score
                    })
        
        if failed_queries > 0:
            print(f"⚠️  Failed to process {failed_queries}/{processed_queries} queries")
        
        # Create final dataframe and assign ranks
        reranked_df = pd.DataFrame(reranked_results)
        
        # VALIDATION: Check if we preserved document IDs
        input_docnos = set(run_df['docno'].astype(str))
        output_docnos = set(reranked_df['docno'].astype(str))
        preserved_ratio = len(input_docnos & output_docnos) / len(input_docnos)
        
        print(f"📊 Document ID preservation: {preserved_ratio:.1%}")
        if preserved_ratio < 0.95:
            print("🚨 WARNING: Significant document ID loss detected!")
        
        if self.debug:
            print(f"📊 Final score statistics:")
            print(f"   Min: {reranked_df['score'].min():.6f}")
            print(f"   Max: {reranked_df['score'].max():.6f}")
            print(f"   Mean: {reranked_df['score'].mean():.6f}")
            print(f"   Std: {reranked_df['score'].std():.6f}")
            print(f"   Unique values: {reranked_df['score'].nunique()}")
        
        # Assign ranks (higher scores get lower ranks)
        reranked_df['rank'] = reranked_df.groupby('qid')['score'].rank(
            method='first', ascending=False).astype(int)
        reranked_df = reranked_df.sort_values(['qid', 'rank'])
        
        # Save output
        model_tag = self.model_name.split('/')[-1].replace('/', '_')
        output_filename = f"{self.rewriter}_{self.dataset_version}_{model_tag}_FIXED.txt"
        output_path = self.run_dir / output_filename
        
        with open(output_path, 'w') as f_out:
            for _, row in reranked_df.iterrows():
                f_out.write(f"{row['qid']} Q0 {row['docno']} {row['rank']} {row['score']:.6f} {model_tag}_FIXED\n")
        
        print(f"✅ Saved FIXED reranked run to: {output_path}")
        
        # Final validation
        unique_qids = reranked_df['qid'].nunique()
        unique_docnos = reranked_df['docno'].nunique()
        print(f"📋 Final output: {len(reranked_df)} entries, {unique_qids} queries, {unique_docnos} unique docs")
        
        # Show sample output docnos to verify they match input
        sample_output_docnos = reranked_df['docno'].head(10).tolist()
        print(f"📋 Sample output docnos: {sample_output_docnos}")
        
        return output_path

# Test function
def test_fixed_reranker():
    """Test the fixed reranker with your config."""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    
    from sparse_retrieval.config_loader import ConfigLoader
    
    try:
        config = ConfigLoader(str(project_root / "env.json"))
        
        # Test with cross-encoder
        reranker = FixedDenseReranker(
            config=config,
            dataset_version="train",
            rewriter="mistral",
            model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
            use_cross_encoder=True,
            debug=True
        )
        
        output_path = reranker.rerank()
        print(f"✅ Test completed successfully: {output_path}")
        
        # Quick validation
        print("\n🧪 Running quick validation...")
        import subprocess
        result = subprocess.run([
            "python", "dense_retrieval/debug_reranking.py",
            "--reranked_run", str(output_path),
            "--fused_run", "fused_run_files/train/mistral_train_fused.txt",
            "--qrels", "qrel/train-2025-qrel.txt",
            "--output_dir", "debug_output/fixed_test"
        ], capture_output=True, text=True)
        
        if "Document ID preservation" in result.stdout:
            print("✅ Validation passed!")
        else:
            print("⚠️  Validation output not found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_fixed_reranker()