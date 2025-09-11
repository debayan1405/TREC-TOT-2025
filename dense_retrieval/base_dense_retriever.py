"""
Base Dense Retriever Framework
Provides common functionality for all dense retrieval stages
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import time
import json
from abc import ABC, abstractmethod

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "sparse_retrieval"))

from config_loader import ConfigLoader
from data_loader import DataLoader

class BaseDenseRetriever(ABC):
    """Base class for all dense retrievers with common functionality"""
    
    def __init__(self, config: ConfigLoader, dataset_version: str, rewriter: str, stage: str, model_name: str):
        """
        Initialize base dense retriever
        
        Args:
            config: Configuration loader
            dataset_version: Dataset version (train, test, etc.)
            rewriter: Query rewriter type (llama, mistral, etc.)
            stage: Stage identifier for this retriever
            model_name: Model name for identification
        """
        
        self.config = config
        self.data_loader = DataLoader(config)
        self.dataset_version = dataset_version
        self.rewriter = rewriter
        self.stage = stage
        self.model_name = model_name
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent
        
        # Setup device for GPU optimization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # PyTerrier Index for document text retrieval
        try:
            import pyterrier as pt
            # Initialize PyTerrier if not already done
            if not hasattr(pt, 'java') or pt.java is None:
                pt.init()
            elif hasattr(pt.java, 'is_started') and not pt.java.is_started():
                pt.init()
            else:
                # For newer versions, just try to init
                try:
                    pt.init()
                except:
                    pass  # Already initialized
            
            self.index = pt.IndexRef.of("/home/ugdf8/IRIS/TREC-TOT-2025/trec-tot-2025-pyterrier-index")
            self.pt_index = pt.IndexFactory.of(self.index)
            print(f"   ✅ PyTerrier index loaded: {self.pt_index.getCollectionStatistics().getNumberOfDocuments()} documents")
        except Exception as e:
            print(f"   ⚠️  PyTerrier index loading failed: {e}")
            self.index = None
            self.pt_index = None
        
        print(f"🚀 Initializing {self.__class__.__name__}:")
        print(f"   Dataset: {dataset_version}")
        print(f"   Rewriter: {rewriter}")
        print(f"   Stage: {stage}")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
    
    def _setup_hardware(self):
        """Setup hardware optimization"""
        cpu_count = os.cpu_count()
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
        
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model (implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _score_documents(self, query_text, doc_texts):
        """Score documents for a query (implemented by subclasses)"""
        pass
    
    def load_input_data(self, file_path: str, top_k: int = None) -> pd.DataFrame:
        """Load TREC run format file for reranking."""
        try:
            results = pd.read_csv(
                file_path, 
                sep=' ', 
                header=None,
                names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name']
            )
            
            # Apply top-k filtering if specified
            if top_k:
                results = results.groupby('qid').head(top_k).reset_index(drop=True)
                
            print(f"Loaded {len(results)} results from {file_path}")
            return results
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            raise
    
    def load_rewritten_queries(self):
        """Load rewritten queries directly from rewritten_queries directory"""
        
        # Construct path to rewritten queries file
        rewritten_file = Path(self.project_root) / "rewritten_queries" / f"{self.rewriter}_{self.dataset_version}_rewritten_queries.jsonl"
        
        if not rewritten_file.exists():
            raise FileNotFoundError(f"Rewritten queries file not found: {rewritten_file}")
        
        # Load JSONL file
        topics_dict = {}
        with open(rewritten_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Convert to int for consistent lookup with results file
                qid = int(data['query_id'])
                query = data['query']
                topics_dict[qid] = query
        
        print(f"   Loaded {len(topics_dict)} rewritten queries from {rewritten_file.name}")
        return topics_dict
    
    def _get_document_texts_safe(self, docnos, debug=False):
        """
        Safely retrieve document texts, with fallbacks for missing documents
        CRITICAL: Returns dict with ORIGINAL docnos as keys (preserving input format)
        """
        texts = {}
        
        if debug:
            print(f"📄 Fetching texts for {len(docnos)} documents...")
        
        successful_retrievals = 0
        
        # Try PyTerrier index with proper text retrieval
        if self.index:
            try:
                import pandas as pd
                import pyterrier as pt
                import pyterrier.text
                
                # Create dataframe for PyTerrier text retrieval
                docno_strings = [str(docno) for docno in docnos]
                docs_df = pd.DataFrame({
                    'qid': ['query'] * len(docno_strings),
                    'docno': docno_strings,
                    'score': [1.0] * len(docno_strings),
                    'rank': list(range(1, len(docno_strings) + 1))
                })
                
                # Use PyTerrier's text retrieval pipeline
                text_pipeline = pyterrier.text.get_text(self.index)
                result_with_text = text_pipeline(docs_df)
                
                # Extract texts
                for _, row in result_with_text.iterrows():
                    docno = row['docno']
                    if 'text' in row and pd.notna(row['text']):
                        text_content = str(row['text']).strip()
                        if len(text_content) > 10:  # Must be meaningful content
                            texts[docno] = text_content
                            successful_retrievals += 1
                        else:
                            texts[docno] = f"Document {docno}: Content placeholder for dense retrieval analysis"
                    else:
                        texts[docno] = f"Document {docno}: Content placeholder for dense retrieval analysis"
                        
            except Exception as e:
                if debug:
                    print(f"   PyTerrier text retrieval failed: {e}")
                
                # Fallback to placeholder texts
                for docno in docnos:
                    texts[str(docno)] = f"Document {docno}: Content placeholder for dense retrieval analysis"
        else:
            # No PyTerrier index available
            for docno in docnos:
                texts[str(docno)] = f"Document {docno}: Content placeholder for dense retrieval analysis"
        
        # Ensure we have text for all requested documents
        for docno in docnos:
            if str(docno) not in texts:
                texts[str(docno)] = f"Document {docno}: Content placeholder for dense retrieval analysis"
        
        # VALIDATION: Ensure we have exactly the same docnos as input  
        input_docnos_str = set([str(docno) for docno in docnos])
        assert set(texts.keys()) == input_docnos_str, f"CRITICAL: Document ID mismatch! Input: {len(docnos)}, Output: {len(texts)}"
        
        if debug:
            print(f"✅ Text retrieval: {successful_retrievals}/{len(docnos)} from index, {len(docnos)-successful_retrievals} fallbacks")
            
            # Show example of successful retrieval
            if successful_retrievals > 0:
                example_docno = next(docno for docno in texts.keys() if "placeholder" not in texts[docno])
                example_text = texts[example_docno][:100] + "..." if len(texts[example_docno]) > 100 else texts[example_docno]
                print(f"   Example retrieved: {example_docno} -> '{example_text}'")
        
        return texts
    
    def rerank_documents(self, input_file, top_k_input=None, top_k_output=1000):
        """
        Main reranking pipeline with proper query and document loading.
        
        Args:
            input_file: Path to input run file
            top_k_input: Max documents per query to load from input
            top_k_output: Max documents per query to output
        
        Returns:
            List of reranked results
        """
        
        print(f"🔄 Starting {self.stage} reranking:")
        print(f"   Input top-k: {top_k_input}")
        print(f"   Output top-k: {top_k_output}")
        
        # Load input data and queries
        results = self.load_input_data(input_file, top_k_input)
        topics_dict = self.load_rewritten_queries()
        
        # Process queries properly grouped by query ID
        reranked_results = []
        unique_qids = results['qid'].unique()
        
        print(f"   Processing {len(unique_qids)} unique queries...")
        failed_queries = 0
        processed_queries = 0
        
        for qid in tqdm(unique_qids, desc=f"Reranking with {self.model_name}"):
            try:
                processed_queries += 1
                
                # Get all documents for this query ID
                query_results = results[results['qid'] == qid]
                
                if len(query_results) == 0:
                    continue
                
                # Get rewritten query text from topics dict
                query_text = topics_dict.get(qid)  # Use integer key directly
                if not query_text:
                    print(f"⚠️  No rewritten query found for QID {qid}")
                    continue
                
                if processed_queries <= 3:  # Debug first few queries
                    print(f"\\n🔍 Query {qid}: '{query_text[:50]}...' with {len(query_results)} docs")
                
                # Get original docnos and preserve them exactly
                original_docnos = query_results['docno'].tolist()
                
                # Get actual document texts using ORIGINAL docnos as keys
                doc_texts_dict = self._get_document_texts_safe(original_docnos, debug=(processed_queries <= 3))
                doc_texts = [doc_texts_dict[str(docno)] for docno in original_docnos]
                
                # Score documents using specific model implementation
                scores = self._score_documents(query_text, doc_texts)
                
                # Validate scores
                if len(scores) != len(original_docnos):
                    print(f"⚠️  Score length mismatch for query {qid}: {len(scores)} scores, {len(original_docnos)} docs")
                    scores = scores[:len(original_docnos)] + [0.0] * max(0, len(original_docnos) - len(scores))
                
                # Create reranked results with ORIGINAL docnos
                for docno, score in zip(original_docnos, scores):
                    reranked_results.append({
                        'qid': str(qid),
                        'Q0': 'Q0',
                        'docno': str(docno),  # PRESERVE ORIGINAL
                        'rank': 1,  # Will be updated after sorting
                        'score': float(score),
                        'run_name': f"{self.model_name}_{self.stage}"
                    })
                
                if processed_queries <= 3:
                    print(f"   ✅ Processed {len(original_docnos)} docs, score range: {min(scores):.4f} to {max(scores):.4f}")
                
            except Exception as e:
                print(f"❌ Failed to process query {qid}: {e}")
                failed_queries += 1
                
                # Add fallback results with ORIGINAL docnos
                for _, row in query_results.iterrows():
                    reranked_results.append({
                        'qid': str(row['qid']),
                        'Q0': 'Q0',
                        'docno': str(row['docno']),  # PRESERVE ORIGINAL
                        'rank': int(row['rank']) if 'rank' in row else 1,
                        'score': float(row['score']) if 'score' in row else 0.0,
                        'run_name': f"{self.model_name}_{self.stage}"
                    })
        
        if failed_queries > 0:
            print(f"⚠️  Failed to process {failed_queries}/{processed_queries} queries")
        
        # Create final dataframe and assign ranks
        reranked_df = pd.DataFrame(reranked_results)
        
        # Assign ranks within each query (higher scores get lower ranks)
        reranked_df['rank'] = reranked_df.groupby('qid')['score'].rank(
            method='first', ascending=False).astype(int)
        
        # Sort by qid and rank, limit to top_k_output per query
        reranked_df = reranked_df.sort_values(['qid', 'rank'])
        final_df = reranked_df.groupby('qid').head(top_k_output).reset_index(drop=True)
        
        # Convert back to list of dictionaries
        final_results = final_df.to_dict('records')
        
        print(f"   ✓ Completed reranking {len(final_results)} documents for {len(unique_qids)} queries")
        return final_results
    
    def save_results(self, results, output_file):
        """Save results to TREC format file"""
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to TREC format
        with open(output_file, 'w') as f:
            for result in results:
                f.write(f"{result['qid']}\tQ0\t{result['docno']}\t{result['rank']}\t{result['score']:.6f}\t{result['run_name']}\n")
        
        print(f"   ✓ Saved {len(results)} results to {output_file}")
    
    def evaluate_results(self, results_file):
        """Evaluate results using the evaluation framework"""
        
        try:
            # Import evaluation framework
            sys.path.append(str(Path(__file__).parent.parent))
            from evaluation import SimpleEvaluator
            
            evaluator = SimpleEvaluator()
            qrels = evaluator.load_qrels(self.dataset_version)
            results = evaluator.load_results([results_file])
            
            if len(results) > 0:
                file_key = list(results.keys())[0]
                metrics = evaluator.evaluate_single(results[file_key], qrels)
                
                print(f"   📊 Evaluation Results:")
                print(f"      NDCG@10: {metrics['ndcg_cut_10']:.4f}")
                print(f"      Recall@1000: {metrics['recall_1000']:.4f}")
                print(f"      P@10: {metrics['P_10']:.4f}")
                
                return metrics
        except Exception as e:
            print(f"   Warning: Could not evaluate results: {e}")
            return None
    
    def get_output_path(self, dataset_version):
        """Get output path for this retriever"""
        return Path(self.config.get_dense_run_directory()) / "run_files" / self.stage / f"{self.rewriter}_{dataset_version}_{self.model_name}.txt"
    
    def get_evaluation_path(self, dataset_version):
        """Get evaluation results path"""
        return Path(self.config.get_dense_run_directory()) / "evaluations" / f"{self.rewriter}_{dataset_version}_{self.stage}_{self.model_name}_eval.json"
