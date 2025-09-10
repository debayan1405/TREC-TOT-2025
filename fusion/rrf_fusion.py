"""
Implements Reciprocal Rank Fusion (RRF) and evaluation for sparse retrieval runs.
This module groups run files by query re-writer, performs RRF for each group,
and evaluates the fused results.
"""

import os
import sys
from pathlib import Path

# Fix Java environment before importing PyTerrier
def setup_java_environment():
    """Set up Java environment for PyTerrier."""
    
    # Find the correct Java installation
    possible_java_paths = [
        "/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm",
        "/home/ugdf8/anaconda3/pkgs/openjdk-17.0.15-h5ddf6bc_0/lib/jvm",
        "/home/ugdf8/anaconda3/pkgs/openjdk-21.0.6-h38aa4c6_0/lib",
        "/home/ugdf8/.vscode/extensions/redhat.java-1.45.0-linux-x64/jre/21.0.8-linux-x86_64",
    ]
    
    java_home = None
    for path in possible_java_paths:
        libjvm_path = Path(path) / "lib" / "server" / "libjvm.so"
        if libjvm_path.exists():
            java_home = path
            print(f"✓ Found Java at: {java_home}")
            break
    
    if not java_home:
        print("ERROR: Could not find a working Java installation")
        sys.exit(1)
    
    # Set environment variables
    os.environ['JAVA_HOME'] = java_home
    os.environ['LD_LIBRARY_PATH'] = f"{java_home}/lib/server:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ['JVM_PATH'] = f"{java_home}/lib/server/libjvm.so"
    
    print(f"✓ Set JAVA_HOME to: {java_home}")
    return java_home

# Setup Java environment before importing PyTerrier
setup_java_environment()

import pandas as pd
import pyterrier as pt
from collections import defaultdict

# Add project root to the Python path to allow sibling imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sparse_retrieval.data_loader import DataLoader
from sparse_retrieval.config_loader import ConfigLoader


class RRFusion:
    """
    Handles the Reciprocal Rank Fusion and evaluation process.
    """

    def __init__(self, config: ConfigLoader, dataset_version: str):
        """
        Initializes the RRFusion process.

        Args:
            config (ConfigLoader): Configuration object.
            dataset_version (str): The dataset to process (e.g., "train", "dev-1").
        """
        self.config = config
        self.data_loader = DataLoader(config)  # Initialize DataLoader
        self.dataset_version = dataset_version
        self.rrf_k = self.config.get_rrf_k()

        # Define and create necessary directories
        self.sparse_run_dir = Path(
            self.config.get_sparse_run_directory()) / self.dataset_version
        self.fusion_run_dir = Path(
            self.config.get_fusion_run_directory()) / self.dataset_version
        self.eval_dir = self.fusion_run_dir / "evals"

        self.fusion_run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PyTerrier with modern API
        if not hasattr(pt, 'java') or not pt.java.started():
            print("🔧 Initializing PyTerrier for RRF fusion...")
            pt.init(mem=4096, logging="WARN")
            print("✓ PyTerrier initialized successfully")

    def _reciprocal_rank_fusion(self, list_of_run_files: list) -> dict:
        """
        Performs Reciprocal Rank Fusion on a list of run files, preserving query IDs.

        Args:
            list_of_run_files (list): A list of paths to run files.

        Returns:
            dict: A dictionary of {(qid, doc_id): score}.
        """
        fused_scores = defaultdict(float)

        for run_file in list_of_run_files:
            df = pd.read_csv(
                run_file,
                sep=r'\s+',
                header=None,
                names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name'],
                dtype={'qid': str, 'docno': str}
            )

            for _, row in df.iterrows():
                fused_scores[(row['qid'], row['docno'])] += 1.0 / \
                    (self.rrf_k + row['rank'])

        return fused_scores

    def _group_runs_by_rewriter(self) -> dict:
        """
        Scans the sparse run directory and groups run files by the rewriter name.
        """
        if not self.sparse_run_dir.exists():
            raise FileNotFoundError(
                f"Sparse run directory not found: {self.sparse_run_dir}")

        all_run_files = list(self.sparse_run_dir.glob("*.txt"))
        rewriter_groups = defaultdict(list)

        for run_file in all_run_files:
            rewriter_name = run_file.name.split('_')[0]
            rewriter_groups[rewriter_name].append(str(run_file))

        print(
            f"Found {len(rewriter_groups)} re-writer groups: {list(rewriter_groups.keys())}")
        return rewriter_groups

    def run_fusion_and_evaluation(self):
        """
        Main execution method to run fusion and evaluation for all re-writer groups.
        """
        rewriter_groups = self._group_runs_by_rewriter()

        if not rewriter_groups:
            print(f"No run files found in {self.sparse_run_dir}. Aborting.")
            return

        # Load Qrels and Topics for the dataset
        qrels = None
        topics = None
        can_evaluate = False
        try:
            qrels = self.data_loader.load_qrels(self.dataset_version)
            # We load 'original' topics as the baseline for qids
            topics = self.data_loader.load_topics(
                self.dataset_version, "original")
            can_evaluate = True
            print(
                f"Successfully loaded qrels and topics for '{self.dataset_version}'")
        except Exception as e:
            print(
                f"Warning: Could not load qrels/topics. Skipping evaluation. Error: {e}")

        # Process each re-writer group
        for rewriter, run_files in rewriter_groups.items():
            print(f"\n--- Processing re-writer: {rewriter} ---")
            print(f"Fusing {len(run_files)} run files...")

            fused_scores = self._reciprocal_rank_fusion(run_files)
            if not fused_scores:
                print(
                    f"Warning: Fusion for '{rewriter}' resulted in no scores. Skipping.")
                continue

            data = [{'qid': qid, 'docno': docno, 'score': score}
                    for (qid, docno), score in fused_scores.items()]
            fused_df = pd.DataFrame(data)

            fused_df['rank'] = fused_df.groupby('qid')['score'].rank(
                method='first', ascending=False).astype(int)
            fused_df = fused_df.sort_values(['qid', 'rank'])

            output_path = self.fusion_run_dir / \
                f"{rewriter}_{self.dataset_version}_fused.txt"

            with open(output_path, 'w') as f_out:
                for _, row in fused_df.iterrows():
                    f_out.write(
                        f"{row['qid']} Q0 {row['docno']} {row['rank']} {row['score']:.6f} {rewriter}_fused\n")

            print(f"Saved fused run file for '{rewriter}' to: {output_path}")

            # Run evaluation
            if can_evaluate and qrels is not None and topics is not None:
                try:
                    eval_df = pt.Experiment(
                        [fused_df],
                        topics,
                        qrels,
                        eval_metrics=self.config.get_eval_metrics(),
                        names=[rewriter]
                    )

                    if eval_df is not None and not eval_df.empty:
                        eval_path = self.eval_dir / \
                            f"{rewriter}_{self.dataset_version}_eval.csv"
                        eval_df.to_csv(eval_path)
                        print(
                            f"Saved evaluation for '{rewriter}' to: {eval_path}")
                        print("Evaluation Results:")
                        print(eval_df.to_string())
                    else:
                        print(
                            f"Warning: Evaluation for '{rewriter}' produced no results.")

                except Exception as e:
                    print(f"Error during evaluation for '{rewriter}': {e}")
