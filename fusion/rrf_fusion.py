"""
Implements Reciprocal Rank Fusion (RRF) and evaluation for sparse retrieval runs.
This module groups run files by query re-writer, performs RRF for each group,
and evaluates the fused results.
"""

# Add project root to the Python path to allow sibling imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sparse_retrieval.config_loader import ConfigLoader

import pandas as pd
import pyterrier as pt
from collections import defaultdict
from pathlib import Path
import sys

# Add parent directory to path to import ConfigLoader
sys.path.append(str(Path(__file__).parent.parent / 'sparse_retrieval'))


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

        if not pt.started():
            pt.init()

    def _reciprocal_rank_fusion(self, list_of_run_files: list) -> dict:
        """
        Performs Reciprocal Rank Fusion on a list of run files.

        Args:
            list_of_run_files (list): A list of paths to run files.

        Returns:
            dict: A dictionary of {doc_id: score} for a single query.
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
                fused_scores[row['docno']] += 1.0 / (self.rrf_k + row['rank'])

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
            # Filename format: {rewriter}_{dataset}_{model}_{k}.txt
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

        # Load Qrels once for the dataset
        can_evaluate = False
        qrels = None
        try:
            qrels_path = self.config.get_qrels_path(self.dataset_version)
            qrels = pt.io.read_qrels(qrels_path)
            can_evaluate = True
            print(
                f"Successfully loaded qrels for '{self.dataset_version}' from {qrels_path}")
        except Exception as e:
            print(
                f"Warning: Could not load qrels for '{self.dataset_version}'. Skipping evaluation. Error: {e}")

        # Process each re-writer group
        for rewriter, run_files in rewriter_groups.items():
            print(f"\n--- Processing re-writer: {rewriter} ---")
            print(f"Fusing {len(run_files)} run files...")

            # Step 1: Perform Fusion
            fused_scores = self._reciprocal_rank_fusion(run_files)

            # Step 2: Format and save the fused results
            sorted_docs = sorted(fused_scores.items(),
                                 key=lambda item: item[1], reverse=True)

            output_path = self.fusion_run_dir / \
                f"{rewriter}_{self.dataset_version}_fused.txt"

            # Assuming a single query ID '1' for all docs in the fusion.
            # This is a common practice for saving fused lists if qid is not tracked.
            with open(output_path, 'w') as f_out:
                for i, (doc_id, score) in enumerate(sorted_docs):
                    rank = i + 1
                    f_out.write(
                        f"1 Q0 {doc_id} {rank} {score:.6f} {rewriter}_fused\n")

            print(f"Saved fused run file for '{rewriter}' to: {output_path}")

            # Step 3: Run evaluation
            if can_evaluate and qrels is not None:
                try:
                    results_df = pt.io.read_results(str(output_path))
                    eval_metrics = self.config.get_eval_metrics()

                    eval_results = pt.Utils.evaluate(
                        results_df, qrels, metrics=eval_metrics)

                    eval_df = pd.DataFrame([eval_results])
                    eval_df.index = [rewriter]

                    eval_path = self.eval_dir / \
                        f"{rewriter}_{self.dataset_version}_eval.csv"
                    eval_df.to_csv(eval_path)
                    print(f"Saved evaluation for '{rewriter}' to: {eval_path}")
                    print("Evaluation Results:")
                    print(eval_df.to_string())

                except Exception as e:
                    print(f"Error during evaluation for '{rewriter}': {e}")
