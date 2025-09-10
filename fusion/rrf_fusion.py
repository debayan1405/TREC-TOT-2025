"""
Core implementation of the Reciprocal Rank Fusion (RRF) algorithm.
"""
import pandas as pd
from typing import List

class RRFFusion:
    """
    Handles the Reciprocal Rank Fusion process.
    """

    def __init__(self, rrf_k: int = 60):
        """
        Initialize the RRF fusion process.

        Args:
            rrf_k (int): The ranking constant for RRF.
        """
        self.rrf_k = rrf_k

    def fuse(self, run_files: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Fuse a list of run files using RRF.

        Args:
            run_files (List[pd.DataFrame]): A list of pandas DataFrames,
                                             each representing a run file.

        Returns:
            pd.DataFrame: A single fused and re-ranked DataFrame.
        """
        if not run_files:
            raise ValueError("run_files list cannot be empty.")

        # Dictionary to store RRF scores for each document
        rrf_scores = {}

        for run_df in run_files:
            for _, row in run_df.iterrows():
                qid = str(row['qid'])
                docno = str(row['docno'])
                rank = int(row['rank'])

                # Initialize qid entry if not present
                if qid not in rrf_scores:
                    rrf_scores[qid] = {}

                # Calculate RRF score for the document
                rrf_score = 1 / (self.rrf_k + rank)

                # Add score to existing document score, or initialize
                rrf_scores[qid][docno] = rrf_scores[qid].get(docno, 0) + rrf_score

        # Convert the scores dictionary to a list of tuples
        fused_results = []
        for qid, doc_scores in rrf_scores.items():
            for docno, score in doc_scores.items():
                fused_results.append((qid, docno, score))

        # Create a DataFrame from the fused results
        fused_df = pd.DataFrame(fused_results, columns=['qid', 'docno', 'score'])

        # Sort by qid, then by score in descending order
        fused_df = fused_df.sort_values(['qid', 'score'], ascending=[True, False])

        # Assign new ranks
        fused_df['rank'] = fused_df.groupby('qid').cumcount() + 1

        return fused_df