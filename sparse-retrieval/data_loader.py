"""
Data loading module for PyTerrier experiments.
Handles loading topics, qrels, and managing index operations.
"""
import json
import pandas as pd
import pyterrier as pt
from pathlib import Path
from typing import Dict, Any, Optional
from config_loader import ConfigLoader


class DataLoader:
    """Handles loading of topics, qrels, and index management."""

    def __init__(self, config: ConfigLoader):
        """
        Initialize data loader.

        Args:
            config (ConfigLoader): Configuration loader instance
        """
        self.config = config
        self._index = None

    def load_topics(self) -> pd.DataFrame:
        """
        Load topics from JSONL file.

        Returns:
            pd.DataFrame: Topics dataframe with columns ['qid', 'query']
        """
        topics_path = Path(self.config.get_topics_path())

        if not topics_path.exists():
            raise FileNotFoundError(f"Topics file not found: {topics_path}")

        topics = []
        try:
            with open(topics_path, 'r', encoding='utf-8') as f:
                for line in f:
                    topic = json.loads(line.strip())
                    if 'q_id' not in topic or 'query' not in topic:
                        raise ValueError(
                            "Each topic must have 'q_id' and 'query' keys")
                    topics.append({
                        'qid': str(topic['q_id']),
                        'query': topic['query']
                    })
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in topics file: {e}")

        if not topics:
            raise ValueError("No topics found in topics file")

        return pd.DataFrame(topics)

    def load_qrels(self) -> pd.DataFrame:
        """
        Load qrels file.

        Returns:
            pd.DataFrame: Qrels dataframe
        """
        qrels_path = Path(self.config.get_qrels_path())

        if not qrels_path.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

        try:
            # Try loading as TREC format qrels
            qrels = pt.io.read_qrels(str(qrels_path))
            return qrels
        except Exception as e:
            raise ValueError(f"Error loading qrels file: {e}")

    def get_index(self) -> Any:
        """
        Get or load Terrier index.

        Returns:
            Terrier index object
        """
        if self._index is None:
            index_path = Path(self.config.get_index_path())

            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")

            try:
                self._index = pt.IndexFactory.of(str(index_path))
            except Exception as e:
                raise ValueError(f"Error loading index: {e}")

        return self._index

    def validate_data_consistency(self, topics: pd.DataFrame, qrels: pd.DataFrame,
                                  version: str = "train") -> None:
        """
        Validate that topics and qrels are consistent for specified version.

        Args:
            topics (pd.DataFrame): Topics dataframe
            qrels (pd.DataFrame): Qrels dataframe  
            version (str): Dataset version for logging purposes
        """
        topic_qids = set(topics['qid'].astype(str))
        qrel_qids = set(qrels['qid'].astype(str))

        if not topic_qids.intersection(qrel_qids):
            raise ValueError(
                f"No matching query IDs found between {version} topics and qrels")

        missing_in_qrels = topic_qids - qrel_qids
        if missing_in_qrels:
            print(
                f"Warning: {len(missing_in_qrels)} {version} topics have no relevance judgments")
