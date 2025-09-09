"""
Data loading module for PyTerrier experiments.
Handles loading topics, qrels, and managing index operations.
Updated to support both original and rewritten query formats with dynamic dataset handling.
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

    def load_topics(self, version: str = "train", source_type: str = "original") -> pd.DataFrame:
        """
        Load topics from JSONL file with support for different sources.

        Args:
            version (str): Dataset version ('train', 'dev-1', 'dev-2', 'dev-3', 'test')
            source_type (str): Type of topics to load ('original', 'rewritten_<model>', 'summarized')

        Returns:
            pd.DataFrame: Topics dataframe with columns ['qid', 'query']
        """
        if source_type == "original":
            topics_path = Path(self.config.get_topics_path(version))
        elif source_type.startswith("rewritten_"):
            # Load rewritten queries from specific model
            model_name = source_type.replace("rewritten_", "")
            rewritten_dir = Path(self.config.get_rewritten_queries_directory())
            topics_path = rewritten_dir / \
                f"{model_name}_{version}_rewritten_queries.jsonl"
        elif source_type == "summarized":
            # Load summarized queries
            rewritten_dir = Path(self.config.get_rewritten_queries_directory())
            topics_path = rewritten_dir / f"{version}_summarized.jsonl"
        else:
            raise ValueError(f"Invalid source_type: {source_type}")

        if not topics_path.exists():
            raise FileNotFoundError(f"Topics file not found: {topics_path}")

        topics = []
        try:
            with open(topics_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        topic = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON in topics file at line {line_num}: {e}")

                    # Handle both q_id and query_id field names
                    if 'q_id' in topic:
                        qid = str(topic['q_id'])
                    elif 'query_id' in topic:
                        qid = str(topic['query_id'])
                    else:
                        raise ValueError(
                            f"Each topic must have 'q_id' or 'query_id' key at line {line_num}")

                    if 'query' not in topic:
                        raise ValueError(
                            f"Each topic must have 'query' key at line {line_num}")

                    topics.append({
                        'qid': qid,
                        'query': topic['query']
                    })
        except Exception as e:
            raise ValueError(f"Error loading topics from {topics_path}: {e}")

        if not topics:
            raise ValueError("No topics found in topics file")

        df = pd.DataFrame(topics)
        print(
            f"Loaded {len(df)} topics from {topics_path.name} (source: {source_type})")
        return df

    def load_qrels(self, version: str = "train") -> pd.DataFrame:
        """
        Load qrels file for specified version.

        Args:
            version (str): Dataset version ('train', 'dev-1', 'dev-2', 'dev-3')

        Returns:
            pd.DataFrame: Qrels dataframe
        """
        qrels_path = Path(self.config.get_qrels_path(version))

        if not qrels_path.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

        try:
            # Try loading as TREC format qrels
            qrels = pt.io.read_qrels(str(qrels_path))
            print(
                f"Loaded {len(qrels)} relevance judgments from {qrels_path.name}")
            return qrels
        except Exception as e:
            raise ValueError(f"Error loading qrels file {qrels_path}: {e}")

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
                print(f"Index loaded successfully from {index_path}")
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
        missing_in_topics = qrel_qids - topic_qids

        if missing_in_qrels:
            print(
                f"Warning: {len(missing_in_qrels)} {version} topics have no relevance judgments")

        if missing_in_topics:
            print(
                f"Warning: {len(missing_in_topics)} {version} qrels have no corresponding topics")

        matching_qids = len(topic_qids.intersection(qrel_qids))
        print(
            f"Data validation for {version}: {matching_qids} matching query IDs")

    def get_available_rewritten_sources(self, version: str) -> List[str]:
        """
        Get list of available rewritten query sources for a dataset version.

        Args:
            version (str): Dataset version

        Returns:
            List[str]: Available source types for this version
        """
        sources = ["original"]  # Original is always available

        rewritten_dir = Path(self.config.get_rewritten_queries_directory())
        if rewritten_dir.exists():
            # Look for rewritten query files
            pattern = f"*_{version}_rewritten_queries.jsonl"
            for file_path in rewritten_dir.glob(pattern):
                # Extract model name from filename: model_version_rewritten_queries.jsonl
                model_name = file_path.stem.replace(
                    f"_{version}_rewritten_queries", "")
                sources.append(f"rewritten_{model_name}")

            # Look for summarized query file
            summarized_file = rewritten_dir / f"{version}_summarized.jsonl"
            if summarized_file.exists():
                sources.append("summarized")

        return sources

    def load_all_query_sources(self, version: str) -> Dict[str, pd.DataFrame]:
        """
        Load all available query sources for a dataset version.

        Args:
            version (str): Dataset version

        Returns:
            Dict[str, pd.DataFrame]: Mapping of source type to topics dataframe
        """
        available_sources = self.get_available_rewritten_sources(version)
        loaded_sources = {}

        for source_type in available_sources:
            try:
                topics_df = self.load_topics(version, source_type)
                loaded_sources[source_type] = topics_df
            except Exception as e:
                print(
                    f"Warning: Could not load {source_type} topics for {version}: {e}")
                continue

        return loaded_sources
