"""
Configuration loader module for PyTerrier experiments.
Handles loading and validation of configuration from env.json file.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List


class ConfigLoader:
    """Handles loading and accessing configuration from env.json file."""

    def __init__(self, config_path: str = "env.json"):
        """
        Initialize configuration loader.

        Args:
            config_path (str): Path to the configuration JSON file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def _validate_config(self) -> None:
        """Validate required configuration keys."""
        required_keys = [
            "index_path", "topics_path", "qrels_path",
            "run_directory", "k_sparse", "eval_metrics"
        ]

        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {missing_keys}")

        # Validate k_sparse is positive integer
        if not isinstance(self.config["k_sparse"], int) or self.config["k_sparse"] <= 0:
            raise ValueError("k_sparse must be a positive integer")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)

    def get_index_path(self) -> str:
        """Get index path."""
        return self.config["index_path"]

    def get_topics_path(self, version: str = "train") -> str:
        """
        Get topics file path for specified version.

        Args:
            version (str): Dataset version ('train', 'dev-1', 'dev-2', 'dev-3', 'test')

        Returns:
            str: Path to topics file
        """
        version_map = {
            "train": self.get_train_topics_path,
            "dev-1": self.get_dev_1_topics_path,
            "dev-2": self.get_dev_2_topics_path,
            "dev-3": self.get_dev_3_topics_path,
            "test": self.get_test_topics_path
        }

        if version not in version_map:
            raise ValueError(
                f"Invalid topics version: {version}. Valid options: {list(version_map.keys())}")

        return version_map[version]()

    def get_train_topics_path(self) -> str:
        """Get train topics file path."""
        return self.config["train_topics_path"]

    def get_dev_1_topics_path(self) -> str:
        """Get dev-1 topics file path."""
        return self.config["dev_1_topics_path"]

    def get_dev_2_topics_path(self) -> str:
        """Get dev-2 topics file path."""
        return self.config["dev_2_topics_path"]

    def get_dev_3_topics_path(self) -> str:
        """Get dev-3 topics file path."""
        return self.config["dev_3_topics_path"]

    def get_test_topics_path(self) -> str:
        """Get test topics file path."""
        return self.config["test_topics_path"]

    def get_qrels_path(self, version: str = "train") -> str:
        """
        Get qrels file path for specified version.

        Args:
            version (str): Dataset version ('train', 'dev-1', 'dev-2', 'dev-3')

        Returns:
            str: Path to qrels file
        """
        version_map = {
            "train": self.get_train_qrels_path,
            "dev-1": self.get_dev_1_qrels_path,
            "dev-2": self.get_dev_2_qrels_path,
            "dev-3": self.get_dev_3_qrels_path
        }

        if version not in version_map:
            raise ValueError(
                f"Invalid qrels version: {version}. Valid options: {list(version_map.keys())}")

        return version_map[version]()

    def get_train_qrels_path(self) -> str:
        """Get qrels file path."""
        return self.config["train_qrels_path"]

    def get_dev_1_qrels_path(self) -> str:
        """Get dev-1 qrels file path."""
        return self.config["dev_1_qrels_path"]

    def get_dev_2_qrels_path(self) -> str:
        """Get dev-2 qrels file path."""
        return self.config["dev_2_qrels_path"]

    def get_dev_3_qrels_path(self) -> str:
        """Get dev-3 qrels file path."""
        return self.config["dev_3_qrels_path"]

    def get_run_directory(self) -> str:
        """Get run directory path."""
        return self.config["run_directory"]

    def get_k_sparse(self) -> int:
        """Get k_sparse value."""
        return self.config["k_sparse"]

    def get_eval_metrics(self) -> List[str]:
        """Get evaluation metrics."""
        return self.config["eval_metrics"]

    def ensure_run_directory_exists(self) -> None:
        """Create run directory if it doesn't exist."""
        run_dir = Path(self.get_run_directory())
        run_dir.mkdir(parents=True, exist_ok=True)
