"""
Configuration loader module for PyTerrier experiments.
Handles loading and validation of configuration from env.json file.
Fixed for new env.json structure with dynamic path support.
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
        self.config_path = self._find_config_path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _find_config_path(self, config_path: str) -> str:
        """Find config file, checking current dir and parent dirs if needed."""
        if os.path.isabs(config_path):
            return config_path

        # Check current directory first
        if os.path.exists(config_path):
            return config_path

        # Check parent directory
        parent_path = Path(__file__).parent.parent / config_path
        if parent_path.exists():
            return str(parent_path)

        # Check two levels up
        grandparent_path = Path(__file__).parent.parent.parent / config_path
        if grandparent_path.exists():
            return str(grandparent_path)

        # If not found, return original path (will cause FileNotFoundError later)
        return config_path

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
        """Validate required configuration keys for new structure."""
        # Check top-level structure
        if "paths" not in self.config:
            raise ValueError("Configuration must contain 'paths' section")

        paths = self.config["paths"]

        # Required path keys
        required_path_keys = [
            "index_path", "sparse_run_directory"
        ]

        # Required topic paths
        topic_path_keys = [
            "train_topics_path", "dev_1_topics_path", "dev_2_topics_path",
            "dev_3_topics_path", "test_topics_path"
        ]

        # Required qrel paths
        qrel_path_keys = [
            "train_qrels_path", "dev1_qrels_path", "dev2_qrels_path", "dev3_qrels_path"
        ]

        all_required_paths = required_path_keys + topic_path_keys + qrel_path_keys

        missing_paths = [key for key in all_required_paths if key not in paths]
        if missing_paths:
            raise ValueError(f"Missing required path keys: {missing_paths}")

        # Validate other required top-level keys
        other_required = ["k_sparse", "eval_metrics"]
        missing_other = [
            key for key in other_required if key not in self.config]
        if missing_other:
            raise ValueError(
                f"Missing required configuration keys: {missing_other}")

        # Validate k_sparse is positive integer
        if not isinstance(self.config["k_sparse"], int) or self.config["k_sparse"] <= 0:
            raise ValueError("k_sparse must be a positive integer")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)

    def get_index_path(self) -> str:
        """Get index path."""
        return self.config["paths"]["index_path"]

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
        return self.config["paths"]["train_topics_path"]

    def get_dev_1_topics_path(self) -> str:
        """Get dev-1 topics file path."""
        return self.config["paths"]["dev_1_topics_path"]

    def get_dev_2_topics_path(self) -> str:
        """Get dev-2 topics file path."""
        return self.config["paths"]["dev_2_topics_path"]

    def get_dev_3_topics_path(self) -> str:
        """Get dev-3 topics file path."""
        return self.config["paths"]["dev_3_topics_path"]

    def get_test_topics_path(self) -> str:
        """Get test topics file path."""
        return self.config["paths"]["test_topics_path"]

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
        """Get train qrels file path."""
        return self.config["paths"]["train_qrels_path"]

    def get_dev_1_qrels_path(self) -> str:
        """Get dev-1 qrels file path."""
        return self.config["paths"]["dev1_qrels_path"]

    def get_dev_2_qrels_path(self) -> str:
        """Get dev-2 qrels file path."""
        return self.config["paths"]["dev2_qrels_path"]

    def get_dev_3_qrels_path(self) -> str:
        """Get dev-3 qrels file path."""
        return self.config["paths"]["dev3_qrels_path"]

    def get_sparse_run_directory(self) -> str:
        """Get sparse run directory path."""
        return self.config["paths"]["sparse_run_directory"]

    def get_combined_run_directory(self) -> str:
        """Get combined run directory path."""
        return self.config["paths"]["combined_run_directory"]

    def get_evaluation_directory(self) -> str:
        """Get evaluation directory path."""
        return self.config["paths"]["evaluation_directory"]

    def get_k_sparse(self) -> int:
        """Get k_sparse value."""
        return self.config["k_sparse"]

    def get_rrf_k(self) -> int:
        """Get RRF_K value."""
        return self.config.get("RRF_K", 2000)

    def get_eval_metrics(self) -> List[str]:
        """Get evaluation metrics."""
        return self.config["eval_metrics"]

    def ensure_run_directory_exists(self) -> None:
        """Create run directory if it doesn't exist."""
        run_dir = Path(self.get_sparse_run_directory())
        run_dir.mkdir(parents=True, exist_ok=True)

    def ensure_combined_run_directory_exists(self) -> None:
        """Create combined run directory if it doesn't exist."""
        combined_dir = Path(self.get_combined_run_directory())
        combined_dir.mkdir(parents=True, exist_ok=True)

    def ensure_evaluation_directory_exists(self) -> None:
        """Create evaluation directory if it doesn't exist."""
        eval_dir = Path(self.get_evaluation_directory())
        eval_dir.mkdir(parents=True, exist_ok=True)

    def get_rewritten_queries_directory(self) -> str:
        """Get rewritten queries directory path."""
        return self.config["paths"]["rewritten_queries_directory"]
