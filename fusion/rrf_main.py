"""
Main execution script for Reciprocal Rank Fusion (RRF).
Updated to allow direct in-script configuration for the dataset version.
"""
import sys
from pathlib import Path
import pandas as pd
from rrf_fusion import RRFFusion

# Add the parent directory to the Python path to allow importing config_loader
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sparse_retrieval.config_loader import ConfigLoader

# =============================================================================
# CONFIGURATION SECTION - MODIFY THIS VARIABLE AS NEEDED
# =============================================================================

# Dataset to run fusion on (CHANGE THIS AS NEEDED)
# Options: "train", "dev-1", "dev-2", "dev-3", "test"
DATASET_TO_FUSE = "train"  # <-- CHANGE THIS LINE TO FUSE A DIFFERENT DATASET

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================

def find_env_path():
    """Finds the env.json file, looking in parent directories."""
    current_dir = Path(__file__).resolve().parent
    for i in range(3):
        env_path = current_dir / "env.json"
        if env_path.exists():
            return str(env_path)
        current_dir = current_dir.parent
    raise FileNotFoundError("env.json not found in the current or parent directories.")

def main(env_path: str, dataset_version: str):
    """
    Main function to run the RRF fusion process.

    Args:
        env_path (str): Path to the env.json configuration file.
        dataset_version (str): The dataset version to process (e.g., 'train').
    """
    print(f"Starting RRF fusion for dataset: {dataset_version}")
    # Load configuration
    config = ConfigLoader(env_path)
    rrf_k = config.get_rrf_k()
    top_k = 2000  # As per your request

    # Define directories
    sparse_run_dir = Path(config.get_sparse_run_directory()) / dataset_version
    fused_run_dir = Path(config.get_fused_run_directory())
    fused_run_dir.mkdir(parents=True, exist_ok=True)

    # Find all run files for the specified dataset version
    run_files_paths = list(sparse_run_dir.glob("*.txt"))
    if not run_files_paths:
        print(f"Error: No run files found in {sparse_run_dir}")
        print("Please ensure you have generated sparse retrieval runs for this dataset first.")
        return

    print(f"Found {len(run_files_paths)} run files to fuse.")

    # Load run files into a list of DataFrames
    run_files = []
    for file_path in run_files_paths:
        try:
            # Load as TREC format (qid Q0 docno rank score run_name)
            df = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name'],
                dtype={'qid': str, 'docno': str}
            )
            run_files.append(df)
        except Exception as e:
            print(f"Warning: Could not load or parse run file {file_path.name}: {e}")

    if not run_files:
        print("Error: No valid run files could be loaded. Aborting fusion.")
        return

    # Perform RRF fusion
    print("Fusing run files using Reciprocal Rank Fusion...")
    fusion = RRFFusion(rrf_k=rrf_k)
    fused_df = fusion.fuse(run_files)

    # Get the top K results
    fused_df = fused_df.groupby('qid').head(top_k)

    # Save the fused run file in TREC format
    output_path = fused_run_dir / f"fused_{dataset_version}_top{top_k}.txt"
    with open(output_path, 'w') as f:
        for _, row in fused_df.iterrows():
            f.write(
                f"{row['qid']}\tQ0\t{row['docno']}\t{row['rank']}\t{row['score']}\tfused_run\n"
            )

    print(f"\nFusion complete!")
    print(f"Fused run file with top {top_k} results saved to: {output_path}")

if __name__ == "__main__":
    # The script now uses the DATASET_TO_FUSE variable defined above
    # instead of command-line arguments.
    env_path = find_env_path()
    main(env_path, DATASET_TO_FUSE)