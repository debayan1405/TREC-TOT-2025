"""
Main execution script for dense re-ranking experiments.
1. Re-ranks a fused sparse run file using multiple dense models.
2. Fuses the dense results using RRF.
3. Evaluates all individual and fused dense runs.
"""
import argparse
import sys
import os
import torch
from pathlib import Path
import pandas as pd
import pyterrier as pt

# Optimize for maximum performance on high-end hardware
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # Use all CPU cores
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())  # Use all CPU threads for PyTorch

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dense_retrieval.dense_reranker import DenseReranker
from sparse_retrieval.config_loader import ConfigLoader
from sparse_retrieval.data_loader import DataLoader
from fusion.rrf_fusion import RRFusion # Re-use for the final fusion step

def main():
    parser = argparse.ArgumentParser(description="Run Dense Re-ranking and Evaluation")
    parser.add_argument("--dataset", required=True, choices=["train", "dev-1", "dev-2", "dev-3", "test"], help="Dataset version to process")
    parser.add_argument("--rewriter", required=True, help="Rewriter version to process (e.g., llama, qwen, original)")
    args = parser.parse_args()

    config = ConfigLoader(str(project_root / "env.json"))
    data_loader = DataLoader(config)
    
    print(f"=== Dense Re-ranking Experiment for Dataset: {args.dataset}, Rewriter: {args.rewriter} ===")

    dense_models = config.config.get("dense_rerank_models", [])
    if not dense_models:
        print("No dense_rerank_models specified in env.json. Aborting.")
        return

    # --- Step 1: Re-rank with each dense model ---
    reranked_run_paths = []
    for model_name in dense_models:
        try:
            reranker = DenseReranker(config, args.dataset, args.rewriter, model_name)
            run_path = reranker.rerank()
            reranked_run_paths.append(run_path)
        except Exception as e:
            print(f"Error re-ranking with {model_name}: {e}")
            continue

    # --- Step 2: Fuse the dense results using RRF ---
    print("\n--- Fusing dense results ---")
    if len(reranked_run_paths) > 1:
        # We can re-use the RRFusion logic, but need to adapt it slightly
        fusion_processor = RRFusion(config, args.dataset)
        fused_scores = fusion_processor._reciprocal_rank_fusion([str(p) for p in reranked_run_paths])

        data = [{'qid': qid, 'docno': docno, 'score': score} for (qid, docno), score in fused_scores.items()]
        fused_df = pd.DataFrame(data)
        fused_df['rank'] = fused_df.groupby('qid')['score'].rank(method='first', ascending=False).astype(int)
        
        fused_run_path = reranker.run_dir / f"{args.rewriter}_{args.dataset}_dense_fused.txt"
        with open(fused_run_path, 'w') as f_out:
            for _, row in fused_df.iterrows():
                f_out.write(f"{row['qid']} Q0 {row['docno']} {row['rank']} {row['score']:.6f} dense_fused\n")
        
        print(f"Saved final fused dense run to: {fused_run_path}")
        reranked_run_paths.append(fused_run_path) # Add for evaluation
    
    # --- Step 3: Evaluate all generated run files ---
    print("\n--- Evaluating all dense run files ---")
    try:
        qrels = data_loader.load_qrels(args.dataset)
        topics = data_loader.load_topics(args.dataset, "original") # Use original topics for eval
        
        run_dfs = [pt.io.read_results(str(p)) for p in reranked_run_paths]
        names = [p.stem for p in reranked_run_paths]
        
        eval_results = pt.Experiment(
            run_dfs,
            topics,
            qrels,
            eval_metrics=config.get_eval_metrics(),
            names=names
        )

        # Save combined evaluation file
        eval_output_path = reranker.eval_dir / f"{args.rewriter}_{args.dataset}_dense_eval_summary.csv"
        eval_results.to_csv(eval_output_path)
        print(f"\nSaved combined evaluation summary to: {eval_output_path}")
        print(eval_results)

    except Exception as e:
        print(f"Evaluation failed: {e}")

    print("\n=== Experiment Completed! ===")

if __name__ == "__main__":
    main()