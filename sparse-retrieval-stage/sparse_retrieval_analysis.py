import pyterrier as pt
import pandas as pd
import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

if not pt.started():
    pt.init()

# ==========================================
# CONFIGURATION
# ==========================================
REWRITERS = ["original", "llama", "qwen", "mistral", "summarised"]
DATASETS = ["train", "dev1", "dev2"]
MODELS = ["BM25", "PL2"]

# ==========================================
# HELPERS
# ==========================================

def load_env(env_path="env.json"):
    with open(env_path, 'r') as f:
        return json.load(f)

def get_qrels(env, dataset):
    key_map = {
        "train": "train_qrels_path",
        "dev1": "dev1_qrels_path",
        "dev2": "dev2_qrels_path",
        "dev3": "dev3_qrels_path"
    }
    return pt.io.read_qrels(env['paths'][key_map[dataset]])

def parse_run_filename(filename):
    """
    Extracts parameters from filename.
    Expected format: {dataset}_BEST_{model}_{params}.run
    """
    basename = os.path.basename(filename)
    parts = basename.replace(".run", "").split("_")
    # This is a basic parser, might need adjustment based on exact filename
    return basename

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    env = load_env()
    base_run_dir = env['paths']['sparse_run_directory']
    eval_dir = env['paths']['evaluation_directory']
    chart_dir = os.path.join(eval_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    metrics = env['eval_metrics']
    
    # Store all results here
    consolidated_results = []

    print("Starting Rewriter Analysis...")

    for rewriter in REWRITERS:
        print(f"\nAnalyzing Rewriter: {rewriter.upper()}")
        
        # We assume run files are organized by rewriter folders or follow a naming pattern
        # Adjust 'search_path' if your files are all in one flat directory
        # Current logic: Look for folders named after rewriters
        rewriter_dir = os.path.join(base_run_dir, rewriter)
        
        if not os.path.exists(rewriter_dir):
            print(f"Warning: Directory not found for {rewriter}: {rewriter_dir}")
            print(f"Skipping {rewriter}...")
            continue

        for dataset in DATASETS:
            qrels = get_qrels(env, dataset)
            
            for model_name in MODELS:
                # Find the best run file for this combo
                # Pattern: {dataset}_BEST_{model}_*.run
                pattern = os.path.join(rewriter_dir, f"{dataset}_BEST_{model_name.lower()}_*.run")
                found_files = glob.glob(pattern)
                
                if not found_files:
                    print(f"  No run file found for {dataset} {model_name}")
                    continue
                
                # Take the most recent if duplicates exist
                found_files.sort(key=os.path.getmtime, reverse=True)
                run_file = found_files[0]
                
                # Extract Params string from filename for CSV
                # e.g., "dev1_BEST_bm25_k1-1.2_b-0.75.run" -> "k1-1.2_b-0.75"
                filename_clean = os.path.basename(run_file)
                params_str = filename_clean.split(f"{model_name.lower()}_")[-1].replace(".run", "")

                # Load Run
                res = pt.io.read_results(run_file)
                
                # Evaluate
                eval_res = pt.Utils.evaluate(res, qrels, metrics=metrics)
                
                # Log Data
                record = {
                    "rewriter": rewriter,
                    "dataset": dataset,
                    "model": model_name,
                    "params": params_str,
                    "filename": filename_clean
                }
                # Add all metrics to record
                record.update(eval_res)
                consolidated_results.append(record)
                
                print(f"  {dataset} {model_name}: NDCG@10={eval_res.get('ndcg_cut_10', 0):.4f}")

    # 1. Save CSV
    df = pd.DataFrame(consolidated_results)
    output_csv = os.path.join(eval_dir, "rewriter_comparison_metrics.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed metrics saved to: {output_csv}")

    # 2. Visualization
    if not df.empty:
        # Filter for plotting (Focus on NDCG@100 and Recip Rank as per prompt interest)
        plot_metrics = ["ndcg_cut_100", "recip_rank", "success_10"]
        
        for metric in plot_metrics:
            if metric not in df.columns: continue
            
            plt.figure(figsize=(12, 6))
            
            # Grouped Bar Chart
            # X=Dataset, Hue=Rewriter, Column=Model (using FacetGrid logic via catplot usually, but simple bar here)
            sns.catplot(
                data=df, 
                kind="bar", 
                x="dataset", 
                y=metric, 
                hue="rewriter", 
                col="model",
                palette="viridis",
                height=5, 
                aspect=1.2
            )
            
            plt.subplots_adjust(top=0.9)
            plt.suptitle(f"Impact of Rewriter on {metric}")
            
            chart_path = os.path.join(chart_dir, f"rewriter_comparison_{metric}.png")
            plt.savefig(chart_path)
            plt.close()
            print(f"Chart saved: {chart_path}")

    # 3. Identify Winner (Based on Dev2 NDCG@100)
    print("\n--- WINNER DETERMINATION (Based on Dev2) ---")
    dev2_df = df[df['dataset'] == 'dev2']
    if not dev2_df.empty:
        best_row = dev2_df.loc[dev2_df['ndcg_cut_100'].idxmax()]
        print(f"Best Configuration:")
        print(f"  Rewriter: {best_row['rewriter']}")
        print(f"  Model:    {best_row['model']}")
        print(f"  Params:   {best_row['params']}")
        print(f"  Score:    {best_row['ndcg_cut_100']:.4f}")
    else:
        print("No Dev2 results found to determine winner.")

if __name__ == "__main__":
    main()