import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# ==========================================
# CONFIGURATION
# ==========================================
# File names as specified
BENCHMARK_FILE = "sparse_benchmark_results.csv"

def load_env(env_path="env.json"):
    if not os.path.exists(env_path):
        # Fallback for colab/different envs
        env_path = "/content/env.json"
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Config file not found at {env_path}")
        
    with open(env_path, 'r') as f:
        return json.load(f)

def main():
    # 1. Load Environment & Paths
    try:
        env = load_env()
        eval_dir = env['paths']['evaluation_directory']
    except Exception as e:
        print(f"Error loading environment: {e}")
        # Fallback for local testing if env.json fails
        eval_dir = "../evaluations" 
        print(f"Using fallback directory: {eval_dir}")

    chart_dir = os.path.join(eval_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    
    csv_path = os.path.join(eval_dir, BENCHMARK_FILE)

    # 2. Load Data
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: Benchmark CSV not found at: {csv_path}")
        print("Please run 'analyze_rewriter_and_dev3.py' first.")
        return

    print(f"Loading data from: {csv_path}")
    df_full = pd.read_csv(csv_path)

    # 3. Filter for Dev3 (The Tournament Set)
    # The benchmark file contains train, dev1, dev2, and dev3.
    # We only want to plot the final pseudo-test results.
    df = df_full[df_full['dataset'] == 'dev3'].copy()

    if df.empty:
        print("No 'dev3' data found in the benchmark CSV. Charts cannot be generated.")
        return

    # 4. Map Column Names
    # ir_measures output often has columns like "nDCG@10" or "Success@10"
    # We check for these or standard aliases
    
    ndcg_col = "nDCG@10" if "nDCG@10" in df.columns else "ndcg_cut_10"
    success_col = "Success@10" if "Success@10" in df.columns else "success_10"
    
    # Validation
    missing_cols = []
    if ndcg_col not in df.columns: missing_cols.append("nDCG@10")
    if success_col not in df.columns: missing_cols.append("Success@10")
    
    if missing_cols:
        print(f"Warning: Missing metric columns {missing_cols}. Available: {df.columns.tolist()}")
        # Continue if at least one exists, else exit
        if len(missing_cols) == 2: return

    # Set style
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------
    # CHART 1: The Leaderboard (nDCG@10)
    # -------------------------------------------------------
    if ndcg_col in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Sort order based on BM25 performance of each rewriter
        bm25_data = df[df['model'] == "BM25"]
        if not bm25_data.empty:
            order = bm25_data.sort_values(ndcg_col, ascending=False)['rewriter'].tolist()
        else:
            order = df['rewriter'].unique().tolist()
        
        ax = sns.barplot(
            data=df, 
            x="rewriter", 
            y=ndcg_col, 
            hue="model", 
            order=order,
            palette="viridis"
        )
        
        plt.title("Dev3 Tournament: Best Rewriter by Ranking Quality (nDCG@10)", fontsize=14)
        plt.xlabel("Query Version", fontsize=12)
        plt.ylabel("nDCG@10", fontsize=12)
        # Dynamic Y-limit with some headroom
        plt.ylim(0, df[ndcg_col].max() * 1.15)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)

        plt.tight_layout()
        path1 = os.path.join(chart_dir, "final_rewriter_tournament_ndcg.png")
        plt.savefig(path1)
        print(f"Leaderboard Chart saved to: {path1}")
        
        # Determine Winner
        winner = order[0]

    # -------------------------------------------------------
    # CHART 2: Success@10 (Recall for 1 doc)
    # -------------------------------------------------------
    if success_col in df.columns:
        plt.figure(figsize=(10, 6))
        
        ax2 = sns.barplot(
            data=df, 
            x="rewriter", 
            y=success_col, 
            hue="model", 
            order=order,
            palette="magma"
        )
        
        plt.title("Dev3 Tournament: Best Rewriter by Recall (Success@10)", fontsize=14)
        plt.xlabel("Query Version", fontsize=12)
        plt.ylabel("Success@10", fontsize=12)
        plt.ylim(0, df[success_col].max() * 1.15)
        
        # Add value labels
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.3f', padding=3)
            
        plt.tight_layout()
        path2 = os.path.join(chart_dir, "final_rewriter_tournament_success.png")
        plt.savefig(path2)
        print(f"Success Chart saved to: {path2}")

    # -------------------------------------------------------
    # Recommendation Output
    # -------------------------------------------------------
    if 'winner' in locals():
        print("\n" + "="*40)
        print("RECOMMENDATION FOR NEXT STAGE")
        print("="*40)
        print(f"Winner: {winner.upper()}")
        print("Action: Proceed to RRF Fusion using:")
        print(f"  1. {winner}_BM25 run file")
        print(f"  2. {winner}_PL2 run file")

if __name__ == "__main__":
    main()