import pyterrier as pt
import pandas as pd
import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ir_measures
from ir_measures import calc_aggregate, Success, nDCG, AP, RR, R, P

# Initialize PyTerrier (RAM optimized for metadata)
os.environ["JAVA_OPTS"] = "-Xmx100g -Xms16g -Dindex.meta.data-source=fileinmem"
if not pt.java.started():
    pt.java.init()

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_REWRITER = "mistral"
RUN_DIR_FOLDER = "mistral-query-run-files"
PREFIX = "mistral"

DATASETS = ["train", "dev1", "dev2", "dev3"]
RRF_K_GRID = [10, 20, 30, 40, 50, 60, 80, 100]

# Metrics to track
MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]
PRIMARY_METRIC = nDCG@10  # Optimization Target

# ==========================================
# HELPERS
# ==========================================

def load_env(env_path="env.json"):
    if not os.path.exists(env_path):
        env_path = "/content/env.json"
    with open(env_path, 'r') as f:
        return json.load(f)

def get_qrels(env, dataset):
    key_map = {
        "train": "train_qrels_path", "dev1": "dev1_qrels_path",
        "dev2": "dev2_qrels_path", "dev3": "dev3_qrels_path"
    }
    return pt.io.read_qrels(env['paths'][key_map[dataset]])

def find_run_file(base_dir, dataset, model):
    """
    Finds the run file: mistral_{dataset}_BEST_{model}_*.run
    """
    pattern = os.path.join(base_dir, f"{PREFIX}_{dataset}_BEST_{model}_*.run")
    files = glob.glob(pattern)
    if not files:
        return None
    # Return most recent
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def evaluate_run(res, qrels):
    qrels_std = qrels.rename(columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"})
    res_std = res.rename(columns={"qid": "query_id", "docno": "doc_id"})
    return calc_aggregate(MEASURES, qrels_std, res_std)

def pandas_rrf(df_list, k=60):
    """
    Performs Reciprocal Rank Fusion on a list of DataFrames using pure Pandas.
    Formula: score = sum(1 / (k + rank))
    """
    fused_parts = []
    
    for df in df_list:
        # Ensure we have a rank column relative to the query
        # We assume df is scored, but let's enforce rank calculation to be safe
        df_sorted = df.sort_values(["qid", "score"], ascending=[True, False])
        df_sorted["rank"] = df_sorted.groupby("qid").cumcount() + 1
        
        # Calculate RRF component
        df_sorted["rrf_score_part"] = 1.0 / (k + df_sorted["rank"])
        fused_parts.append(df_sorted[["qid", "docno", "rrf_score_part"]])
        
    # Concatenate all parts
    combined = pd.concat(fused_parts)
    
    # Sum scores by qid, docno
    fused_df = combined.groupby(["qid", "docno"], as_index=False)["rrf_score_part"].sum()
    fused_df = fused_df.rename(columns={"rrf_score_part": "score"})
    
    # Add Rank column required for TREC output
    fused_df = fused_df.sort_values(["qid", "score"], ascending=[True, False])
    fused_df["rank"] = fused_df.groupby("qid").cumcount() + 1
    
    # Add dummy 'Q0' and 'system' columns for standard format if needed (PyTerrier adds them usually)
    fused_df["Q0"] = "Q0"
    fused_df["system"] = "RRF_fusion"
    
    return fused_df

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    env = load_env()
    
    sparse_dir = env['paths']['sparse_run_directory']
    mistral_run_dir = os.path.join(sparse_dir, RUN_DIR_FOLDER)
    
    # Output Directories
    fusion_out_dir = env['paths']['fusion_run_directory']
    os.makedirs(fusion_out_dir, exist_ok=True)
    
    eval_dir = env['paths']['evaluation_directory']
    chart_dir = os.path.join(eval_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    summary_log = []
    tuning_log = [] # New log to track sensitivity of K

    print(f"{'='*60}")
    print(f"RRF FUSION ANALYSIS: {TARGET_REWRITER.upper()}")
    print(f"{'='*60}")

    for dataset in DATASETS:
        print(f"\nProcessing Dataset: {dataset.upper()}")
        
        # 1. Load Input Runs
        bm25_path = find_run_file(mistral_run_dir, dataset, "bm25")
        pl2_path = find_run_file(mistral_run_dir, dataset, "pl2")
        
        if not bm25_path or not pl2_path:
            print(f"  [SKIP] Missing run files for {dataset}")
            continue
            
        print(f"  BM25 Input: {os.path.basename(bm25_path)}")
        print(f"  PL2  Input: {os.path.basename(pl2_path)}")
        
        res_bm25 = pt.io.read_results(bm25_path)
        res_pl2 = pt.io.read_results(pl2_path)
        qrels = get_qrels(env, dataset)

        # 2. Baseline Evaluation
        metrics_bm25 = evaluate_run(res_bm25, qrels)
        metrics_pl2 = evaluate_run(res_pl2, qrels)
        
        score_bm25 = metrics_bm25[PRIMARY_METRIC]
        score_pl2 = metrics_pl2[PRIMARY_METRIC]
        
        # Identify Best Single Model
        if score_bm25 >= score_pl2:
            best_single_model = "BM25"
            best_single_metrics = metrics_bm25
        else:
            best_single_model = "PL2"
            best_single_metrics = metrics_pl2
            
        print(f"  Baseline (Best Single): {best_single_model} ({score_bm25 if score_bm25>score_pl2 else score_pl2:.4f})")

        # 3. Tune RRF
        print(f"  Tuning RRF k {RRF_K_GRID}...")
        best_k = -1
        best_rrf_res = None
        best_rrf_metrics = {}
        best_rrf_score = -1.0
        
        for k in RRF_K_GRID:
            # Replaced pt.rrf() with robust pandas implementation
            res_fused = pandas_rrf([res_bm25, res_pl2], k=k)
            
            metrics_fused = evaluate_run(res_fused, qrels)
            score_fused = metrics_fused[PRIMARY_METRIC]
            
            # Log for plotting
            tuning_log.append({
                "dataset": dataset,
                "k": k,
                "score": score_fused
            })
            
            if score_fused > best_rrf_score:
                best_rrf_score = score_fused
                best_k = k
                best_rrf_res = res_fused
                best_rrf_metrics = metrics_fused

        # 4. Calculate Gains & Log
        # Gain based on PRIMARY Metric (nDCG@10)
        gain_pct = ((best_rrf_score - best_single_metrics[PRIMARY_METRIC]) / best_single_metrics[PRIMARY_METRIC]) * 100.0
        
        print(f"  Winner RRF k={best_k} | Score: {best_rrf_score:.4f} | Gain: {gain_pct:+.2f}%")
        
        # Save Fused Run
        run_filename = f"{TARGET_REWRITER}_{dataset}_RRF_k-{best_k}.run"
        save_path = os.path.join(fusion_out_dir, run_filename)
        pt.io.write_results(best_rrf_res, save_path)
        
        # Log comparison for all metrics
        for m in MEASURES:
            m_str = str(m)
            single_val = best_single_metrics[m]
            rrf_val = best_rrf_metrics[m]
            delta = ((rrf_val - single_val) / single_val) * 100.0 if single_val > 0 else 0.0
            
            summary_log.append({
                "dataset": dataset,
                "metric": m_str,
                "Single_Model": best_single_model,
                "Single_Score": single_val,
                "RRF_k": best_k,
                "RRF_Score": rrf_val,
                "Gain_Pct": delta
            })

    # 5. Save Data
    df = pd.DataFrame(summary_log)
    csv_path = os.path.join(eval_dir, "mistral_fusion_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFusion Comparison Data saved to: {csv_path}")
    
    # 6. Visualizations
    
    # Chart A: Sensitivity Plot (Metric vs K) -- NEW
    if tuning_log:
        df_tuning = pd.DataFrame(tuning_log)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_tuning, x="k", y="score", hue="dataset", marker="o")
        plt.title(f"RRF Parameter Sensitivity (nDCG@10)")
        plt.xlabel("RRF Constant (k)")
        plt.ylabel("nDCG@10")
        plt.grid(True)
        chart_path_sens = os.path.join(chart_dir, "mistral_rrf_sensitivity.png")
        plt.savefig(chart_path_sens)
        print(f"Sensitivity Chart saved to: {chart_path_sens}")

    # Chart B: Side-by-Side Comparison (Gain)
    plot_metrics = ["nDCG@10", "Success@10"]
    if not df.empty:
        plt.figure(figsize=(12, 6))
        
        # Filter Data
        df_plot = df[df['metric'].isin(plot_metrics)].copy()
        
        # Melt for seaborn side-by-side
        rows = []
        for _, row in df_plot.iterrows():
            rows.append({
                "Dataset": row['dataset'], 
                "Metric": row['metric'], 
                "System": f"Single ({row['Single_Model']})", 
                "Score": row['Single_Score']
            })
            rows.append({
                "Dataset": row['dataset'], 
                "Metric": row['metric'], 
                "System": "RRF Fusion", 
                "Score": row['RRF_Score']
            })
            
        df_viz = pd.DataFrame(rows)
        
        sns.catplot(
            data=df_viz, 
            kind="bar",
            x="Dataset", 
            y="Score", 
            hue="System", 
            col="Metric",
            palette="Paired",
            height=5, 
            aspect=1.2,
            sharey=False
        )
        
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Mistral: Single Model vs. RRF Fusion")
        
        chart_path = os.path.join(chart_dir, "mistral_fusion_gain_chart.png")
        plt.savefig(chart_path)
        print(f"Chart saved to: {chart_path}")

if __name__ == "__main__":
    main()