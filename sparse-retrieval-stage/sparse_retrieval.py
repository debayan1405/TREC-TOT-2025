import pyterrier as pt
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initialize PyTerrier
if not pt.started():
    pt.init()

# ==========================================
# CONFIGURATION
# ==========================================
# The order of datasets to fine-tune on
DATASET_SEQUENCE = ["train", "dev1", "dev2", "dev3"]

# 1. Global Grid (Used for the first dataset to find rough area)
GLOBAL_BM25_GRID = {
    "k1": [0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0],
    "b": [0.3, 0.5, 0.6, 0.75, 0.9, 1.0]
}
GLOBAL_PL2_GRID = {
    "c": [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0]
}

# 2. Refinement Settings (Used for subsequent datasets)
# We will search: [best - delta, best, best + delta]
# You can adjust steps to make it more granular
REFINE_STEPS = 5  # Number of points in the local grid
REFINE_RADIUS_K1 = 0.4  # Search +/- 0.4 around previous best
REFINE_RADIUS_B = 0.2   # Search +/- 0.2 around previous best
REFINE_RADIUS_C = 2.0   # Search +/- 2.0 around previous best

# ==========================================
# HELPERS
# ==========================================

def load_env(env_path="env.json"):
    with open(env_path, 'r') as f:
        return json.load(f)

def get_dataset_paths(env, dataset_key):
    paths = env['paths']
    key_map = {
        "train": ("train_queries_path", "train_qrels_path"),
        "dev1": ("dev_1_queries_path", "dev1_qrels_path"),
        "dev2": ("dev_2_queries_path", "dev2_qrels_path"),
        "dev3": ("dev_3_queries_path", "dev3_qrels_path"),
        "test": ("test_queries_path", "test_qrels_path")
    }
    q_key, qrel_key = key_map[dataset_key]
    return paths[q_key], paths[qrel_key]

def load_queries(query_path):
    queries_df = pd.read_json(query_path, lines=True)
    if 'text' in queries_df.columns:
        queries_df = queries_df.rename(columns={'text': 'query'})
    queries_df['qid'] = queries_df['qid'].astype(str)
    return queries_df

def generate_refined_grid(center, radius, steps, min_val=0.01):
    """
    Creates a linear space grid centered around 'center'.
    """
    start = max(min_val, center - radius)
    end = center + radius
    # Ensure unique values and sort
    grid = np.linspace(start, end, steps)
    return sorted(list(set(np.round(grid, 2))))

def save_trec_run(model, topics, run_dir, filename):
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, filename)
    pt.io.write_results(model.transform(topics), output_path)

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    env = load_env()
    index = pt.IndexFactory.of(env['paths']['index_path'])
    
    run_dir = env['paths']['sparse_run_directory']
    eval_dir = env['paths']['evaluation_directory']
    best_params_path = env['paths']['best_params_path']
    chart_dir = os.path.join(eval_dir, "charts")
    
    metrics = env['eval_metrics']
    target_metric = metrics[0]

    # State variables to hold the "Previous Best"
    # Initialize with None so we know to use Global Grid first
    current_best_bm25 = {"k1": None, "b": None}
    current_best_pl2 = {"c": None}
    
    history_log = []

    print(f"Starting Sequential Optimization: {DATASET_SEQUENCE}")
    print(f"Target Metric: {target_metric}")

    for dataset in DATASET_SEQUENCE:
        print(f"\n{'='*40}")
        print(f"PROCESSING DATASET: {dataset}")
        print(f"{'='*40}")

        # Load Data
        q_path, qrel_path = get_dataset_paths(env, dataset)
        topics = load_queries(q_path)
        qrels = pt.io.read_qrels(qrel_path)

        # -----------------------------------------------
        # 1. Determine Search Grids
        # -----------------------------------------------
        if current_best_bm25["k1"] is None:
            # First run: Use Global Grid
            bm25_k1_grid = GLOBAL_BM25_GRID["k1"]
            bm25_b_grid = GLOBAL_BM25_GRID["b"]
            pl2_c_grid = GLOBAL_PL2_GRID["c"]
            print("Using GLOBAL Grid search space.")
        else:
            # Subsequent runs: Refine around previous best
            print(f"Initializing search around previous best: BM25 {current_best_bm25}, PL2 {current_best_pl2}")
            bm25_k1_grid = generate_refined_grid(current_best_bm25["k1"], REFINE_RADIUS_K1, REFINE_STEPS)
            bm25_b_grid = generate_refined_grid(current_best_bm25["b"], REFINE_RADIUS_B, REFINE_STEPS, min_val=0.1)
            pl2_c_grid = generate_refined_grid(current_best_pl2["c"], REFINE_RADIUS_C, REFINE_STEPS)

        # -----------------------------------------------
        # 2. Optimize BM25
        # -----------------------------------------------
        print(f"\n--- Optimizing BM25 on {dataset} ---")
        best_score = -1
        local_best_bm25 = {}
        
        # We collect data for heatmap
        heatmap_data = []

        for k1 in bm25_k1_grid:
            for b in bm25_b_grid:
                # Run experiment
                model = pt.BatchRetrieve(index, wmodel="BM25", controls={"c": k1, "bm25.b": b})
                res = pt.Experiment([model], topics, qrels, eval_metrics=[target_metric], verbose=False)
                score = res.iloc[0][target_metric]
                
                heatmap_data.append({"k1": k1, "b": b, "score": score})
                
                if score > best_score:
                    best_score = score
                    local_best_bm25 = {"k1": k1, "b": b, "score": score}

        print(f"Best BM25 on {dataset}: {local_best_bm25}")
        
        # Update State
        current_best_bm25["k1"] = local_best_bm25["k1"]
        current_best_bm25["b"] = local_best_bm25["b"]
        
        # Save Run
        best_bm25_model = pt.BatchRetrieve(index, wmodel="BM25", 
                                         controls={"c": local_best_bm25['k1'], "bm25.b": local_best_bm25['b']})
        save_trec_run(best_bm25_model, topics, run_dir, 
                      f"{dataset}_BEST_bm25_k1-{local_best_bm25['k1']}_b-{local_best_bm25['b']}.run")

        # Plot Heatmap
        df_bm25 = pd.DataFrame(heatmap_data)
        pivot = df_bm25.pivot(index="k1", columns="b", values="score")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".4f")
        plt.title(f"BM25 Optimization ({dataset})\nRefined around previous best")
        save_chart(plt.gcf(), chart_dir, f"{dataset}_bm25_optimization.png")

        # -----------------------------------------------
        # 3. Optimize PL2
        # -----------------------------------------------
        print(f"\n--- Optimizing PL2 on {dataset} ---")
        best_score_pl2 = -1
        local_best_pl2 = {}
        line_data = []

        for c in pl2_c_grid:
            model = pt.BatchRetrieve(index, wmodel="PL2", controls={"c": c})
            res = pt.Experiment([model], topics, qrels, eval_metrics=[target_metric], verbose=False)
            score = res.iloc[0][target_metric]
            
            line_data.append({"c": c, "score": score})
            
            if score > best_score_pl2:
                best_score_pl2 = score
                local_best_pl2 = {"c": c, "score": score}

        print(f"Best PL2 on {dataset}: {local_best_pl2}")

        # Update State
        current_best_pl2["c"] = local_best_pl2["c"]

        # Save Run
        best_pl2_model = pt.BatchRetrieve(index, wmodel="PL2", controls={"c": local_best_pl2['c']})
        save_trec_run(best_pl2_model, topics, run_dir, 
                      f"{dataset}_BEST_pl2_c-{local_best_pl2['c']}.run")

        # Plot Line
        df_pl2 = pd.DataFrame(line_data)
        plt.figure(figsize=(8, 6))
        plt.plot(df_pl2['c'], df_pl2['score'], marker='o')
        plt.title(f"PL2 Optimization ({dataset})")
        plt.xlabel("Parameter c")
        plt.ylabel(target_metric)
        plt.grid(True)
        save_chart(plt.gcf(), chart_dir, f"{dataset}_pl2_optimization.png")

        # -----------------------------------------------
        # 4. Log Results
        # -----------------------------------------------
        history_log.append({
            "dataset": dataset,
            "model": "BM25",
            "best_params": f"k1={local_best_bm25['k1']}, b={local_best_bm25['b']}",
            "metric_value": local_best_bm25["score"]
        })
        history_log.append({
            "dataset": dataset,
            "model": "PL2",
            "best_params": f"c={local_best_pl2['c']}",
            "metric_value": local_best_pl2["score"]
        })

    # Final Save of History
    print(f"\nSaving optimization history to {best_params_path}")
    pd.DataFrame(history_log).to_csv(best_params_path, index=False)

def save_chart(fig, chart_dir, filename):
    os.makedirs(chart_dir, exist_ok=True)
    output_path = os.path.join(chart_dir, filename)
    fig.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    main()