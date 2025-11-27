import pyterrier as pt
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Initialize PyTerrier
# UPDATED: Allocated 500GB to JVM to leverage your 700GB+ RAM availability
if not pt.started():
    pt.init(mem="500g")

# ==========================================
# CONFIGURATION
# ==========================================
# The order of datasets to fine-tune on
DATASET_SEQUENCE = ["train", "dev1", "dev2", "dev3"]

# 1. Global Grid (Used for the first dataset to find rough area)
GLOBAL_BM25_GRID = {
    "k1": [0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0],
    "b": [0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0]
}
GLOBAL_PL2_GRID = {
    "c": [0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
}

# 2. Refinement Settings
REFINE_STEPS = 5
REFINE_RADIUS_K1 = 0.4
REFINE_RADIUS_B = 0.2
REFINE_RADIUS_C = 2.0

# 3. System Resources
# UPDATED: Increased to 60 workers to utilize 120 CPU cores.
# The 500GB heap allocation ensures each worker has enough RAM for its index instance.
MAX_WORKERS = 60

# Thread Local Storage to hold the Index instance per thread
thread_local = threading.local()

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
    if 'query_id' in queries_df.columns:
        queries_df = queries_df.rename(columns={'query_id': 'qid'})
    if 'text' in queries_df.columns:
        queries_df = queries_df.rename(columns={'text': 'query'})
    queries_df['qid'] = queries_df['qid'].astype(str)
    queries_df = pt.rewrite.tokenise()(queries_df)
    return queries_df

def generate_refined_grid(center, radius, steps, min_val=0.01):
    start = max(min_val, center - radius)
    end = center + radius
    grid = np.linspace(start, end, steps)
    return sorted(list(set(np.round(grid, 2))))

def save_trec_run(model, topics, run_dir, filename):
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, filename)
    pt.io.write_results(model.transform(topics), output_path)

def save_chart(fig, chart_dir, filename):
    os.makedirs(chart_dir, exist_ok=True)
    output_path = os.path.join(chart_dir, filename)
    fig.savefig(output_path)
    plt.close(fig)

# ==========================================
# CORE EVALUATION LOGIC
# ==========================================

def get_thread_index(index_path):
    """
    Returns a thread-local instance of the index.
    This ensures each thread has its own file pointers, preventing race conditions.
    """
    if not hasattr(thread_local, "index"):
        # print(f"DEBUG: Opening new index instance for thread {threading.get_ident()}")
        thread_local.index = pt.IndexFactory.of(index_path)
    return thread_local.index

def evaluate_single_config(index_path, topics, qrels, model_type, params, target_metric):
    """
    Runs a single configuration safely using a thread-local index.
    """
    try:
        # Get the index specific to this thread
        index = get_thread_index(index_path)
        
        if model_type == "BM25":
            # Using new pt.terrier.Retriever API
            model = pt.terrier.Retriever(index, wmodel="BM25", 
                                       controls={"c": params['k1'], "bm25.b": params['b']}, 
                                       verbose=False)
        elif model_type == "PL2":
            model = pt.terrier.Retriever(index, wmodel="PL2", 
                                       controls={"c": params['c']}, 
                                       verbose=False)
        
        # 1. Retrieve
        res = model.transform(topics)
        
        # 2. Evaluate
        metrics = pt.Utils.evaluate(res, qrels, metrics=[target_metric])
        score = metrics[target_metric]
        
        return {**params, "score": score}
        
    except Exception as e:
        # Catch errors to prevent killing the whole pool, but print them
        print(f"Error evaluating {model_type} {params}: {e}")
        return {**params, "score": -1.0}

def run_grid_search(index_path, topics, qrels, model_type, param_grid_list, target_metric, desc):
    """
    Executes the grid search in parallel.
    Passes 'index_path' instead of 'index' object to allow thread-local instantiation.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(evaluate_single_config, index_path, topics, qrels, model_type, p, target_metric): p 
            for p in param_grid_list
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="cfg"):
            result = future.result()
            results.append(result)
            
    return results

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    env = load_env()
    index_path = env['paths']['index_path']
    
    # Verify index exists before starting
    if not os.path.exists(index_path):
        print(f"Error: Index path not found: {index_path}")
        return

    run_dir = env['paths']['sparse_run_directory']
    eval_dir = env['paths']['evaluation_directory']
    best_params_path = env['paths']['best_params_path']
    chart_dir = os.path.join(eval_dir, "charts")
    
    metrics = env['eval_metrics']
    target_metric = metrics[0]

    current_best_bm25 = {"k1": None, "b": None}
    current_best_pl2 = {"c": None}
    
    history_log = []

    print(f"Starting Sequential Optimization: {DATASET_SEQUENCE}")
    print(f"Target Metric: {target_metric}")
    print(f"Parallel Workers: {MAX_WORKERS}")

    for dataset in DATASET_SEQUENCE:
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset}")
        print(f"{'='*60}")

        # Load Data
        q_path, qrel_path = get_dataset_paths(env, dataset)
        topics = load_queries(q_path)
        qrels = pt.io.read_qrels(qrel_path)

        # -----------------------------------------------
        # 1. Prepare Search Grids
        # -----------------------------------------------
        bm25_configs = []
        pl2_configs = []

        if current_best_bm25["k1"] is None:
            print(">> Phase: Global Search")
            for k1 in GLOBAL_BM25_GRID["k1"]:
                for b in GLOBAL_BM25_GRID["b"]:
                    bm25_configs.append({"k1": k1, "b": b})
            
            for c in GLOBAL_PL2_GRID["c"]:
                pl2_configs.append({"c": c})
        else:
            print(f">> Phase: Refined Search (Center: BM25 {current_best_bm25}, PL2 {current_best_pl2})")
            k1_vals = generate_refined_grid(current_best_bm25["k1"], REFINE_RADIUS_K1, REFINE_STEPS)
            b_vals = generate_refined_grid(current_best_bm25["b"], REFINE_RADIUS_B, REFINE_STEPS, min_val=0.1)
            c_vals = generate_refined_grid(current_best_pl2["c"], REFINE_RADIUS_C, REFINE_STEPS)
            
            for k1 in k1_vals:
                for b in b_vals:
                    bm25_configs.append({"k1": k1, "b": b})
            
            for c in c_vals:
                pl2_configs.append({"c": c})

        # -----------------------------------------------
        # 2. Optimize BM25
        # -----------------------------------------------
        print(f"\n--- Optimizing BM25 ({len(bm25_configs)} configs) ---")
        bm25_results = run_grid_search(index_path, topics, qrels, "BM25", bm25_configs, target_metric, f"BM25 Grid ({dataset})")
        
        df_bm25 = pd.DataFrame(bm25_results)
        # Handle cases where all might have failed
        if df_bm25['score'].max() < 0:
            print("CRITICAL ERROR: All BM25 runs failed. Check logs.")
            return

        best_bm25_row = df_bm25.loc[df_bm25['score'].idxmax()]
        
        local_best_bm25 = {
            "k1": best_bm25_row['k1'],
            "b": best_bm25_row['b'],
            "score": best_bm25_row['score']
        }
        print(f"Best BM25 on {dataset}: {local_best_bm25}")
        
        current_best_bm25["k1"] = local_best_bm25["k1"]
        current_best_bm25["b"] = local_best_bm25["b"]

        # Save Best Run (Main thread can open its own index safely)
        main_index = pt.IndexFactory.of(index_path)
        best_bm25_model = pt.terrier.Retriever(main_index, wmodel="BM25", 
                                         controls={"c": local_best_bm25['k1'], "bm25.b": local_best_bm25['b']})
        save_trec_run(best_bm25_model, topics, run_dir, 
                      f"{dataset}_BEST_bm25_k1-{local_best_bm25['k1']:.2f}_b-{local_best_bm25['b']:.2f}.run")

        # Plot Heatmap
        pivot = df_bm25.pivot(index="k1", columns="b", values="score")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".4f")
        plt.title(f"BM25 Optimization ({dataset})\nRefined around previous best")
        save_chart(plt.gcf(), chart_dir, f"{dataset}_bm25_optimization.png")

        # -----------------------------------------------
        # 3. Optimize PL2
        # -----------------------------------------------
        print(f"\n--- Optimizing PL2 ({len(pl2_configs)} configs) ---")
        pl2_results = run_grid_search(index_path, topics, qrels, "PL2", pl2_configs, target_metric, f"PL2 Grid ({dataset})")
        
        df_pl2 = pd.DataFrame(pl2_results)
        if df_pl2['score'].max() < 0:
            print("CRITICAL ERROR: All PL2 runs failed. Check logs.")
            return

        best_pl2_row = df_pl2.loc[df_pl2['score'].idxmax()]
        
        local_best_pl2 = {
            "c": best_pl2_row['c'],
            "score": best_pl2_row['score']
        }
        print(f"Best PL2 on {dataset}: {local_best_pl2}")

        current_best_pl2["c"] = local_best_pl2["c"]

        # Save Best Run
        best_pl2_model = pt.terrier.Retriever(main_index, wmodel="PL2", controls={"c": local_best_pl2['c']})
        save_trec_run(best_pl2_model, topics, run_dir, 
                      f"{dataset}_BEST_pl2_c-{local_best_pl2['c']:.2f}.run")

        # Plot Line
        plt.figure(figsize=(10, 6))
        df_pl2_sorted = df_pl2.sort_values(by="c")
        plt.plot(df_pl2_sorted['c'], df_pl2_sorted['score'], marker='o', linestyle='-')
        plt.title(f"PL2 Optimization ({dataset})")
        plt.xlabel("Parameter c")
        plt.ylabel(target_metric)
        plt.grid(True)
        save_chart(plt.gcf(), chart_dir, f"{dataset}_pl2_optimization.png")

        # -----------------------------------------------
        # 4. Log History
        # -----------------------------------------------
        history_log.append({
            "dataset": dataset,
            "model": "BM25",
            "k1": local_best_bm25['k1'],
            "b": local_best_bm25['b'],
            "param_c": None,
            "metric": target_metric,
            "value": local_best_bm25["score"]
        })
        history_log.append({
            "dataset": dataset,
            "model": "PL2",
            "k1": None, 
            "b": None,
            "param_c": local_best_pl2['c'],
            "metric": target_metric,
            "value": local_best_pl2["score"]
        })

    print(f"\nSaving optimization history to {best_params_path}")
    pd.DataFrame(history_log).to_csv(best_params_path, index=False)
    print("Optimization Complete.")

if __name__ == "__main__":
    main()