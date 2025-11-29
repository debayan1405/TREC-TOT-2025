import pyterrier as pt
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Initialize PyTerrier with large heap
import os
os.environ["JAVA_OPTS"] = "-Xmx600g -Xms200g"
if not pt.java.started():
    pt.java.init()

# ==========================================
# RESUME CONFIGURATION
# ==========================================
# Only run for dev2 since train and dev1 are done
DATASET_SEQUENCE = ["dev2"]

# PREVIOUS BEST VALUES (From Dev1)
# The script will use these to center the grid search for Dev2
STARTING_BEST_BM25 = {"k1": 0.01, "b": 0.65}
STARTING_BEST_PL2 = {"c": 2.50}

# Refinement Settings
REFINE_STEPS = 5
REFINE_RADIUS_K1 = 0.4
REFINE_RADIUS_B = 0.2
REFINE_RADIUS_C = 2.0

# System Resources
MAX_WORKERS = 1

# ==========================================
# HELPERS
# ==========================================

def load_env(env_path="env.json"):
    # Adjusted path handling to look in current dir or fallback
    if not os.path.exists(env_path):
        # Fallback for colab/different envs if needed, though usually ./env.json
        env_path = "./env.json" 
    with open(env_path, 'r') as f:
        return json.load(f)

def get_dataset_paths(env, dataset_key):
    # Hardcoded paths for emergency resume as requested
    if dataset_key == "dev2":
        return "./rewritten-queries/qwen_dev-2_rewritten_queries.jsonl", "./qrel/dev2-2025-qrel.txt"

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

def evaluate_single_config(index, topics_dict, qrels, model_type, params, target_metric):
    try:
        topics = pd.DataFrame(topics_dict)
        
        if model_type == "BM25":
            model = pt.terrier.Retriever(index, wmodel="BM25", 
                                       controls={"c": params['k1'], "bm25.b": params['b']}, 
                                       verbose=False)
        elif model_type == "PL2":
            model = pt.terrier.Retriever(index, wmodel="PL2", 
                                       controls={"c": params['c']}, 
                                       verbose=False)
        
        res = model.transform(topics)
        
        # Evaluate using ir_measures
        import ir_measures
        from ir_measures import calc_aggregate
        
        qrels_formatted = qrels.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})
        res_formatted = res.rename(columns={'qid': 'query_id', 'docno': 'doc_id'})
        
        metric_map = {
            "ndcg_cut_10": "nDCG@10",
            "ndcg_cut_100": "nDCG@100", # Added this as it might be used
            "ndcg_cut_1000": "nDCG@1000", 
            "recall_1000": "R@1000",
            "P_10": "P@10"
        }
        metric_name = metric_map.get(target_metric, target_metric)
        metric_obj = ir_measures.parse_measure(metric_name)
        metrics_dict = calc_aggregate([metric_obj], qrels_formatted, res_formatted)
        score = metrics_dict[metric_obj]
        
        return {**params, "score": score}
        
    except Exception as e:
        print(f"Error evaluating {model_type} {params}: {e}")
        return {**params, "score": -1.0}

def run_grid_search(index, topics, qrels, model_type, param_grid_list, target_metric, desc):
    results = []
    topics_dict = topics.to_dict('list')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(evaluate_single_config, index, topics_dict, qrels, model_type, p, target_metric): p 
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
    
    if not os.path.exists(index_path):
        print(f"Error: Index path not found: {index_path}")
        return

    # -------------------------------------------------------------------------
    # OPTIMIZATION: Load Meta Index into RAM
    # This resolves the "Structure meta reading data file directly from disk" warning
    # and significantly improves lookup speed.
    # -------------------------------------------------------------------------
    print("Configuring index to load MetaIndex into RAM (fileinmem)...")
    pt.ApplicationSetup.setProperty("index.meta.data-source", "fileinmem")

    print(f"Loading index from: {index_path}")
    index = pt.IndexFactory.of(index_path)

    run_dir = env['paths']['sparse_run_directory']
    eval_dir = env['paths']['evaluation_directory']
    chart_dir = os.path.join(eval_dir, "charts")
    
    metrics = env['eval_metrics']
    target_metric = metrics[0]

    # Initialize with the Last Known Best Values from Dev1
    current_best_bm25 = STARTING_BEST_BM25
    current_best_pl2 = STARTING_BEST_PL2
    
    print(f"Resuming Optimization on: {DATASET_SEQUENCE}")
    print(f"Initialized with Previous Best (Dev1): BM25 {current_best_bm25}, PL2 {current_best_pl2}")
    print(f"Target Metric: {target_metric}")

    for dataset in DATASET_SEQUENCE:
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset}")
        print(f"{'='*60}")

        q_path, qrel_path = get_dataset_paths(env, dataset)
        topics = load_queries(q_path)
        qrels = pt.io.read_qrels(qrel_path)

        # -----------------------------------------------
        # 1. Prepare Search Grids (REFINED ONLY)
        # -----------------------------------------------
        bm25_configs = []
        pl2_configs = []

        print(f">> Phase: Refined Search (Center: BM25 {current_best_bm25}, PL2 {current_best_pl2})")
        
        # Refined Grid Generation
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
        bm25_results = run_grid_search(index, topics, qrels, "BM25", bm25_configs, target_metric, f"BM25 Grid ({dataset})")
        
        df_bm25 = pd.DataFrame(bm25_results)
        if df_bm25['score'].max() < 0:
            print("CRITICAL ERROR: All BM25 runs failed.")
            return

        best_bm25_row = df_bm25.loc[df_bm25['score'].idxmax()]
        
        local_best_bm25 = {
            "k1": best_bm25_row['k1'],
            "b": best_bm25_row['b'],
            "score": best_bm25_row['score']
        }
        print(f"Best BM25 on {dataset}: {local_best_bm25}")
        
        # Save Best Run
        best_bm25_model = pt.terrier.Retriever(index, wmodel="BM25", 
                                         controls={"c": local_best_bm25['k1'], "bm25.b": local_best_bm25['b']},
                                         verbose=True)
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
        pl2_results = run_grid_search(index, topics, qrels, "PL2", pl2_configs, target_metric, f"PL2 Grid ({dataset})")
        
        df_pl2 = pd.DataFrame(pl2_results)
        if df_pl2['score'].max() < 0:
            print("CRITICAL ERROR: All PL2 runs failed.")
            return

        best_pl2_row = df_pl2.loc[df_pl2['score'].idxmax()]
        local_best_pl2 = {
            "c": best_pl2_row['c'],
            "score": best_pl2_row['score']
        }
        print(f"Best PL2 on {dataset}: {local_best_pl2}")

        # Save Best Run
        best_pl2_model = pt.terrier.Retriever(index, wmodel="PL2", controls={"c": local_best_pl2['c']}, verbose=True)
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

    print("Optimization Complete.")

if __name__ == "__main__":
    main()