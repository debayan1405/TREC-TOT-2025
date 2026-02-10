import pyterrier as pt
import pandas as pd
import json
import os
import re
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# SYSTEM SETUP
# ==========================================
# Initialize PyTerrier with large heap
os.environ["JAVA_OPTS"] = "-Xmx600g -Xms200g"
if not pt.java.started():
    pt.java.init()

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent

# 1. Absolute Paths (Large external assets)
INDEX_PATH = "/media/12TB/shared/datasets/indices/trec-tot-2025/trec-tot-2025-pyterrier-index"
QRELS_ROOT = (SCRIPT_DIR / "../qrel").resolve()

# 2. Relative Paths (Project structure)
# Updated to direct outputs to sparse-retrieval-stage-1 subfolders
PATHS = {
    "rewritten_dir": (SCRIPT_DIR / "../rewritten-queries").resolve(),
    "original_dir": (SCRIPT_DIR / "../original_queries").resolve(),
    # Output paths updated as requested
    "run_output_dir": (SCRIPT_DIR / "runs/sparse-retrieval-stage-1").resolve(),
    "charts_dir": (SCRIPT_DIR / "evaluations/sparse-retrieval-stage-1/charts").resolve(),
    "best_params_log": (SCRIPT_DIR / "evaluations/sparse-retrieval-stage-1/best_sparse_params.csv").resolve()
}

# 3. QREL Mappings
QRELS_FILES = {
    "train": os.path.join(QRELS_ROOT, "train-2025-qrel.txt"),
    "dev1": os.path.join(QRELS_ROOT, "dev1-2025-qrel.txt"),
    "dev2": os.path.join(QRELS_ROOT, "dev2-2025-qrel.txt"),
    "dev3": os.path.join(QRELS_ROOT, "dev3-2025-qrel.txt"),
    "test": os.path.join(QRELS_ROOT, "test-2025-qrel.txt")
}

# 4. Optimization Settings
# Order of datasets to tune on sequentially
DATASET_SEQUENCE = ["train", "dev1", "dev2"]

# Metrics - UPDATED TO RECALL@1000
TARGET_METRIC = "recall_1000"

# Grid Search Configs
GLOBAL_BM25_GRID = {
    "k1": [0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0],
    "b": [0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0]
}
GLOBAL_PL2_GRID = {
    "c": [0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
}

REFINE_STEPS = 5
REFINE_RADIUS_K1 = 0.4
REFINE_RADIUS_B = 0.2
REFINE_RADIUS_C = 2.0

MAX_WORKERS = 1  # Single thread to prevent index decompression errors

# ==========================================
# FILE PARSING LOGIC
# ==========================================
def scan_query_variants():
    """
    Scans rewritten-queries and original-queries directories.
    Returns a dictionary structured as:
    {
        "variant_name": {
            "train": Path(...),
            "dev1": Path(...),
            ...
        }
    }
    """
    variants = {}

    # 1. Scan Rewritten Directory
    # Pattern A: Rewritten -> {model}_{dataset}_rewritten_queries.jsonl
    # Pattern B: Summarized -> {dataset}_summarized_{model}.jsonl
    if PATHS["rewritten_dir"].exists():
        for f in PATHS["rewritten_dir"].glob("*.jsonl"):
            name = f.name
            
            # Check Pattern A (Rewritten)
            match_rw = re.match(r"^(?P<model>.*?)_(?P<dataset>.*?)_rewritten_queries\.jsonl$", name)
            if match_rw:
                model = match_rw.group("model")
                dataset = match_rw.group("dataset")
                variant_key = f"rewritten-{model}"
                
                if variant_key not in variants: variants[variant_key] = {}
                variants[variant_key][dataset] = f
                continue

            # Check Pattern B (Summarized)
            match_sum = re.match(r"^(?P<dataset>.*?)_summarized_(?P<model>.*?)\.jsonl$", name)
            if match_sum:
                model = match_sum.group("model")
                dataset = match_sum.group("dataset")
                variant_key = f"summarized-{model}"
                
                if variant_key not in variants: variants[variant_key] = {}
                variants[variant_key][dataset] = f
                continue

    # 2. Scan Original Directory
    # Pattern: {dataset}-original.jsonl
    if PATHS["original_dir"].exists():
        for f in PATHS["original_dir"].glob("*-original.jsonl"):
            name = f.name
            match_orig = re.match(r"^(?P<dataset>.*?)-original\.jsonl$", name)
            if match_orig:
                dataset = match_orig.group("dataset")
                variant_key = "original"
                if variant_key not in variants: variants[variant_key] = {}
                variants[variant_key][dataset] = f

    return variants

# ==========================================
# DATA LOADING
# ==========================================
def load_queries(query_path):
    """Load queries from JSONL and format for PyTerrier."""
    queries_df = pd.read_json(query_path, lines=True)
    
    # Normalize columns
    if 'query_id' in queries_df.columns:
        queries_df = queries_df.rename(columns={'query_id': 'qid'})
    if 'text' in queries_df.columns:
        queries_df = queries_df.rename(columns={'text': 'query'})
        
    # Ensure string types
    queries_df['qid'] = queries_df['qid'].astype(str)
    
    # PyTerrier Tokenization
    queries_df = pt.rewrite.tokenise()(queries_df)
    return queries_df

def generate_refined_grid(center, radius, steps, min_val=0.01):
    start = max(min_val, center - radius)
    end = center + radius
    grid = np.linspace(start, end, steps)
    return sorted(list(set(np.round(grid, 2))))

def save_trec_run(model, topics, run_dir, filename):
    os.makedirs(run_dir, exist_ok=True)
    output_path = run_dir / filename
    pt.io.write_results(model.transform(topics), str(output_path))

def save_chart(fig, chart_dir, filename):
    os.makedirs(chart_dir, exist_ok=True)
    output_path = chart_dir / filename
    fig.savefig(str(output_path))
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
        
        # Retrieve
        res = model.transform(topics)
        
        # Evaluate using ir_measures
        import ir_measures
        from ir_measures import calc_aggregate
        
        # Map DataFrame columns to ir_measures expectations
        qrels_formatted = qrels.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})
        res_formatted = res.rename(columns={'qid': 'query_id', 'docno': 'doc_id'})
        
        # Normalize Metric Name
        metric_map = {
            "ndcg_cut_10": "nDCG@10",
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
            results.append(future.result())
            
    return results

# ==========================================
# MAIN LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Sparse Retrieval Optimizer (BM25 & PL2)")
    parser.add_argument("--variant", type=str, required=True, 
                        help="Name of the query variant to optimize (e.g., 'rewritten-llama', 'original', 'summarized-mistral')")
    args = parser.parse_args()

    # 1. Setup Index
    if not os.path.exists(INDEX_PATH):
        print(f"Error: Index path not found: {INDEX_PATH}")
        return

    print(f"Loading index from: {INDEX_PATH}")
    index = pt.IndexFactory.of(INDEX_PATH, properties={"index.meta.data-source": "fileinmem"})
    print(f"Index loaded: {index.getCollectionStatistics().getNumberOfDocuments()} docs")

    # 2. Select Query Variant
    variants = scan_query_variants()
    if not variants:
        print("No query files found in ../rewritten-queries or ../original_queries")
        return

    selected_variant_name = args.variant
    if selected_variant_name not in variants:
        print(f"Error: Variant '{selected_variant_name}' not found.")
        print("Available variants:")
        for v in sorted(variants.keys()):
            print(f" - {v}")
        sys.exit(1)
        
    selected_variant_files = variants[selected_variant_name]
    
    # 3. Validation
    missing_datasets = [ds for ds in DATASET_SEQUENCE if ds not in selected_variant_files]
    if missing_datasets:
        print(f"Warning: Selected variant '{selected_variant_name}' is missing files for: {missing_datasets}")
        confirm = input("Continue anyway? (y/n): ")
        if confirm.lower() != 'y':
            return
        # Filter sequence to only available datasets
        active_sequence = [ds for ds in DATASET_SEQUENCE if ds in selected_variant_files]
    else:
        active_sequence = DATASET_SEQUENCE

    # 4. Optimization Loop
    print(f"\nStarting Optimization for: {selected_variant_name}")
    print(f"Metric: {TARGET_METRIC} (Recall@1000)")
    print(f"Sequence: {active_sequence}")
    
    current_best_bm25 = {"k1": None, "b": None}
    current_best_pl2 = {"c": None}
    history_log = []

    for dataset in active_sequence:
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset}")
        print(f"{'='*60}")

        # Load Data
        query_file = selected_variant_files[dataset]
        qrel_file = QRELS_FILES.get(dataset)
        
        if not qrel_file or not os.path.exists(qrel_file):
            print(f"Skipping {dataset}: QREL file not found at {qrel_file}")
            continue

        topics = load_queries(str(query_file))
        qrels = pt.io.read_qrels(qrel_file)

        # -----------------------------------------------
        # Prepare Grids
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
        # Optimize BM25
        # -----------------------------------------------
        print(f"\n--- Optimizing BM25 ({len(bm25_configs)} configs) ---")
        bm25_results = run_grid_search(index, topics, qrels, "BM25", bm25_configs, TARGET_METRIC, f"BM25 Grid")
        
        df_bm25 = pd.DataFrame(bm25_results)
        if df_bm25.empty or df_bm25['score'].max() < 0:
            print("CRITICAL ERROR: BM25 runs failed.")
            continue

        best_bm25_row = df_bm25.loc[df_bm25['score'].idxmax()]
        local_best_bm25 = {"k1": best_bm25_row['k1'], "b": best_bm25_row['b'], "score": best_bm25_row['score']}
        
        print(f"Best BM25 on {dataset}: {local_best_bm25}")
        current_best_bm25.update({"k1": local_best_bm25["k1"], "b": local_best_bm25["b"]})

        # Save Run
        best_bm25_model = pt.terrier.Retriever(index, wmodel="BM25", 
                                         controls={"c": local_best_bm25['k1'], "bm25.b": local_best_bm25['b']})
        run_filename = f"{dataset}_BEST_bm25_k1-{local_best_bm25['k1']:.2f}_b-{local_best_bm25['b']:.2f}_{selected_variant_name}.run"
        save_trec_run(best_bm25_model, topics, PATHS["run_output_dir"], run_filename)

        # Plot Heatmap
        pivot = df_bm25.pivot(index="k1", columns="b", values="score")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".4f")
        plt.title(f"BM25 Optimization ({dataset} - {selected_variant_name})\nMetric: {TARGET_METRIC}")
        save_chart(plt.gcf(), PATHS["charts_dir"], f"{dataset}_bm25_{selected_variant_name}_{TARGET_METRIC}.png")

        # -----------------------------------------------
        # Optimize PL2
        # -----------------------------------------------
        print(f"\n--- Optimizing PL2 ({len(pl2_configs)} configs) ---")
        pl2_results = run_grid_search(index, topics, qrels, "PL2", pl2_configs, TARGET_METRIC, f"PL2 Grid")
        
        df_pl2 = pd.DataFrame(pl2_results)
        if df_pl2.empty or df_pl2['score'].max() < 0:
            print("CRITICAL ERROR: PL2 runs failed.")
            continue

        best_pl2_row = df_pl2.loc[df_pl2['score'].idxmax()]
        local_best_pl2 = {"c": best_pl2_row['c'], "score": best_pl2_row['score']}
        
        print(f"Best PL2 on {dataset}: {local_best_pl2}")
        current_best_pl2["c"] = local_best_pl2["c"]

        # Save Run
        best_pl2_model = pt.terrier.Retriever(index, wmodel="PL2", controls={"c": local_best_pl2['c']})
        run_filename = f"{dataset}_BEST_pl2_c-{local_best_pl2['c']:.2f}_{selected_variant_name}.run"
        save_trec_run(best_pl2_model, topics, PATHS["run_output_dir"], run_filename)

        # Plot Line
        plt.figure(figsize=(10, 6))
        df_pl2_sorted = df_pl2.sort_values(by="c")
        plt.plot(df_pl2_sorted['c'], df_pl2_sorted['score'], marker='o', linestyle='-')
        plt.title(f"PL2 Optimization ({dataset} - {selected_variant_name})\nMetric: {TARGET_METRIC}")
        plt.xlabel("Parameter c")
        plt.grid(True)
        save_chart(plt.gcf(), PATHS["charts_dir"], f"{dataset}_pl2_{selected_variant_name}_{TARGET_METRIC}.png")

        # Log History
        history_log.append({
            "variant": selected_variant_name,
            "dataset": dataset,
            "model": "BM25",
            "k1": local_best_bm25['k1'],
            "b": local_best_bm25['b'],
            "param_c": None,
            "metric": TARGET_METRIC,
            "value": local_best_bm25["score"]
        })
        history_log.append({
            "variant": selected_variant_name,
            "dataset": dataset,
            "model": "PL2",
            "k1": None, 
            "b": None,
            "param_c": local_best_pl2['c'],
            "metric": TARGET_METRIC,
            "value": local_best_pl2["score"]
        })

    # Save History
    if history_log:
        os.makedirs(PATHS["best_params_log"].parent, exist_ok=True)
        # Append to existing log if exists, else create new
        mode = 'a' if PATHS["best_params_log"].exists() else 'w'
        header = not PATHS["best_params_log"].exists()
        pd.DataFrame(history_log).to_csv(PATHS["best_params_log"], mode=mode, header=header, index=False)
        print(f"\nOptimization history updated at {PATHS['best_params_log']}")
    else:
        print("\nNo history to save.")

if __name__ == "__main__":
    main()