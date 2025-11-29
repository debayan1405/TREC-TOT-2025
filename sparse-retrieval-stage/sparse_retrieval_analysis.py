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

# Initialize PyTerrier
# UPDATED: Added -Dindex.meta.data-source=fileinmem to JAVA_OPTS.
# This forces the JVM to load the metadata structure into RAM immediately upon index load.
os.environ["JAVA_OPTS"] = "-Xmx100g -Xms16g -Dindex.meta.data-source=fileinmem"

if not pt.java.started():
    pt.java.init()

# ==========================================
# CONFIGURATION
# ==========================================
# Mapping your specific folder structure
REWRITER_CONFIG = {
    "llama": {
        "folder": "llama-query-run-files",
        "prefix": "llama",
        "query_key": "llama"
    },
    "mistral": {
        "folder": "mistral-query-run-files",
        "prefix": "mistral",
        "query_key": "mistral"
    },
    "qwen": {
        "folder": "qwen-query-run-files",
        "prefix": "qwen",
        "query_key": "qwen"
    },
    "summarized": {
        "folder": "summarized-query-run-files",
        "prefix": "summarized",
        "query_key": "summarized" 
    },
    "original": {
        "folder": "original-query-run-files",
        "prefix": "original_query",
        "query_key": "original"
    }
}

DATASETS_EXISTING = ["train", "dev1", "dev2"]
TARGET_TEST_SET = "dev3"

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
        "train": "train_qrels_path",
        "dev1": "dev1_qrels_path",
        "dev2": "dev2_qrels_path",
        "dev3": "dev3_qrels_path",
        "test": "test_qrels_path"
    }
    return pt.io.read_qrels(env['paths'][key_map[dataset]])

def parse_filename_params(filename, model):
    """
    Extracts k1, b for BM25 or c for PL2 from the filename.
    """
    params = {}
    parts = filename.replace(".run", "").split("_")
    
    for p in parts:
        if "-" in p:
            k, v = p.split("-")
            try:
                params[k] = float(v)
            except:
                pass
    return params

def calculate_custom_metrics(res, qrels):
    """
    Wraps ir_measures.calc_aggregate directly.
    """
    measures = [
        nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10
    ]
    
    # Rename columns to match ir_measures expectations
    qrels_std = qrels.rename(columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"})
    res_std = res.rename(columns={"qid": "query_id", "docno": "doc_id"})
    
    # Run calculation directly
    return calc_aggregate(measures, qrels_std, res_std)

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    env = load_env()
    
    # Setup Paths
    base_run_dir = env['paths']['sparse_run_directory']
    eval_dir = env['paths']['evaluation_directory']
    chart_dir = os.path.join(eval_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    
    # Load Index (for Dev3 run)
    index_path = env['paths']['index_path']
    print(f"Loading Index for Dev3 runs: {index_path}")
    
    # Redundant but harmless check to ensure property is set
    # This might work for some versions: index.setProperty("index.meta.data-source", "fileinmem")
    pt.ApplicationSetup.setProperty("index.meta.data-source", "fileinmem")
    index = pt.IndexFactory.of(index_path)

    all_results = []
    best_configurations = []

    print(f"{'='*60}")
    print("STARTING REWRITER TOURNAMENT & GENERALIZATION TEST")
    print(f"{'='*60}")

    for rewriter, config in REWRITER_CONFIG.items():
        print(f"\n>>> Processing Rewriter: {rewriter.upper()}")
        
        rewriter_dir = os.path.join(base_run_dir, config['folder'])
        prefix = config['prefix']
        
        # We need to track the 'dev2' params to use them for 'dev3'
        dev2_params_bm25 = None
        dev2_params_pl2 = None
        
        # 1. Analyze Existing Runs (Train, Dev1, Dev2)
        for model in ["bm25", "pl2"]:
            for dataset in DATASETS_EXISTING:
                # Construct expected filename pattern
                pattern = os.path.join(rewriter_dir, f"{prefix}_{dataset}_BEST_{model}_*.run")
                files = glob.glob(pattern)
                
                if not files:
                    print(f"  [WARN] Missing run file: {pattern}")
                    continue
                
                # Take the most recent one
                files.sort(key=os.path.getmtime, reverse=True)
                run_file = files[0]
                
                # Parse Params
                params = parse_filename_params(os.path.basename(run_file), model)
                
                # Store dev2 params for later use
                if dataset == "dev2":
                    if model == "bm25": dev2_params_bm25 = params
                    if model == "pl2": dev2_params_pl2 = params

                # Load & Evaluate
                qrels = get_qrels(env, dataset)
                res = pt.io.read_results(run_file)
                
                metrics = calculate_custom_metrics(res, qrels)
                
                record = {
                    "rewriter": rewriter,
                    "dataset": dataset,
                    "model": model.upper(),
                    "params": str(params),
                    "filename": os.path.basename(run_file)
                }
                # Flatten metrics
                for k, v in metrics.items():
                    record[str(k)] = v
                
                all_results.append(record)
                ndcg_10_val = metrics.get(nDCG@10, 0)
                print(f"  {dataset.ljust(6)} {model.upper().ljust(4)} | nDCG@10: {ndcg_10_val:.4f}")

        # 2. Run Pseudo-Test on Dev3
        print(f"  --- Running Dev3 Pseudo-Test ---")
        
        # Load Dev3 Queries
        q_key = config['query_key']
        if 'query_variations' in env['paths'] and q_key in env['paths']['query_variations']:
             q_path = env['paths']['query_variations'][q_key]['dev3']
        else:
             q_path = f"../queries/dev3-{q_key}.jsonl" 
             
        if not os.path.exists(q_path):
             print(f"  [ERR] Query file not found: {q_path}")
             continue

        topics = pd.read_json(q_path, lines=True)
        if 'query_id' in topics.columns: topics = topics.rename(columns={'query_id': 'qid'})
        if 'text' in topics.columns: topics = topics.rename(columns={'text': 'query'})
        topics['qid'] = topics['qid'].astype(str)
        topics = pt.rewrite.tokenise()(topics)
        
        qrels_dev3 = get_qrels(env, "dev3")

        # Execute BM25 Dev3
        if dev2_params_bm25:
            print(f"  Running BM25 Retrieval on Dev3...")
            bm25_model = pt.terrier.Retriever(index, wmodel="BM25", 
                                            controls={"c": dev2_params_bm25.get('k1'), "bm25.b": dev2_params_bm25.get('b')},
                                            verbose=True)
            res_bm25 = bm25_model.transform(topics)
            
            # Save Run
            run_name = f"{prefix}_dev3_BEST_bm25_k1-{dev2_params_bm25.get('k1'):.2f}_b-{dev2_params_bm25.get('b'):.2f}.run"
            save_path = os.path.join(rewriter_dir, run_name)
            pt.io.write_results(res_bm25, save_path)
            
            # Evaluate
            metrics_bm25 = calculate_custom_metrics(res_bm25, qrels_dev3)
            
            record = {
                "rewriter": rewriter,
                "dataset": "dev3",
                "model": "BM25",
                "params": str(dev2_params_bm25),
                "filename": run_name
            }
            for k, v in metrics_bm25.items(): record[str(k)] = v
            all_results.append(record)
            
            best_configurations.append({
                "rewriter": rewriter,
                "model": "BM25",
                "params": dev2_params_bm25,
                "dev3_path": save_path,
                "ndcg_10": metrics_bm25.get(nDCG@10),
                "success_10": metrics_bm25.get(Success@10)
            })
            print(f"  dev3   BM25 | nDCG@10: {metrics_bm25.get(nDCG@10, 0):.4f} (Gen. Test)")

        # Execute PL2 Dev3
        if dev2_params_pl2:
            print(f"  Running PL2 Retrieval on Dev3...")
            pl2_model = pt.terrier.Retriever(index, wmodel="PL2", 
                                           controls={"c": dev2_params_pl2.get('c')},
                                           verbose=True)
            res_pl2 = pl2_model.transform(topics)
            
            # Save Run
            run_name = f"{prefix}_dev3_BEST_pl2_c-{dev2_params_pl2.get('c'):.2f}.run"
            save_path = os.path.join(rewriter_dir, run_name)
            pt.io.write_results(res_pl2, save_path)
            
            # Evaluate
            metrics_pl2 = calculate_custom_metrics(res_pl2, qrels_dev3)
            
            record = {
                "rewriter": rewriter,
                "dataset": "dev3",
                "model": "PL2",
                "params": str(dev2_params_pl2),
                "filename": run_name
            }
            for k, v in metrics_pl2.items(): record[str(k)] = v
            all_results.append(record)
            
            best_configurations.append({
                "rewriter": rewriter,
                "model": "PL2",
                "params": dev2_params_pl2,
                "dev3_path": save_path,
                "ndcg_10": metrics_pl2.get(nDCG@10),
                "success_10": metrics_pl2.get(Success@10)
            })
            print(f"  dev3   PL2  | nDCG@10: {metrics_pl2.get(nDCG@10, 0):.4f} (Gen. Test)")

    # 3. Save Consolidated CSVs
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(eval_dir, "sparse_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull Benchmark Metrics saved to: {csv_path}")
    
    config_df = pd.DataFrame(best_configurations)
    best_config_path = os.path.join(eval_dir, "best_sparse_configurations.csv")
    config_df.to_csv(best_config_path, index=False)
    print(f"Best Configurations saved to: {best_config_path}")

    # 4. Visualizations
    if not df.empty:
        plt.figure(figsize=(14, 6))
        
        # Plot 1: BM25 Comparison
        bm25_df = df[df['model'] == "BM25"]
        if not bm25_df.empty:
            plt.subplot(1, 2, 1)
            y_metric = str(nDCG@10) 
            sns.barplot(data=bm25_df, x="dataset", y=y_metric, hue="rewriter", palette="viridis")
            plt.title("BM25 Performance Stability")
            plt.ylim(0, 1.0)
        
        # Plot 2: PL2 Comparison
        pl2_df = df[df['model'] == "PL2"]
        if not pl2_df.empty:
            plt.subplot(1, 2, 2)
            y_metric = str(nDCG@10)
            sns.barplot(data=pl2_df, x="dataset", y=y_metric, hue="rewriter", palette="magma")
            plt.title("PL2 Performance Stability")
            plt.ylim(0, 1.0)
        
        plt.tight_layout()
        chart_path = os.path.join(chart_dir, "rewriter_stability_ndcg10.png")
        plt.savefig(chart_path)
        print(f"Chart saved: {chart_path}")
        
        # Plot 3: Success@10 for Dev3 Only
        dev3_df = df[df['dataset'] == 'dev3']
        y_metric_succ = str(Success@10)
        if not dev3_df.empty and y_metric_succ in dev3_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=dev3_df, x="rewriter", y=y_metric_succ, hue="model")
            plt.title("Dev3 Pseudo-Test: Success@10 (Recall)")
            plt.ylim(0, 1.0)
            chart_path2 = os.path.join(chart_dir, "dev3_success_comparison.png")
            plt.savefig(chart_path2)
            print(f"Chart saved: {chart_path2}")

if __name__ == "__main__":
    main()