import pyterrier as pt
import pandas as pd
import json
import os
import argparse
import sys
import ir_measures
from ir_measures import calc_aggregate, nDCG, R, P, RR, Success

# ==========================================
# CONFIGURATION
# ==========================================
# Metrics mapping
MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]

# ==========================================
# HELPERS
# ==========================================

def load_env(env_path="env.json"):
    if not os.path.exists(env_path):
        # Fallback relative location
        env_path = os.path.join(os.path.dirname(__file__), "env.json")
    
    if not os.path.exists(env_path):
        # Fallback to colab path if script is running there
        env_path = "/content/env.json"
        
    if not os.path.exists(env_path):
        print(f"Warning: env.json not found at {env_path}. Some dataset shortcuts might fail.")
        return {}
        
    with open(env_path, 'r') as f:
        return json.load(f)

def get_qrels_path(env, dataset_name):
    """
    Resolves dataset name (e.g., 'test', 'dev1') to a QREL path from env.json.
    """
    if not env:
        return None
        
    key_map = {
        "train": "train_qrels_path",
        "dev1": "dev1_qrels_path",
        "dev2": "dev2_qrels_path",
        "dev3": "dev3_qrels_path",
        "test": "test_qrels_path"
    }
    
    if dataset_name in key_map:
        path_key = key_map[dataset_name]
        return env['paths'].get(path_key)
    return None

def evaluate(run_path, qrels_path):
    print(f"Loading Run File: {run_path}")
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Run file not found: {run_path}")
    
    run_df = pt.io.read_results(run_path)
    
    print(f"Loading QREL File: {qrels_path}")
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"QREL file not found: {qrels_path}")
        
    qrels_df = pt.io.read_qrels(qrels_path)
    
    # Standardize column names for ir_measures
    # PyTerrier: qid, docno, score / label
    # ir_measures: query_id, doc_id, score / relevance
    
    q_std = qrels_df.rename(columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"})
    r_std = run_df.rename(columns={"qid": "query_id", "docno": "doc_id"})
    
    # Calculate
    print("Calculating metrics...")
    metrics = calc_aggregate(MEASURES, q_std, r_std)
    
    return metrics

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a TREC run file against QRELs.")
    
    # Required: Run File
    parser.add_argument("run_file", type=str, help="Path to the .run file you want to evaluate.")
    
    # Optional: QREL File OR Dataset Name
    parser.add_argument("--qrels", type=str, help="Path to the .txt QREL file.")
    parser.add_argument("--dataset", type=str, choices=["train", "dev1", "dev2", "dev3", "test"], 
                        help="Name of the dataset (e.g., 'test') to automatically fetch QRELs from env.json.")
    
    parser.add_argument("--save", type=str, help="Path to save the metrics as a JSON file.", default=None)

    args = parser.parse_args()
    
    # Logic to resolve QRELs
    qrels_path = args.qrels
    
    if not qrels_path:
        if args.dataset:
            env = load_env()
            qrels_path = get_qrels_path(env, args.dataset)
            if not qrels_path:
                print(f"Error: Could not find QREL path for dataset '{args.dataset}' in env.json.")
                sys.exit(1)
        else:
            print("Error: You must provide either --qrels <path> or --dataset <name>.")
            sys.exit(1)

    try:
        metrics = evaluate(args.run_file, qrels_path)
        
        print("\n" + "="*40)
        print("EVALUATION RESULTS")
        print("="*40)
        
        # Sort keys for cleaner output (ir_measures keys are objects, convert to str)
        sorted_keys = sorted(metrics.keys(), key=lambda x: str(x))
        
        results_json = {}
        
        for m in sorted_keys:
            val = metrics[m]
            print(f"{str(m):<15} : {val:.4f}")
            results_json[str(m)] = val
            
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(results_json, f, indent=4)
            print(f"\nMetrics saved to: {args.save}")
            
    except Exception as e:
        print(f"\nAn error occurred during evaluation:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# Running Command: python evaluate_run_file.py my_run_file.run --dataset test --save results.json