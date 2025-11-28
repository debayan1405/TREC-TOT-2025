import pyterrier as pt
import pandas as pd
import json
import os
import sys

if not pt.started():
    pt.init(mem="32g")

# ==========================================
# CONFIGURATION
# ==========================================
# NORMALLY, you would parse the CSV from the previous step to get these automatically.
# For this script, I will allow you to plug them in, or I can read the 'best_sparse_params.csv'
# But since we are comparing rewriters now, let's look at the 'rewriter_comparison_metrics.csv'

METRIC_TO_OPTIMIZE = "ndcg_cut_100"

# ==========================================
# HELPERS
# ==========================================
def load_env(env_path="env.json"):
    with open(env_path, 'r') as f:
        return json.load(f)

def get_best_config_from_csv(csv_path):
    """
    Reads the rewriter comparison CSV and finds the row with max score on dev2.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Analysis CSV not found at {csv_path}. Run analyze_rewriter_performance.py first.")
    
    df = pd.read_csv(csv_path)
    # Filter for dev2
    dev2_df = df[df['dataset'] == 'dev2']
    
    if dev2_df.empty:
        raise ValueError("No dev2 data in CSV.")
        
    best_row = dev2_df.loc[dev2_df[METRIC_TO_OPTIMIZE].idxmax()]
    return best_row

def parse_params(param_str):
    """
    Converts string 'k1-1.2_b-0.75' into dict {'k1': 1.2, 'b': 0.75}
    Converts string 'c-10.5' into dict {'c': 10.5}
    """
    params = {}
    parts = param_str.split("_")
    for p in parts:
        key, val = p.split("-")
        params[key] = float(val)
    return params

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    env = load_env()
    index = pt.IndexFactory.of(env['paths']['index_path'])
    
    eval_dir = env['paths']['evaluation_directory']
    analysis_csv = os.path.join(eval_dir, "rewriter_comparison_metrics.csv")
    run_dir = env['paths']['sparse_run_directory']
    
    print("--- Dev3 Pseudo-Test Validation ---")
    
    # 1. Identify Best Config
    try:
        best_cfg = get_best_config_from_csv(analysis_csv)
    except Exception as e:
        print(f"Error: {e}")
        return

    rewriter = best_cfg['rewriter']
    model_name = best_cfg['model']
    param_str = best_cfg['params']
    dev2_score = best_cfg[METRIC_TO_OPTIMIZE]
    
    print(f"Winning Config (from Dev2):")
    print(f"  Rewriter: {rewriter}")
    print(f"  Model:    {model_name}")
    print(f"  Params:   {param_str}")
    print(f"  Dev2 {METRIC_TO_OPTIMIZE}: {dev2_score:.4f}")
    
    # 2. Load Dev3 Data for that Rewriter
    # We access the path via the new env structure
    query_path = env['paths']['query_variations'][rewriter]['dev3']
    qrel_path = env['paths']['dev3_qrels_path']
    
    print(f"Loading Dev3 Queries: {query_path}")
    topics = pt.io.read_topics(query_path, format='singlelinejson') # Assuming JSONL
    # Normalize QID
    topics['qid'] = topics['qid'].astype(str) 
    topics = pt.rewrite.tokenise()(topics)
    
    qrels = pt.io.read_qrels(qrel_path)
    
    # 3. Instantiate Model
    params = parse_params(param_str)
    
    if model_name == "BM25":
        # Handle param mapping
        controls = {"c": params.get('k1', 1.2), "bm25.b": params.get('b', 0.75)}
        model = pt.BatchRetrieve(index, wmodel="BM25", controls=controls)
    elif model_name == "PL2":
        controls = {"c": params.get('c', 1.0)}
        model = pt.BatchRetrieve(index, wmodel="PL2", controls=controls)
        
    # 4. Run on Dev3
    print("Running retrieval on Dev3...")
    res = model.transform(topics)
    
    # 5. Evaluate
    metrics = env['eval_metrics']
    eval_res = pt.Utils.evaluate(res, qrels, metrics=metrics)
    
    dev3_score = eval_res[METRIC_TO_OPTIMIZE]
    print(f"  Dev3 {METRIC_TO_OPTIMIZE}: {dev3_score:.4f}")
    
    # 6. Analyze Generalization
    delta = dev3_score - dev2_score
    pct_change = (delta / dev2_score) * 100.0
    
    print(f"\nGeneralization Analysis:")
    print(f"  Change: {delta:+.4f} ({pct_change:+.2f}%)")
    
    if pct_change < -10.0:
        print("  WARNING: Significant performance drop (>10%). Possible overfitting to Dev2.")
    elif pct_change > 0:
        print("  POSITIVE: Performance improved on unseen data.")
    else:
        print("  STABLE: Performance is consistent with tuning set.")

    # Save Dev3 Run
    run_filename = f"{rewriter}_dev3_pseudo_test_{model_name}_{param_str}.run"
    save_path = os.path.join(run_dir, run_filename)
    pt.io.write_results(res, save_path)
    print(f"Saved Dev3 run file to: {save_path}")

if __name__ == "__main__":
    main()